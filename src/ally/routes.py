import asyncio
import uuid
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, BackgroundTasks, Body, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import PlainTextResponse

from ally import pipeline
from ally.audio import InvalidAudio, probe
from ally.db import now
from ally.export import signoff_stats, to_json, to_markdown
from ally.risk import has_alert, new_flag_id
from ally.schemas import (
    Language,
    NotePatch,
    RiskFlag,
    RiskFlagCreate,
    RiskFlagPatch,
    Role,
    Segment,
    SegmentPatch,
    Session,
    SessionCreated,
    SessionNote,
    SessionSummary,
    SignOff,
    Transcript,
    WebhookResult,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])
ALLOWED_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm", ".amr", ".mp4"}
MB = 1024 * 1024


def _get(request: Request, session_id: str) -> dict:
    row = request.app.state.store.get(session_id)
    if row is None:
        raise HTTPException(404, "session not found")
    return row


def _draft(request: Request, session_id: str) -> dict:
    row = _get(request, session_id)
    if row["status"] != "draft":
        raise HTTPException(409, f"session is {row['status']}; edits are only allowed in draft")
    return row


def _signed(request: Request, session_id: str) -> Session:
    row = _get(request, session_id)
    if row["status"] != "signed":
        raise HTTPException(409, "export requires a signed-off session")
    return Session(**row)


@router.post("", status_code=202, response_model=SessionCreated, summary="Upload a call recording")
async def create_session(
    request: Request,
    background: BackgroundTasks,
    audio: Annotated[
        UploadFile, File(description="Call recording: mp3, wav, m4a, aac, ogg, opus, flac, webm, amr")
    ],
    language: Annotated[
        Language,
        Form(
            description="auto = detect from clips across the whole call (default). Or set it if known: "
            "hi = Hindi-English, mr = Marathi(-English), ml = Malayalam(-English), en = English"
        ),
    ] = "auto",
):
    state = request.app.state
    settings = state.settings
    ext = Path(audio.filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            415, f"unsupported file type '{ext or 'none'}'; allowed: {', '.join(sorted(ALLOWED_EXT))}"
        )
    if state.llm is None:
        raise HTTPException(503, "LLM_API_KEY not configured")
    if language == "auto":
        if "gemini" not in state.speech:
            raise HTTPException(503, "language 'auto' needs the gemini speech provider (set LLM_API_KEY)")
    elif (provider := settings.speech_routes.get(language)) not in state.speech:
        raise HTTPException(503, f"speech provider '{provider}' for language '{language}' not configured")

    session_id = uuid.uuid4().hex[:12]
    path = pipeline.audio_path(settings, session_id, audio.filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        # ponytail: Starlette spools the whole body before this runs; cap request size at the proxy in prod.
        with path.open("wb") as out:
            while chunk := await audio.read(MB):
                size += len(chunk)
                if size > settings.max_upload_mb * MB:
                    raise HTTPException(413, f"file exceeds {settings.max_upload_mb} MB")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "empty file")
        meta = await asyncio.to_thread(probe, path)
    except InvalidAudio as e:
        path.unlink(missing_ok=True)
        raise HTTPException(415, str(e)) from e
    except HTTPException:
        path.unlink(missing_ok=True)
        raise
    state.store.create(session_id, language, Path(audio.filename).name, meta.model_dump())
    background.add_task(pipeline.run, state, session_id)
    return SessionCreated(id=session_id, status="queued")


@router.get("", response_model=list[SessionSummary], summary="List sessions, newest first")
def list_sessions(request: Request):
    return request.app.state.store.list_all()


@router.get("/{session_id}", response_model=Session, summary="Status, transcript, note and risk flags")
def get_session(request: Request, session_id: str):
    return _get(request, session_id)


@router.patch("/{session_id}/note", response_model=SessionNote, summary="Edit note fields (draft only)")
def edit_note(request: Request, session_id: str, patch: NotePatch):
    store = request.app.state.store
    row = _draft(request, session_id)
    changes = patch.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(400, "no fields to update")
    note = SessionNote(**{**row["note"], **changes})
    for field, value in changes.items():
        store.add_edit(session_id, "note", field, row["note"].get(field), value)
    store.update(session_id, note=note.model_dump())
    return note


@router.patch(
    "/{session_id}/segments/{segment_id}", response_model=Segment, summary="Edit a segment's English text"
)
def edit_segment(request: Request, session_id: str, segment_id: str, patch: SegmentPatch):
    store = request.app.state.store
    row = _draft(request, session_id)
    seg = next((s for s in row["transcript"]["segments"] if s["id"] == segment_id), None)
    if seg is None:
        raise HTTPException(404, "segment not found")
    store.add_edit(session_id, "segment", segment_id, seg["text_en"], patch.text_en)
    seg["text_en"] = patch.text_en
    store.update(session_id, transcript=row["transcript"])
    return seg


@router.patch("/{session_id}/speakers", response_model=Transcript, summary="Re-map speakers to roles")
def edit_speakers(
    request: Request,
    session_id: str,
    roles: Annotated[dict[int, Role], Body(examples=[{"0": "client", "1": "counselor"}])],
):
    store = request.app.state.store
    row = _draft(request, session_id)
    segments = row["transcript"]["segments"]
    if unknown := set(roles) - {s["speaker"] for s in segments}:
        raise HTTPException(422, f"unknown speakers: {sorted(unknown)}")
    before = {s["speaker"]: s["role"] for s in segments if s["speaker"] in roles}
    for s in segments:
        s["role"] = roles.get(s["speaker"], s["role"])
    store.add_edit(session_id, "speakers", "roles", before, roles)
    store.update(session_id, transcript=row["transcript"])
    return row["transcript"]


def _save_flags(store, session_id: str, flags: list[dict]) -> None:
    store.update(session_id, risk_flags=flags, risk_alert=has_alert([RiskFlag(**f) for f in flags]))


@router.patch(
    "/{session_id}/risk-flags/{flag_id}", response_model=RiskFlag, summary="Review or edit a risk flag"
)
def edit_flag(request: Request, session_id: str, flag_id: str, patch: RiskFlagPatch):
    store = request.app.state.store
    row = _draft(request, session_id)
    flag = next((f for f in row["risk_flags"] if f["id"] == flag_id), None)
    if flag is None:
        raise HTTPException(404, "risk flag not found")
    changes = patch.model_dump(exclude_unset=True, exclude_none=True)
    if not changes:
        raise HTTPException(400, "no fields to update")
    store.add_edit(session_id, "risk_flag", flag_id, {k: flag.get(k) for k in changes}, changes)
    flag.update(changes)
    _save_flags(store, session_id, row["risk_flags"])
    return flag


@router.post(
    "/{session_id}/risk-flags", status_code=201, response_model=RiskFlag, summary="Add a missed risk flag"
)
def add_flag(request: Request, session_id: str, body: RiskFlagCreate):
    store = request.app.state.store
    row = _draft(request, session_id)
    flag = RiskFlag(id=new_flag_id(), source="counselor", status="confirmed", **body.model_dump())
    store.add_edit(session_id, "risk_flag", flag.id, None, flag.model_dump())
    _save_flags(store, session_id, [*row["risk_flags"], flag.model_dump()])
    return flag


@router.post("/{session_id}/sign-off", response_model=Session, summary="Lock the session as final")
def sign_off(request: Request, session_id: str, body: SignOff):
    store = request.app.state.store
    row = _draft(request, session_id)
    if any(f["severity"] == "high" and f["status"] == "pending" for f in row["risk_flags"]):
        raise HTTPException(409, "confirm or dismiss every high-severity risk flag before signing off")
    stats = signoff_stats(row["draft_note"], row["note"], store.edit_count(session_id))
    store.update(session_id, status="signed", signed_by=body.counselor, signed_at=now(), signoff=stats)
    return store.get(session_id)


@router.get(
    "/{session_id}/export",
    summary="Export a signed session",
    responses={200: {"content": {"application/json": {}, "text/markdown": {}}}},
)
def export(
    request: Request,
    session_id: str,
    fmt: Annotated[Literal["json", "markdown"], Query(alias="format")] = "json",
    include_transcript: bool = False,
):
    session = _signed(request, session_id)
    if fmt == "markdown":
        return PlainTextResponse(to_markdown(session, include_transcript), media_type="text/markdown")
    return to_json(session, include_transcript)


@router.post(
    "/{session_id}/export/webhook", response_model=WebhookResult, summary="POST the JSON export to the CRM"
)
async def export_webhook(request: Request, session_id: str, include_transcript: bool = False):
    state = request.app.state
    if not state.settings.export_webhook_url:
        raise HTTPException(503, "EXPORT_WEBHOOK_URL not configured")
    session = _signed(request, session_id)
    try:
        resp = await state.http.post(
            state.settings.export_webhook_url, json=to_json(session, include_transcript), timeout=10
        )
    except httpx.HTTPError as e:
        raise HTTPException(502, f"webhook unreachable ({type(e).__name__})") from e
    if resp.status_code >= 400:
        raise HTTPException(502, f"webhook returned HTTP {resp.status_code}")
    return WebhookResult(status_code=resp.status_code)
