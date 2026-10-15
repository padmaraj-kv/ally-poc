import json
from difflib import SequenceMatcher

from ally.schemas import RiskFlag, Session


def stamp(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _visible_flags(session: Session) -> list[RiskFlag]:
    return [f for f in session.risk_flags if f.status != "dismissed"]


def to_json(session: Session, include_transcript: bool = False) -> dict:
    data = {
        "session": {
            "id": session.id,
            "language": session.language,
            "created_at": session.created_at,
            "duration_s": session.audio_meta.duration_s if session.audio_meta else None,
            "signed_by": session.signed_by,
            "signed_at": session.signed_at,
        },
        "note": session.note.model_dump() if session.note else None,
        "risk_flags": [f.model_dump() for f in _visible_flags(session)],
    }
    if include_transcript and session.transcript:
        data["transcript"] = [
            s.model_dump(include={"id", "start", "end", "role", "text", "text_en"})
            for s in session.transcript.segments
        ]
    return data


def to_markdown(session: Session, include_transcript: bool = False) -> str:
    note = session.note
    segments = session.transcript.segments if session.transcript else []
    starts = {s.id: s.start for s in segments}

    def refs(ids: list[str]) -> str:
        times = [stamp(starts[i]) for i in ids if i in starts]
        return f" _({', '.join(times)})_" if times else ""

    def section(title: str, rows: list[str]) -> list[str]:
        return [f"## {title}", "", *(rows or ["- None recorded"]), ""]

    out = [
        "# Counselling session note",
        "",
        f"- Session: {session.id}",
        f"- Language: {session.language}",
        f"- Recorded: {session.created_at}",
        f"- Signed by: {session.signed_by or 'not signed'} ({session.signed_at or '-'})",
        "",
        "## Summary",
        "",
        note.summary,
        "",
    ]
    out += section(
        "Themes", [f"- **{t.label.replace('_', ' ')}**: {t.detail}{refs(t.segment_ids)}" for t in note.themes]
    )
    out += section("Client concerns", [f"- {i.text}{refs(i.segment_ids)}" for i in note.concerns])
    out += section("Progress", [f"- {i.text}{refs(i.segment_ids)}" for i in note.progress])
    out += section("Follow-ups", [f"- {i.text}{refs(i.segment_ids)}" for i in note.follow_ups])
    out += section("Tone", [f"- {i.text}{refs(i.segment_ids)}" for i in note.tone])
    out += section(
        "Risk flags",
        [
            f"- **{f.category.replace('_', ' ')}** ({f.severity}, {f.status}): {f.rationale}"
            + (f' "{f.quote}"' if f.quote else "")
            + (f" Counselor note: {f.counselor_note}" if f.counselor_note else "")
            + refs(f.segment_ids)
            for f in _visible_flags(session)
        ],
    )
    out += ["## Tags", "", ", ".join(note.tags) or "None", ""]
    if include_transcript:
        out += ["## Transcript (English)", ""]
        out += [f"[{stamp(s.start)}] {s.role.title()}: {s.text_en or s.text}" for s in segments]
    return "\n".join(out).rstrip() + "\n"


def signoff_stats(draft: dict, final: dict, edit_count: int) -> dict:
    changed = sorted(k for k in draft if draft[k] != final.get(k))
    ratio = SequenceMatcher(
        None,
        json.dumps(draft, sort_keys=True, ensure_ascii=False),
        json.dumps(final, sort_keys=True, ensure_ascii=False),
    ).ratio()
    return {"changed_fields": changed, "similarity": round(ratio, 3), "edit_count": edit_count}
