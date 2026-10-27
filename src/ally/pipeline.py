import asyncio
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ally import audio, prompts
from ally.audio import InvalidAudio
from ally.config import Settings
from ally.llm import LLMError
from ally.notes import generate_note
from ally.risk import assess_risk, has_alert
from ally.schemas import AudioMeta, Segment
from ally.speech import SpeechError
from ally.translate import translate

log = logging.getLogger(__name__)
KNOWN_ERRORS = (InvalidAudio, LLMError, SpeechError)


class PipelineError(RuntimeError):
    pass


class SpeakerRole(BaseModel):
    speaker: int
    role: Literal["counselor", "client"]


class RoleGuess(BaseModel):
    speakers: list[SpeakerRole]
    confidence: Literal["low", "medium", "high"]
    rationale: str


def audio_path(settings: Settings, session_id: str, filename: str) -> Path:
    return settings.data_dir / "audio" / f"{session_id}{Path(filename).suffix.lower()}"


def main_language(heard: dict[str, int]) -> tuple[str, list[str]]:
    """Clip counts per language -> the call's label: the most-heard Indian language, else English."""
    if not any(lang != "none" for lang in heard):
        raise PipelineError("no speech detected")
    indic = [lang for lang in heard if lang in ("hi", "mr", "ml")]
    language = max(indic, key=lambda lang: heard[lang]) if indic else "en"
    warnings = ["other language heard in some clips; check the transcript"] if "other" in heard else []
    return language, warnings


def pick_provider(settings: Settings, duration_s: float) -> str:
    return (
        settings.long_call_speech
        if duration_s > settings.long_call_minutes * 60
        else settings.speech_provider
    )


async def assign_roles(llm, segments: list[Segment], counselor_channel: int) -> list[str]:
    if any(s.channel is not None for s in segments):
        for s in segments:
            s.role = "counselor" if s.speaker == counselor_channel else "client"
        return []
    chunks = sorted({s.chunk for s in segments})
    groups = [[s for s in segments if s.chunk == c] for c in chunks]
    results = await asyncio.gather(*(_label_roles(llm, g, len(groups) > 1) for g in groups))
    if len(groups) > 1:
        # speaker numbers restart in every chunk; the role is what carries across
        for s in segments:
            s.speaker = {"counselor": 0, "client": 1}.get(s.role, 10 * (s.chunk + 1) + s.speaker)
    return list(dict.fromkeys(w for r in results for w in r))


async def _label_roles(llm, segments: list[Segment], chunked: bool) -> list[str]:
    where = f" in minutes {int(segments[0].start // 60)}-{int(segments[-1].end // 60)}" if chunked else ""
    speakers = sorted({s.speaker for s in segments})
    if len(speakers) < 2:
        return [f"only one speaker detected{where}; roles left unknown"]
    guess = await llm.structured(
        prompts.ROLES_SYSTEM, prompts.roles_user(role_sample(segments)), RoleGuess, effort="low"
    )
    # diarization can split one person into several numbers (Sarvam does on a language switch)
    roles = {r.speaker: r.role for r in guess.speakers if r.speaker in speakers}
    if "counselor" not in roles.values():
        roles = {sp: "counselor" if sp == segments[0].speaker else "client" for sp in speakers}
    for s in segments:
        s.role = roles.get(s.speaker, "unknown")
    warnings = [f"speaker {sp} not labelled{where}; check speakers" for sp in speakers if sp not in roles]
    if guess.confidence == "low":
        warnings.append(f"counselor/client roles uncertain{where}; check speakers")
    return warnings


def role_sample(segments: list[Segment], first: int = 40, per_speaker: int = 8) -> list[Segment]:
    """The opening plus each speaker's first lines, so voices that only appear late are seen too."""
    seen: dict[int, int] = {}
    sample = []
    for i, s in enumerate(segments):
        seen[s.speaker] = seen.get(s.speaker, 0) + 1
        if i < first or seen[s.speaker] <= per_speaker:
            sample.append(s)
    return sample


class Timer:
    def __init__(self):
        self.stage = "start"
        self.timings: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def __call__(self, stage: str):
        self.stage = stage
        t = time.perf_counter()
        yield
        self.timings[stage] = round(time.perf_counter() - t, 2)

    def finish(self) -> dict[str, float]:
        self.timings["total"] = round(time.perf_counter() - self._start, 2)
        return self.timings


async def run(state, session_id: str) -> None:
    store, settings, llm = state.store, state.settings, state.llm
    row = store.get(session_id)
    language = row["language"]
    meta = AudioMeta(**row["audio_meta"])
    src = audio_path(settings, session_id, row["filename"])
    files, warnings, timer = {src}, [], Timer()
    store.update(session_id, status="processing")
    try:
        heard = {}
        if language == "auto":
            # labels the call for the prompts and the record; the engine is chosen by length, not language
            with timer("detect"):
                heard = await state.speech["gemini"].detect_languages(src, meta)
            language, w = main_language(heard)
            warnings += w
            store.update(session_id, language=language)
        route = pick_provider(settings, meta.duration_s)
        if route not in state.speech:
            raise PipelineError(f"speech provider '{route}' not configured")
        provider = state.speech[route]
        with timer("audio"):
            prepared = await asyncio.to_thread(audio.prepare, src, meta, provider.name)
            files.add(prepared)
        with timer("speech"):
            transcript = await provider.transcribe(prepared, meta, language)
            if not transcript.segments:
                raise PipelineError("no speech detected")
            transcript.languages = sorted(lang for lang in heard if lang != "none")
        segments = transcript.segments
        with timer("translate"):
            warnings += await translate(llm, segments, language, settings.translate_batch_size)
        with timer("roles"):
            warnings += await assign_roles(llm, segments, settings.counselor_channel)
        with timer("notes_risk"):
            (note, w_note), (flags, w_risk) = await asyncio.gather(
                generate_note(llm, segments, language), assess_risk(llm, segments, language)
            )
        warnings += w_note + w_risk
        store.update(
            session_id,
            status="draft",
            transcript=transcript.model_dump(),
            draft_note=note.model_dump(),
            note=note.model_dump(),
            risk_flags=[f.model_dump() for f in flags],
            risk_alert=has_alert(flags),
            timings=timer.finish(),
            warnings=warnings,
        )
        log.info("session %s draft in %.1fs", session_id, timer.timings["total"])
    except Exception as e:
        reason = (
            str(e)[:200]
            if isinstance(e, KNOWN_ERRORS + (PipelineError,))
            else f"internal error ({type(e).__name__})"
        )
        log.error("session %s failed at %s: %s", session_id, timer.stage, type(e).__name__)
        store.update(
            session_id,
            status="failed",
            error=f"{timer.stage}: {reason}",
            timings=timer.finish(),
            warnings=warnings,
        )
    finally:
        if not settings.keep_audio:
            for f in files:
                f.unlink(missing_ok=True)
