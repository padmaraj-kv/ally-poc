import asyncio
import base64
import re
import time
from collections import Counter
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel

from ally import audio, prompts
from ally.audio import mime_type
from ally.schemas import AudioMeta, Segment, Transcript

DEEPGRAM_LANG = {"hi": "multi", "mr": "mr", "en": "en"}
SARVAM_LANG = {"hi": "hi-IN", "mr": "mr-IN", "ml": "ml-IN", "en": "en-IN"}
SARVAM_MAX_WAIT_S = 30 * 60
INLINE_LIMIT = 14 * 1024 * 1024  # raw bytes; base64 adds a third, request cap is 20 MB
# ponytail: fixed 10-min cuts can split a sentence; snap to the nearest pause if boundary words go missing.
CHUNK_S = 10 * 60
GEMINI_TIMEOUT_S = 600
CLIP_EVERY_S, CLIP_S = 60, 10
CLIPS_PER_REQUEST = 15  # parallel requests; one 72-clip request took 73 s


class SpeechError(RuntimeError):
    pass


def parse_deepgram(body: dict, language: str, stereo: bool) -> Transcript:
    utterances = sorted(
        body["results"].get("utterances", []), key=lambda u: (u["start"], u.get("channel", 0))
    )
    segments = []
    for u in utterances:
        text = u["transcript"].strip()
        if not text:
            continue
        channel = u.get("channel", 0)
        segments.append(
            Segment(
                id=f"u{len(segments) + 1}",
                speaker=channel if stereo else u.get("speaker", 0),
                channel=channel if stereo else None,
                start=round(u["start"], 2),
                end=round(u["end"], 2),
                text=text,
            )
        )
    return Transcript(provider="deepgram", language=language, segments=segments)


class DeepgramSpeech:
    name = "deepgram"

    def __init__(self, api_key: str, base_url: str, model: str, http: httpx.AsyncClient):
        self.api_key, self.base_url, self.model, self.http = api_key, base_url.rstrip("/"), model, http

    async def transcribe(self, path: Path, meta: AudioMeta, language: str) -> Transcript:
        params = {
            "model": self.model,
            "language": DEEPGRAM_LANG.get(language, language),
            "smart_format": "true",
            "punctuate": "true",
            "utterances": "true",
        }
        params |= {"multichannel": "true"} if meta.stereo else {"diarize_model": "latest"}
        try:
            resp = await self.http.post(
                f"{self.base_url}/v1/listen",
                params=params,
                content=path.read_bytes(),
                headers={"Authorization": f"Token {self.api_key}", "Content-Type": mime_type(path)},
                timeout=600,
            )
        except httpx.HTTPError as e:
            raise SpeechError(f"deepgram unreachable ({type(e).__name__})") from e
        if resp.status_code >= 400:
            raise SpeechError(f"deepgram HTTP {resp.status_code}")
        return parse_deepgram(resp.json(), language, meta.stereo)


class AudioSegment(BaseModel):
    speaker: int
    start: str
    end: str
    text: str
    text_en: str


class AudioTranscript(BaseModel):
    segments: list[AudioSegment]


class ClipLanguage(BaseModel):
    clip: int
    languages: list[Literal["hi", "mr", "ml", "en", "other", "none"]]


class LanguageScan(BaseModel):
    clips: list[ClipLanguage]


def mp3_part(data: bytes) -> dict:
    if len(data) > INLINE_LIMIT:
        raise SpeechError("audio too large for inline Gemini request (20 MB cap); Files API not implemented")
    return {"type": "input_audio", "input_audio": {"data": base64.b64encode(data).decode(), "format": "mp3"}}


def clock(value: str) -> float:
    """'MM:SS', 'H:MM:SS' or plain seconds → seconds; unparseable → 0."""
    try:
        total = 0.0
        for part in value.strip().split(":"):
            total = total * 60 + float(part)
        return total
    except ValueError:
        return 0.0


class GeminiSpeech:
    name = "gemini"

    def __init__(self, llm):
        self.llm = llm

    async def detect_languages(self, path: Path, meta: AudioMeta) -> Counter:
        """Clips across the whole call (a switch can come late) -> clips each language is heard in."""
        if meta.duration_s <= CHUNK_S:
            windows = [(0, None)]
        else:
            windows = [(t, CLIP_S) for t in range(0, int(meta.duration_s), CLIP_EVERY_S)]
        clips = await asyncio.to_thread(lambda: [audio.clip_mp3(path, t, d) for t, d in windows])
        labelled = list(zip(windows, clips, strict=True))
        batches = [labelled[i : i + CLIPS_PER_REQUEST] for i in range(0, len(labelled), CLIPS_PER_REQUEST)]
        scans = await asyncio.gather(*(self._scan(batch) for batch in batches))
        return Counter(lang for scan in scans for c in scan.clips for lang in set(c.languages))

    async def _scan(self, batch: list) -> LanguageScan:
        user = [{"type": "text", "text": prompts.langid_user(len(batch))}]
        for i, ((t, _), data) in enumerate(batch, 1):
            user += [{"type": "text", "text": f"Clip {i} ({t // 60:02d}:{t % 60:02d})"}, mp3_part(data)]
        return await self.llm.structured(
            prompts.LANGID_SYSTEM, user, LanguageScan, effort="low", timeout=GEMINI_TIMEOUT_S
        )

    async def transcribe(self, path: Path, meta: AudioMeta, language: str) -> Transcript:
        starts = list(range(0, int(meta.duration_s), CHUNK_S)) or [0]
        parts = await asyncio.gather(
            *(self._chunk(path, language, i, len(starts), start) for i, start in enumerate(starts))
        )
        segments = [s for part in parts for s in part]
        for i, s in enumerate(segments, 1):
            s.id = f"u{i}"
        return Transcript(provider="gemini", language=language, segments=segments)

    async def _chunk(self, path: Path, language: str, i: int, n: int, start: int) -> list[Segment]:
        if n == 1:
            data = path.read_bytes()
        else:
            data = await asyncio.to_thread(audio.clip_mp3, path, start, CHUNK_S)
        user = [{"type": "text", "text": prompts.audio_user(language, i + 1, n, start)}, mp3_part(data)]
        out = await self.llm.structured(
            prompts.AUDIO_SYSTEM, user, AudioTranscript, effort="low", timeout=GEMINI_TIMEOUT_S
        )
        return [
            Segment(
                id="",
                speaker=s.speaker,
                start=round(start + clock(s.start), 2),
                end=round(start + clock(s.end), 2),
                text=s.text.strip(),
                text_en=s.text_en.strip(),
                chunk=i,
            )
            for s in out.segments
            if s.text.strip()
        ]


def parse_sarvam(body: dict, language: str) -> Transcript:
    segments = []
    for e in (body.get("diarized_transcript") or {}).get("entries") or []:
        text = e["transcript"].strip()
        if not text:
            continue
        speaker = re.sub(r"\D", "", str(e.get("speaker_id", "")))
        segments.append(
            Segment(
                id=f"u{len(segments) + 1}",
                speaker=int(speaker or 0),
                start=round(e["start_time_seconds"], 2),
                end=round(e["end_time_seconds"], 2),
                text=text,
            )
        )
    return Transcript(provider="sarvam", language=language, segments=segments)


class SarvamSpeech:
    """Batch STT: create job, upload to a presigned URL, start, poll, download the result JSON."""

    name = "sarvam"

    def __init__(self, api_key: str, base_url: str, model: str, http: httpx.AsyncClient, poll_s: float = 5):
        self.api_key, self.base_url, self.model = api_key, base_url.rstrip("/"), model
        self.http, self.poll_s = http, poll_s

    async def transcribe(self, path: Path, meta: AudioMeta, language: str) -> Transcript:
        return parse_sarvam(await self.result(path, language), language)

    async def result(self, path: Path, language: str) -> dict:
        jobs = "/speech-to-text/job/v1"
        params = {
            "model": self.model,
            # English words stay in Latin script (like Deepgram multi); translation is our own LLM step
            "mode": "codemix",
            "language_code": SARVAM_LANG.get(language, "unknown"),
            "with_timestamps": True,
            "with_diarization": True,
        }
        job_id = (await self._call("POST", jobs, {"job_parameters": params}))["job_id"]
        urls = await self._call("POST", f"{jobs}/upload-files", {"job_id": job_id, "files": [path.name]})
        await self._call(
            "PUT",
            urls["upload_urls"][path.name]["file_url"],
            content=path.read_bytes(),
            headers={"x-ms-blob-type": "BlockBlob", "Content-Type": mime_type(path)},
        )
        await self._call("POST", f"{jobs}/{job_id}/start")
        deadline = time.monotonic() + SARVAM_MAX_WAIT_S
        while (status := await self._call("GET", f"{jobs}/{job_id}/status"))["job_state"] != "Completed":
            if status["job_state"] == "Failed":
                raise SpeechError("sarvam job failed")
            if time.monotonic() > deadline:
                raise SpeechError("sarvam job timed out")
            await asyncio.sleep(self.poll_s)
        names = [o["file_name"] for d in status.get("job_details", []) for o in d.get("outputs", [])]
        if not names:
            raise SpeechError("sarvam job returned no output")
        links = await self._call("POST", f"{jobs}/download-files", {"job_id": job_id, "files": names[:1]})
        return await self._call("GET", links["download_urls"][names[0]]["file_url"])

    async def _call(self, method: str, url: str, json: dict | None = None, **kw):
        """API paths get the key; presigned blob URLs (absolute) must not."""
        if url.startswith("/"):
            url = self.base_url + url
            kw["headers"] = {"api-subscription-key": self.api_key}
        try:
            resp = await self.http.request(method, url, json=json, timeout=300, **kw)
        except httpx.HTTPError as e:
            raise SpeechError(f"sarvam unreachable ({type(e).__name__})") from e
        if resp.status_code >= 400:
            raise SpeechError(f"sarvam HTTP {resp.status_code}")
        return resp.json() if resp.content else None
