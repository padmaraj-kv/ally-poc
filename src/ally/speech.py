import base64
from pathlib import Path

import httpx
from pydantic import BaseModel

from ally import prompts
from ally.audio import mime_type
from ally.schemas import AudioMeta, Segment, Transcript

DEEPGRAM_LANG = {"hi": "multi", "mr": "mr", "en": "en"}
INLINE_LIMIT = 14 * 1024 * 1024  # raw bytes; base64 adds a third, request cap is 20 MB


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

    async def transcribe(self, path: Path, meta: AudioMeta, language: str) -> Transcript:
        data = path.read_bytes()
        if len(data) > INLINE_LIMIT:
            raise SpeechError(
                "audio too large for inline Gemini request (20 MB cap); Files API not implemented"
            )
        user = [
            {"type": "text", "text": prompts.audio_user(language)},
            {
                "type": "input_audio",
                "input_audio": {"data": base64.b64encode(data).decode(), "format": "mp3"},
            },
        ]
        out = await self.llm.structured(prompts.AUDIO_SYSTEM, user, AudioTranscript, effort="low")
        segments = [
            Segment(
                id=f"u{i}",
                speaker=s.speaker,
                start=clock(s.start),
                end=clock(s.end),
                text=s.text.strip(),
                text_en=s.text_en.strip(),
            )
            for i, s in enumerate((s for s in out.segments if s.text.strip()), 1)
        ]
        return Transcript(provider="gemini", language=language, segments=segments)
