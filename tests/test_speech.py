import json

import httpx
import pytest

from ally.schemas import AudioMeta
from ally.speech import (
    AudioSegment,
    AudioTranscript,
    DeepgramSpeech,
    GeminiSpeech,
    SpeechError,
    clock,
    parse_deepgram,
)

MONO = {
    "results": {
        "utterances": [
            {"start": 3.0, "end": 4.5, "speaker": 1, "channel": 0, "transcript": "मुझे बहुत tension है"},
            {"start": 0.2, "end": 2.0, "speaker": 0, "channel": 0, "transcript": "Hello, kaise ho?"},
            {"start": 5.0, "end": 5.1, "speaker": 0, "channel": 0, "transcript": "  "},
        ]
    }
}
STEREO = {
    "results": {
        "utterances": [
            {"start": 1.0, "end": 2.0, "channel": 1, "speaker": 0, "transcript": "ठीक है"},
            {"start": 0.0, "end": 0.9, "channel": 0, "speaker": 0, "transcript": "Hello"},
        ]
    }
}


def meta(stereo: bool) -> AudioMeta:
    return AudioMeta(
        container="wav",
        codec="pcm_s16le",
        channels=2 if stereo else 1,
        sample_rate=8000,
        duration_s=5,
        stereo=stereo,
    )


def test_parse_mono_sorted_ids_and_speakers():
    t = parse_deepgram(MONO, "hi", stereo=False)
    assert [(s.id, s.speaker, s.channel) for s in t.segments] == [("u1", 0, None), ("u2", 1, None)]
    assert t.segments[0].text == "Hello, kaise ho?"
    assert t.provider == "deepgram"


def test_parse_stereo_speaker_is_channel():
    t = parse_deepgram(STEREO, "hi", stereo=True)
    assert [(s.speaker, s.channel) for s in t.segments] == [(0, 0), (1, 1)]


async def test_deepgram_request_params(tmp_path):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = request.url
        seen["headers"] = request.headers
        return httpx.Response(200, json=MONO)

    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        dg = DeepgramSpeech("k", "https://api.in.deepgram.com", "nova-3", http)
        await dg.transcribe(path, meta(stereo=False), "hi")
        params = seen["url"].params
        assert seen["url"].host == "api.in.deepgram.com"
        assert params["language"] == "multi"
        assert params["diarize_model"] == "latest"
        assert "multichannel" not in params
        assert seen["headers"]["authorization"] == "Token k"
        assert seen["headers"]["content-type"] == "audio/wav"

        await dg.transcribe(path, meta(stereo=True), "mr")
        params = seen["url"].params
        assert params["language"] == "mr"
        assert params["multichannel"] == "true"
        assert "diarize_model" not in params


async def test_deepgram_http_error(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF")
    transport = httpx.MockTransport(lambda r: httpx.Response(401, json={"err_msg": "bad key"}))
    async with httpx.AsyncClient(transport=transport) as http:
        with pytest.raises(SpeechError, match="HTTP 401"):
            await DeepgramSpeech("k", "https://x", "nova-3", http).transcribe(path, meta(False), "hi")


def test_clock():
    assert clock("01:05") == 65
    assert clock("1:02:03") == 3723
    assert clock("7.5") == 7.5
    assert clock("garbage") == 0


class AudioLLM:
    def __init__(self):
        self.user = None

    async def structured(self, system, user, schema, **kw):
        self.user = user
        return AudioTranscript(
            segments=[
                AudioSegment(speaker=0, start="00:00", end="00:04", text="നമസ്കാരം", text_en="Hello"),
                AudioSegment(
                    speaker=1, start="00:05", end="00:09", text="എനിക്ക് വിഷമം ഉണ്ട്", text_en="I feel sad"
                ),
            ]
        )


async def test_gemini_speech_sends_audio_and_fills_english(tmp_path):
    path = tmp_path / "a.gemini.mp3"
    path.write_bytes(b"ID3fake")
    llm = AudioLLM()
    t = await GeminiSpeech(llm).transcribe(path, meta(False), "ml")
    part = next(p for p in llm.user if p["type"] == "input_audio")
    assert part["input_audio"]["format"] == "mp3"
    assert json.dumps(part)  # serialisable
    assert [(s.id, s.start, s.end, s.text_en) for s in t.segments] == [
        ("u1", 0, 4, "Hello"),
        ("u2", 5, 9, "I feel sad"),
    ]
    assert t.provider == "gemini"


async def test_gemini_rejects_oversized_audio(tmp_path):
    path = tmp_path / "big.gemini.mp3"
    path.write_bytes(b"0" * (15 * 1024 * 1024))
    with pytest.raises(SpeechError, match="20 MB"):
        await GeminiSpeech(AudioLLM()).transcribe(path, meta(False), "ml")
