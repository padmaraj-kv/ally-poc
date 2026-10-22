import json

import httpx
import pytest

from ally.schemas import AudioMeta
from ally.speech import (
    AudioSegment,
    AudioTranscript,
    ClipLanguage,
    DeepgramSpeech,
    GeminiSpeech,
    LanguageScan,
    SarvamSpeech,
    SpeechError,
    clock,
    parse_deepgram,
    parse_sarvam,
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
        self.kw = kw
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


async def test_gemini_uses_long_timeout(tmp_path):
    path = tmp_path / "a.gemini.mp3"
    path.write_bytes(b"ID3fake")
    llm = AudioLLM()
    await GeminiSpeech(llm).transcribe(path, meta(False), "ml")
    assert llm.kw["timeout"] >= 600


async def test_gemini_chunks_long_calls_and_offsets_times(tmp_path, monkeypatch):
    cuts = []
    monkeypatch.setattr("ally.audio.clip_mp3", lambda path, start, dur: cuts.append((start, dur)) or b"ID3")
    path = tmp_path / "long.gemini.mp3"
    path.write_bytes(b"ID3fake")
    llm = AudioLLM()
    long_call = meta(False).model_copy(update={"duration_s": 25 * 60})
    t = await GeminiSpeech(llm).transcribe(path, long_call, "ml")
    assert cuts == [(0, 600), (600, 600), (1200, 600)]
    assert [s.id for s in t.segments] == ["u1", "u2", "u3", "u4", "u5", "u6"]
    assert [s.start for s in t.segments] == [0, 5, 600, 605, 1200, 1205]
    assert [s.chunk for s in t.segments] == [0, 0, 1, 1, 2, 2]
    assert "part 3 of 3" in llm.user[0]["text"]


class ScanLLM:
    def __init__(self, scan):
        self.scan, self.users = scan, []

    async def structured(self, system, user, schema, **kw):
        self.users.append(user)
        return self.scan


async def test_detect_languages_samples_whole_call_and_counts_clips(monkeypatch):
    cuts = []
    monkeypatch.setattr("ally.audio.clip_mp3", lambda path, start, dur: cuts.append((start, dur)) or b"ID3")
    scan = LanguageScan(
        clips=[ClipLanguage(clip=1, languages=["en"]), ClipLanguage(clip=2, languages=["en", "ml"])]
    )
    llm = ScanLLM(scan)
    hour = meta(False).model_copy(update={"duration_s": 3600})
    heard = await GeminiSpeech(llm).detect_languages("a.wav", hour)
    assert len(cuts) == 60 and cuts[-1] == (3540, 10)
    assert [sum(p["type"] == "input_audio" for p in u) for u in llm.users] == [15, 15, 15, 15]
    assert heard == {"en": 8, "ml": 4}  # the same 2-clip scan is replayed for each of the 4 requests


async def test_detect_languages_sends_short_call_whole(monkeypatch):
    cuts = []
    monkeypatch.setattr("ally.audio.clip_mp3", lambda path, start, dur: cuts.append((start, dur)) or b"ID3")
    llm = ScanLLM(LanguageScan(clips=[ClipLanguage(clip=1, languages=["hi", "en"])]))
    await GeminiSpeech(llm).detect_languages("a.wav", meta(False).model_copy(update={"duration_s": 300}))
    assert cuts == [(0, None)]


SARVAM_RESULT = {
    "transcript": "unused",
    "language_code": "ml-IN",
    "diarized_transcript": {
        "entries": [
            {
                "transcript": "Hello, Ally helpline",
                "start_time_seconds": 0.4,
                "end_time_seconds": 2.1,
                "speaker_id": "0",
            },
            {"transcript": "  ", "start_time_seconds": 2.2, "end_time_seconds": 2.3, "speaker_id": "1"},
            {
                "transcript": "എനിക്ക് tension ആണ്",
                "start_time_seconds": 2.5,
                "end_time_seconds": 4.0,
                "speaker_id": "1",
            },
        ]
    },
}


def test_parse_sarvam_diarized_entries():
    t = parse_sarvam(SARVAM_RESULT, "ml")
    assert [(s.id, s.speaker, s.start, s.text) for s in t.segments] == [
        ("u1", 0, 0.4, "Hello, Ally helpline"),
        ("u2", 1, 2.5, "എനിക്ക് tension ആണ്"),
    ]
    assert t.provider == "sarvam" and t.segments[0].text_en is None


def sarvam_transport(calls, state="Completed"):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, request))
        path = request.url.path
        if path == "/speech-to-text/job/v1":
            return httpx.Response(200, json={"job_id": "j1", "job_state": "Accepted"})
        if path.endswith("/upload-files"):
            return httpx.Response(
                200, json={"upload_urls": {"call.sarvam.mp3": {"file_url": "https://blob.test/in"}}}
            )
        if path == "/in":
            return httpx.Response(201)
        if path.endswith("/start"):
            return httpx.Response(200, json={"job_state": "Pending"})
        if path.endswith("/status"):
            return httpx.Response(
                200, json={"job_state": state, "job_details": [{"outputs": [{"file_name": "0.json"}]}]}
            )
        if path.endswith("/download-files"):
            return httpx.Response(
                200, json={"download_urls": {"0.json": {"file_url": "https://blob.test/out"}}}
            )
        if path == "/out":
            return httpx.Response(200, json=SARVAM_RESULT)
        return httpx.Response(404)

    return httpx.MockTransport(handler)


async def test_sarvam_batch_flow(tmp_path):
    path = tmp_path / "call.sarvam.mp3"
    path.write_bytes(b"ID3fake")
    calls = []
    async with httpx.AsyncClient(transport=sarvam_transport(calls)) as http:
        t = await SarvamSpeech("k", "https://api.test", "saaras:v4", http, poll_s=0).transcribe(
            path, meta(False), "ml"
        )
    assert [s.text for s in t.segments] == ["Hello, Ally helpline", "എനിക്ക് tension ആണ്"]
    create = json.loads(calls[0][2].content)["job_parameters"]
    # always auto: a 22 Oct test showed Sarvam follows an en -> ml switch mid-file on its own
    assert create["mode"] == "codemix" and create["language_code"] == "unknown"
    assert create["with_diarization"] is True and create["model"] == "saaras:v4"
    assert calls[0][2].headers["api-subscription-key"] == "k"
    put = next(r for m, p, r in calls if m == "PUT")
    assert put.headers["x-ms-blob-type"] == "BlockBlob" and "api-subscription-key" not in put.headers
    assert put.content == b"ID3fake"


async def test_sarvam_failed_job(tmp_path):
    path = tmp_path / "call.sarvam.mp3"
    path.write_bytes(b"ID3fake")
    async with httpx.AsyncClient(transport=sarvam_transport([], state="Failed")) as http:
        with pytest.raises(SpeechError, match="sarvam job failed"):
            await SarvamSpeech("k", "https://api.test", "saaras:v4", http, poll_s=0).transcribe(
                path, meta(False), "ml"
            )
