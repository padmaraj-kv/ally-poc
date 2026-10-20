from types import SimpleNamespace

import pytest
from conftest import FakeLLM, FakeSpeech, _seg, wav_bytes

from ally import audio
from ally.config import Settings
from ally.db import Store
from ally.pipeline import PipelineError, assign_roles, audio_path, choose_route, run
from ally.speech import SpeechError


def setup(tmp_path, speech=None, llm=None, language="hi", channels=1, extra=(), **kw):
    settings = Settings(_env_file=None, data_dir=tmp_path, **kw)
    store = Store(tmp_path / "ally.db")
    path = audio_path(settings, "s1", "call.wav")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(wav_bytes(channels=channels))
    store.create("s1", language, "call.wav", audio.probe(path).model_dump())
    speech = speech or FakeSpeech()
    state = SimpleNamespace(
        store=store, settings=settings, speech={p.name: p for p in (speech, *extra)}, llm=llm or FakeLLM()
    )
    return state, path


async def test_happy_path_to_draft(tmp_path):
    state, path = setup(tmp_path)
    await run(state, "s1")
    row = state.store.get("s1")
    assert row["status"] == "draft", row["error"]
    segs = row["transcript"]["segments"]
    assert [s["text_en"] for s in segs] == ["EN u1", "EN u2", "EN u3", "EN u4"]
    assert [s["role"] for s in segs] == ["counselor", "client", "counselor", "client"]
    assert row["note"] == row["draft_note"]
    assert row["note"]["concerns"][0]["segment_ids"] == ["u1"]
    assert row["risk_alert"] is True
    assert [f["source"] for f in row["risk_flags"]] == ["llm"]
    assert {"audio", "speech", "translate", "roles", "notes_risk", "total"} <= set(row["timings"])
    assert not path.exists()


async def test_stereo_roles_follow_channel_without_llm(tmp_path):
    segs = [_seg("u1", 0, "Hello", channel=0), _seg("u2", 1, "Namaste", channel=1)]
    llm = FakeLLM()
    state, _ = setup(tmp_path, speech=FakeSpeech(segs), llm=llm, channels=2, counselor_channel=1)
    await run(state, "s1")
    roles = [s["role"] for s in state.store.get("s1")["transcript"]["segments"]]
    assert roles == ["client", "counselor"]
    assert "RoleGuess" not in llm.calls


async def test_single_speaker_warns_and_keeps_unknown(tmp_path):
    state, _ = setup(tmp_path, speech=FakeSpeech([_seg("u1", 0, "Hello"), _seg("u2", 0, "Haan")]))
    await run(state, "s1")
    row = state.store.get("s1")
    assert row["status"] == "draft"
    assert {s["role"] for s in row["transcript"]["segments"]} == {"unknown"}
    assert "only one speaker detected; roles left unknown" in row["warnings"]


async def test_speech_error_fails_with_stage(tmp_path):
    state, path = setup(tmp_path, speech=FakeSpeech(error=SpeechError("deepgram HTTP 500")))
    await run(state, "s1")
    row = state.store.get("s1")
    assert (row["status"], row["error"]) == ("failed", "speech: deepgram HTTP 500")
    assert not path.exists()


async def test_unexpected_error_message_not_leaked(tmp_path):
    state, _ = setup(tmp_path, speech=FakeSpeech(error=KeyError("client said something private")))
    await run(state, "s1")
    assert state.store.get("s1")["error"] == "speech: internal error (KeyError)"


async def test_no_speech_fails(tmp_path):
    state, _ = setup(tmp_path, speech=FakeSpeech([]))
    await run(state, "s1")
    assert state.store.get("s1")["error"] == "speech: no speech detected"


async def test_english_skips_translation_and_keep_audio(tmp_path):
    llm = FakeLLM()
    state, path = setup(tmp_path, llm=llm, language="en", keep_audio=True)
    await run(state, "s1")
    assert "TranslationBatch" not in llm.calls
    assert path.exists()


async def test_gemini_route_transcodes_and_cleans_up(tmp_path):
    segs = [_seg("u1", 0, "നമസ്കാരം", text_en="Hello"), _seg("u2", 1, "വിഷമം", text_en="Sad")]
    llm = FakeLLM()
    state, path = setup(tmp_path, speech=FakeSpeech(segs, name="gemini"), llm=llm, language="ml")
    await run(state, "s1")
    sent = state.speech["gemini"].calls[0][0]
    assert sent.suffix == ".mp3"
    assert not sent.exists() and not path.exists()
    assert "TranslationBatch" not in llm.calls
    assert state.store.get("s1")["status"] == "draft"


ROUTES = {"hi": "deepgram", "mr": "gemini", "en": "deepgram", "ml": "gemini"}


def test_choose_route():
    assert choose_route({"hi": 50, "en": 20}, ROUTES, "gemini") == ("hi", "deepgram", [])
    assert choose_route({"en": 55, "ml": 5}, ROUTES, "gemini") == ("ml", "gemini", [])
    assert choose_route({"en": 60}, ROUTES, "gemini") == ("en", "deepgram", [])
    assert choose_route({"hi": 40, "ml": 3}, ROUTES, "gemini")[:2] == ("hi", "gemini")
    lang, route, warnings = choose_route({"en": 10, "other": 2}, ROUTES, "gemini")
    assert (lang, route) == ("en", "gemini") and "unsupported" in warnings[0]


def test_choose_route_no_speech():
    with pytest.raises(PipelineError, match="no speech"):
        choose_route({"none": 4}, ROUTES, "gemini")


async def test_auto_language_detects_then_routes(tmp_path):
    segs = [_seg("u1", 0, "Hello", text_en="Hello"), _seg("u2", 1, "വിഷമം", text_en="Sad")]
    gemini = FakeSpeech(segs, name="gemini", heard={"en": 50, "ml": 4})
    state, _ = setup(tmp_path, language="auto", extra=[gemini], speech_routes=ROUTES)
    await run(state, "s1")
    row = state.store.get("s1")
    assert row["status"] == "draft", row["error"]
    assert row["language"] == "ml"
    assert row["transcript"]["languages"] == ["en", "ml"]
    assert gemini.calls and not state.speech["deepgram"].calls
    assert "detect" in row["timings"]


async def test_roles_per_chunk_renumber_speakers():
    segs = [_seg("u1", 0, "Hi"), _seg("u2", 1, "Hello"), _seg("u3", 1, "Ok"), _seg("u4", 0, "Hmm")]
    segs[2].chunk = segs[3].chunk = 1

    class PerChunk:
        async def structured(self, system, user, schema, **kw):
            from ally.pipeline import RoleGuess

            counselor = 0 if "[u1 " in user else 1  # chunk 1 numbers the counselor as speaker 1
            return RoleGuess(counselor_speaker=counselor, confidence="high", rationale="")

    assert await assign_roles(PerChunk(), segs, 0) == []
    assert [(s.role, s.speaker) for s in segs] == [
        ("counselor", 0),
        ("client", 1),
        ("counselor", 0),
        ("client", 1),
    ]
