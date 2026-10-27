from types import SimpleNamespace

import pytest
from conftest import FakeLLM, FakeSpeech, _seg, wav_bytes

from ally import audio
from ally.config import Settings
from ally.db import Store
from ally.pipeline import PipelineError, assign_roles, audio_path, main_language, pick_provider, run
from ally.speech import SpeechError


def setup(tmp_path, speech=None, llm=None, language="hi", channels=1, extra=(), **kw):
    kw.setdefault("speech_provider", (speech or FakeSpeech()).name)
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


def test_main_language():
    assert main_language({"hi": 50, "en": 20}) == ("hi", [])
    assert main_language({"en": 55, "ml": 5}) == ("ml", [])
    assert main_language({"en": 60}) == ("en", [])
    lang, warnings = main_language({"en": 10, "other": 2})
    assert lang == "en" and "other language" in warnings[0]


def test_main_language_no_speech():
    with pytest.raises(PipelineError, match="no speech"):
        main_language({"none": 4})


def test_pick_provider_sends_long_calls_to_gemini():
    settings = Settings(_env_file=None)
    assert pick_provider(settings, 600) == "sarvam"
    assert pick_provider(settings, 601) == "gemini"


async def test_auto_language_labels_and_short_call_uses_default_provider(tmp_path):
    sarvam = FakeSpeech(name="sarvam")
    gemini = FakeSpeech(name="gemini", heard={"en": 50, "ml": 4})
    state, _ = setup(tmp_path, speech=sarvam, language="auto", extra=[gemini])
    await run(state, "s1")
    row = state.store.get("s1")
    assert row["status"] == "draft", row["error"]
    assert row["language"] == "ml"
    assert row["transcript"]["languages"] == ["en", "ml"]
    assert sarvam.calls and not gemini.calls
    assert "detect" in row["timings"]


async def test_long_call_goes_to_gemini(tmp_path):
    segs = [_seg("u1", 0, "Hello", text_en="Hello"), _seg("u2", 1, "വിഷമം", text_en="Sad")]
    sarvam, gemini = FakeSpeech(name="sarvam"), FakeSpeech(segs, name="gemini")
    state, _ = setup(tmp_path, speech=sarvam, language="ml", extra=[gemini], long_call_minutes=0)
    await run(state, "s1")
    assert gemini.calls and not sarvam.calls
    assert state.store.get("s1")["status"] == "draft"


async def test_roles_per_chunk_renumber_speakers():
    segs = [_seg("u1", 0, "Hi"), _seg("u2", 1, "Hello"), _seg("u3", 1, "Ok"), _seg("u4", 0, "Hmm")]
    segs[2].chunk = segs[3].chunk = 1

    class PerChunk:
        async def structured(self, system, user, schema, **kw):
            from ally.pipeline import RoleGuess, SpeakerRole

            counselor = 0 if "[u1 " in user else 1  # chunk 1 numbers the counselor as speaker 1
            return RoleGuess(
                speakers=[
                    SpeakerRole(speaker=s, role="counselor" if s == counselor else "client") for s in (0, 1)
                ],
                confidence="high",
                rationale="",
            )

    assert await assign_roles(PerChunk(), segs, 0) == []
    assert [(s.role, s.speaker) for s in segs] == [
        ("counselor", 0),
        ("client", 1),
        ("counselor", 0),
        ("client", 1),
    ]


class MapRoles:
    """Labels speakers from a fixed map and records what it was shown."""

    def __init__(self, roles):
        self.roles, self.user = roles, ""

    async def structured(self, system, user, schema, **kw):
        from ally.pipeline import RoleGuess, SpeakerRole

        self.user = user
        return RoleGuess(
            speakers=[SpeakerRole(speaker=s, role=r) for s, r in self.roles.items()],
            confidence="high",
            rationale="",
        )


async def test_roles_label_every_speaker_when_diarization_splits_people():
    # Sarvam renumbered both voices after a language switch: 0/1 in English, 2/3 in Malayalam
    segs = [_seg("u1", 0, "Hello"), _seg("u2", 1, "Hi"), _seg("u3", 2, "ശരി"), _seg("u4", 3, "ഉണ്ട്")]
    llm = MapRoles({0: "counselor", 1: "client", 2: "counselor", 3: "client"})
    assert await assign_roles(llm, segs, 0) == []
    assert [s.role for s in segs] == ["counselor", "client", "counselor", "client"]


async def test_roles_unlabelled_speaker_stays_unknown_with_warning():
    segs = [_seg("u1", 0, "Hello"), _seg("u2", 1, "Hi"), _seg("u3", 2, "Hmm")]
    warnings = await assign_roles(MapRoles({0: "counselor", 1: "client"}), segs, 0)
    assert segs[2].role == "unknown"
    assert warnings == ["speaker 2 not labelled; check speakers"]


async def test_roles_prompt_includes_speakers_that_only_appear_late():
    segs = [_seg(f"u{i}", i % 2, f"line {i}") for i in range(1, 51)] + [_seg("u51", 5, "late voice")]
    llm = MapRoles({0: "counselor", 1: "client", 5: "counselor"})
    await assign_roles(llm, segs, 0)
    assert "late voice" in llm.user
    assert segs[-1].role == "counselor"


async def test_roles_all_counselor_falls_back_with_warning():
    segs = [_seg("u1", 0, "Hello"), _seg("u2", 1, "I want to die")]
    warnings = await assign_roles(MapRoles({0: "counselor", 1: "counselor"}), segs, 0)
    assert [s.role for s in segs] == ["counselor", "client"]
    assert warnings == ["counselor/client roles uncertain; check speakers"]
