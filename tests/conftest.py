import io
import math
import struct
import wave

import pytest


def wav_bytes(channels: int = 1, seconds: float = 1.0, rate: int = 8000, same: bool = False) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(int(seconds * rate)):
            a = int(8000 * math.sin(2 * math.pi * 440 * i / rate))
            b = a if same else int(8000 * math.sin(2 * math.pi * 300 * i / rate))
            frames += struct.pack("<h", a) if channels == 1 else struct.pack("<hh", a, b)
        w.writeframes(bytes(frames))
    return buf.getvalue()


@pytest.fixture
def make_wav(tmp_path):
    def make(name: str = "a.wav", **kw):
        path = tmp_path / name
        path.write_bytes(wav_bytes(**kw))
        return path

    return make


class ScriptedLLM:
    """Mimics LLM.structured: replays responses, one retry when check() reports issues."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    async def structured(self, system, user, schema, *, effort=None, check=None):
        self.calls.append({"system": system, "user": user, "schema": schema, "effort": effort})
        out = self.responses.pop(0)
        out = out(user) if callable(out) else out
        if check and check(out) and self.responses:
            return await self.structured(system, user, schema, effort=effort)
        return out


def make_note(ids=("u1",)):
    from ally.schemas import NoteItem, SessionNote, Theme

    def item(text):
        return NoteItem(text=text, segment_ids=list(ids))

    return SessionNote(
        summary="Client discussed exam stress and family conflict.",
        themes=[Theme(label="academic", detail="Exam pressure", segment_ids=list(ids))],
        concerns=[item("Exam pressure")],
        progress=[],
        follow_ups=[item("Daily 20-minute walk")],
        tone=[item("Client: anxious early, calmer later")],
        tags=["exam-stress"],
    )


def _seg(id, speaker, text, channel=None, text_en=None):
    from ally.schemas import Segment

    return Segment(
        id=id,
        speaker=speaker,
        channel=channel,
        start=float(id[1:]) * 5,
        end=float(id[1:]) * 5 + 4,
        text=text,
        text_en=text_en,
    )


HINGLISH = [
    _seg("u1", 0, "Hello, main Ally helpline se baat kar rahi hoon. Aap kaise ho?"),
    _seg("u2", 1, "कभी कभी लगता है कि जीने का कोई मतलब नहीं है"),
    _seg("u3", 0, "Kya aapke mann mein khud ko hurt karne ka khayal aaya hai?"),
    _seg("u4", 1, "नहीं, बस बहुत tension है exams की"),
]


class FakeSpeech:
    def __init__(self, segments=None, error=None, name="deepgram", heard=None):
        self.segments = HINGLISH if segments is None else segments
        self.error = error
        self.name = name
        self.heard = heard or {}
        self.calls = []

    async def detect_languages(self, path, meta):
        from collections import Counter

        return Counter(self.heard)

    async def transcribe(self, path, meta, language):
        from ally.schemas import Transcript

        self.calls.append((path, language))
        if self.error:
            raise self.error
        return Transcript(
            provider=self.name, language=language, segments=[s.model_copy() for s in self.segments]
        )


class FakeLLM:
    """Dispatches on schema: echo translations, speaker 0 is counselor, canned note and one high risk flag."""

    def __init__(self):
        self.calls = []

    async def structured(self, system, user, schema, *, effort=None, check=None):
        import re

        from ally.pipeline import RoleGuess
        from ally.risk import RiskFinding, RiskReport
        from ally.schemas import SessionNote
        from ally.translate import TranslatedItem, TranslationBatch

        self.calls.append(schema.__name__)
        if schema is TranslationBatch:
            ids = [i.strip() for i in user.rsplit("ids: ", 1)[1].split(",")]
            return TranslationBatch(items=[TranslatedItem(id=i, en=f"EN {i}") for i in ids])
        if schema is RoleGuess:
            return RoleGuess(counselor_speaker=0, confidence="high", rationale="Greets with helpline name")
        if schema is SessionNote:
            return make_note(re.findall(r"\[(u\d+) ", user)[:1])
        if schema is RiskReport:
            return RiskReport(
                findings=[
                    RiskFinding(
                        category="suicidal_ideation",
                        severity="high",
                        quote="जीने का कोई मतलब नहीं",
                        rationale="Passive ideation",
                        segment_ids=["u2"],
                    )
                ]
            )
        raise AssertionError(f"unexpected schema {schema}")
