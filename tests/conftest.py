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
