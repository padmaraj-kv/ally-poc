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
