"""Synthesise a Hinglish counselling role-play with macOS `say` (Rishi en_IN, Lekha hi_IN).

Writes samples/session_stereo.wav (counselor left, client right), samples/session_mono.wav and
samples/session_mono.m4a, all 8 kHz like telephony. macOS only.
"""

import subprocess
import tempfile
import wave
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "samples"
RATE = 8000
GAP = b"\x00\x00" * int(RATE * 0.4)
VOICES = {"counselor": "Rishi", "client": "Lekha"}
SCRIPT = [
    ("counselor", "Hello, this is Rishi from the Ally helpline. Aap kaise ho? How can I help you today?"),
    ("client", "नमस्ते। मैं ठीक नहीं हूँ, बहुत tension है। Exams आने वाले हैं और रात को नींद नहीं आती।"),
    ("counselor", "I hear you. Exams ka pressure kab se feel ho raha hai?"),
    ("client", "पिछले दो महीने से। Parents बोलते हैं कि marks अच्छे नहीं आए तो सब बेकार है। घर पे रोज़ झगड़ा होता है।"),
    ("counselor", "That sounds really hard. Kya aapne kisi se is baare mein baat ki hai?"),
    ("client", "नहीं, किसी से नहीं। कभी कभी लगता है कि जीने का कोई मतलब नहीं है।"),
    (
        "counselor",
        "Thank you for telling me that. Kya aapke mann mein khud ko hurt karne ka khayal aaya hai?",
    ),
    ("client", "नहीं, ऐसा कुछ plan नहीं है। बस बहुत थक गई हूँ।"),
    (
        "counselor",
        "Okay. Let's make a small plan. Roz bees minute walk, aur sone se pehle phone band. "
        "Next week hum phir baat karenge. Agar kabhi bahut bura lage, Tele MANAS one four four one six "
        "pe call karna.",
    ),
    ("client", "ठीक है, मैं try करूँगी। Thank you।"),
]


def speak(text: str, voice: str, tmp: Path) -> bytes:
    aiff, wav_path = tmp / "line.aiff", tmp / "line.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(
        ["afconvert", "-f", "WAVE", "-d", f"LEI16@{RATE}", "-c", "1", str(aiff), str(wav_path)], check=True
    )
    with wave.open(str(wav_path)) as w:
        return w.readframes(w.getnframes())


def write_wav(path: Path, channels: int, frames: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(frames)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    left, right, mono = bytearray(), bytearray(), bytearray()
    with tempfile.TemporaryDirectory() as d:
        for role, text in SCRIPT:
            pcm = speak(text, VOICES[role], Path(d)) + GAP
            silence = bytes(len(pcm))
            left += pcm if role == "counselor" else silence
            right += pcm if role == "client" else silence
            mono += pcm
    stereo = bytearray()
    for i in range(0, len(left), 2):
        stereo += left[i : i + 2] + right[i : i + 2]
    write_wav(OUT / "session_stereo.wav", 2, bytes(stereo))
    write_wav(OUT / "session_mono.wav", 1, bytes(mono))
    m4a = OUT / "session_mono.m4a"
    subprocess.run(
        ["afconvert", "-f", "m4af", "-d", "aac", str(OUT / "session_mono.wav"), str(m4a)], check=True
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
