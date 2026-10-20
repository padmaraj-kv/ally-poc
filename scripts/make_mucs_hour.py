"""Build ~1-hour 8 kHz Hindi-English test calls from the MUCS 2021 test set (OpenSLR 104, CC BY-SA 4.0).

Downloads Hindi-English_test.tar.gz (~444 MB) once, then joins whole lectures until the target length,
downsampled to 8 kHz mono like telephony. Writes samples/mucs/mucs_hi_<n>.wav plus a reference transcript
(.txt with [MM:SS] lines, .json with segments) whose times are shifted to the joined file.

Usage: .venv/bin/python scripts/make_mucs_hour.py [--minutes 60] [--files 1]
"""

import argparse
import json
import shutil
import ssl
import tarfile
import urllib.request
import wave
from collections import defaultdict
from pathlib import Path

import av
import certifi

URL = "https://openslr.elda.org/resources/104/Hindi-English_test.tar.gz"
OUT = Path(__file__).resolve().parent.parent / "samples" / "mucs"
RATE = 8000
GAP_S = 1.0


def fetch(out: Path) -> Path:
    archive = out / "Hindi-English_test.tar.gz"
    if not archive.exists():
        print(f"downloading {URL} (~444 MB)")
        tmp = archive.with_suffix(".part")
        # python.org builds ship without system CAs; certifi comes with httpx
        ctx = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(URL, context=ctx) as resp, tmp.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp.rename(archive)
    raw = out / "raw"
    if not raw.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(raw, filter="data")
    return raw


def load_reference(raw: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Kaldi segments + text → {recording_id: [(start, end, text), ...]} sorted by start."""
    segments = next(raw.rglob("segments"))
    text = dict(
        line.rstrip("\n").split(" ", 1)
        for line in next(raw.rglob("text")).open(encoding="utf-8")
        if " " in line
    )
    by_rec = defaultdict(list)
    for line in segments.open(encoding="utf-8"):
        utt, rec, start, end = line.split()
        if utt in text:
            by_rec[rec].append((float(start), float(end), text[utt].strip()))
    return {rec: sorted(v) for rec, v in by_rec.items()}


def pcm_8k_mono(path: Path) -> bytes:
    resampler = av.AudioResampler(format="s16", layout="mono", rate=RATE)
    out = bytearray()
    with av.open(str(path)) as container:
        for frame in container.decode(container.streams.audio[0]):
            for f in resampler.resample(frame):
                out += bytes(f.planes[0])[: f.samples * 2]
        for f in resampler.resample(None):
            out += bytes(f.planes[0])[: f.samples * 2]
    return bytes(out)


def stamp(s: float) -> str:
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{int(s % 60):02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--files", type=int, default=1, help="how many joined files to build")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    raw = fetch(OUT)
    ref = load_reference(raw)
    wavs = {p.stem: p for p in raw.rglob("*.wav")}
    recs = [r for r in sorted(ref) if r in wavs]
    print(f"{len(recs)} recordings with transcripts")

    target = args.minutes * 60
    for n in range(1, args.files + 1):
        pcm, segs, offset = bytearray(), [], 0.0
        while recs and offset < target:
            rec = recs.pop(0)
            audio = pcm_8k_mono(wavs[rec])
            segs += [
                {"recording": rec, "start": round(s + offset, 2), "end": round(e + offset, 2), "text": t}
                for s, e, t in ref[rec]
            ]
            pcm += audio + bytes(int(RATE * GAP_S) * 2)
            offset += len(audio) / 2 / RATE + GAP_S
        if not segs:
            break
        base = OUT / f"mucs_hi_{n}"
        with wave.open(str(base.with_suffix(".wav")), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(bytes(pcm))
        base.with_suffix(".json").write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
        base.with_suffix(".txt").write_text(
            "\n".join(f"[{stamp(s['start'])}] {s['text']}" for s in segs) + "\n", encoding="utf-8"
        )
        print(f"wrote {base}.wav  {offset / 60:.1f} min, {len(segs)} reference segments")
        if offset < target:
            print("ran out of recordings before the target length")
            break


if __name__ == "__main__":
    main()
