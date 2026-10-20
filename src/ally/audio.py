import io
from array import array
from pathlib import Path

import av
from av.error import FFmpegError

from ally.schemas import AudioMeta

DEEPGRAM_CODECS = {"mp3", "mp3float", "aac", "flac", "opus", "libopus", "vorbis"}
SILENCE = 2 * 100**2  # per-sample energy of both channels below ~-50 dBFS
SAME_RATIO = 1e-4  # residual energy (L-R) vs (L+R); ~1 for real two-speaker stereo
MIME = {
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
}


class InvalidAudio(ValueError):
    pass


def probe(path: Path) -> AudioMeta:
    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise InvalidAudio("no audio stream")
            stream = container.streams.audio[0]
            ctx = stream.codec_context
            if container.duration:
                duration = container.duration / av.time_base
            elif stream.duration:
                duration = float(stream.duration * stream.time_base)
            else:
                duration = 0.0
            info = {
                "container": container.format.name,
                "codec": ctx.name,
                "channels": ctx.channels,
                "sample_rate": ctx.sample_rate,
                "duration_s": round(float(duration), 2),
            }
        stereo = info["channels"] == 2 and not _identical_channels(path)
    except FFmpegError as e:
        raise InvalidAudio(f"unreadable audio ({type(e).__name__})") from e
    return AudioMeta(**info, stereo=stereo)


# ponytail: residual-energy check in pure Python (~1 s on 48 kHz input); numpy if it ever shows in timings.
def _identical_channels(path: Path, voiced_s: float = 30, max_s: float = 120) -> bool:
    """True when both channels carry the same signal (lossy codecs leave tiny differences)."""
    diff = total = 0.0
    voiced = 0
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        rate = stream.codec_context.sample_rate
        resampler = av.AudioResampler(format="s16p", layout="stereo")
        for frame in container.decode(stream):
            if voiced >= voiced_s * rate or (frame.time is not None and frame.time > max_s):
                break
            for f in resampler.resample(frame):
                left = array("h", bytes(f.planes[0])[: f.samples * 2])
                right = array("h", bytes(f.planes[1])[: f.samples * 2])
                energy = sum(a * a + b * b for a, b in zip(left, right, strict=True))
                if energy < SILENCE * f.samples:
                    continue
                d = sum((a - b) ** 2 for a, b in zip(left, right, strict=True))
                diff += d
                total += 2 * energy - d  # (L+R)^2 summed, since (L-R)^2 + (L+R)^2 = 2(L^2 + R^2)
                voiced += f.samples
    return diff <= SAME_RATIO * total


def prepare(path: Path, meta: AudioMeta, provider: str) -> Path:
    """Return a file the provider accepts, transcoding only when needed."""
    if provider == "gemini":
        out = path.with_suffix(".gemini.mp3")
        _transcode(path, out, "libmp3lame", "mp3", rate=16000, layout="mono", bit_rate=32000)
        return out
    if meta.codec in DEEPGRAM_CODECS or meta.codec.startswith("pcm_"):
        return path
    out = path.with_suffix(".flac")
    _transcode(path, out, "flac", "flac")
    return out


def clip_mp3(path: Path, start: float, dur: float | None) -> bytes:
    """16 kHz mono MP3 of [start, start + dur); dur None = to the end."""
    buf = io.BytesIO()
    _transcode(path, buf, "libmp3lame", "mp3", 16000, "mono", 32000, start=start, dur=dur)
    return buf.getvalue()


def mime_type(path: Path) -> str:
    return MIME.get(path.suffix.lower(), "application/octet-stream")


def _transcode(
    src: Path, dst, codec: str, fmt: str, rate=None, layout=None, bit_rate=None, start=0.0, dur=None
) -> None:
    target = dst if isinstance(dst, io.BytesIO) else str(dst)
    with av.open(str(src)) as inp, av.open(target, "w", format=fmt) as out:
        ist = inp.streams.audio[0]
        if start:
            inp.seek(int(start * av.time_base))  # lands on or before start; earlier frames skipped below
        rate = rate or ist.codec_context.sample_rate
        layout = layout or ist.codec_context.layout.name
        ost = out.add_stream(codec, rate=rate, layout=layout)
        if bit_rate:
            ost.bit_rate = bit_rate
        resampler = av.AudioResampler(format=ost.codec_context.format.name, layout=layout, rate=rate)
        for frame in inp.decode(ist):
            t = frame.time or 0.0
            if t + frame.samples / frame.sample_rate <= start:
                continue
            if dur is not None and t >= start + dur:
                break
            for f in resampler.resample(frame):
                for pkt in ost.encode(f):
                    out.mux(pkt)
        for f in resampler.resample(None):
            for pkt in ost.encode(f):
                out.mux(pkt)
        for pkt in ost.encode(None):
            out.mux(pkt)
