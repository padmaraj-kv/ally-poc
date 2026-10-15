from pathlib import Path

import av
from av.error import FFmpegError

from ally.schemas import AudioMeta

DEEPGRAM_CODECS = {"mp3", "mp3float", "aac", "flac", "opus", "libopus", "vorbis"}
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


# ponytail: exact match on the first 30 s; switch to correlation if near-duplicate channels show up.
def _identical_channels(path: Path, seconds: float = 30) -> bool:
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16p", layout="stereo")
        for frame in container.decode(stream):
            if frame.time is not None and frame.time > seconds:
                break
            for f in resampler.resample(frame):
                n = f.samples * 2
                if bytes(f.planes[0])[:n] != bytes(f.planes[1])[:n]:
                    return False
    return True


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


def mime_type(path: Path) -> str:
    return MIME.get(path.suffix.lower(), "application/octet-stream")


def _transcode(src: Path, dst: Path, codec: str, fmt: str, rate=None, layout=None, bit_rate=None) -> None:
    with av.open(str(src)) as inp, av.open(str(dst), "w", format=fmt) as out:
        ist = inp.streams.audio[0]
        rate = rate or ist.codec_context.sample_rate
        layout = layout or ist.codec_context.layout.name
        ost = out.add_stream(codec, rate=rate, layout=layout)
        if bit_rate:
            ost.bit_rate = bit_rate
        resampler = av.AudioResampler(format=ost.codec_context.format.name, layout=layout, rate=rate)
        for frame in inp.decode(ist):
            for f in resampler.resample(frame):
                for pkt in ost.encode(f):
                    out.mux(pkt)
        for f in resampler.resample(None):
            for pkt in ost.encode(f):
                out.mux(pkt)
        for pkt in ost.encode(None):
            out.mux(pkt)
