import wave

import av
import pytest
from conftest import wav_bytes

from ally import audio
from ally.audio import InvalidAudio, prepare, probe


def test_probe_mono(make_wav):
    meta = probe(make_wav(channels=1, seconds=2))
    assert (meta.channels, meta.sample_rate, meta.stereo) == (1, 8000, False)
    assert meta.duration_s == pytest.approx(2.0, abs=0.1)


def test_probe_true_stereo(make_wav):
    assert probe(make_wav(channels=2)).stereo is True


def test_probe_fake_stereo_is_mono(make_wav):
    meta = probe(make_wav(channels=2, same=True))
    assert meta.channels == 2
    assert meta.stereo is False


def test_probe_rejects_garbage(tmp_path):
    path = tmp_path / "x.wav"
    path.write_bytes(b"not audio at all" * 100)
    with pytest.raises(InvalidAudio):
        probe(path)


def test_probe_rejects_no_audio_stream(tmp_path):
    path = tmp_path / "video.mp4"
    with av.open(str(path), "w") as c:
        s = c.add_stream("mpeg4", rate=1)
        s.width = s.height = 16
        s.pix_fmt = "yuv420p"
        for pkt in s.encode(av.VideoFrame(16, 16, "yuv420p")):
            c.mux(pkt)
        for pkt in s.encode(None):
            c.mux(pkt)
    with pytest.raises(InvalidAudio, match="no audio"):
        probe(path)


def test_prepare_deepgram_passthrough(make_wav):
    path = make_wav()
    assert prepare(path, probe(path), "deepgram") == path


def test_prepare_deepgram_transcodes_amr(make_wav, tmp_path):
    amr = tmp_path / "call.amr"
    audio._transcode(make_wav(), amr, "libopencore_amrnb", "amr", rate=8000, layout="mono", bit_rate=12200)
    out = prepare(amr, probe(amr), "deepgram")
    assert out.suffix == ".flac"
    assert probe(out).codec == "flac"


def test_prepare_gemini_mp3_mono(make_wav):
    path = make_wav(channels=2)
    meta = probe(prepare(path, probe(path), "gemini"))
    assert (meta.container, meta.channels, meta.sample_rate) == ("mp3", 1, 16000)


def test_mime_type(tmp_path):
    assert audio.mime_type(tmp_path / "a.M4A") == "audio/mp4"
    assert audio.mime_type(tmp_path / "a.xyz") == "application/octet-stream"


@pytest.mark.parametrize(
    ("name", "codec", "fmt", "rate", "bit_rate"),
    [("a.m4a", "aac", "ipod", 16000, 128000), ("a.ogg", "libopus", "ogg", 16000, 32000)],
)
def test_lossy_fake_stereo_is_mono(make_wav, tmp_path, name, codec, fmt, rate, bit_rate):
    out = tmp_path / name
    src = make_wav(channels=2, same=True, seconds=2)
    audio._transcode(src, out, codec, fmt, rate=rate, layout="stereo", bit_rate=bit_rate)
    assert probe(out).stereo is False


def test_true_stereo_after_leading_silence(tmp_path):
    path = tmp_path / "ring.wav"
    silence = bytes(8000 * 4 * 31)  # 31 s of digital silence on both channels
    tones = wav_bytes(channels=2, seconds=1)[44:]
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(8000)
        w.writeframes(silence + tones)
    assert probe(path).stereo is True


def test_clip_mp3_cuts_window(make_wav, tmp_path):
    out = tmp_path / "clip.mp3"
    out.write_bytes(audio.clip_mp3(make_wav(seconds=5), 2, 1.5))
    m = audio.probe(out)
    assert m.codec.startswith("mp3") and 1.2 < m.duration_s < 1.9


def test_clip_mp3_whole_file(make_wav, tmp_path):
    out = tmp_path / "all.mp3"
    out.write_bytes(audio.clip_mp3(make_wav(seconds=3), 0, None))
    assert 2.7 < audio.probe(out).duration_s < 3.3


def test_prepare_sarvam_mp3(make_wav):
    path = make_wav(seconds=2)
    out = prepare(path, probe(path), "sarvam")
    assert out.name.endswith(".sarvam.mp3") and probe(out).codec.startswith("mp3")
