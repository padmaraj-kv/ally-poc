import av
import pytest

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
