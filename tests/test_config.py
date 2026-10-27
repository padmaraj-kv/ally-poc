from ally.config import Settings


def test_settings_parse_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECH_PROVIDER", "deepgram")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.speech_provider == "deepgram"
    assert s.deepgram_api_key.get_secret_value() == "dg-secret"
    assert "dg-secret" not in repr(s)


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.deepgram_base_url == "https://api.in.deepgram.com"
    assert (s.speech_provider, s.long_call_speech, s.long_call_minutes) == ("sarvam", "gemini", 10)


def test_sarvam_speaker_count_and_keyterms(monkeypatch):
    defaults = Settings(_env_file=None)
    assert defaults.sarvam_num_speakers == 0 and defaults.keyterms() == []
    monkeypatch.setenv("SARVAM_KEYTERMS", "Tele MANAS, Thrissur ,")
    monkeypatch.setenv("SARVAM_NUM_SPEAKERS", "2")
    s = Settings(_env_file=None)
    assert s.keyterms() == ["Tele MANAS", "Thrissur"]
    assert s.sarvam_num_speakers == 2


def test_blank_env_values_fall_back_to_defaults(monkeypatch):
    monkeypatch.setenv("SARVAM_NUM_SPEAKERS", "")
    assert Settings(_env_file=None).sarvam_num_speakers == 0
