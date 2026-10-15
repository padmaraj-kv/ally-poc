from ally.config import Settings


def test_settings_parse_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SPEECH_ROUTES", '{"hi": "gemini"}')
    monkeypatch.setenv("DEEPGRAM_API_KEY", "dg-secret")
    s = Settings(_env_file=None, data_dir=tmp_path)
    assert s.speech_routes == {"hi": "gemini"}
    assert s.deepgram_api_key.get_secret_value() == "dg-secret"
    assert "dg-secret" not in repr(s)


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.deepgram_base_url == "https://api.in.deepgram.com"
    assert s.speech_routes["ml"] == "gemini"
