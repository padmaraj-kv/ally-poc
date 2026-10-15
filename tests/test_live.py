import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ally.config import Settings
from ally.main import create_app

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
pytestmark = pytest.mark.live


@pytest.mark.parametrize("name", ["session_stereo.wav", "session_mono.m4a"])
def test_live_pipeline(tmp_path, name):
    settings = Settings(data_dir=tmp_path)
    if not (settings.deepgram_api_key.get_secret_value() and settings.llm_api_key.get_secret_value()):
        pytest.skip("needs DEEPGRAM_API_KEY and LLM_API_KEY in .env")
    sample = SAMPLES / name
    if not sample.exists():
        pytest.skip("run scripts/make_fixtures.py first")
    with TestClient(create_app(settings)) as client:
        start = time.monotonic()
        created = client.post(
            "/sessions", files={"audio": (name, sample.read_bytes())}, data={"language": "hi"}
        )
        assert created.status_code == 202, created.text
        session = client.get(f"/sessions/{created.json()['id']}").json()  # TestClient ran the background task
    elapsed = time.monotonic() - start
    print(name, session["timings"], session["warnings"])
    assert session["status"] == "draft", session["error"]
    segments = session["transcript"]["segments"]
    assert len(segments) >= 6
    assert all(s["text_en"] for s in segments)
    assert {s["role"] for s in segments} == {"counselor", "client"}
    assert session["note"]["summary"]
    assert any(f["category"] in {"suicidal_ideation", "acute_distress"} for f in session["risk_flags"])
    assert elapsed < 300
