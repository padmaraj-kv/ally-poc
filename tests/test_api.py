import json

import httpx
import pytest
from conftest import FakeLLM, FakeSpeech, wav_bytes
from fastapi.testclient import TestClient

from ally.config import Settings
from ally.db import Store
from ally.main import create_app


def build(tmp_path, **kw):
    settings = Settings(
        _env_file=None,
        data_dir=tmp_path,
        max_upload_mb=1,
        speech_provider="deepgram",
        export_webhook_url=kw.pop("webhook", "https://crm.example/hook"),
    )
    speech = kw.pop("speech", {"deepgram": FakeSpeech(), "gemini": FakeSpeech(name="gemini")})
    return create_app(settings, speech=speech, llm=kw.pop("llm", FakeLLM()))


@pytest.fixture
def client(tmp_path):
    with TestClient(build(tmp_path)) as c:
        yield c


def upload(client, data=None, name="call.wav", language="hi"):
    files = {"audio": (name, wav_bytes() if data is None else data, "audio/wav")}
    return client.post("/sessions", files=files, data={"language": language})


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "speech": ["deepgram", "gemini"], "llm": True}


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("call.txt", None, 415),
        ("call.wav", b"", 400),
        ("call.wav", b"garbage!" * 1000, 415),
        ("call.wav", b"0" * (1024 * 1024 + 1), 413),
    ],
)
def test_upload_rejections_leave_nothing_behind(client, tmp_path, name, data, status):
    assert upload(client, data, name).status_code == status
    assert client.get("/sessions").json() == []
    assert list((tmp_path / "audio").iterdir()) == []


def test_missing_providers_503(tmp_path):
    with TestClient(build(tmp_path, speech={})) as c:
        assert upload(c).status_code == 503
    with TestClient(build(tmp_path, llm=None)) as c:
        assert upload(c).json()["detail"] == "LLM_API_KEY not configured"


def test_unknown_session_404(client):
    assert client.get("/sessions/nope").status_code == 404


def test_restart_marks_inflight_failed(tmp_path):
    (tmp_path / "audio").mkdir()
    meta = {
        "container": "wav",
        "codec": "pcm_s16le",
        "channels": 1,
        "sample_rate": 8000,
        "duration_s": 1,
        "stereo": False,
    }
    Store(tmp_path / "ally.db").create("old", "hi", "a.wav", meta)
    with TestClient(build(tmp_path)) as c:
        body = c.get("/sessions/old").json()
    assert (body["status"], body["error"]) == ("failed", "interrupted by restart")


def test_review_signoff_export_flow(client):
    created = upload(client)
    assert created.status_code == 202
    sid = created.json()["id"]
    s = client.get(f"/sessions/{sid}").json()
    assert s["status"] == "draft"
    assert s["risk_alert"] is True

    assert client.patch(f"/sessions/{sid}/note", json={"summary": "Edited."}).json()["summary"] == "Edited."
    assert client.patch(f"/sessions/{sid}/note", json={}).status_code == 400
    assert (
        client.patch(
            f"/sessions/{sid}/segments/u2", json={"text_en": "Sometimes life feels pointless"}
        ).status_code
        == 200
    )
    assert client.patch(f"/sessions/{sid}/segments/u99", json={"text_en": "x"}).status_code == 404
    roles = client.patch(f"/sessions/{sid}/speakers", json={"0": "client", "1": "counselor"}).json()
    assert roles["segments"][0]["role"] == "client"
    assert client.patch(f"/sessions/{sid}/speakers", json={"7": "client"}).status_code == 422

    blocked = client.post(f"/sessions/{sid}/sign-off", json={"counselor": "Asha"})
    assert blocked.status_code == 409
    assert "high-severity" in blocked.json()["detail"]

    flag_id = s["risk_flags"][0]["id"]
    assert (
        client.patch(f"/sessions/{sid}/risk-flags/{flag_id}", json={"status": "dismissed"}).status_code == 200
    )
    assert client.get(f"/sessions/{sid}").json()["risk_alert"] is False
    added = client.post(
        f"/sessions/{sid}/risk-flags",
        json={"category": "acute_distress", "severity": "high", "rationale": "Exam panic"},
    )
    assert added.status_code == 201
    assert added.json()["source"] == "counselor"
    assert client.get(f"/sessions/{sid}").json()["risk_alert"] is True

    assert client.get(f"/sessions/{sid}/export").status_code == 409
    signed = client.post(f"/sessions/{sid}/sign-off", json={"counselor": "Asha"}).json()
    assert signed["status"] == "signed"
    assert "summary" in signed["signoff"]["changed_fields"]
    assert signed["signoff"]["edit_count"] == 5
    assert client.patch(f"/sessions/{sid}/note", json={"summary": "late"}).status_code == 409

    assert client.get(f"/sessions/{sid}/export").json()["note"]["summary"] == "Edited."
    md = client.get(f"/sessions/{sid}/export", params={"format": "markdown", "include_transcript": True})
    assert md.headers["content-type"].startswith("text/markdown")
    assert "Sometimes life feels pointless" in md.text

    seen = {}

    def hook(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(204)

    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(hook))
    assert client.post(f"/sessions/{sid}/export/webhook").json() == {"status_code": 204}
    assert seen["body"]["session"]["signed_by"] == "Asha"


def test_webhook_unconfigured_503(tmp_path):
    with TestClient(build(tmp_path, webhook="")) as c:
        assert c.post("/sessions/x/export/webhook").status_code == 503


def test_failed_webhook_502(client):
    sid = upload(client).json()["id"]
    flag_id = client.get(f"/sessions/{sid}").json()["risk_flags"][0]["id"]
    client.patch(f"/sessions/{sid}/risk-flags/{flag_id}", json={"status": "confirmed"})
    client.post(f"/sessions/{sid}/sign-off", json={"counselor": "Asha"})
    client.app.state.http = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    assert client.post(f"/sessions/{sid}/export/webhook").status_code == 502


def test_startup_purges_leftover_audio(tmp_path):
    (tmp_path / "audio").mkdir()
    leftover = tmp_path / "audio" / "crashed.wav"
    leftover.write_bytes(wav_bytes())
    with TestClient(build(tmp_path)):
        assert not leftover.exists()


def test_httpx_request_logs_silenced(tmp_path):
    import logging

    with TestClient(build(tmp_path)):
        assert logging.getLogger("httpx").level == logging.WARNING


def test_ui_served_at_root(client):
    page = client.get("/")
    assert page.status_code == 200
    assert page.headers["content-type"].startswith("text/html")
    assert "Ally" in page.text


def test_upload_without_language_autodetects(tmp_path):
    speech = {"deepgram": FakeSpeech(), "gemini": FakeSpeech(name="gemini", heard={"hi": 3, "en": 1})}
    with TestClient(build(tmp_path, speech=speech)) as c:
        sid = c.post("/sessions", files={"audio": ("call.wav", wav_bytes(), "audio/wav")}).json()["id"]
        s = c.get(f"/sessions/{sid}").json()
    assert (s["status"], s["language"], s["transcript"]["provider"]) == ("draft", "hi", "deepgram")


def test_autodetect_needs_gemini(tmp_path):
    with TestClient(build(tmp_path, speech={"deepgram": FakeSpeech()})) as c:
        r = c.post("/sessions", files={"audio": ("call.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 503 and "auto" in r.json()["detail"]
