from ally.db import Store


def test_crud_and_json_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    store.create("s1", "hi", "call.wav", {"channels": 1})
    store.update("s1", status="draft", timings={"total": 1.5}, risk_alert=True)
    row = store.get("s1")
    assert row["status"] == "draft"
    assert row["audio_meta"] == {"channels": 1}
    assert row["timings"] == {"total": 1.5}
    assert row["risk_flags"] == []
    assert row["risk_alert"] is True
    assert store.get("missing") is None


def test_list_newest_first(tmp_path):
    store = Store(tmp_path / "t.db")
    store.create("s1", "hi", "a.wav", {})
    store.create("s2", "en", "b.wav", {})
    assert [r["id"] for r in store.list_all()] == ["s2", "s1"]


def test_fail_interrupted_marks_only_inflight(tmp_path):
    store = Store(tmp_path / "t.db")
    for sid in ("q", "p", "d"):
        store.create(sid, "hi", "a.wav", {})
    store.update("p", status="processing")
    store.update("d", status="draft")
    assert store.fail_interrupted() == 2
    assert store.get("p")["status"] == "failed"
    assert store.get("p")["error"] == "interrupted by restart"
    assert store.get("d")["status"] == "draft"


def test_edits_counted(tmp_path):
    store = Store(tmp_path / "t.db")
    store.create("s1", "hi", "a.wav", {})
    store.add_edit("s1", "note", "summary", "old", "new")
    assert store.edit_count("s1") == 1
