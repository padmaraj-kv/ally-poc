from conftest import HINGLISH, make_note

from ally.export import signoff_stats, to_json, to_markdown
from ally.schemas import AudioMeta, RiskFlag, Session, Transcript


def flag(id, severity, status, source="llm", rationale="r"):
    return RiskFlag(
        id=id,
        category="suicidal_ideation",
        severity=severity,
        quote="q",
        rationale=rationale,
        segment_ids=["u2"],
        source=source,
        status=status,
    )


def session():
    segs = [
        s.model_copy(update={"text_en": f"EN {s.id}", "role": "counselor" if s.speaker == 0 else "client"})
        for s in HINGLISH
    ]
    return Session(
        id="s1",
        status="signed",
        language="hi",
        filename="call.wav",
        created_at="2026-10-15T10:00:00+00:00",
        updated_at="2026-10-15T10:05:00+00:00",
        risk_alert=True,
        audio_meta=AudioMeta(
            container="wav", codec="pcm_s16le", channels=1, sample_rate=8000, duration_s=42, stereo=False
        ),
        transcript=Transcript(provider="deepgram", language="hi", segments=segs),
        note=make_note(["u2"]),
        risk_flags=[
            flag("a", "high", "confirmed", rationale="Passive ideation"),
            flag("b", "low", "dismissed", rationale="Dismissed one"),
            flag("c", "medium", "pending", source="keyword", rationale="Keyword hit"),
        ],
        signed_by="Asha",
        signed_at="2026-10-15T10:05:00+00:00",
    )


def test_markdown_sections_and_flag_filtering():
    md = to_markdown(session())
    for heading in (
        "## Summary",
        "## Themes",
        "## Client concerns",
        "## Follow-ups",
        "## Risk flags",
        "## Tags",
    ):
        assert heading in md
    assert "Exam pressure _(00:10)_" in md
    assert "Passive ideation" in md and "pending" in md
    assert "Dismissed one" not in md
    assert "## Transcript" not in md
    assert "Signed by: Asha" in md


def test_markdown_with_transcript():
    md = to_markdown(session(), include_transcript=True)
    assert "[00:05] Counselor: EN u1" in md


def test_json_shape():
    data = to_json(session())
    assert data["session"]["signed_by"] == "Asha"
    assert data["session"]["duration_s"] == 42
    assert [f["id"] for f in data["risk_flags"]] == ["a", "c"]
    assert "transcript" not in data
    assert to_json(session(), include_transcript=True)["transcript"][0]["text_en"] == "EN u1"


def test_signoff_stats():
    draft = make_note().model_dump()
    final = {**draft, "summary": "Rewritten by counselor."}
    stats = signoff_stats(draft, final, edit_count=1)
    assert stats["changed_fields"] == ["summary"]
    assert 0 < stats["similarity"] < 1
    assert signoff_stats(draft, draft, 0) == {"changed_fields": [], "similarity": 1.0, "edit_count": 0}
