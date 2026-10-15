from conftest import ScriptedLLM

from ally.risk import RiskFinding, RiskReport, assess_risk, has_alert, keyword_hits
from ally.schemas import RiskFlag, Segment


def seg(id, text, text_en=None, role="client"):
    return Segment(id=id, speaker=1, role=role, start=0, end=1, text=text, text_en=text_en)


def test_keyword_hits_across_scripts():
    segs = [
        seg("u1", "I sometimes want to kill myself"),
        seg("u2", "कभी कभी लगता है कि जीने का कोई मतलब नहीं है"),
        seg("u3", "എനിക്ക് മരിക്കണം"),
        seg("u4", "papa roz maarta hai"),
        seg("u5", "It beats meditation, honestly"),
    ]
    hits = {(s.id, cat) for cat, _, s in keyword_hits(segs)}
    assert ("u1", "suicidal_ideation") in hits
    assert ("u2", "suicidal_ideation") in hits
    assert ("u3", "suicidal_ideation") in hits
    assert ("u4", "abuse_violence") in hits
    assert not any(i == "u5" for i, _ in hits)


def test_keyword_hit_once_per_segment_and_category():
    hits = keyword_hits([seg("u1", "suicide, I am suicidal", "suicide")])
    assert len([h for h in hits if h[0] == "suicidal_ideation"]) == 1


async def test_assess_merges_llm_and_uncovered_keyword_hits():
    segs = [
        seg("u1", "Kya aap safe ho?", "Are you safe?", role="counselor"),
        seg("u2", "मैं आत्महत्या के बारे में सोचती हूँ", "I think about suicide"),
        seg("u3", "घर पे दारू पीके मारता है", "He drinks and hits me at home"),
    ]
    finding = RiskFinding(
        category="suicidal_ideation",
        severity="high",
        quote="आत्महत्या के बारे में सोचती",
        rationale="Client reports suicidal thoughts",
        segment_ids=["u2", "u42"],
    )
    llm = ScriptedLLM(RiskReport(findings=[finding]), RiskReport(findings=[finding]))
    flags, warnings = await assess_risk(llm, segs, "hi")
    assert llm.calls[0]["effort"] == "medium"
    assert "I think about suicide" in llm.calls[0]["user"]
    llm_flags = [f for f in flags if f.source == "llm"]
    kw_flags = [f for f in flags if f.source == "keyword"]
    assert llm_flags[0].segment_ids == ["u2"]
    assert {f.segment_ids[0] for f in kw_flags} == {"u3"}
    assert all(f.severity == "medium" and f.status == "pending" for f in kw_flags)
    assert warnings == ["1 unknown citation(s) removed from risk flags"]
    assert has_alert(flags)


def test_has_alert_ignores_dismissed():
    flag = RiskFlag(id="a", category="self_harm", severity="high", quote="q", rationale="r", source="llm")
    assert has_alert([flag])
    flag.status = "dismissed"
    assert not has_alert([flag])
