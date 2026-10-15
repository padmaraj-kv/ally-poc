from conftest import ScriptedLLM, make_note

from ally.notes import generate_note, note_items
from ally.schemas import Segment

SEGS = [
    Segment(id="u1", speaker=0, role="counselor", start=0, end=2, text="Hello", text_en="Hello"),
    Segment(
        id="u2", speaker=1, role="client", start=65, end=70, text="बहुत tension है", text_en="Lots of stress"
    ),
]


async def test_note_uses_english_transcript_and_medium_effort():
    llm = ScriptedLLM(make_note(["u2"]))
    note, warnings = await generate_note(llm, SEGS, "hi")
    assert warnings == []
    assert note.concerns[0].segment_ids == ["u2"]
    call = llm.calls[0]
    assert call["effort"] == "medium"
    assert "[u2 01:05 client] Lots of stress" in call["user"]
    assert "Hindi-English" in call["user"]


async def test_unknown_citations_retry_then_stripped():
    bad = make_note(["u1", "u99"])
    llm = ScriptedLLM(bad, bad)
    note, warnings = await generate_note(llm, SEGS, "hi")
    assert len(llm.calls) == 2
    assert all(i in {"u1", "u2"} for item in note_items(note) for i in item.segment_ids)
    assert warnings == ["4 unknown citation(s) removed from the note"]


async def test_retry_fixes_citations():
    llm = ScriptedLLM(make_note(["u7"]), make_note(["u1"]))
    note, warnings = await generate_note(llm, SEGS, "hi")
    assert warnings == []
    assert note.themes[0].segment_ids == ["u1"]
