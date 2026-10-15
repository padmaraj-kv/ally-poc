from conftest import ScriptedLLM

from ally.schemas import Segment
from ally.translate import TranslatedItem, TranslationBatch, translate


def segs(n):
    return [Segment(id=f"u{i}", speaker=i % 2, start=i, end=i + 1, text=f"नमस्ते {i}") for i in range(1, n + 1)]


def echo(user):
    ids = [i.strip() for i in user.rsplit("ids: ", 1)[1].split(",")]
    return TranslationBatch(items=[TranslatedItem(id=i, en=f"EN {i}") for i in ids])


async def test_batches_with_context():
    s = segs(5)
    llm = ScriptedLLM(echo, echo, echo)
    assert await translate(llm, s, "hi", batch_size=2) == []
    assert [x.text_en for x in s] == [f"EN u{i}" for i in range(1, 6)]
    assert len(llm.calls) == 3
    second = llm.calls[1]["user"]
    assert "<context>" in second and "[u2" in second.split("<segments>")[0]
    assert llm.calls[0]["effort"] == "low"


async def test_missing_id_retries_then_keeps_original():
    s = segs(2)
    partial = TranslationBatch(items=[TranslatedItem(id="u1", en="Hello 1")])
    llm = ScriptedLLM(partial, partial)
    warnings = await translate(llm, s, "hi", batch_size=10)
    assert len(llm.calls) == 2
    assert s[0].text_en == "Hello 1"
    assert s[1].text_en == s[1].text
    assert warnings == ["1 segment(s) left untranslated; original text kept"]


async def test_retry_fixes_missing_id():
    s = segs(2)
    partial = TranslationBatch(items=[TranslatedItem(id="u1", en="Hello 1")])
    llm = ScriptedLLM(partial, echo)
    assert await translate(llm, s, "hi", batch_size=10) == []
    assert s[1].text_en == "EN u2"


async def test_extra_and_duplicate_ids_ignored():
    s = segs(2)
    noisy = TranslationBatch(
        items=[
            TranslatedItem(id="u1", en="first"),
            TranslatedItem(id="u1", en="second"),
            TranslatedItem(id="u9", en="ghost"),
            TranslatedItem(id="u2", en="two"),
        ]
    )
    llm = ScriptedLLM(noisy)
    assert await translate(llm, s, "hi", batch_size=10) == []
    assert [x.text_en for x in s] == ["first", "two"]


async def test_english_copies_text_without_llm():
    s = segs(2)
    llm = ScriptedLLM()
    assert await translate(llm, s, "en", batch_size=10) == []
    assert [x.text_en for x in s] == [x.text for x in s]
    assert llm.calls == []


async def test_prefilled_segments_skipped():
    s = segs(2)
    s[0].text_en = "already"
    llm = ScriptedLLM(echo)
    await translate(llm, s, "ml", batch_size=10)
    assert s[0].text_en == "already"
    assert "ids: u2" in llm.calls[0]["user"]
