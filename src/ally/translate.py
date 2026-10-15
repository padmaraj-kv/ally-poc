import asyncio
import logging
from collections import Counter

from pydantic import BaseModel

from ally import prompts
from ally.llm import LLMError
from ally.schemas import Segment

log = logging.getLogger(__name__)


class TranslatedItem(BaseModel):
    id: str
    en: str


class TranslationBatch(BaseModel):
    items: list[TranslatedItem]


async def translate(llm, segments: list[Segment], language: str, batch_size: int) -> list[str]:
    """Fill `text_en` in place; returns warnings."""
    if language == "en":
        for s in segments:
            s.text_en = s.text_en or s.text
        return []
    todo = [i for i, s in enumerate(segments) if s.text_en is None]
    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    missing = sum(await asyncio.gather(*(_batch(llm, segments, b, language) for b in batches)))
    return [f"{missing} segment(s) left untranslated; original text kept"] if missing else []


async def _batch(llm, segments: list[Segment], idx: list[int], language: str) -> int:
    batch = [segments[i] for i in idx]
    context = segments[max(0, idx[0] - 3) : idx[0]]
    ids = [s.id for s in batch]

    def check(out: TranslationBatch) -> list[str]:
        got = Counter(item.id for item in out.items)
        issues = []
        if missing := [i for i in ids if i not in got]:
            issues.append("missing ids: " + ", ".join(missing))
        if extra := sorted(set(got) - set(ids)):
            issues.append("unexpected ids: " + ", ".join(extra))
        if dups := sorted(i for i, n in got.items() if n > 1):
            issues.append("duplicate ids: " + ", ".join(dups))
        return issues

    try:
        out = await llm.structured(
            prompts.TRANSLATE_SYSTEM,
            prompts.translate_user(language, context, batch),
            TranslationBatch,
            effort="low",
            check=check,
        )
    except LLMError as e:
        log.warning("translation batch %s..%s failed: %s", ids[0], ids[-1], e)
        out = TranslationBatch(items=[])
    english: dict[str, str] = {}
    for item in out.items:
        if item.id in ids and item.id not in english and item.en.strip():
            english[item.id] = item.en.strip()
    for s in batch:
        s.text_en = english.get(s.id, s.text)
    return sum(1 for s in batch if s.id not in english)
