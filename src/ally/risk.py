import re
import uuid

from pydantic import BaseModel

from ally import prompts
from ally.schemas import RiskCategory, RiskFlag, Segment, Severity


class RiskFinding(BaseModel):
    category: RiskCategory
    severity: Severity
    quote: str
    rationale: str
    segment_ids: list[str]


class RiskReport(BaseModel):
    findings: list[RiskFinding]


# ponytail: small curated phrase net for recall; swap for a clinician-reviewed lexicon before real use.
KEYWORDS: dict[str, list[str]] = {
    "suicidal_ideation": [
        "kill myself",
        "end my life",
        "suicide",
        "suicidal",
        "want to die",
        "don't want to live",
        "no reason to live",
        "better off dead",
        "marna chahta",
        "marna chahti",
        "mar jaun",
        "mar jaana hai",
        "jeena nahi",
        "jeene ka koi matlab nahi",
        "khudkushi",
        "aatmahatya",
        "atmahatya",
        "आत्महत्या",
        "खुदकुशी",
        "मरना चाहत",
        "मर जाऊं",
        "जीना नहीं",
        "जीने का कोई मतलब नहीं",
        "जीने का मन नहीं",
        "मरायचं",
        "जगायचं नाही",
        "ആത്മഹത്യ",
        "മരിക്കണം",
        "ജീവിക്കണ്ട",
        "ചാകണം",
    ],
    "self_harm": [
        "cut myself",
        "hurt myself",
        "self harm",
        "self-harm",
        "burn myself",
        "khud ko nuksan",
        "khud ko chot",
        "खुद को नुकसान",
        "खुद को चोट",
        "हाथ काट",
    ],
    "harm_to_others": [
        "kill him",
        "kill her",
        "kill them",
        "maar dalunga",
        "maar daalunga",
        "जान से मार",
        "मार डालूंगा",
    ],
    "abuse_violence": [
        "beats me",
        "hits me",
        "hit me",
        "domestic violence",
        "maarta hai",
        "peet-ta hai",
        "pitai",
        "मारता है",
        "पीटता",
        "मारहाण",
        "തല്ലും",
    ],
    "child_safety": ["touched me", "touches me", "molest", "galat tarike se chhua", "गलत तरीके से छु"],
    "substance_misuse": [
        "overdose",
        "sleeping pills",
        "nasha",
        "daru",
        "sharab",
        "नशा",
        "शराब",
        "दारू",
        "മദ്യം",
    ],
    "acute_distress": [
        "can't take it anymore",
        "can't go on",
        "hopeless",
        "panic attack",
        "bardasht nahi",
        "ghabrahat",
        "बर्दाश्त नहीं",
        "घबराहट",
        "सहन होत नाही",
    ],
}


def _pattern(phrase: str) -> re.Pattern:
    # \b is unreliable around Indic combining marks, so non-ASCII phrases match as substrings.
    if phrase.isascii():
        return re.compile(rf"\b{re.escape(phrase)}\b", re.IGNORECASE)
    return re.compile(re.escape(phrase))


PATTERNS = [(cat, phrase, _pattern(phrase)) for cat, phrases in KEYWORDS.items() for phrase in phrases]


def keyword_hits(segments: list[Segment]) -> list[tuple[str, str, Segment]]:
    hits, seen = [], set()
    for s in segments:
        text = f"{s.text}\n{s.text_en or ''}"
        for cat, phrase, pattern in PATTERNS:
            if (s.id, cat) not in seen and pattern.search(text):
                seen.add((s.id, cat))
                hits.append((cat, phrase, s))
    return hits


def new_flag_id() -> str:
    return uuid.uuid4().hex[:8]


def has_alert(flags: list[RiskFlag]) -> bool:
    return any(f.severity == "high" and f.status != "dismissed" for f in flags)


async def assess_risk(llm, segments: list[Segment], language: str) -> tuple[list[RiskFlag], list[str]]:
    known = {s.id for s in segments}

    def check(report: RiskReport) -> list[str]:
        bad = sorted({i for f in report.findings for i in f.segment_ids if i not in known})
        return [f"unknown segment ids cited: {', '.join(bad)}"] if bad else []

    report = await llm.structured(
        prompts.RISK_SYSTEM, prompts.risk_user(language, segments), RiskReport, effort="medium", check=check
    )
    flags, removed = [], 0
    for f in report.findings:
        ids = [i for i in f.segment_ids if i in known]
        removed += len(f.segment_ids) - len(ids)
        flags.append(
            RiskFlag(id=new_flag_id(), source="llm", segment_ids=ids, **f.model_dump(exclude={"segment_ids"}))
        )
    covered = {i for f in flags for i in f.segment_ids}
    for cat, phrase, s in keyword_hits(segments):
        if s.id not in covered:
            flags.append(
                RiskFlag(
                    id=new_flag_id(),
                    category=cat,
                    severity="medium",
                    quote=s.text[:300],
                    segment_ids=[s.id],
                    rationale=f"Keyword match '{phrase}', not flagged by the model; review in context.",
                    source="keyword",
                )
            )
    return flags, [f"{removed} unknown citation(s) removed from risk flags"] if removed else []
