from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# "auto" until detection picks the main language
Language = Literal["auto", "hi", "mr", "ml", "en"]
Role = Literal["counselor", "client", "unknown"]
Status = Literal["queued", "processing", "draft", "signed", "failed"]
Severity = Literal["low", "medium", "high"]
FlagStatus = Literal["pending", "confirmed", "dismissed"]
RiskCategory = Literal[
    "suicidal_ideation",
    "self_harm",
    "harm_to_others",
    "abuse_violence",
    "child_safety",
    "substance_misuse",
    "acute_distress",
    "other",
]
ThemeLabel = Literal[
    "anxiety",
    "low_mood",
    "stress",
    "family_conflict",
    "relationships",
    "academic",
    "work_career",
    "grief_loss",
    "trauma",
    "substance_use",
    "sleep",
    "self_esteem",
    "health_somatic",
    "financial",
    "safety",
    "other",
]


class AudioMeta(BaseModel):
    container: str
    codec: str
    channels: int
    sample_rate: int
    duration_s: float
    stereo: bool = Field(description="Two distinct channels, so each channel is one speaker")


class Segment(BaseModel):
    id: str
    speaker: int
    role: Role = "unknown"
    channel: int | None = None
    start: float
    end: float
    text: str
    text_en: str | None = None
    # Gemini numbers speakers per 10-min chunk; roles are matched per chunk, then renumbered
    chunk: int = Field(0, exclude=True)


class Transcript(BaseModel):
    provider: str
    language: Language
    languages: list[str] = Field([], description="Languages heard by auto-detection")
    segments: list[Segment]


# LLM-facing models below: no non-None defaults, so strict JSON schema works across providers.
class NoteItem(BaseModel):
    text: str
    segment_ids: list[str] = Field(description="Ids of transcript segments supporting this item, e.g. ['u4']")


class Theme(BaseModel):
    label: ThemeLabel
    detail: str
    segment_ids: list[str]


class SessionNote(BaseModel):
    summary: str = Field(description="4-6 sentence counselor-facing summary of the session")
    themes: list[Theme]
    concerns: list[NoteItem] = Field(description="Client's presenting concerns")
    progress: list[NoteItem] = Field(description="Progress, insight or coping noted; empty if none")
    follow_ups: list[NoteItem] = Field(description="Agreed or recommended next steps; empty if none")
    tone: list[NoteItem] = Field(
        description="Tone/affect per speaker, e.g. 'Client: anxious early, calmer later'"
    )
    tags: list[str] = Field(description="3-8 short lowercase tags for CRM filtering, e.g. 'exam-stress'")


class RiskFlag(BaseModel):
    id: str
    category: RiskCategory
    severity: Severity
    quote: str
    rationale: str
    segment_ids: list[str] = []
    source: Literal["llm", "keyword", "counselor"]
    status: FlagStatus = "pending"
    counselor_note: str | None = None


class SessionCreated(BaseModel):
    id: str
    status: Status


class SessionSummary(BaseModel):
    id: str
    status: Status
    language: Language
    filename: str
    created_at: str
    risk_alert: bool


class Session(SessionSummary):
    updated_at: str
    audio_meta: AudioMeta | None = None
    transcript: Transcript | None = None
    draft_note: SessionNote | None = None
    note: SessionNote | None = None
    risk_flags: list[RiskFlag] = []
    timings: dict[str, float] = {}
    warnings: list[str] = []
    error: str | None = None
    signed_by: str | None = None
    signed_at: str | None = None
    signoff: dict | None = None


class NotePatch(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"summary": "Counselor-edited summary.", "tags": ["exam-stress"]}]}
    )

    summary: str | None = None
    themes: list[Theme] | None = None
    concerns: list[NoteItem] | None = None
    progress: list[NoteItem] | None = None
    follow_ups: list[NoteItem] | None = None
    tone: list[NoteItem] | None = None
    tags: list[str] | None = None


class SegmentPatch(BaseModel):
    text_en: str = Field(min_length=1)


class RiskFlagPatch(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={"examples": [{"status": "confirmed", "counselor_note": "Safety plan agreed."}]}
    )

    status: FlagStatus | None = None
    severity: Severity | None = None
    category: RiskCategory | None = None
    rationale: str | None = None
    counselor_note: str | None = None


class RiskFlagCreate(BaseModel):
    category: RiskCategory
    severity: Severity
    rationale: str
    quote: str = ""
    segment_ids: list[str] = []
    counselor_note: str | None = None


class SignOff(BaseModel):
    counselor: str = Field(min_length=1, examples=["Asha Menon"])


class WebhookResult(BaseModel):
    status_code: int
