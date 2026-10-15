# Ally POC — Counselling Session Note Generator (backend)

Date: 2026-10-15 · Status: approved in chat (sections 1–3), implementation authorised

## 1. Goal

A Python backend that turns a recorded counselling phone call into an English, diarized transcript plus structured,
evidence-linked session notes and risk flags, lets a counselor review/edit/sign off, and exports the signed result.
Tested end to end through Swagger (`/docs`). Single-tenant POC; production gaps are listed in §11.

Source requirements: the "AI-Powered Counselling Session Note Generator" spec (all functional requirements plus the
optional enhancements) and the "Counselling Note Generator Playbook" doc. Languages: Hindi-English, Marathi-English,
Malayalam(-English), English.

## 2. Decisions

| Area | Choice | Why |
|---|---|---|
| Speech (hi, mr, en) | Deepgram Nova-3 pre-recorded REST via `httpx`, India endpoint `api.in.deepgram.com` | User has a Deepgram key; India residency; SDK had 4 majors in 18 months, REST is stable |
| Speech (ml) | Gemini audio input via the OpenAI-compatible endpoint | Deepgram has no Malayalam model; same key as the LLM |
| Hindi-English | Deepgram `language=multi` (code-switching set includes `hi`) | |
| Marathi | Deepgram `language=mr` (monolingual) | Known gap: no Marathi-English code-switching in Deepgram; route is configurable to `gemini` |
| LLM | One `AsyncOpenAI` client, configurable `base_url`/`model`; default Gemini `gemini-3.8-flash` | Free tier with enough TPM; Groq (8K TPM) only fits calls ≲20 min |
| English transcript | LLM translation of Deepgram utterances in batches, ids preserved | Deepgram does not translate |
| Structured output | `chat.completions.parse` (strict) → fallback `json_object` + Pydantic validation → one retry with errors | Cross-provider robustness |
| Audio | PyAV (`av<19`, bundled FFmpeg) for probe/transcode; no denoising | No system ffmpeg; neural enhancement hurts ASR |
| Jobs | FastAPI `BackgroundTasks` in-process, status polling | POC; no queue/worker |
| Storage | stdlib `sqlite3`, WAL, JSON columns, connection per call | Zero infra |
| Config | `pydantic-settings` from `.env` | |

## 3. Pipeline

`POST /sessions` stores the upload, probes it, creates the session (`queued`), returns `202`, and schedules
`pipeline.run(session_id)`:

1. **Audio prep** (`audio.py`, blocking work in `asyncio.to_thread`)
   - Probe: container, codec, duration, channels, sample rate. Unreadable → `415` at upload time.
   - Fake-stereo check: decode the first 30 s to planar s16 and compare channel bytes; identical → treat as mono.
   - Transcode only when needed: codecs Deepgram does not accept (e.g. AMR) → FLAC; Gemini route → mono 16 kHz MP3
     (fallback WAV) so base64 stays under the 20 MB inline request limit (larger → stage error naming the limit).
2. **Speech** (`speech.py`): `SPEECH_ROUTES` maps language → provider (`hi`/`mr`/`en` → `deepgram`, `ml` → `gemini`).
   - Deepgram params: `model=nova-3`, `language` (`hi→multi`, `mr→mr`, `en→en`), `smart_format`, `punctuate`,
     `utterances=true`; true stereo → `multichannel=true` (speaker = channel); mono → `diarize_model=latest`.
     Parse `results.utterances` → segments sorted by start, ids `u1..uN`.
   - Gemini: one structured call with the audio part; returns segments `{speaker, start "MM:SS", end, text, text_en}`.
   - Both return `Transcript(provider, language, segments)`.
3. **Translate** (`translate.py`): segments with `text_en is None` and language ≠ `en` go to the LLM in batches of
   `TRANSLATE_BATCH_SIZE` (120), each with up to 3 earlier segments as read-only context, concurrency bounded by the
   LLM semaphore. Output `{items:[{id, en}]}`; ids must equal the batch ids (one retry with feedback). Still missing →
   `text_en` falls back to the original text and a session warning is recorded. English sessions copy `text` → `text_en`.
4. **Roles** (in `pipeline.py`): stereo → `COUNSELOR_CHANNEL` (default 0) is
   counselor, other channel client. Mono → one LLM call over the first 40 English segments returns
   `{counselor_speaker, confidence, rationale}`; others → `client`. Counselor can re-map later.
5. **Notes ∥ Risk** (`asyncio.gather`):
   - **Notes** (`notes.py`): English transcript with roles and ids → `SessionNote`. Themes come from a counselling
     taxonomy (anxiety, low mood, stress/"tension", family conflict, relationship, academic, work/career, grief,
     trauma, substance use, sleep, self-esteem, health/somatic, financial, safety) or `other`. Every item cites
     segment ids; unknown ids → one retry with feedback, then stripped with a warning.
   - **Risk** (`risk.py`): separate recall-first LLM pass over role + original + English text → flags
     `{category, severity, quote, rationale, segment_ids}`; categories: `suicidal_ideation`, `self_harm`,
     `harm_to_others`, `abuse_violence`, `child_safety`, `substance_misuse`, `acute_distress`, `other`.
     Plus a multilingual keyword net (English, romanised Hindi, Devanagari Hindi/Marathi, Malayalam) over original
     and English text; keyword hits on segments not already covered by an LLM flag become `medium` flags with
     `source=keyword`. No scores. Any `high` flag sets `risk_alert=true`.
6. **Finish**: status `draft`, per-stage `timings` (seconds) and `total`, audio deleted unless `KEEP_AUDIO`.
   Any exception → status `failed`, `error = "<stage>: <short message>"` (no secrets, no provider bodies).

Startup marks sessions left in `queued`/`processing` as `failed` ("interrupted by restart").

## 4. LLM wrapper (`llm.py`)

`LLM.structured(system, user, schema, *, effort=None, check=None) -> schema`

- `user` is `str` or OpenAI content parts (for audio).
- Global `asyncio.Semaphore(LLM_CONCURRENCY)`.
- Ladder: `parse()` strict → on `BadRequestError` use `create(response_format={"type":"json_object"})` with the JSON
  schema in the system prompt → `model_validate_json`. Validation error or `check(result)` issues → one retry with
  the issues appended as a user turn. Second failure → `LLMError`.
- `LengthFinishReasonError` → `LLMError` (callers size batches to avoid it).
- `reasoning_effort` passed only when set (`low` translate/roles, `medium` risk/notes).
- Prompts are constants in `prompts.py`; transcript data wrapped in XML tags with an instruction to treat it as data.
- Schemas stay cross-provider safe: no non-`None` defaults in LLM-facing models, no free-form dicts, `Literal` enums,
  `list[str]` for ids.

## 5. Data model

**Lifecycle** `queued → processing → draft → signed`, or `failed`. Edits only in `draft` (else `409`).

**`sessions`**: `id TEXT PK, status, language, created_at, updated_at, filename, audio_meta JSON, transcript JSON,
draft_note JSON, note JSON, risk_flags JSON, timings JSON, warnings JSON, error, risk_alert INT, signed_by,
signed_at, signoff JSON`.

**`edits`**: `id INTEGER PK, session_id, at, target (note|segment|speakers|risk_flag), ref, before JSON, after JSON`.

**Pydantic** (`schemas.py`):
- `Segment{id, speaker:int, role: counselor|client|unknown, channel:int|None, start:float, end:float, text,
  text_en:str|None}`; `Transcript{provider, language, segments}`.
- `NoteItem{text, segment_ids}`; `Theme{label, detail, segment_ids}`;
  `SessionNote{summary, themes, concerns, progress, follow_ups, tone: list[NoteItem], tags: list[str]}`.
- `RiskFlag{id, category, severity: low|medium|high, quote, rationale, segment_ids, source: llm|keyword|counselor,
  status: pending|confirmed|dismissed, counselor_note:str|None}`.

## 6. API (Swagger at `/docs`)

| Method | Path | Behaviour |
|---|---|---|
| GET | `/health` | `{status, deepgram_configured, llm_configured}` |
| POST | `/sessions` | multipart `audio`, `language` (`hi`/`mr`/`ml`/`en`, default `hi`), `counselor_channel` (opt). Extension allowlist (`mp3 wav m4a aac ogg opus flac webm amr mp4`) → `415`; empty → `400`; > `MAX_UPLOAD_MB` → `413`; unreadable → `415`; provider key for route missing → `503`. Returns `202 {id, status}` |
| GET | `/sessions` | summaries, newest first |
| GET | `/sessions/{id}` | full session |
| PATCH | `/sessions/{id}/note` | partial `SessionNote`; given fields replaced; one edit row per field |
| PATCH | `/sessions/{id}/segments/{seg_id}` | `{text_en}` |
| PATCH | `/sessions/{id}/speakers` | `{"<speaker>": "counselor"\|"client"\|"unknown"}` |
| PATCH | `/sessions/{id}/risk-flags/{flag_id}` | any of `status, severity, category, rationale, counselor_note` |
| POST | `/sessions/{id}/risk-flags` | counselor-added flag (`source=counselor`, `status=confirmed`) |
| POST | `/sessions/{id}/sign-off` | `{counselor}`; `409` if any `high` flag is `pending`; stores `signoff` = per-field changed + similarity of draft vs signed note (difflib) |
| GET | `/sessions/{id}/export?format=json\|markdown&include_transcript=false` | signed only (`409` otherwise) |
| POST | `/sessions/{id}/export/webhook` | POSTs JSON export to `EXPORT_WEBHOOK_URL` (config, not request — no SSRF); `503` if unset, `502` on failure |

Unknown session → `404`. Markdown export: header (id, language, date, signed by/at), summary, themes, concerns,
progress, follow-ups, risk flags (confirmed + counselor-added; dismissed omitted), tone, tags, optional transcript
lines `[MM:SS] Role: text_en`.

## 7. Config (`.env`)

`DEEPGRAM_API_KEY`, `DEEPGRAM_BASE_URL=https://api.in.deepgram.com`, `DEEPGRAM_MODEL=nova-3`,
`LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`, `LLM_API_KEY`, `LLM_MODEL=gemini-3.8-flash`,
`LLM_TIMEOUT_S=120`, `LLM_MAX_RETRIES=3`, `LLM_CONCURRENCY=4`, `TRANSLATE_BATCH_SIZE=120`,
`SPEECH_ROUTES={"hi":"deepgram","mr":"deepgram","en":"deepgram","ml":"gemini"}`, `COUNSELOR_CHANNEL=0`,
`DATA_DIR=./data`, `KEEP_AUDIO=false`, `MAX_UPLOAD_MB=200`, `EXPORT_WEBHOOK_URL=`, `LOG_LEVEL=INFO`.

## 8. Layout

```
pyproject.toml  .env.example  .gitignore  README.md
src/ally/  main.py config.py db.py schemas.py routes.py pipeline.py audio.py speech.py llm.py
           prompts.py translate.py notes.py risk.py export.py
scripts/make_fixtures.py      # macOS say (Rishi en_IN, Lekha hi_IN) + afconvert → samples/ (gitignored)
tests/  conftest.py (fakes) test_api.py test_pipeline.py test_audio.py test_speech.py test_llm.py
        test_risk.py test_export.py test_live.py (-m live)
tasks/todo.md tasks/lessons.md
```

`create_app(settings=None, speech=None, llm=None)` builds the app; providers live on `app.state` so tests inject fakes.

## 9. Errors & logging

Stage failure → `failed` + short error. Upload/config errors as in §6. stdlib `logging`, one line per stage
(`session_id stage seconds`); transcript text never logged. openai SDK `max_retries`/`timeout` from config; Deepgram
`httpx` timeout 600 s.

## 10. Testing

Offline by default: fake speech provider + fake LLM (returns canned schema instances, records calls), Deepgram via
`httpx.MockTransport`, audio tests on tiny WAVs generated with stdlib `wave`. Covers upload validation, lifecycle and
`409`s, Deepgram parsing (mono/stereo), Gemini segment parsing, translation id check + fallback, citation check,
keyword net, sign-off gate, export rendering, webhook. `pytest -m live` (needs keys + macOS fixtures) runs the real
path on the synthetic Hinglish call and asserts `draft` with total < 300 s.

## 11. Out of scope / production gaps

Auth and multi-tenancy, encryption at rest, a durable job queue, telephony webhooks/URL ingestion, Sarvam adapter
(Marathi-English code-switching, Indic translation), Gemini Files API for > 20 MB audio, CRM-specific adapters,
consent capture, retention jobs, tracing (Langfuse/OTel), eval harness on the `edits` table, real-time alerting on
`risk_alert`.
