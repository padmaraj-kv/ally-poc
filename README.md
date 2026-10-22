# Ally POC: counselling session note generator

A FastAPI backend that turns a recorded counselling phone call into:

- an English transcript, split by speaker and labelled counselor or client
- structured notes covering the summary, themes, concerns, progress, follow-ups, tone and tags, with each item linked to the transcript lines that support it
- risk flags for a counselor to review

A counselor then edits, reviews the risk flags and signs off. The signed session can be exported as JSON or Markdown, or POSTed to a CRM webhook. You test it through Swagger.

Design spec: [`docs/superpowers/specs/2026-10-15-counselling-note-poc-design.md`](docs/superpowers/specs/2026-10-15-counselling-note-poc-design.md)

## Pipeline

```
POST /sessions ─► probe (PyAV) ─► 202 {id}
                    │ background task
                    ▼
 detect ─► audio prep ─► speech ─► translate ─► roles ─► notes ∥ risk ─► draft ─► review ─► sign-off ─► export
 (clips      (transcode    Deepgram    LLM, batched,  channel or   LLM + keyword     PATCH ...   gate: no      JSON / MD
  across      only when    (hi/en)     ids checked    LLM guess,   net               edits only  pending high   / webhook
  the call)   needed)      Gemini                     per chunk                      in draft    flags
                           (ml/mr/mixed)
```

Each stage is timed into `timings`, which is how you check the 5-minute budget. Any failure sets `status=failed` and records `error="<stage>: <reason>"`.

## Quick start

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env   # add DEEPGRAM_API_KEY and LLM_API_KEY
.venv/bin/uvicorn ally.main:app --reload
open http://127.0.0.1:8000        # test UI
open http://127.0.0.1:8000/docs   # Swagger
```

FFmpeg doesn't need to be installed separately: PyAV wheels bundle it. The LLM default is the Gemini free tier through its OpenAI-compatible endpoint. Free-tier prompts may be used for training, so send synthetic audio only.

Sample audio, macOS only: `.venv/bin/python scripts/make_fixtures.py` writes `samples/session_stereo.wav`, `session_mono.wav` and `session_mono.m4a`. Together they make a 60-second Hindi-English role-play voiced by `say -v Rishi` (counselor) and `say -v Lekha` (client), at telephony 8 kHz.

## Walkthrough in Swagger

1. **`POST /sessions`:** pick a file. `language` defaults to `auto`, which detects it; set `hi`, `mr`, `ml` or `en` only if you know it. Returns `202 {"id": ...}`.
2. **`GET /sessions/{id}`:** poll until `status` is `draft`. The response holds the transcript (`text` as spoken, `text_en` in English), the `note`, the `risk_flags` and `risk_alert`, plus `timings` and `warnings`.
3. **Review:**
   - `PATCH /sessions/{id}/note`: send any subset of note fields.
   - `PATCH /sessions/{id}/segments/{seg_id}`: fix a translation (`{"text_en": "..."}`).
   - `PATCH /sessions/{id}/speakers`: swap roles (`{"0": "client", "1": "counselor"}`).
   - `PATCH /sessions/{id}/risk-flags/{flag_id}`: set `status` to `confirmed` or `dismissed`, or edit the flag.
   - `POST /sessions/{id}/risk-flags`: add a flag the model missed.
4. **`POST /sessions/{id}/sign-off`** with `{"counselor": "Asha"}`. This returns `409` while any high-severity flag is still `pending`. Once signed, the session is locked and `signoff` records which fields the counselor changed.
5. **Export:** `GET /sessions/{id}/export?format=markdown&include_transcript=true`, or `POST /sessions/{id}/export/webhook` to send it to `EXPORT_WEBHOOK_URL`.

## Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `DEEPGRAM_API_KEY` | | Required for `hi`, `mr` and `en` |
| `DEEPGRAM_BASE_URL` | `https://api.in.deepgram.com` | India endpoint (Hyderabad) |
| `SARVAM_API_KEY` / `SARVAM_MODEL` | / `saaras:v4` | Enables the `sarvam` provider; route languages to it in `SPEECH_ROUTES` |
| `LLM_BASE_URL` / `LLM_MODEL` | Gemini `gemini-3.8-flash` | Any OpenAI-compatible endpoint works. Groq `openai/gpt-oss-120b` only fits calls up to about 20 min on its free tier (8K tokens per minute) |
| `LLM_API_KEY` | | Required |
| `LLM_CONCURRENCY` / `TRANSLATE_BATCH_SIZE` | `4` / `120` | Lower these if you hit free-tier rate limits |
| `SPEECH_ROUTES` | `{"hi":"deepgram","mr":"gemini","en":"deepgram","ml":"gemini"}` | Maps language to provider |
| `MIXED_SPEECH_ROUTE` | `gemini` | Provider for auto-detected calls whose languages need different providers |
| `COUNSELOR_CHANNEL` | `0` | Which channel is the counselor in true-stereo recordings |
| `KEEP_AUDIO` | `false` | Audio is deleted once processing ends |
| `MAX_UPLOAD_MB` | `200` | |
| `EXPORT_WEBHOOK_URL` | | Fixed in config rather than taken from the request, which rules out server-side request forgery |

## Design notes

- **Deepgram over REST, not its SDK:** the SDK has had four major versions since May 2025. The request is Nova-3 with `diarize_model=latest`; true-stereo files use `multichannel=true` instead, so speaker = channel.
- **Language detection (`language=auto`):** callers can switch language anywhere in the call, so detection never looks at the opening alone (Whisper's first-30-seconds detection is the known pitfall). It takes a 10-second clip every minute across the whole call, or the whole call when it is 10 minutes or shorter. Gemini lists the languages in each clip, in parallel requests of 15 clips. A 71-minute call takes about 25 s.
- **One engine per call:** if only Hindi and English are heard, the call goes to Deepgram. Any Malayalam, Marathi or unsupported language sends the whole call to Gemini. Switching engines partway would mix two independent speaker numberings.
- **Sarvam (optional):** batch STT in `codemix` mode with diarization, for any of the 23 Indian languages. It costs about ₹0.75 per audio minute, and the free tier's ₹100 covers about 130 minutes. Flow: create job → upload to a presigned URL → start → poll every 5 s → download the result JSON. Translation stays on our LLM step, which avoids paying Sarvam's translate rate. On the 1-minute sample it separated the two voices that Deepgram's mono diarization had merged (speech 8.4 s vs 0.7 s).
- **Hindi-English** uses Nova-3 `language=multi`, which handles code-switching and tags each word's language.
- **Marathi** defaults to Gemini: Deepgram's `mr` model is Marathi-only and mishears English words.
- **Malayalam:** Deepgram has no Malayalam model, so `ml` goes to Gemini audio, which returns segments with English text already filled in.
  - Timestamps are only approximate (MM:SS).
  - Calls are sent in 10-minute parts, in parallel. Gemini numbers speakers afresh in each part, so roles (counselor or client) are decided per part and speakers are renumbered by role across the call.
  - Parts are cut at fixed times, so a sentence can be split at a boundary.
- **Fake stereo:** many telephony exports copy one signal to both channels. Probing compares the energy of the difference between channels over the first 30 seconds of non-silent audio, which tolerates AAC and Opus rounding. Matching channels are treated as mono and diarized, rather than split into two identical "speakers".
- **No denoising:** neural speech enhancement measurably hurts recognition accuracy. Audio is only transcoded when a provider can't accept it, such as AMR to FLAC for Deepgram.
- **Translation:**
  - Segments are batched, with three earlier segments sent as read-only context.
  - The code checks that every segment id comes back exactly once. One retry feeds back the problems, then any missing segments keep their original text and a warning is recorded.
- **Notes read both languages:** each line gives the original words and the English translation. A study of code-mixed summarisation (CS-Sum, 2025) found translate-then-summarise doesn't help and that summaries tend to drop the non-English parts, so the prompt asks for every language to be covered.
- **Evidence links:** every note item and risk flag cites segment ids. Unknown ids get one retry and are then stripped, with a warning. Markdown exports show the cited timestamps.
- **Risk:**
  - A separate recall-first LLM pass reads both the original-language and English text.
  - A small multilingual keyword net (English, romanised Hindi, Devanagari, Marathi, Malayalam) covers lines the model didn't flag.
  - No scores. Severity is low, medium or high. A non-dismissed high flag sets `risk_alert`.
  - The keyword list is a starting point and needs clinician review.
- **Structured output:** `chat.completions.parse` in strict mode. If the provider rejects the schema, it falls back to `json_object` plus Pydantic validation, then one retry that feeds the validation errors back.
- **Storage:** SQLite (`data/ally.db`). An `edits` table logs every counselor change. On sign-off, `signoff` records which fields changed, how similar the final note is to the draft, and the edit count; this is the raw material for evals.
- **Restarts:** a restart marks sessions that were in flight as `failed`, so they don't hang in `processing`.

## Tests

```bash
.venv/bin/pytest -q                    # offline: fakes for speech and LLM, ~2 s
.venv/bin/pytest -m live -s            # real Deepgram + LLM on samples/ (needs .env keys and fixtures)
.venv/bin/ruff check . && .venv/bin/ruff format --check .
```

## Layout

```
src/ally/
  main.py       app factory, lifespan, /health
  routes.py     /sessions API
  pipeline.py   stage orchestration, roles, timings
  audio.py      probe, fake-stereo check, transcode
  speech.py     Deepgram REST + Gemini audio providers
  llm.py        OpenAI-compatible structured-output wrapper
  translate.py  notes.py  risk.py  prompts.py
  export.py     JSON / Markdown / sign-off stats
  db.py         SQLite store
  schemas.py    Pydantic models
  config.py     settings
scripts/make_fixtures.py
```

## Not built yet (production gaps)

- **Access and data protection:** auth and per-organisation tenancy, encryption at rest, consent capture, retention jobs.
- **Processing:** a durable job queue in place of in-process background tasks, and ingestion from telephony webhooks or URLs.
- **Language coverage:**
  - A Sarvam `saaras` adapter for Marathi-English and Indic translation with India residency.
  - Benchmark Sarvam on long, real Malayalam and Marathi calls; if it holds up, route every language to it and drop language routing.
- **Integrations:** CRM-specific adapters.
- **Operations:** tracing (Langfuse or OpenTelemetry), an eval harness built on the `edits` table, and real-time alerting on `risk_alert`.
