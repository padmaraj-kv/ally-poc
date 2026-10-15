import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from ally.config import Settings
from ally.db import Store
from ally.llm import LLM
from ally.routes import router
from ally.speech import DeepgramSpeech, GeminiSpeech

DESCRIPTION = """
Turns a recorded counselling call into an English transcript, evidence-linked notes and risk flags.

1. `POST /sessions` with the recording and its language. Returns `202` and an id.
2. Poll `GET /sessions/{id}` until `status` is `draft` (or `failed`, with `error`).
3. Review: `PATCH` the note, segments, speakers and risk flags.
4. `POST /sessions/{id}/sign-off` (high-severity flags must be confirmed or dismissed first).
5. `GET /sessions/{id}/export` or `POST /sessions/{id}/export/webhook`.
"""


def create_app(settings: Settings | None = None, speech: dict | None = None, llm=None) -> FastAPI:
    settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)  # request lines would log webhook URLs
        (settings.data_dir / "audio").mkdir(parents=True, exist_ok=True)
        app.state.settings = settings
        app.state.store = Store(settings.data_dir / "ally.db")
        if n := app.state.store.fail_interrupted():
            logging.getLogger("ally").warning("marked %d interrupted session(s) as failed", n)
        if not settings.keep_audio:  # nothing is in flight at startup; drop leftovers from a crash
            for f in (settings.data_dir / "audio").iterdir():
                f.unlink(missing_ok=True)
        app.state.http = httpx.AsyncClient()
        if llm is not None:
            app.state.llm = llm
        else:
            app.state.llm = LLM.from_settings(settings) if settings.llm_api_key.get_secret_value() else None
        app.state.speech = (
            speech if speech is not None else _providers(settings, app.state.http, app.state.llm)
        )
        yield
        await app.state.http.aclose()

    app = FastAPI(
        title="Ally counselling notes (POC)", version="0.1.0", description=DESCRIPTION, lifespan=lifespan
    )
    app.include_router(router)

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok", "speech": sorted(app.state.speech), "llm": app.state.llm is not None}

    return app


def _providers(settings: Settings, http: httpx.AsyncClient, llm) -> dict:
    providers = {}
    if key := settings.deepgram_api_key.get_secret_value():
        providers["deepgram"] = DeepgramSpeech(key, settings.deepgram_base_url, settings.deepgram_model, http)
    if llm is not None:
        providers["gemini"] = GeminiSpeech(llm)  # needs an audio-capable model (Gemini)
    return providers


app = create_app()
