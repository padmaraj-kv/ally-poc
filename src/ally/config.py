from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    deepgram_api_key: SecretStr = SecretStr("")
    deepgram_base_url: str = "https://api.in.deepgram.com"
    deepgram_model: str = "nova-3"

    sarvam_api_key: SecretStr = SecretStr("")
    sarvam_base_url: str = "https://api.sarvam.ai"
    sarvam_model: str = "saaras:v4"

    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_api_key: SecretStr = SecretStr("")
    llm_model: str = "gemini-3.8-flash"
    llm_timeout_s: float = 120
    llm_max_retries: int = 3
    llm_concurrency: int = 4
    translate_batch_size: int = 120

    # one engine per call: speech_provider, or long_call_speech above long_call_minutes
    speech_provider: str = "sarvam"
    long_call_speech: str = "gemini"
    long_call_minutes: float = 10
    counselor_channel: int = 0
    data_dir: Path = Path("data")
    keep_audio: bool = False
    max_upload_mb: int = 200
    export_webhook_url: str = ""
    log_level: str = "INFO"
