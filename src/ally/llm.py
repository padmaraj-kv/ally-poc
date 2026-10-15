import asyncio
import json
import logging
from collections.abc import Callable
from typing import TypeVar

import openai
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

from ally.config import Settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, client, model: str, concurrency: int = 4):
        self.client = client
        self.model = model
        self._sem = asyncio.Semaphore(concurrency)
        # ponytail: one sticky downgrade per process; per-model capability flags if we mix providers.
        self._strict = True

    @classmethod
    def from_settings(cls, s: Settings) -> "LLM":
        client = AsyncOpenAI(
            base_url=s.llm_base_url,
            api_key=s.llm_api_key.get_secret_value(),
            timeout=s.llm_timeout_s,
            max_retries=s.llm_max_retries,
        )
        return cls(client, s.llm_model, s.llm_concurrency)

    async def structured(
        self,
        system: str,
        user: str | list[dict],
        schema: type[T],
        *,
        effort: str | None = None,
        check: Callable[[T], list[str]] | None = None,
        timeout: float | None = None,
    ) -> T:
        """Return a validated `schema` instance; one retry with feedback on invalid output or check issues."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        extra = {k: v for k, v in (("reasoning_effort", effort), ("timeout", timeout)) if v}
        best = None
        for attempt in range(2):
            try:
                result = await self._complete(messages, schema, extra)
            except ValidationError as e:
                issues = [f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:5]]
            else:
                best = result
                issues = check(result) if check else []
                if not issues:
                    return result
            if attempt == 0:
                feedback = "Your previous answer had problems:\n- " + "\n- ".join(issues[:20])
                messages = [
                    *messages,
                    {"role": "user", "content": feedback + "\nReturn the complete corrected JSON."},
                ]
        if best is None:
            raise LLMError(f"{schema.__name__}: invalid output after retry")
        return best  # caller handles any remaining check issues

    async def _complete(self, messages: list[dict], schema: type[T], extra: dict) -> T:
        try:
            async with self._sem:
                if self._strict:
                    try:
                        resp = await self.client.chat.completions.parse(
                            model=self.model, messages=messages, response_format=schema, **extra
                        )
                    except openai.BadRequestError:
                        log.warning("strict json_schema rejected; falling back to json_object")
                        self._strict = False
                    else:
                        msg = resp.choices[0].message
                        if msg.parsed is None:
                            raise LLMError(
                                f"{schema.__name__}: model refused ({msg.refusal or 'empty reply'})"
                            )
                        return msg.parsed
                resp = await self.client.chat.completions.create(
                    model=self.model,
                    messages=_with_schema(messages, schema),
                    response_format={"type": "json_object"},
                    **extra,
                )
        except openai.LengthFinishReasonError as e:
            raise LLMError(f"{schema.__name__}: output truncated") from e
        except openai.ContentFilterFinishReasonError as e:
            raise LLMError(f"{schema.__name__}: blocked by provider content filter") from e
        except openai.APIStatusError as e:
            raise LLMError(f"{schema.__name__}: provider HTTP {e.status_code}") from e
        except openai.APITimeoutError as e:
            raise LLMError(f"{schema.__name__}: provider timed out") from e
        except openai.APIConnectionError as e:
            raise LLMError(f"{schema.__name__}: provider unreachable") from e
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"{schema.__name__}: output truncated")
        if choice.finish_reason == "content_filter":
            raise LLMError(f"{schema.__name__}: blocked by provider content filter")
        return schema.model_validate_json(_strip_fences(choice.message.content or ""))


def _with_schema(messages: list[dict], schema: type[BaseModel]) -> list[dict]:
    hint = "\n\nRespond with one JSON object matching this JSON schema:\n" + json.dumps(
        schema.model_json_schema()
    )
    return [{**messages[0], "content": messages[0]["content"] + hint}, *messages[1:]]


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text
