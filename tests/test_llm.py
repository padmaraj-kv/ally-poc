from types import SimpleNamespace

import httpx2
import openai
import pytest
from pydantic import BaseModel, ValidationError

from ally.llm import LLM, LLMError


class Answer(BaseModel):
    value: int


class StubCompletions:
    def __init__(self, parse=(), create=()):
        self.results = {"parse": list(parse), "create": list(create)}
        self.calls = []

    async def _next(self, kind, kw):
        self.calls.append((kind, kw))
        r = self.results[kind].pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    async def parse(self, **kw):
        return await self._next("parse", kw)

    async def create(self, **kw):
        return await self._next("create", kw)


def make_llm(**results):
    comp = StubCompletions(**results)
    return LLM(SimpleNamespace(chat=SimpleNamespace(completions=comp)), "m"), comp


def parsed(obj, refusal=None):
    msg = SimpleNamespace(parsed=obj, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")])


def raw(text, finish="stop"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason=finish)]
    )


def http_error(cls, status):
    resp = httpx2.Response(status, request=httpx2.Request("POST", "http://llm"))
    return cls("boom", response=resp, body=None)


def validation_error():
    try:
        Answer.model_validate({})
    except ValidationError as e:
        return e


async def test_strict_success_passes_effort():
    llm, comp = make_llm(parse=[parsed(Answer(value=1))])
    assert (await llm.structured("sys", "hi", Answer, effort="low")).value == 1
    kind, kw = comp.calls[0]
    assert kind == "parse"
    assert kw["response_format"] is Answer
    assert kw["reasoning_effort"] == "low"


async def test_bad_request_falls_back_to_json_object_and_sticks():
    llm, comp = make_llm(
        parse=[http_error(openai.BadRequestError, 400)],
        create=[raw('```json\n{"value": 2}\n```'), raw('{"value": 3}')],
    )
    assert (await llm.structured("sys", "hi", Answer)).value == 2
    assert (await llm.structured("sys", "hi", Answer)).value == 3
    assert [k for k, _ in comp.calls] == ["parse", "create", "create"]
    create_kw = comp.calls[1][1]
    assert create_kw["response_format"] == {"type": "json_object"}
    assert '"value"' in create_kw["messages"][0]["content"]


async def test_invalid_output_retries_once_with_feedback():
    llm, comp = make_llm(parse=[validation_error(), parsed(Answer(value=4))])
    assert (await llm.structured("sys", "hi", Answer)).value == 4
    retry_messages = comp.calls[1][1]["messages"]
    assert len(comp.calls[0][1]["messages"]) == 2
    assert retry_messages[-1]["role"] == "user"
    assert "problems" in retry_messages[-1]["content"]


async def test_check_issues_retry_then_return_last():
    llm, comp = make_llm(parse=[parsed(Answer(value=5)), parsed(Answer(value=6))])
    result = await llm.structured("sys", "hi", Answer, check=lambda a: ["unknown id u9"])
    assert result.value == 6
    assert "unknown id u9" in comp.calls[1][1]["messages"][-1]["content"]


async def test_two_invalid_outputs_raise():
    llm, _ = make_llm(parse=[validation_error(), validation_error()])
    with pytest.raises(LLMError, match="invalid output"):
        await llm.structured("sys", "hi", Answer)


async def test_refusal_raises():
    llm, _ = make_llm(parse=[parsed(None, refusal="no")])
    with pytest.raises(LLMError, match="refused"):
        await llm.structured("sys", "hi", Answer)


async def test_http_error_is_short_llm_error():
    llm, _ = make_llm(parse=[http_error(openai.RateLimitError, 429)])
    with pytest.raises(LLMError, match="HTTP 429"):
        await llm.structured("sys", "hi", Answer)
