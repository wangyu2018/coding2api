"""openai_compat 客户端：凭证归一化、请求头、chat 转发、models 拉取。"""

from __future__ import annotations

import json

import httpx
import pytest

from src.provider.base import ErrKind, EventKind, UpstreamProtocolViolation
from src.provider.openai_compat.client import (
    EP_CHAT,
    EP_MODELS,
    OpenAICompatClient,
    UpstreamHTTPError,
    _base_url,
    _headers,
    normalize_credential,
    prepare_body,
)
from tests.helpers.http import chunk_payload as chunk
from tests.helpers.http import sse_text_done as _sse
from tests.provider.openai_compat._support import CREDENTIAL
from tests.provider.openai_compat._support import client as _client


class TestNormalizeCredential:
    def test_rejects_non_dict(self):
        with pytest.raises(UpstreamProtocolViolation):
            normalize_credential("not-a-dict")  # type: ignore[arg-type]

    @pytest.mark.parametrize("raw", [
        {},
        {"api_key": "k"},
        {"base_url": ""},
        {"base_url": "localhost:8080"},
        {"base_url": "ftp://x"},
        {"base_url": 123},
    ])
    def test_rejects_missing_or_bad_base_url(self, raw):
        with pytest.raises(UpstreamProtocolViolation, match="base_url"):
            normalize_credential(raw)

    def test_normalizes_http_and_https_strips_trailing_slash(self):
        http = normalize_credential({"base_url": "http://h:8080/v1/"})
        assert http["base_url"] == "http://h:8080/v1"
        https = normalize_credential({"base_url": "https://h/v1///"})
        assert https["base_url"] == "https://h/v1"

    def test_base_url_aliases(self):
        assert normalize_credential({"baseUrl": "https://h"})["base_url"] == "https://h"
        assert normalize_credential({"api_base": "https://h"})["base_url"] == "https://h"

    def test_api_key_aliases_and_default(self):
        assert normalize_credential({"base_url": "https://h"})["api_key"] == ""
        assert normalize_credential(
            {"base_url": "https://h", "api_key": " k1 "})["api_key"] == "k1"
        assert normalize_credential(
            {"base_url": "https://h", "apiKey": "k2"})["api_key"] == "k2"
        assert normalize_credential(
            {"base_url": "https://h", "key": "k3"})["api_key"] == "k3"

    def test_models_variants(self):
        base = {"base_url": "https://h"}
        assert normalize_credential({**base, "models": None})["models"] == []
        assert normalize_credential(
            {**base, "models": "a, b ,a"})["models"] == ["a", "b"]
        assert normalize_credential(
            {**base, "model": "solo"})["models"] == ["solo"]
        assert normalize_credential(
            {**base, "models": [" x ", "y", "x", " "]})["models"] == ["x", "y"]

    def test_models_rejects_bad_shapes(self):
        base = {"base_url": "https://h"}
        with pytest.raises(UpstreamProtocolViolation, match="models"):
            normalize_credential({**base, "models": 42})
        with pytest.raises(UpstreamProtocolViolation, match="models"):
            normalize_credential({**base, "models": ["ok", 7]})


class TestHelpers:
    def test_base_url_valid(self):
        assert _base_url({"base_url": "https://h/v1"}) == "https://h/v1"

    @pytest.mark.parametrize("cred", [{}, {"base_url": "ftp://h"},
                                      {"base_url": 42}])
    def test_base_url_bad(self, cred):
        with pytest.raises(UpstreamProtocolViolation):
            _base_url(cred)

    def test_headers_with_and_without_key(self):
        assert _headers({"api_key": "sk-1"})["Authorization"] == "Bearer sk-1"
        assert "Authorization" not in _headers({})
        assert "Authorization" not in _headers({"api_key": ""})
        assert "Authorization" not in _headers({"api_key": 42})

    def test_prepare_body(self):
        messages = [{"role": "user", "content": "hi"}]
        payload = {"messages": messages, "model": "wrong", "stream": False,
                   "temperature": 0.3}
        body = prepare_body(payload, "glm-4.6")
        assert body["model"] == "glm-4.6" and body["stream"] is True
        assert body["temperature"] == 0.3
        # messages 必须深拷贝：改上游体不得回写引擎持有的 request.raw
        body["messages"][0]["content"] = "mutated"
        assert messages[0]["content"] == "hi"


class TestStreamChat:
    async def test_streams_events_and_sends_bearer(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["path"] = request.url.path
            seen["auth"] = request.headers.get("authorization")
            body = json.loads(request.content)
            seen["model"] = body["model"]
            seen["stream"] = body["stream"]
            return httpx.Response(200, text=_sse(
                chunk({"content": "hi"}), chunk({}, finish="stop")))

        events = [e async for e in _client(handler).stream_chat(
            CREDENTIAL, {"messages": []}, "glm-4.6")]
        assert seen["path"] == f"/v1{EP_CHAT}"
        assert seen["auth"] == "Bearer sk-test"
        assert seen["model"] == "glm-4.6" and seen["stream"] is True
        kinds = {e.kind for e in events}
        assert EventKind.CONTENT in kinds and EventKind.FINISH in kinds

    async def test_anonymous_when_no_key(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["auth"] = request.headers.get("authorization")
            return httpx.Response(200, text=_sse(chunk({"content": "hi"})))

        cred = {"base_url": "https://compat.test/v1", "api_key": ""}
        events = [e async for e in _client(handler).stream_chat(
            cred, {"messages": []}, "m")]
        assert seen["auth"] is None
        assert any(e.kind is EventKind.CONTENT for e in events)

    async def test_error_status_raises_with_classify(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, content=b'{"error":{"message":"bad key"}}')

        with pytest.raises(UpstreamHTTPError) as caught:
            [e async for e in _client(handler).stream_chat(
                CREDENTIAL, {"messages": []}, "m")]
        assert caught.value.status == 401
        assert caught.value.kind() is ErrKind.DEAD


class TestFetchModels:
    async def test_happy_path_and_item_filtering(self):
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == f"/v1{EP_MODELS}"
            assert request.headers.get("authorization") == "Bearer sk-test"
            return httpx.Response(200, json={"data": [
                {"id": "glm-4.6", "name": "GLM 4.6"},
                {"id": "deepseek-chat"},
                {"name": "missing id"},
                {"id": ""},
                {"id": 42},
                "not-a-dict",
            ]})

        models = await _client(handler).fetch_models(CREDENTIAL)
        assert [m.id for m in models] == ["glm-4.6", "deepseek-chat"]
        assert models[0].name == "GLM 4.6"
        # 非字符串 name 兜底为空串而不是整个条目被丢
        assert models[1].name == ""

    async def test_empty_list_raises(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        with pytest.raises(UpstreamProtocolViolation, match="no models"):
            await _client(handler).fetch_models(CREDENTIAL)

    @pytest.mark.parametrize("payload", [
        {},
        {"data": "not-a-list"},
        ["top-level-not-dict"],
    ])
    async def test_missing_data_list_raises(self, payload):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=payload)

        with pytest.raises(UpstreamProtocolViolation, match="data list"):
            await _client(handler).fetch_models(CREDENTIAL)

    async def test_non_json_raises(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"not json")

        with pytest.raises(UpstreamProtocolViolation, match="non-JSON"):
            await _client(handler).fetch_models(CREDENTIAL)

    async def test_error_status_raises(self):
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, content=b"forbidden")

        with pytest.raises(UpstreamHTTPError) as caught:
            await _client(handler).fetch_models(CREDENTIAL)
        assert caught.value.status == 403

    async def test_aclose_closes_clients(self):
        client = OpenAICompatClient()
        await client.aclose()  # 幂等关闭，不抛
