"""OpenAICompatProvider 委托、模型白名单与装配。"""

from __future__ import annotations

import httpx
import pytest

from src.provider.base import ErrKind, EventKind
from src.provider.openai_compat import OpenAICompatProvider
from src.provider.openai_compat.client import (
    OpenAICompatClient,
    UpstreamHTTPError,
)
from tests.helpers.http import chunk_payload as chunk
from tests.helpers.http import sse_text_done as _sse
from tests.provider.openai_compat._support import CREDENTIAL
from tests.provider.openai_compat._support import client as _client


async def test_provider_probe_quota_is_unknown_not_exhausted():
    quota = await OpenAICompatProvider().probe_quota({})
    assert quota.probe_failed is True
    assert quota.total is None and quota.remaining is None
    assert quota.probed_at is not None


def test_provider_import_credential_normalizes():
    out = OpenAICompatProvider().import_credential(
        {"base_url": "https://h/v1/", "api_key": " k ", "models": "a,b"})
    assert out == {"base_url": "https://h/v1", "api_key": "k",
                   "models": ["a", "b"]}


def test_provider_classify_delegates():
    assert OpenAICompatProvider().classify(401, b"") is ErrKind.DEAD


def test_provider_default_client_and_id():
    provider = OpenAICompatProvider()
    assert provider.id == "openai_compat"
    assert isinstance(provider.client, OpenAICompatClient)


async def test_list_models_static_whitelist_skips_upstream():
    """凭证给了 models 白名单就不打 /models（零探测、可只暴露子集）。"""
    def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("静态白名单下不应请求上游")

    provider = OpenAICompatProvider(client=_client(handler))
    models = await provider.list_models(
        {"base_url": "https://compat.test/v1", "models": ["glm-4.6", "glm-4.5"]})
    assert [m.id for m in models] == ["glm-4.6", "glm-4.5"]
    await provider.aclose()


async def test_list_models_empty_whitelist_falls_back_to_fetch():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "m1"}]})

    provider = OpenAICompatProvider(client=_client(handler))
    cred = {"base_url": "https://compat.test/v1", "models": []}
    assert [m.id for m in await provider.list_models(cred)] == ["m1"]
    # models 缺字段同样回退拉取
    assert [m.id for m in await provider.list_models(
        {"base_url": "https://compat.test/v1"})] == ["m1"]
    await provider.aclose()


async def test_provider_stream_chat_with_and_without_pacer():
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sse(chunk({"content": "hi"}, finish="stop")))

    class Pacer:
        def __init__(self) -> None:
            self.turns = 0
            self.released = 0

        async def wait_turn(self, key=None) -> None:
            self.turns += 1

        def release(self, key=None) -> None:
            self.released += 1

    pacer = Pacer()
    provider = OpenAICompatProvider(client=_client(handler), pacer=pacer)
    events = [e async for e in provider.stream_chat(
        CREDENTIAL, {"messages": []}, "glm-4.6")]
    assert any(e.kind is EventKind.CONTENT for e in events)
    assert pacer.turns == 1 and pacer.released == 1
    await provider.aclose()

    plain = OpenAICompatProvider(client=_client(handler))
    _ = [e async for e in plain.stream_chat(
        CREDENTIAL, {"messages": []}, "glm-4.6")]
    await plain.aclose()


async def test_provider_stream_chat_releases_pacer_on_error():
    """上游抛错也要归还名额，否则该桶被当成永远在途而失去节流。"""
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, content=b"bad key")

    class Pacer:
        def __init__(self) -> None:
            self.released = 0

        async def wait_turn(self, key=None) -> None:
            pass

        def release(self, key=None) -> None:
            self.released += 1

    pacer = Pacer()
    provider = OpenAICompatProvider(client=_client(handler), pacer=pacer)
    with pytest.raises(UpstreamHTTPError):
        [e async for e in provider.stream_chat(
            CREDENTIAL, {"messages": []}, "glm-4.6")]
    assert pacer.released == 1
    await provider.aclose()


def test_openai_compat_in_known_providers():
    """注册面：`model@openai_compat` 定向与按渠道代理解析都走 KNOWN_PROVIDERS。"""
    from src.engine.model_resolver import KNOWN_PROVIDERS, resolve

    assert "openai_compat" in KNOWN_PROVIDERS
    target = resolve("glm-4.6@openai_compat", "glm-4.6")
    assert target.providers == ("openai_compat",) and target.forced


def test_build_app_registers_openai_compat(tmp_path):
    """端到端装配：默认 registry 含 openai_compat provider。"""
    from src.config import Settings
    from src.main import build_app
    from tests.helpers.constants import SECRET

    settings = Settings(_env_file=None, APP_SECRET=SECRET, DATA_DIR=str(tmp_path),
                        ADMIN_USERNAMES="root")
    app = build_app(settings)
    assert "openai_compat" in app.state.executor._deps.providers


def test_import_credential_end_to_end_via_api(tmp_path, monkeypatch):
    """管理台导入 openai_compat 凭证 → 落库规范化数据并列出模型。"""
    from fastapi.testclient import TestClient

    from src.auth.session import create_session_token
    from src.config import Settings
    from src.main import build_app
    from src.provider.base import Model
    from tests.helpers.constants import SECRET

    async def fake_fetch_models(self, credential_data):
        return [Model(id="glm-4.6")]

    monkeypatch.setattr(OpenAICompatClient, "fetch_models", fake_fetch_models)
    settings = Settings(_env_file=None, APP_SECRET=SECRET, DATA_DIR=str(tmp_path),
                        ADMIN_USERNAMES="root")
    app = build_app(settings)
    with TestClient(app) as client:
        client.cookies.set("coding2api_session", create_session_token("root", SECRET))
        created = client.post("/api/credentials", json={
            "provider": "openai_compat",
            "credential": {"base_url": "https://open.bigmodel.cn/api/paas/v4",
                           "api_key": "zhipu-key", "models": ["glm-4.6"]},
            "nickname": "智谱 Coding Plan",
        })
        assert created.status_code == 200, created.text
        credential_id = created.json()["id"]
        stored = app.state.credentials.credential_data(credential_id)
        assert stored["base_url"] == "https://open.bigmodel.cn/api/paas/v4"
        assert stored["api_key"] == "zhipu-key"
        assert stored["models"] == ["glm-4.6"]

        # 坏凭证在导入时就被拒，不落库
        rejected = client.post("/api/credentials", json={
            "provider": "openai_compat", "credential": {"api_key": "k"}})
        assert rejected.status_code >= 400
