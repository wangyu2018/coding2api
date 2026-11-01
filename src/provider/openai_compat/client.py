"""OpenAI 兼容上游客户端：凭证驱动的转发（base_url/api_key 随凭证传入）。

与 Kilo 同为标准 OpenAI SSE，但三条关键差异决定它是**凭证渠道**：

* **鉴权**——`Authorization: Bearer <api_key>` 随凭证注入；凭证不带
  api_key 时匿名转发（内网自建网关如 devin-2api 裸跑的场景）；
* **模型列表**——凭证里给了 `models` 白名单就直接用它：一次探测不发，
  也允许管理员只暴露上游模型池的子集（如只在客户端挂 `glm-4.6`）；
  留空时回退 `GET {base_url}/models`；
* **错误语义**——401/403 是 key 坏了 → DEAD；402 余额不足 → CREDIT；
  429/404 归模型级；与 Kilo 的匿名渠道判法相反（见 events.py 注释）。
"""

from __future__ import annotations

import copy
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ...engine.sse import iter_frames
from ...provider import base
from ...provider.base import (
    ErrKind,
    Event,
    Model,
    Quota,
    UpstreamProtocolViolation,
)
from ...tasks.pacer import paced
from ..lazy_clients import LazyClients
from . import events as oc_events

EP_CHAT = "/chat/completions"
EP_MODELS = "/models"


def normalize_credential(raw: dict[str, Any]) -> dict[str, Any]:
    """导入凭证 → 规范化 credential_data。

    接受 `{base_url, api_key?, models?}`（别名：`apiBase`/`api_key`/`key`/
    `model` 单串、`models` 逗号串），拒绝非 http(s) 或不完整的输入。
    拒绝发生在导入时，让用户立刻看到，而不是落库一条永远打不通的凭证。
    """
    if not isinstance(raw, dict):
        raise UpstreamProtocolViolation("credential must be an object")
    base_url = raw.get("base_url") or raw.get("baseUrl") or raw.get("api_base") or ""
    base_url = str(base_url).strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        raise UpstreamProtocolViolation(
            "credential.base_url must be an http(s) upstream URL")
    api_key = raw.get("api_key") or raw.get("apiKey") or raw.get("key") or ""
    models = _normalize_models(raw.get("models", raw.get("model")))
    return {"base_url": base_url, "api_key": str(api_key).strip(), "models": models}


def _normalize_models(raw: Any) -> list[str]:
    """models 白名单归一化：列表 / 单串 / 逗号串 → 去重后的有序列表。"""
    if raw is None:
        return []
    items: list[Any]
    if isinstance(raw, str):
        items = [item.strip() for item in raw.split(",")]
    elif isinstance(raw, list):
        items = raw
    else:
        raise UpstreamProtocolViolation("credential.models must be a list or string")
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise UpstreamProtocolViolation("credential.models entries must be strings")
        model = item.strip()
        if model and model not in out:
            out.append(model)
    return out


def _base_url(credential_data: dict[str, Any]) -> str:
    """credential_data → base_url；缺字段按凭证损坏显式失败。"""
    base_url = credential_data.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
        raise UpstreamProtocolViolation("credential missing valid base_url")
    return base_url


def _headers(credential_data: dict[str, Any]) -> dict[str, str]:
    """请求头：有 key 才带 Authorization（兼容端点对假头敏感，见 Kilo 实测）。"""
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    api_key = credential_data.get("api_key")
    if isinstance(api_key, str) and api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def prepare_body(payload: dict[str, Any], model: str) -> dict[str, Any]:
    """上游请求体：深拷 messages、强制 stream、写死 model（同 Kilo/Zen 约定）。"""
    body = dict(payload)
    body["messages"] = copy.deepcopy(payload.get("messages"))
    body["model"] = model
    body["stream"] = True
    return body


class UpstreamHTTPError(base.UpstreamHTTPError):
    """兼容上游非 2xx；kind() 走本渠道的状态码规则。"""

    classify_status = staticmethod(oc_events.classify_status)


class OpenAICompatClient(LazyClients):
    """上游 HTTP 客户端：base_url / api_key 全部来自 credential_data。"""

    def __init__(
        self,
        *,
        stream_client: httpx.AsyncClient | None = None,
        short_client: httpx.AsyncClient | None = None,
        proxy: str | None = None,
    ) -> None:
        self._stream_client = stream_client
        self._short_client = short_client
        self._proxy = proxy

    async def stream_chat(self, credential_data: dict[str, Any],
                          payload: dict[str, Any], model: str) -> AsyncIterator[Event]:
        """POST {base_url}/chat/completions 并逐事件产出中立 Event。"""
        async with self._stream.stream(
            "POST", f"{_base_url(credential_data)}{EP_CHAT}",
            json=prepare_body(payload, model), headers=_headers(credential_data),
        ) as response:
            if response.status_code >= 400:
                raw = await base.read_body_bounded(response)
                raise UpstreamHTTPError(response.status_code, raw,
                                        **base.retry_hints(response.headers))
            async for frame in iter_frames(response.aiter_bytes()):
                for event in oc_events.parse_all_events(frame):
                    yield event

    async def fetch_models(self, credential_data: dict[str, Any]) -> list[Model]:
        """GET {base_url}/models → `data[].id` 全量透传（不做免费过滤）。

        返回空列表时显式失败，让上层用上次成功的缓存兜底，避免该渠道
        从模型列表里消失。
        """
        response = await self._short.get(
            f"{_base_url(credential_data)}{EP_MODELS}", headers=_headers(credential_data))
        if response.status_code >= 400:
            raise UpstreamHTTPError(response.status_code,
                                    await base.read_body_bounded(response))
        try:
            data = response.json()
        except ValueError as error:
            raise UpstreamProtocolViolation("non-JSON models response") from error
        items = data.get("data") if isinstance(data, dict) else None
        if not isinstance(items, list):
            raise UpstreamProtocolViolation("models response missing data list")
        models: list[Model] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            name = item.get("name")
            if isinstance(model_id, str) and model_id:
                models.append(Model(id=model_id,
                                    name=name if isinstance(name, str) else ""))
        if not models:
            raise UpstreamProtocolViolation("models api returned no models")
        return models


@dataclass(slots=True)
class OpenAICompatProvider:
    """Provider 协议实现（细接口）。

    只实现对话与模型列表三条能力，签到/成长/刷新靠「方法缺失」自然跳过
    （CheckinTask / GrowthTask / RefreshTask 都用 getattr 探测）。
    """

    client: OpenAICompatClient = field(default_factory=OpenAICompatClient)
    pacer: Any | None = None

    id: str = "openai_compat"

    def import_credential(self, raw: dict) -> dict:
        """{base_url, api_key?, models?} → 规范化 credential_data。"""
        return normalize_credential(raw)

    def classify(self, status: int, body: bytes) -> ErrKind:
        return oc_events.classify_status(status, body)

    async def probe_quota(self, credential_data: dict) -> Quota:
        """各家计费接口无统一形状 → 恒 unknown（绝不返回 total=None 被判已耗尽）。"""
        return Quota(probe_failed=True, probed_at=int(time.time()))

    async def list_models(self, credential_data: dict) -> list[Model]:
        """凭证白名单优先；留空回退上游 `/models`。"""
        static = credential_data.get("models")
        if isinstance(static, list) and static:
            return [Model(id=str(model_id)) for model_id in static]
        return await self.client.fetch_models(credential_data)

    async def stream_chat(self, credential_data: dict, payload: dict,
                          model: str) -> AsyncIterator[Event]:
        async with paced(self.pacer, "openai_compat"):
            async for event in self.client.stream_chat(credential_data, payload, model):
                yield event

    async def aclose(self) -> None:
        await self.client.aclose()
