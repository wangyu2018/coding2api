"""openai_compat 测试共享装配助手。"""

from __future__ import annotations

from src.provider.openai_compat.client import OpenAICompatClient
from tests.helpers.http import provider_client

__all__ = ["client"]


def client(handler, **kw) -> OpenAICompatClient:
    """用 `handler` 造 OpenAICompatClient（流/短请求共用同一 MockTransport）。"""
    return provider_client(OpenAICompatClient, handler, **kw)


CREDENTIAL = {
    "base_url": "https://compat.test/v1",
    "api_key": "sk-test",
    "models": ["glm-4.6"],
}
"""一份标准凭证形状：测试集中用一处定义，避免各文件散落字面量。"""
