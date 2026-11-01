"""OpenAI 兼容上游 SSE → 中立事件映射。

兼容端点（GLM / DeepSeek / MiniMax / 豆包方舟 / devin-2api / one-api 等）都走
标准 `data: {chat.completion.chunk}` + `data: [DONE]`。差异只在两处，全部
参数化处理：

* 思考字段名不统一——DeepSeek/GLM 用 `reasoning_content`，Kilo 系网关用
  `reasoning`，两个键都认；
* 流内错误信封 `{"error": {...}}` 的 `code` 可能是 int 或字符串（
  `insufficient_balance` 之类）——只有 int 码进分类码表，其余按 OTHER。
"""

from __future__ import annotations

from typing import Any

from ...engine.sse import SSEFrame
from ...provider.base import (
    ErrKind,
    Event,
    EventKind,
    UpstreamProtocolViolation,  # noqa: F401 - 测试经本模块访问
    Usage,
)
from ..openai_chunk import openai_event, openai_events, payload_of, usage_from_openai

_REASONING_KEYS = ("reasoning_content", "reasoning")


def parse_frame(frame: SSEFrame) -> Event | None:
    """单帧 → 单个代表性事件；`[DONE]` 与无内容帧返回 None。"""
    payload = payload_of(frame, "OpenAI-Compat")
    return None if payload is None else openai_event(
        payload, reasoning_keys=_REASONING_KEYS, error_event=_error_event,
        usage=_usage)


def parse_all_events(frame: SSEFrame) -> list[Event]:
    """一帧拆成多个事件：同一 chunk 可同时携带正文/思考与 finish_reason。"""
    return openai_events(frame, "OpenAI-Compat", reasoning_keys=_REASONING_KEYS,
                         error_event=_error_event, usage=_usage)


def _error_event(error: dict[str, Any]) -> Event:
    """流内错误信封（`{"error": {...}}`）。

    `error.code` 可能是 int 也可能是字符串（如 `insufficient_balance`）；
    只有 int 才进 error_code，其余按消息分类。
    """
    raw_code = error.get("code")
    code = (raw_code if isinstance(raw_code, int) and not isinstance(raw_code, bool)
            else None)
    message = error.get("message")
    return Event(
        kind=EventKind.ERROR,
        error_code=code,
        error_message=str(message) if message is not None else "",
        error_kind=classify_error_code(code),
    )


def _usage(raw: dict[str, Any]) -> Usage:
    return usage_from_openai(raw)


def classify_status(status: int, body: bytes = b"") -> ErrKind:
    """HTTP 状态码分类（**带凭证**渠道，与 Kilo 的匿名语义相反）。

    * 401/403 → DEAD：key 无效/被拒是凭证本身坏了，硬禁用等用户改凭证，
      归 INVALID 会反复拿坏 key 重试；
    * 402 → CREDIT：余额不足（GLM/DeepSeek 欠费）按额度耗尽冷却；
    * 429 → MODEL：限流一般点名具体模型（共享池同 Kilo 实测），模型级冷却
      不连累同凭证的其他模型；
    * 404 → MODEL：模型名打错或上游没上架该模型，模型级跳过；
    * 400/422 → REQUEST：请求体形状问题，不罚凭证不罚模型；
    * 5xx → SOFT：上游瞬时故障，短冷却重试。
    """
    if status in (401, 403):
        return ErrKind.DEAD
    if status == 402:
        return ErrKind.CREDIT
    if status in (429, 404):
        return ErrKind.MODEL
    if status in (400, 422):
        return ErrKind.REQUEST
    if 500 <= status < 600:
        return ErrKind.SOFT
    return ErrKind.OTHER


def classify_error_code(code: int | None) -> ErrKind:
    """流内 int 错误码 → 错误分类，与 `classify_status` 同口径。"""
    if code in (401, 403):
        return ErrKind.DEAD
    if code == 402:
        return ErrKind.CREDIT
    if code in (429, 404):
        return ErrKind.MODEL
    if code in (400, 422):
        return ErrKind.REQUEST
    if code is not None and 500 <= code < 600:
        return ErrKind.SOFT
    return ErrKind.OTHER
