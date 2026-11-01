"""openai_compat SSE 事件映射与错误分类。"""

from __future__ import annotations

import pytest

from src.provider.base import ErrKind, EventKind, UpstreamProtocolViolation
from src.provider.openai_compat import events
from tests.helpers.http import chunk_payload as chunk
from tests.helpers.http import sse_frame


class TestParseFrame:
    def test_empty_and_done_frames_are_none(self):
        assert events.parse_frame(sse_frame("")) is None
        assert events.parse_frame(sse_frame("[DONE]")) is None

    def test_content_and_reasoning_variants(self):
        content = events.parse_frame(sse_frame(chunk({"content": "hi"})))
        assert content is not None and content.kind is EventKind.CONTENT
        # DeepSeek/GLM 风格 reasoning_content
        r1 = events.parse_frame(sse_frame(chunk({"reasoning_content": "t"})))
        assert r1 is not None and r1.kind is EventKind.REASONING
        # Kilo 系风格 reasoning
        r2 = events.parse_frame(sse_frame(chunk({"reasoning": "t"})))
        assert r2 is not None and r2.kind is EventKind.REASONING

    def test_error_envelope_to_error_event(self):
        payload = {"error": {"code": 429, "message": "slow down"}}
        event = events.parse_frame(sse_frame(payload))
        assert event is not None and event.kind is EventKind.ERROR
        assert event.error_code == 429 and event.error_kind is ErrKind.MODEL

    def test_error_envelope_string_code(self):
        """字符串 code（insufficient_balance 之类）不进 error_code，按 OTHER。"""
        payload = {"error": {"code": "insufficient_balance", "message": "欠费"}}
        event = events.parse_frame(sse_frame(payload))
        assert event is not None and event.kind is EventKind.ERROR
        assert event.error_code is None and event.error_kind is ErrKind.OTHER
        assert event.error_message == "欠费"

    def test_error_envelope_without_message(self):
        payload = {"error": {"code": 500}}
        event = events.parse_frame(sse_frame(payload))
        assert event is not None and event.kind is EventKind.ERROR
        assert event.error_message == "" and event.error_kind is ErrKind.SOFT

    def test_bad_json_raises_protocol_violation(self):
        with pytest.raises(UpstreamProtocolViolation):
            events.parse_frame(sse_frame("{not json"))

    def test_non_object_payload_raises_protocol_violation(self):
        with pytest.raises(UpstreamProtocolViolation):
            events.parse_frame(sse_frame("[1,2]"))


class TestParseAllEvents:
    def test_merged_frame_emits_content_usage_finish(self):
        """正文 + finish + usage 同帧必须拆成三个事件，finish 不能丢。"""
        payload = chunk({"content": "hi"}, finish="stop",
                        usage={"prompt_tokens": 3, "completion_tokens": 4})
        kinds = [e.kind for e in events.parse_all_events(sse_frame(payload))]
        assert EventKind.CONTENT in kinds
        assert EventKind.USAGE in kinds
        assert EventKind.FINISH in kinds

    def test_usage_only_frame(self):
        payload = {"choices": [], "usage": {"prompt_tokens": 1}}
        got = events.parse_all_events(sse_frame(payload))
        assert [e.kind for e in got] == [EventKind.USAGE]
        assert got[0].usage.input_tokens == 1

    def test_finish_only_frame(self):
        payload = chunk({}, finish="stop")
        got = events.parse_all_events(sse_frame(payload))
        assert [e.kind for e in got] == [EventKind.FINISH]
        assert got[0].finish_reason == "stop"

    def test_done_and_empty_frames_emit_nothing(self):
        assert events.parse_all_events(sse_frame("[DONE]")) == []
        assert events.parse_all_events(sse_frame("")) == []


class TestClassifyStatus:
    """带凭证渠道：401/403 是 key 坏了 → DEAD；与 Kilo 匿名渠道相反。"""

    @pytest.mark.parametrize("status,expected", [
        (401, ErrKind.DEAD),
        (403, ErrKind.DEAD),
        (402, ErrKind.CREDIT),
        (429, ErrKind.MODEL),
        (404, ErrKind.MODEL),
        (400, ErrKind.REQUEST),
        (422, ErrKind.REQUEST),
        (500, ErrKind.SOFT),
        (503, ErrKind.SOFT),
        (599, ErrKind.SOFT),
        (418, ErrKind.OTHER),
        (302, ErrKind.OTHER),
        (200, ErrKind.OTHER),
    ])
    def test_status_table(self, status, expected):
        assert events.classify_status(status, b"") is expected


class TestClassifyErrorCode:
    @pytest.mark.parametrize("code,expected", [
        (401, ErrKind.DEAD),
        (403, ErrKind.DEAD),
        (402, ErrKind.CREDIT),
        (429, ErrKind.MODEL),
        (404, ErrKind.MODEL),
        (400, ErrKind.REQUEST),
        (422, ErrKind.REQUEST),
        (500, ErrKind.SOFT),
        (599, ErrKind.SOFT),
        (418, ErrKind.OTHER),
        (None, ErrKind.OTHER),
    ])
    def test_code_table(self, code, expected):
        assert events.classify_error_code(code) is expected
