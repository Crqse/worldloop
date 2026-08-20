"""Tests for :class:`worldloop_data.llm_policy.OpenAICompatibleClient`.

B3 close-out: mock-transport unit tests — success, 429/5xx/timeout error
mapping, usage back-fill, multi-key env resolution. NO real API calls.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import pytest

from worldloop_data.llm_policy import (
    LLMAuthError,
    LLMClientError,
    LLMProtocolError,
    LLMRateLimitError,
    LLMRequest,
    LLMServerError,
    LLMTimeoutError,
    OpenAICompatibleClient,
    _extract_json_object,
    _first_api_key,
)

_ENV = "TEST_LLM_CLIENT_KEY"
_ACTION_JSON = '{"action_type": "REST", "params": {}, "reason_code": "OK"}'


def _envelope(
    content: str = _ACTION_JSON,
    prompt_tokens: int = 123,
    completion_tokens: int = 45,
    finish_reason: str = "stop",
) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    ).encode("utf-8")


class _RecordingTransport:
    """Mock transport that records the request and returns a canned
    (status, body) — or raises — without touching the network.
    """

    def __init__(
        self,
        status: int = 200,
        body: bytes = b"",
        exc: Optional[BaseException] = None,
    ) -> None:
        self.status = status
        self.body = body
        self.exc = exc
        self.calls: List[Dict[str, Any]] = []

    def __call__(
        self,
        url: str,
        body: bytes,
        headers: Dict[str, str],
        timeout_seconds: float,
    ) -> Tuple[int, bytes]:
        self.calls.append(
            {
                "url": url,
                "payload": json.loads(body.decode("utf-8")),
                "headers": dict(headers),
                "timeout": timeout_seconds,
            }
        )
        if self.exc is not None:
            raise self.exc
        return self.status, self.body


def _client(transport: _RecordingTransport, **kw: Any) -> OpenAICompatibleClient:
    kw.setdefault("base_url", "https://api.example.test/v1")
    kw.setdefault("api_key_env", _ENV)
    return OpenAICompatibleClient(transport=transport, **kw)


def _request(**kw: Any) -> LLMRequest:
    kw.setdefault("prompt", '{"tick": 1}')
    kw.setdefault("model", "test-model")
    return LLMRequest(**kw)


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


class TestSuccess:
    def test_success_parses_action_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        transport = _RecordingTransport(200, _envelope())
        resp = _client(transport).complete(_request())
        assert resp.json_body == {
            "action_type": "REST",
            "params": {},
            "reason_code": "OK",
        }
        assert resp.parse_error is None
        assert resp.finish_reason == "stop"
        assert resp.raw_text == _ACTION_JSON

    def test_request_shape_and_url(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        transport = _RecordingTransport(200, _envelope())
        client = _client(transport, timeout_seconds=30.0)
        client.complete(
            _request(system_prompt="sys", temperature=0.0, max_tokens=256)
        )
        call = transport.calls[0]
        assert call["url"] == "https://api.example.test/v1/chat/completions"
        assert call["timeout"] == 30.0
        assert call["payload"]["model"] == "test-model"
        assert call["payload"]["temperature"] == 0.0
        assert call["payload"]["max_tokens"] == 256
        assert call["payload"]["stream"] is False
        assert call["payload"]["messages"] == [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": '{"tick": 1}'},
        ]
        assert call["headers"]["Authorization"] == "Bearer sk-single"

    def test_usage_backfilled_from_response(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        transport = _RecordingTransport(
            200, _envelope(prompt_tokens=321, completion_tokens=99)
        )
        resp = _client(transport).complete(_request())
        assert resp.input_tokens == 321
        assert resp.output_tokens == 99

    def test_missing_usage_defaults_to_zero(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        body = json.dumps(
            {"choices": [{"message": {"content": _ACTION_JSON}}]}
        ).encode("utf-8")
        resp = _client(_RecordingTransport(200, body)).complete(_request())
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.json_body is not None

    def test_fenced_json_content_tolerated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        fenced = f"```json\n{_ACTION_JSON}\n```"
        resp = _client(_RecordingTransport(200, _envelope(content=fenced))).complete(
            _request()
        )
        assert resp.json_body is not None
        assert resp.json_body["action_type"] == "REST"


# ---------------------------------------------------------------------------
# HTTP / transport error mapping
# ---------------------------------------------------------------------------


class TestErrorMapping:
    def test_429_raises_rate_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        with pytest.raises(LLMRateLimitError):
            _client(_RecordingTransport(429, b'{"error": "slow down"}')).complete(
                _request()
            )

    @pytest.mark.parametrize("status", [500, 502, 503])
    def test_5xx_raises_server_error(
        self, status: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        with pytest.raises(LLMServerError):
            _client(_RecordingTransport(status, b"upstream sad")).complete(_request())

    @pytest.mark.parametrize("status", [401, 403])
    def test_auth_rejection_names_env_var_only(
        self, status: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-SECRET-VALUE")
        with pytest.raises(LLMAuthError) as excinfo:
            _client(_RecordingTransport(status, b"{}")).complete(_request())
        assert _ENV in str(excinfo.value)
        assert "sk-SECRET-VALUE" not in str(excinfo.value)

    def test_timeout_maps_to_llm_timeout_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        transport = _RecordingTransport(exc=TimeoutError("timed out"))
        with pytest.raises(LLMTimeoutError):
            _client(transport, timeout_seconds=30.0).complete(_request())

    def test_url_error_with_timeout_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.error

        monkeypatch.setenv(_ENV, "sk-single")
        transport = _RecordingTransport(
            exc=urllib.error.URLError(TimeoutError("timed out"))
        )
        with pytest.raises(LLMTimeoutError):
            _client(transport).complete(_request())

    def test_url_error_other_maps_to_client_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import urllib.error

        monkeypatch.setenv(_ENV, "sk-single")
        transport = _RecordingTransport(
            exc=urllib.error.URLError(ConnectionRefusedError("no"))
        )
        with pytest.raises(LLMClientError):
            _client(transport).complete(_request())

    def test_non_json_envelope_raises_protocol_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        with pytest.raises(LLMProtocolError):
            _client(_RecordingTransport(200, b"<html>gateway</html>")).complete(
                _request()
            )

    def test_empty_choices_raises_protocol_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, "sk-single")
        with pytest.raises(LLMProtocolError):
            _client(
                _RecordingTransport(200, json.dumps({"choices": []}).encode())
            ).complete(_request())

    def test_bad_content_json_returns_parse_error_no_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Content-level JSON failure maps to LLMResponse.parse_error
        (existing semantics), NOT an exception."""
        monkeypatch.setenv(_ENV, "sk-single")
        resp = _client(
            _RecordingTransport(200, _envelope(content="I choose REST!"))
        ).complete(_request())
        assert resp.json_body is None
        assert resp.parse_error is not None
        assert resp.raw_text == "I choose REST!"


# ---------------------------------------------------------------------------
# API key resolution (multi-key env values)
# ---------------------------------------------------------------------------


class TestKeyResolution:
    def test_multi_key_takes_first_nonempty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, " , sk-first , sk-second,sk-third")
        transport = _RecordingTransport(200, _envelope())
        _client(transport).complete(_request())
        assert transport.calls[0]["headers"]["Authorization"] == "Bearer sk-first"

    def test_single_key_passthrough(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(_ENV, "sk-only")
        assert _first_api_key(_ENV) == "sk-only"

    def test_unset_env_raises_auth_error_naming_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(_ENV, raising=False)
        with pytest.raises(LLMAuthError) as excinfo:
            _client(_RecordingTransport(200, _envelope())).complete(_request())
        assert _ENV in str(excinfo.value)

    def test_all_empty_entries_raise_auth_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(_ENV, " ,, , ")
        with pytest.raises(LLMAuthError):
            _first_api_key(_ENV)

    def test_key_never_in_error_message(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        secret = "sk-DO-NOT-LEAK-0xCAFE"
        monkeypatch.setenv(_ENV, secret)
        transport = _RecordingTransport(500, b"boom")
        with pytest.raises(LLMServerError) as excinfo:
            _client(transport).complete(_request())
        assert secret not in str(excinfo.value)


# ---------------------------------------------------------------------------
# Content extraction helper
# ---------------------------------------------------------------------------


class TestExtractJsonObject:
    def test_plain_object(self) -> None:
        body, err = _extract_json_object('{"a": 1}')
        assert body == {"a": 1}
        assert err is None

    def test_fenced_with_language_tag(self) -> None:
        body, err = _extract_json_object('```json\n{"a": 1}\n```')
        assert body == {"a": 1}
        assert err is None

    def test_non_object_json_rejected(self) -> None:
        body, err = _extract_json_object("[1, 2, 3]")
        assert body is None
        assert err is not None and "json_not_object" in err

    def test_garbage_rejected(self) -> None:
        body, err = _extract_json_object("not json at all")
        assert body is None
        assert err is not None and "json_decode_error" in err
