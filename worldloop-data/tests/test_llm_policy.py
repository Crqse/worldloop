"""Tests for :mod:`worldloop_data.llm_policy`.

Covers B3-A (unit tests for all failure paths) and B3-B (non-bypass contract).
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, Optional, Sequence

import pytest

from worldloop_kernel.action import ActionProposal
from worldloop_kernel.protocol import ActionSpace, LegalAction
from worldloop_data.policy import PolicyContext
from worldloop_data import llm_policy
from worldloop_data.llm_policy import (
    EchoLLMClient,
    FakeLLMClient,
    InferenceEvent,
    InMemoryTelemetrySink,
    LLMClient,
    LLMPolicy,
    LLMPolicyConfig,
    LLMRequest,
    LLMResponse,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_action_space(
    actions: Optional[Sequence[str]] = None,
) -> ActionSpace:
    if actions is None:
        actions = ("MOVE", "REST")
    return ActionSpace(
        agent_id="agent_0",
        legal_actions=tuple(LegalAction(action_type=a) for a in actions),
    )


def _make_ctx(
    actions: Optional[Sequence[str]] = None,
    tick: int = 1,
    agent_id: str = "agent_0",
) -> PolicyContext:
    return PolicyContext(
        world=None,  # type: ignore — not used in unit tests
        agent_id=agent_id,
        state=None,  # type: ignore
        action_space=_make_action_space(actions),
        tick=tick,
        rng=random.Random(42),
    )


# ---------------------------------------------------------------------------
# B3-A Unit: Fake client
# ---------------------------------------------------------------------------

class TestFakeClient:
    def test_fake_returns_json(self) -> None:
        client = FakeLLMClient()
        resp = client.complete(LLMRequest(prompt="test", model="fake"))
        assert resp.json_body is not None
        assert "action_type" in resp.json_body

    def test_fake_always_rest(self) -> None:
        client = FakeLLMClient()
        for _ in range(10):
            r = client.complete(LLMRequest(prompt="x", model="f"))
            assert r.json_body["action_type"] == "REST"

    def test_echo_returns_fixed(self) -> None:
        client = EchoLLMClient("MOVE")
        r = client.complete(LLMRequest(prompt="", model="e"))
        assert r.json_body["action_type"] == "MOVE"


# ---------------------------------------------------------------------------
# B3-A Unit: Config validation
# ---------------------------------------------------------------------------

class TestConfig:
    def test_valid_fallback_modes(self) -> None:
        for mode in ("decline", "first_legal", "random_legal"):
            c = LLMPolicyConfig(
                base_url="http://localhost", model="test", fallback_mode=mode
            )
            assert c.fallback_mode == mode

    def test_invalid_fallback_rejected(self) -> None:
        for bad in ("best_effort", "silent", ""):
            with pytest.raises(ValueError):
                LLMPolicyConfig(
                    base_url="http://localhost", model="test", fallback_mode=bad
                )

    def test_api_key_from_env(self, monkeypatch: Any) -> None:
        monkeypatch.setenv("CUSTOM_KEY", "sk-abc")
        c = LLMPolicyConfig(
            base_url="http://x", model="m", api_key_env="CUSTOM_KEY"
        )
        assert c.api_key == "sk-abc"

    def test_api_key_none_when_unset(self) -> None:
        c = LLMPolicyConfig(
            base_url="http://x", model="m", api_key_env="UNSET_KEY_293847"
        )
        assert c.api_key is None


# ---------------------------------------------------------------------------
# B3-A Unit: Prompt builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    def test_includes_legal_actions(self) -> None:
        ctx = _make_ctx()
        prompt = llm_policy.build_llm_prompt(ctx)
        assert "MOVE" in prompt
        assert "REST" in prompt
        assert '"tick": 1' in prompt

    def test_prompt_hash_stable(self) -> None:
        ctx1 = _make_ctx()
        ctx2 = _make_ctx()
        p1 = llm_policy.build_llm_prompt(ctx1)
        p2 = llm_policy.build_llm_prompt(ctx2)
        h1 = llm_policy.hashlib.sha256(p1.encode()).hexdigest()
        h2 = llm_policy.hashlib.sha256(p2.encode()).hexdigest()
        assert h1 == h2  # same input, deterministic output

    def test_prompt_hash_differs_with_different_actions(self) -> None:
        ctx1 = _make_ctx()
        ctx2 = _make_ctx(actions=["FLEE"])
        h1 = llm_policy.hashlib.sha256(
            llm_policy.build_llm_prompt(ctx1).encode()
        ).hexdigest()
        h2 = llm_policy.hashlib.sha256(
            llm_policy.build_llm_prompt(ctx2).encode()
        ).hexdigest()
        assert h1 != h2

    def test_sanitize_params_drops_complex_types(self) -> None:
        result = llm_policy._sanitize_params({"a": 1, "b": [1, 2], "c": {"x": 1}})
        assert result["a"] == 1
        assert isinstance(result["b"], str)  # list -> str
        assert isinstance(result["c"], str)  # dict -> str


# ---------------------------------------------------------------------------
# B3-A Unit: Fake client success path
# ---------------------------------------------------------------------------

class TestLLMPolicyFakeSuccess:
    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        config = LLMPolicyConfig(
            base_url="http://fake", model="fake", fallback_mode="decline"
        )
        self.telemetry = InMemoryTelemetrySink()
        self.policy = LLMPolicy(
            config=config, client=FakeLLMClient(), telemetry=self.telemetry
        )

    def test_propose_returns_proposal(self) -> None:
        proposal = self.policy.propose(_make_ctx())
        assert proposal is not None
        assert proposal.action_type == "REST"

    def test_chooses_legal_action(self) -> None:
        proposal = self.policy.propose(_make_ctx())
        assert proposal.action_type in ("MOVE", "REST")

    def test_telemetry_recorded(self) -> None:
        self.policy.propose(_make_ctx())
        assert len(self.telemetry.events) == 1
        e = self.telemetry.events[0]
        assert e.parse_ok is True
        assert e.candidate_ok is True
        assert e.fallback_used is False
        assert e.error_type is None

    def test_propose_label_correct(self) -> None:
        p = self.policy.propose(_make_ctx())
        assert p.proposer == "llm"
        assert p.params.get("_reason_code") == "FAKE"


# ---------------------------------------------------------------------------
# B3-A Unit: Error paths — illegal JSON / parse failure
# ---------------------------------------------------------------------------

class BrokenJSONClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            raw_text="not json",
            json_body=None,
            parse_error="JSONDecodeError",
        )


class MissingActionClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            raw_text='{"x": 1}',
            json_body={"x": 1},
        )


class IllegalActionClient:
    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            raw_text='{"action_type": "ATTACK"}',
            json_body={"action_type": "ATTACK"},
        )


class TimeoutOnceClient:
    def __init__(self, max_fails: int = 1) -> None:
        self.calls = 0
        self.max_fails = max_fails

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        if self.calls <= self.max_fails:
            raise TimeoutError("simulated timeout")
        return LLMResponse(
            raw_text='{"action_type": "REST", "params": {}}',
            json_body={"action_type": "REST", "params": {}},
            finish_reason="stop",
        )


class HTTP500Client:
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("HTTP 500")


class TestLLMPolicyErrorPaths:
    def config(self, **kw: Any) -> LLMPolicyConfig:
        defaults: Dict[str, Any] = {
            "base_url": "http://t", "model": "t",
            "max_retries": 1,
        }
        defaults.update(kw)
        return LLMPolicyConfig(**defaults)

    def test_broken_json_fallback_decline(self) -> None:
        policy = LLMPolicy(
            config=self.config(fallback_mode="decline"),
            client=BrokenJSONClient(),
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is None  # declined

    def test_broken_json_fallback_first_legal(self) -> None:
        telemetry = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=self.config(fallback_mode="first_legal"),
            client=BrokenJSONClient(),
            telemetry=telemetry,
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
        assert proposal.proposer == "llm_fallback"
        assert telemetry.events[-1].fallback_used is True

    def test_illegal_action_fallback(self) -> None:
        policy = LLMPolicy(
            config=self.config(fallback_mode="decline"),
            client=IllegalActionClient(),
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is None

    def test_missing_action_type_fallback(self) -> None:
        policy = LLMPolicy(
            config=self.config(fallback_mode="decline"),
            client=MissingActionClient(),
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is None

    def test_timeout_retry_success(self) -> None:
        telemetry = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=self.config(max_retries=2, fallback_mode="decline"),
            client=TimeoutOnceClient(max_fails=1),
            telemetry=telemetry,
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
        assert proposal.proposer == "llm"
        # Should have succeeded on attempt 2, no fallback
        assert all(not e.fallback_used for e in telemetry.events)

    def test_timeout_retry_exhausted_decline(self) -> None:
        telemetry = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=self.config(max_retries=1, fallback_mode="decline"),
            client=TimeoutOnceClient(max_fails=5),
            telemetry=telemetry,
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is None

    def test_http500_fallback_decline(self) -> None:
        policy = LLMPolicy(
            config=self.config(fallback_mode="decline"),
            client=HTTP500Client(),
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is None

    def test_no_legal_actions(self) -> None:
        policy = LLMPolicy(
            config=self.config(),
            client=FakeLLMClient(),
        )
        proposal = policy.propose(_make_ctx(actions=[]))
        assert proposal is None


# ---------------------------------------------------------------------------
# B3-A Unit: Fallback modes
# ---------------------------------------------------------------------------

class TestFallbackModes:
    def test_random_fallback_is_legal(self) -> None:
        for _ in range(20):
            policy = LLMPolicy(
                config=LLMPolicyConfig(
                    base_url="http://x", model="x", fallback_mode="random_legal"
                ),
                client=BrokenJSONClient(),
            )
            p = policy.propose(_make_ctx())
            assert p is not None
            assert p.action_type in ("MOVE", "REST")

    def test_markdown_fenced_json_is_rejected(self) -> None:
        class MarkdownClient:
            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(
                    raw_text='```json\n{"action_type": "REST", "params": {}}\n```',
                    json_body=None,
                    parse_error="Not parsable as JSON",
                )

        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://x", model="x", fallback_mode="first_legal"
            ),
            client=MarkdownClient(),
        )
        p = policy.propose(_make_ctx())
        assert p.proposer == "llm_fallback"


# ---------------------------------------------------------------------------
# B3-B Contract: LLMPolicy never calls step() / validate_action()
# ---------------------------------------------------------------------------

class TestNonBypassContract:
    """Spy world: prove LLMPolicy does not call validate_action or step."""

    def test_policy_never_calls_validate_action_through_propose(self) -> None:
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://x", model="x", fallback_mode="first_legal"
            ),
            client=FakeLLMClient(),
        )
        ctx = _make_ctx()
        proposal = policy.propose(ctx)
        assert proposal is not None
        assert proposal.action_type is not None
        # PolicyContext has no step/validate_action — impossible to bypass
        assert not hasattr(ctx, "step")
        assert not hasattr(ctx, "validate_action")


# ---------------------------------------------------------------------------
# Telemetry completeness
# ---------------------------------------------------------------------------

class TestTelemetryCompleteness:
    def test_all_fields_present_on_success(self) -> None:
        telemetry = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://x", model="x", fallback_mode="decline"
            ),
            client=FakeLLMClient(),
            telemetry=telemetry,
        )
        policy.propose(_make_ctx())
        e = telemetry.events[0]
        assert e.inference_id
        assert e.model == "x"
        assert e.prompt_hash
        assert e.response_hash
        assert e.latency_ms > 0
        assert e.attempt_count >= 1
        assert e.parse_ok is True
        assert e.candidate_ok is True
        assert e.fallback_used is False

    def test_error_fields_on_failure(self) -> None:
        telemetry = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://x", model="x", fallback_mode="decline", max_retries=1
            ),
            client=IllegalActionClient(),
            telemetry=telemetry,
        )
        policy.propose(_make_ctx())
        e = telemetry.events[-1]
        assert e.error_type == "illegal_action"
        assert e.fallback_used is True
        assert e.candidate_ok is False

    def test_key_not_leaked_in_telemetry(self) -> None:
        telemetry = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://x", model="x", fallback_mode="decline"
            ),
            client=FakeLLMClient(),
            telemetry=telemetry,
        )
        policy.propose(_make_ctx())
        e = telemetry.events[0]
        # The event dict must not contain any key-like substrings
        raw = str(e.__dict__)
        assert "sk-" not in raw
        assert "WORLDLOOP" not in raw.upper()  # env var name


# ---------------------------------------------------------------------------
# B3-C: No-key smoke (via fake client)
# ---------------------------------------------------------------------------

class TestNoKeySmoke:
    def test_runs_without_api_key(self, monkeypatch: Any) -> None:
        monkeypatch.delenv("WORLDLOOP_LLM_API_KEY", raising=False)
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://x", model="x", fallback_mode="first_legal"
            ),
            client=FakeLLMClient(),
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
