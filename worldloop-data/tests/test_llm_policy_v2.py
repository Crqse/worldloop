"""Phase 2 (Beta correction §6) — V2 telemetry emission from LLMPolicy.

Verifies that :class:`LLMPolicy.propose` emits:

- One :class:`InferenceAttemptEvent` per LLM API call (including
  retries) to sinks implementing :class:`TelemetrySinkV2`.
- One :class:`InferenceDecisionEvent` per (agent, tick) decision,
  aggregating all attempts (token SUM per §6.2 — NOT last-attempt
  overwrite), with validation split and effective backend.

These tests are orthogonal to the legacy ``InferenceEvent`` tests in
``test_llm_policy.py`` (Phase 0 governance freeze compat). The legacy
event captures only the last attempt's fields; the V2 events capture
every attempt with proper accumulation.
"""

from __future__ import annotations

import random
from typing import Any, Optional, Sequence

import pytest

from worldloop_kernel.action import ActionProposal
from worldloop_kernel.protocol import ActionSpace, LegalAction
from worldloop_data.policy import PolicyContext
from worldloop_data.llm_policy import (
    EchoLLMClient,
    FakeLLMClient,
    InMemoryTelemetrySink,
    LLMClient,
    LLMPolicy,
    LLMPolicyConfig,
    LLMRequest,
    LLMResponse,
)
from worldloop_data.telemetry import (
    RunTier,
    ValidationSummary,
    InferenceAttemptEvent,
    InferenceDecisionEvent,
    check_evidence_fail_closed,
    default_run_level_config,
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
        world=None,  # type: ignore — legacy path for these tests
        agent_id=agent_id,
        state=None,  # type: ignore
        action_space=_make_action_space(actions),
        tick=tick,
        rng=random.Random(42),
    )


class _CountingClient:
    """Fake client that returns unparseable JSON N times then succeeds.

    Each call (including failures) reports token usage, so the test can
    verify that the decision event SUMS tokens across all attempts
    rather than taking the last attempt's value (§6.2 — sum, not
    overwrite). This mirrors real provider behaviour: input tokens are
    charged even when the output is unparseable.
    """

    backend_class: str = "fake"

    def __init__(self, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self.calls = 0
        # Per-call token values to verify sum-not-overwrite.
        self._input_tokens_per_call = [10, 20, 30]
        self._output_tokens_per_call = [5, 7, 9]

    def complete(self, request: LLMRequest) -> LLMResponse:
        idx = self.calls
        self.calls += 1
        tokens_idx = min(idx, len(self._input_tokens_per_call) - 1)
        input_tokens = self._input_tokens_per_call[tokens_idx]
        output_tokens = self._output_tokens_per_call[tokens_idx]
        if idx < self._fail_times:
            # Return unparseable response WITH token usage (provider
            # charges for input even on parse failure).
            return LLMResponse(
                raw_text="not json",
                json_body=None,
                parse_error=f"json_decode_error: simulated on call {idx + 1}",
                finish_reason="stop",
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        return LLMResponse(
            raw_text='{"action_type": "REST", "params": {}}',
            json_body={"action_type": "REST", "params": {}},
            finish_reason="stop",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


class _TimeoutCountingClient:
    """Fake client that raises TimeoutError N times then succeeds.

    Used to test the exception path (no token usage on failed calls).
    """

    backend_class: str = "fake"

    def __init__(self, fail_times: int = 0) -> None:
        self._fail_times = fail_times
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        idx = self.calls
        self.calls += 1
        if idx < self._fail_times:
            raise TimeoutError(f"simulated timeout on call {idx + 1}")
        return LLMResponse(
            raw_text='{"action_type": "REST", "params": {}}',
            json_body={"action_type": "REST", "params": {}},
            finish_reason="stop",
            input_tokens=30,
            output_tokens=9,
        )


class _BrokenJSONClient:
    backend_class: str = "fake"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            raw_text="not json",
            json_body=None,
            parse_error="json_decode_error: simulated",
        )


class _IllegalActionClient:
    backend_class: str = "fake"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            raw_text='{"action_type": "ATTACK"}',
            json_body={"action_type": "ATTACK"},
            finish_reason="stop",
            input_tokens=15,
            output_tokens=8,
        )


def _make_policy(
    client: LLMClient,
    *,
    fallback_mode: str = "decline",
    max_retries: int = 1,
    run_tier: RunTier = RunTier.DEV,
    requested_backend: Optional[str] = None,
    telemetry: Optional[InMemoryTelemetrySink] = None,
    enforce_fail_closed: bool = True,
) -> tuple[LLMPolicy, InMemoryTelemetrySink]:
    sink = telemetry if telemetry is not None else InMemoryTelemetrySink()
    policy = LLMPolicy(
        config=LLMPolicyConfig(
            base_url="http://x",
            model="test-model",
            fallback_mode=fallback_mode,
            max_retries=max_retries,
        ),
        client=client,
        telemetry=sink,
        run_tier=run_tier,
        requested_backend=requested_backend,
        enforce_fail_closed=enforce_fail_closed,
    )
    return policy, sink


# ---------------------------------------------------------------------------
# §6.1 — Per-attempt event emission
# ---------------------------------------------------------------------------

class TestAttemptEventEmission:
    def test_single_success_emits_one_attempt(self) -> None:
        """One successful LLM call → one InferenceAttemptEvent."""
        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        assert len(sink.attempts) == 1
        a = sink.attempts[0]
        assert a.attempt_index == 1
        assert a.parse_status == "ok"
        assert a.retry_disposition == "success"
        assert a.error_category is None
        assert a.effective_backend == "fake"
        assert a.mock_call is False
        assert a.input_tokens > 0
        assert a.output_tokens > 0

    def test_retry_then_success_emits_two_attempts(self) -> None:
        """Fail once, retry, succeed → two InferenceAttemptEvents."""
        client = _TimeoutCountingClient(fail_times=1)
        policy, sink = _make_policy(client, max_retries=2)
        policy.propose(_make_ctx())
        assert len(sink.attempts) == 2
        assert sink.attempts[0].attempt_index == 1
        assert sink.attempts[0].retry_disposition == "retry"
        assert sink.attempts[0].error_category == "TimeoutError"
        assert sink.attempts[0].parse_status == "no_response"
        assert sink.attempts[1].attempt_index == 2
        assert sink.attempts[1].retry_disposition == "success"
        assert sink.attempts[1].error_category is None
        assert sink.attempts[1].parse_status == "ok"

    def test_retry_backoff_recorded_per_attempt(self) -> None:
        """Retry backoff is recorded on the retrying attempt (not the
        first)."""
        client = _TimeoutCountingClient(fail_times=2)
        policy, sink = _make_policy(client, max_retries=3)
        policy.propose(_make_ctx())
        # First attempt: no backoff (it's the initial call).
        assert sink.attempts[0].retry_backoff_ms == 0.0
        # Second attempt: backoff from first failure (0.5 * 1 = 0.5s).
        assert sink.attempts[1].retry_backoff_ms > 0.0
        # Third attempt: backoff from second failure (0.5 * 2 = 1.0s).
        assert sink.attempts[2].retry_backoff_ms > sink.attempts[1].retry_backoff_ms

    def test_all_retries_exhausted_emits_all_attempts(self) -> None:
        """All retries fail → all attempt events emitted before fallback."""
        client = _TimeoutCountingClient(fail_times=10)  # always fails
        policy, sink = _make_policy(
            client, max_retries=2, fallback_mode="first_legal"
        )
        policy.propose(_make_ctx())
        # max_retries=2 → 3 total attempts (1 + 2 retries).
        assert len(sink.attempts) == 3
        for a in sink.attempts:
            assert a.error_category == "TimeoutError"
            assert a.parse_status == "no_response"
        # Last attempt should be give_up, not retry.
        assert sink.attempts[-1].retry_disposition == "give_up"


# ---------------------------------------------------------------------------
# §6.1 — Per-decision event emission
# ---------------------------------------------------------------------------

class TestDecisionEventEmission:
    def test_single_success_emits_one_decision(self) -> None:
        """One successful decision → one InferenceDecisionEvent."""
        policy, sink = _make_policy(FakeLLMClient())
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
        assert len(sink.decisions) == 1
        d = sink.decisions[0]
        assert d.final_origin == "llm"
        assert d.fallback_mode is None
        assert d.attempt_count == 1
        assert d.llm_origin_validation is not None
        assert d.llm_origin_validation.parse_ok is True
        assert d.llm_origin_validation.candidate_ok is True
        assert d.llm_origin_validation.matched_action_type == "REST"
        assert d.fallback_origin_validation is None
        assert d.run_tier == RunTier.DEV.value

    def test_decision_carries_prompt_hashes(self) -> None:
        """Decision event carries all prompt component hashes (even on
        the legacy path — empty for observation/scenario, populated for
        legal_action_space / template / config)."""
        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        # Legacy path: observation / scenario / system hashes are empty.
        assert d.observation_hash == ""
        assert d.scenario_contract_hash == ""
        assert d.system_prompt_hash == ""
        assert d.user_message_hash == ""
        # But legal_action_space / template / config hashes are populated.
        assert d.legal_action_space_hash.startswith("sha256:")
        assert d.prompt_template_hash.startswith("sha256:")
        assert d.inference_config_hash.startswith("sha256:")
        # combined_prompt_hash is populated from the legacy build_llm_prompt hash.
        assert d.combined_prompt_hash  # non-empty

    def test_decision_aggregates_tokens_across_attempts(self) -> None:
        """§6.2: total tokens = SUM across all attempts, not last."""
        client = _CountingClient(fail_times=2)
        policy, sink = _make_policy(client, max_retries=3)
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        # 3 attempts with input_tokens [10, 20, 30] → sum = 60.
        assert d.attempt_count == 3
        assert d.total_input_tokens == 60  # 10 + 20 + 30, NOT 30
        assert d.total_output_tokens == 21  # 5 + 7 + 9, NOT 9

    def test_fallback_decision_emits_fallback_origin(self) -> None:
        """Broken JSON + first_legal fallback → final_origin=fallback."""
        policy, sink = _make_policy(
            _BrokenJSONClient(), fallback_mode="first_legal", max_retries=0
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
        assert proposal.proposer.endswith("_fallback")
        d = sink.decisions[0]
        assert d.final_origin == "fallback"
        assert d.fallback_mode == "first_legal"
        assert d.fallback_origin_validation is not None
        assert d.fallback_origin_validation.parse_ok is True
        assert d.fallback_origin_validation.candidate_ok is True
        assert d.fallback_origin_validation.matched_action_type == "MOVE"
        # LLM-origin validation captures the failed parse.
        assert d.llm_origin_validation is not None
        assert d.llm_origin_validation.parse_ok is False
        assert d.llm_origin_validation.candidate_ok is False

    def test_decline_decision_emits_declined_origin(self) -> None:
        """Broken JSON + decline fallback → final_origin=declined."""
        policy, sink = _make_policy(
            _BrokenJSONClient(), fallback_mode="decline", max_retries=0
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is None
        d = sink.decisions[0]
        assert d.final_origin == "declined"
        assert d.fallback_mode is None
        assert d.fallback_origin_validation is None
        # LLM-origin validation still captures the failed parse.
        assert d.llm_origin_validation is not None
        assert d.llm_origin_validation.parse_ok is False

    def test_no_legal_actions_emits_empty_decision(self) -> None:
        """No legal actions → decision short-circuits with 0 attempts."""
        policy, sink = _make_policy(FakeLLMClient())
        proposal = policy.propose(_make_ctx(actions=[]))
        assert proposal is None
        d = sink.decisions[0]
        assert d.final_origin == "declined"
        assert d.attempt_count == 0
        assert d.total_input_tokens == 0
        assert d.total_output_tokens == 0
        assert d.attempts == ()

    def test_illegal_action_triggers_fallback_with_split_validation(
        self,
    ) -> None:
        """LLM returns illegal action → fallback with split validation."""
        policy, sink = _make_policy(
            _IllegalActionClient(), fallback_mode="first_legal", max_retries=0
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
        assert proposal.proposer.endswith("_fallback")
        d = sink.decisions[0]
        assert d.final_origin == "fallback"
        # LLM parsed OK but candidate was illegal.
        assert d.llm_origin_validation is not None
        assert d.llm_origin_validation.parse_ok is True
        assert d.llm_origin_validation.candidate_ok is False
        # Fallback picked a legal action.
        assert d.fallback_origin_validation is not None
        assert d.fallback_origin_validation.candidate_ok is True


# ---------------------------------------------------------------------------
# §6.3 — Validation split (LLM-origin vs fallback-origin)
# ---------------------------------------------------------------------------

class TestValidationSplit:
    def test_llm_success_populates_llm_origin_only(self) -> None:
        """LLM success → llm_origin_validation populated, fallback None."""
        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        assert d.llm_origin_validation is not None
        assert d.fallback_origin_validation is None

    def test_fallback_populates_fallback_origin_only(self) -> None:
        """Fallback → fallback_origin_validation populated, llm None
        (for the final decision; llm_validation may still capture the
        failed LLM attempt)."""
        policy, sink = _make_policy(
            _BrokenJSONClient(), fallback_mode="first_legal", max_retries=0
        )
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        # final_origin=fallback means fallback_origin_validation is the
        # authoritative validation; llm_origin_validation captures the
        # failed LLM parse (still populated for diagnostics).
        assert d.final_origin == "fallback"
        assert d.fallback_origin_validation is not None
        assert d.fallback_origin_validation.candidate_ok is True


# ---------------------------------------------------------------------------
# §6.4 — Effective backend
# ---------------------------------------------------------------------------

class TestEffectiveBackend:
    def test_fake_client_records_fake_backend(self) -> None:
        """FakeLLMClient → effective_backend='fake' on decision + attempts."""
        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        assert d.effective_backend == "fake"
        assert d.requested_backend == "fake"
        assert all(a.effective_backend == "fake" for a in sink.attempts)

    def test_requested_backend_override_to_real(self) -> None:
        """Caller asserts requested_backend='real' even with fake client
        (used for testing the fail-closed path).

        Uses enforce_fail_closed=False so the decision is recorded for
        inspection without raising. Production runs leave fail-closed on.
        """
        policy, sink = _make_policy(
            FakeLLMClient(),
            requested_backend="real",
            enforce_fail_closed=False,
        )
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        # requested_backend is the caller's assertion.
        assert d.requested_backend == "real"
        # effective_backend is derived from attempts (client is fake).
        assert d.effective_backend == "fake"
        assert d.mock_calls == 0

    def test_evidence_fail_closed_raises_on_fake_when_real_requested(
        self,
    ) -> None:
        """EVIDENCE tier + requested=real + effective=fake → propose()
        raises EvidenceFailClosedError automatically (§6.4).

        The fail-closed check is now integrated into _finalize_decision
        (Attempt 7): the policy exits non-zero instead of silently
        recording an invalid evidence decision.
        """
        policy, sink = _make_policy(
            FakeLLMClient(),
            run_tier=RunTier.EVIDENCE,
            requested_backend="real",
        )
        with pytest.raises(Exception) as exc_info:
            policy.propose(_make_ctx())
        # Error mentions effective_backend mismatch, no API key material.
        assert "effective_backend" in str(exc_info.value).lower() or \
               "fallback" in str(exc_info.value).lower()
        assert "sk-" not in str(exc_info.value).lower()
        # Decision was still recorded in the sink (for forensics) before
        # the fail-closed raise.
        assert len(sink.decisions) == 1
        d = sink.decisions[0]
        assert d.requested_backend == "real"
        assert d.effective_backend == "fake"


# ---------------------------------------------------------------------------
# §6.1 — Run tier context
# ---------------------------------------------------------------------------

class TestRunTierContext:
    def test_dev_tier_default(self) -> None:
        """Default run_tier is DEV."""
        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        assert sink.decisions[0].run_tier == RunTier.DEV.value

    def test_smoke_tier_recorded(self) -> None:
        """SMOKE tier recorded on decision event."""
        policy, sink = _make_policy(
            FakeLLMClient(), run_tier=RunTier.SMOKE
        )
        policy.propose(_make_ctx())
        assert sink.decisions[0].run_tier == RunTier.SMOKE.value

    def test_evidence_tier_recorded(self) -> None:
        """EVIDENCE tier recorded on decision event."""
        policy, sink = _make_policy(
            FakeLLMClient(), run_tier=RunTier.EVIDENCE
        )
        policy.propose(_make_ctx())
        assert sink.decisions[0].run_tier == RunTier.EVIDENCE.value


# ---------------------------------------------------------------------------
# §6.2 — Latency aggregation
# ---------------------------------------------------------------------------

class TestLatencyAggregation:
    def test_decision_latency_spans_first_to_last_attempt(self) -> None:
        """total_latency_ms = last_finished - first_started (end-to-end)."""
        client = _TimeoutCountingClient(fail_times=1)
        policy, sink = _make_policy(client, max_retries=2)
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        # End-to-end latency should be > 0 (includes retry backoff).
        assert d.total_latency_ms > 0.0
        # retry_backoff_total_ms should include the backoff before
        # the second attempt (0.5 * 1 = 0.5s = 500ms).
        assert d.retry_backoff_total_ms >= 400.0  # allow scheduling jitter

    def test_attempt_latency_records_per_call(self) -> None:
        """Each attempt records its own latency_ms."""
        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        a = sink.attempts[0]
        assert a.latency_ms > 0.0
        assert a.started_at <= a.finished_at


# ---------------------------------------------------------------------------
# §6.1 — Decision identity + schema
# ---------------------------------------------------------------------------

class TestDecisionIdentity:
    def test_decision_id_shared_across_attempts(self) -> None:
        """All attempts in a decision share the same decision_id."""
        client = _TimeoutCountingClient(fail_times=2)
        policy, sink = _make_policy(client, max_retries=3)
        policy.propose(_make_ctx())
        d = sink.decisions[0]
        assert d.decision_id  # non-empty
        # All 3 attempts share the same decision_id.
        assert len(sink.attempts) == 3
        assert all(a.decision_id == d.decision_id for a in sink.attempts)

    def test_attempt_ids_unique(self) -> None:
        """Each attempt has a unique attempt_id."""
        client = _TimeoutCountingClient(fail_times=2)
        policy, sink = _make_policy(client, max_retries=3)
        policy.propose(_make_ctx())
        ids = {a.attempt_id for a in sink.attempts}
        assert len(ids) == 3  # all unique

    def test_schema_version_recorded(self) -> None:
        """Decision event records TELEMETRY_SCHEMA_VERSION."""
        from worldloop_data.telemetry import TELEMETRY_SCHEMA_VERSION

        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        assert sink.decisions[0].schema_version == TELEMETRY_SCHEMA_VERSION

    def test_episode_id_propagated(self) -> None:
        """episode_id from policy config is recorded on decision."""
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://x", model="m", fallback_mode="decline"
            ),
            client=FakeLLMClient(),
            telemetry=sink,
            episode_id="ep_42",
        )
        policy.propose(_make_ctx())
        assert sink.decisions[0].episode_id == "ep_42"


# ---------------------------------------------------------------------------
# §6.1 — Legacy + V2 coexistence
# ---------------------------------------------------------------------------

class TestLegacyV2Coexistence:
    def test_both_legacy_and_v2_events_emitted(self) -> None:
        """A single InMemoryTelemetrySink receives BOTH legacy events
        (one per decision) AND V2 events (attempts + decision)."""
        policy, sink = _make_policy(FakeLLMClient())
        policy.propose(_make_ctx())
        # Legacy: one InferenceEvent per decision.
        assert len(sink.events) == 1
        # V2: one InferenceAttemptEvent + one InferenceDecisionEvent.
        assert len(sink.attempts) == 1
        assert len(sink.decisions) == 1

    def test_legacy_event_carries_last_attempt_fields(self) -> None:
        """Legacy event captures the LAST attempt's tokens (buggy but
        preserved for compat). V2 decision event captures the SUM."""
        client = _CountingClient(fail_times=2)
        policy, sink = _make_policy(client, max_retries=3)
        policy.propose(_make_ctx())
        legacy = sink.events[0]
        decision = sink.decisions[0]
        # Legacy event has the last attempt's tokens (30).
        assert legacy.input_tokens == 30
        assert legacy.output_tokens == 9
        # V2 decision event has the SUM (60).
        assert decision.total_input_tokens == 60
        assert decision.total_output_tokens == 21
