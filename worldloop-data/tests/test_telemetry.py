"""Phase 2 / Beta correction §6 — tests for telemetry event schema.

Validates the split of ``InferenceEvent`` into
:class:`InferenceAttemptEvent` (per LLM call) and
:class:`InferenceDecisionEvent` (per agent per tick, aggregates
attempts), with:

- Token accumulation across all attempts (§6.2 — sum, NOT last-attempt
  overwrite).
- Validation split: LLM-origin vs fallback-origin (§6.3).
- Evidence-tier fail-closed (§6.4): requested real / effective mock →
  raises :class:`EvidenceFailClosedError`.
- Effective backend derivation from attempts.
- Hash helpers stable across runs.
- ``latency_split`` aggregator (§6.2 — provider per attempt, retry
  backoff, decision e2e, fallback decision, successful LLM-origin).
"""

from __future__ import annotations

from typing import Tuple

import pytest

from worldloop_data.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    EvidenceFailClosedError,
    InferenceAttemptEvent,
    InferenceDecisionEvent,
    RunLevelConfig,
    RunTier,
    ValidationSummary,
    accumulate_attempt_tokens,
    build_decision_from_attempts,
    check_evidence_fail_closed,
    check_evidence_fail_closed_batch,
    default_run_level_config,
    derive_effective_backend,
    hash_inference_config,
    hash_legal_action_space,
    hash_prompt_template,
    latency_split,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_attempt(
    *,
    decision_id: str = "d0",
    attempt_index: int = 1,
    input_tokens: int = 10,
    output_tokens: int = 5,
    cached_tokens: int = 0,
    latency_ms: float = 100.0,
    retry_backoff_ms: float = 0.0,
    started_at: float = 0.0,
    finished_at: float = 0.1,
    effective_backend: str = "real",
    mock_call: bool = False,
    error_category: str | None = None,
    finish_reason: str | None = "stop",
    response_hash: str = "sha256:abc",
    parse_status: str = "ok",
    retry_disposition: str = "success",
) -> InferenceAttemptEvent:
    return InferenceAttemptEvent(
        attempt_id=f"a{attempt_index}",
        decision_id=decision_id,
        attempt_index=attempt_index,
        provider="openai_compatible",
        endpoint_class="OpenAICompatibleClient",
        model="test-model",
        provider_request_id="req-123",
        started_at=started_at,
        finished_at=finished_at,
        latency_ms=latency_ms,
        retry_backoff_ms=retry_backoff_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        finish_reason=finish_reason,
        response_hash=response_hash,
        parse_status=parse_status,
        parse_error=None,
        error_category=error_category,
        retry_disposition=retry_disposition,
        effective_backend=effective_backend,
        mock_call=mock_call,
    )


def _make_prompt_hashes() -> dict[str, str]:
    return {
        "observation_hash": "sha256:obs",
        "legal_action_space_hash": "sha256:legal",
        "system_prompt_hash": "sha256:sys",
        "scenario_contract_hash": "sha256:scenario",
        "user_message_hash": "sha256:user",
        "combined_prompt_hash": "sha256:combined",
        "prompt_template_hash": "sha256:prompt_template:0.1.0",
    }


# ---------------------------------------------------------------------------
# 1. Schema invariants
# ---------------------------------------------------------------------------


class TestSchemaInvariants:
    def test_schema_version_is_string(self):
        assert isinstance(TELEMETRY_SCHEMA_VERSION, str)
        assert TELEMETRY_SCHEMA_VERSION == "0.1.0"

    def test_run_tier_values(self):
        assert RunTier.DEV.value == "dev"
        assert RunTier.SMOKE.value == "smoke"
        assert RunTier.EVIDENCE.value == "evidence"
        assert RunTier.SAFETY_DEMO.value == "safety_demo"

    def test_inference_attempt_event_is_frozen(self):
        a = _make_attempt()
        with pytest.raises(Exception):
            a.input_tokens = 999  # type: ignore[misc]

    def test_inference_decision_event_is_frozen(self):
        d = build_decision_from_attempts(
            (_make_attempt(),),
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.EVIDENCE,
        )
        with pytest.raises(Exception):
            d.final_origin = "fallback"  # type: ignore[misc]

    def test_validation_summary_is_frozen(self):
        v = ValidationSummary(parse_ok=True, candidate_ok=True)
        with pytest.raises(Exception):
            v.parse_ok = False  # type: ignore[misc]

    def test_run_level_config_is_frozen(self):
        c = RunLevelConfig(tier=RunTier.EVIDENCE)
        with pytest.raises(Exception):
            c.tier = RunTier.DEV  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Token accumulation (§6.2 — sum, NOT last-attempt overwrite)
# ---------------------------------------------------------------------------


class TestTokenAccumulation:
    def test_empty_attempts_returns_zeros(self):
        result = accumulate_attempt_tokens(())
        assert result == (0, 0, 0, 0.0, 0.0)

    def test_single_attempt_tokens_passed_through(self):
        a = _make_attempt(input_tokens=15, output_tokens=8, cached_tokens=3)
        result = accumulate_attempt_tokens((a,))
        assert result == (15, 8, 3, 0.0, 100.0)

    def test_multiple_attempts_summed_not_overwritten(self):
        """§6.2: total tokens = sum across ALL attempts, not last."""
        a1 = _make_attempt(
            attempt_index=1,
            input_tokens=10,
            output_tokens=5,
            started_at=0.0,
            finished_at=0.1,
            latency_ms=100.0,
        )
        a2 = _make_attempt(
            attempt_index=2,
            input_tokens=12,
            output_tokens=7,
            started_at=0.2,
            finished_at=0.35,
            latency_ms=150.0,
            retry_backoff_ms=50.0,
        )
        a3 = _make_attempt(
            attempt_index=3,
            input_tokens=14,
            output_tokens=9,
            started_at=0.4,
            finished_at=0.55,
            latency_ms=150.0,
            retry_backoff_ms=50.0,
        )
        total_input, total_output, total_cached, retry_total, total_latency = (
            accumulate_attempt_tokens((a1, a2, a3))
        )
        # Sum, not last attempt.
        assert total_input == 36  # 10 + 12 + 14, NOT 14
        assert total_output == 21  # 5 + 7 + 9, NOT 9
        assert total_cached == 0
        # Retry backoff summed.
        assert retry_total == 100.0  # 0 + 50 + 50
        # End-to-end latency = last_finished - first_started.
        assert total_latency == 550.0  # (0.55 - 0.0) * 1000


# ---------------------------------------------------------------------------
# 3. build_decision_from_attempts correctness
# ---------------------------------------------------------------------------


class TestBuildDecisionFromAttempts:
    def test_decision_carries_all_prompt_hashes(self):
        hashes = _make_prompt_hashes()
        d = build_decision_from_attempts(
            (_make_attempt(),),
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=hashes,
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.EVIDENCE,
        )
        assert d.observation_hash == "sha256:obs"
        assert d.legal_action_space_hash == "sha256:legal"
        assert d.system_prompt_hash == "sha256:sys"
        assert d.scenario_contract_hash == "sha256:scenario"
        assert d.user_message_hash == "sha256:user"
        assert d.combined_prompt_hash == "sha256:combined"
        assert d.prompt_template_hash == "sha256:prompt_template:0.1.0"

    def test_decision_token_fields_are_sums(self):
        attempts = (
            _make_attempt(attempt_index=1, input_tokens=10, output_tokens=5),
            _make_attempt(attempt_index=2, input_tokens=20, output_tokens=10),
        )
        d = build_decision_from_attempts(
            attempts,
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.EVIDENCE,
        )
        assert d.total_input_tokens == 30
        assert d.total_output_tokens == 15
        assert d.attempt_count == 2

    def test_decision_effective_backend_all_real(self):
        attempts = (
            _make_attempt(effective_backend="real"),
            _make_attempt(effective_backend="real"),
        )
        d = build_decision_from_attempts(
            attempts,
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.EVIDENCE,
        )
        assert d.effective_backend == "real"
        assert d.mock_calls == 0

    def test_decision_effective_backend_one_fake_makes_fake(self):
        """If ANY attempt is fake, the decision is fake (§6.4)."""
        attempts = (
            _make_attempt(effective_backend="real"),
            _make_attempt(effective_backend="fake", attempt_index=2),
        )
        d = build_decision_from_attempts(
            attempts,
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.EVIDENCE,
        )
        assert d.effective_backend == "fake"

    def test_decision_effective_backend_mock_overrides_fake(self):
        """If any attempt is mock, effective is mock (highest priority)."""
        attempts = (
            _make_attempt(effective_backend="real"),
            _make_attempt(effective_backend="fake", attempt_index=2),
            _make_attempt(effective_backend="mock", attempt_index=3, mock_call=True),
        )
        d = build_decision_from_attempts(
            attempts,
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.EVIDENCE,
        )
        assert d.effective_backend == "mock"
        assert d.mock_calls == 1

    def test_decision_empty_attempts_effective_equals_requested(self):
        d = build_decision_from_attempts(
            (),
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="declined",
            fallback_mode=None,
            llm_origin_validation=None,
            fallback_origin_validation=None,
            requested_backend="fake",
            run_tier=RunTier.DEV,
        )
        assert d.effective_backend == "fake"
        assert d.mock_calls == 0
        assert d.attempt_count == 0
        assert d.total_input_tokens == 0

    def test_decision_validation_split_llm_origin(self):
        """When final_origin == 'llm', only llm_origin_validation is set."""
        d = build_decision_from_attempts(
            (_make_attempt(),),
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(
                parse_ok=True,
                candidate_ok=True,
                matched_action_type="rest",
            ),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.EVIDENCE,
        )
        assert d.llm_origin_validation is not None
        assert d.llm_origin_validation.matched_action_type == "rest"
        assert d.fallback_origin_validation is None
        assert d.fallback_mode is None

    def test_decision_validation_split_fallback_origin(self):
        """When final_origin == 'fallback', only fallback_origin_validation is set."""
        d = build_decision_from_attempts(
            (_make_attempt(),),
            decision_id="d0",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="fallback",
            fallback_mode="first_legal",
            llm_origin_validation=None,
            fallback_origin_validation=ValidationSummary(
                parse_ok=True,
                candidate_ok=True,
                matched_action_type="rest",
            ),
            requested_backend="real",
            run_tier=RunTier.SMOKE,
        )
        assert d.fallback_origin_validation is not None
        assert d.llm_origin_validation is None
        assert d.fallback_mode == "first_legal"


# ---------------------------------------------------------------------------
# 4. derive_effective_backend
# ---------------------------------------------------------------------------


class TestDeriveEffectiveBackend:
    def test_empty_returns_requested(self):
        backend, mocks = derive_effective_backend((), "real")
        assert backend == "real"
        assert mocks == 0

    def test_all_real_returns_real(self):
        attempts = (_make_attempt(effective_backend="real"),)
        backend, mocks = derive_effective_backend(attempts, "real")
        assert backend == "real"
        assert mocks == 0

    def test_one_fake_makes_fake(self):
        attempts = (
            _make_attempt(effective_backend="real"),
            _make_attempt(effective_backend="fake", attempt_index=2),
        )
        backend, mocks = derive_effective_backend(attempts, "real")
        assert backend == "fake"

    def test_mock_overrides(self):
        attempts = (
            _make_attempt(effective_backend="real"),
            _make_attempt(effective_backend="fake", attempt_index=2),
            _make_attempt(
                effective_backend="mock",
                attempt_index=3,
                mock_call=True,
            ),
        )
        backend, mocks = derive_effective_backend(attempts, "real")
        assert backend == "mock"
        assert mocks == 1


# ---------------------------------------------------------------------------
# 5. default_run_level_config
# ---------------------------------------------------------------------------


class TestDefaultRunLevelConfig:
    def test_dev_allows_fallback_no_fail_closed(self):
        c = default_run_level_config(RunTier.DEV)
        assert c.tier is RunTier.DEV
        assert c.fallback_allowed is True
        assert c.fallback_origin_rate_max == 1.0
        assert c.fail_closed_on_mock is False
        assert c.fail_closed_on_missing_key is False

    def test_smoke_allows_fallback_with_cap(self):
        c = default_run_level_config(RunTier.SMOKE)
        assert c.fallback_allowed is True
        assert c.fallback_origin_rate_max == 0.5
        assert c.fail_closed_on_mock is True
        assert c.fail_closed_on_missing_key is False

    def test_evidence_forbids_fallback_and_fail_closed(self):
        c = default_run_level_config(RunTier.EVIDENCE)
        assert c.fallback_allowed is False
        assert c.fallback_origin_rate_max == 0.0
        assert c.fail_closed_on_mock is True
        assert c.fail_closed_on_missing_key is True

    def test_safety_demo_allows_fallback_no_fail_closed(self):
        c = default_run_level_config(RunTier.SAFETY_DEMO)
        assert c.fallback_allowed is True
        assert c.fallback_origin_rate_max == 1.0
        assert c.fail_closed_on_mock is False


# ---------------------------------------------------------------------------
# 6. Evidence fail-closed (§6.4)
# ---------------------------------------------------------------------------


class TestEvidenceFailClosed:
    def _make_real_decision(
        self,
        *,
        effective_backend: str = "real",
        mock_calls: int = 0,
        final_origin: str = "llm",
        attempts: Tuple[InferenceAttemptEvent, ...] = (),
    ) -> InferenceDecisionEvent:
        """Build a decision with explicit backend fields for fail-closed tests."""
        return InferenceDecisionEvent(
            decision_id="d0",
            tick=0,
            agent_id="a0",
            attempts=attempts,
            final_origin=final_origin,
            requested_backend="real",
            effective_backend=effective_backend,
            mock_calls=mock_calls,
            run_tier=RunTier.EVIDENCE.value,
        )

    def test_non_evidence_tier_no_check(self):
        """Non-evidence tiers never raise."""
        d = self._make_real_decision(effective_backend="mock", mock_calls=5)
        d = InferenceDecisionEvent(
            decision_id="d0",
            tick=0,
            agent_id="a0",
            attempts=(),
            final_origin="llm",
            requested_backend="real",
            effective_backend="mock",
            mock_calls=5,
            run_tier=RunTier.DEV.value,
        )
        config = default_run_level_config(RunTier.DEV)
        # Should NOT raise.
        check_evidence_fail_closed(d, config)

    def test_evidence_requested_fake_no_check(self):
        """Evidence tier but requested=fake → no fail-closed (only real-triggered)."""
        d = InferenceDecisionEvent(
            decision_id="d0",
            tick=0,
            agent_id="a0",
            attempts=(),
            final_origin="llm",
            requested_backend="fake",
            effective_backend="fake",
            mock_calls=0,
            run_tier=RunTier.EVIDENCE.value,
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        check_evidence_fail_closed(d, config)  # no raise

    def test_evidence_real_real_no_mock_passes(self):
        d = self._make_real_decision(
            effective_backend="real",
            mock_calls=0,
            attempts=(_make_attempt(effective_backend="real"),),
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        check_evidence_fail_closed(d, config)  # no raise

    def test_evidence_real_mock_raises(self):
        """§6.4: requested=real + effective=mock → raise."""
        d = self._make_real_decision(
            effective_backend="mock",
            mock_calls=2,
            attempts=(
                _make_attempt(effective_backend="mock", mock_call=True),
            ),
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        with pytest.raises(EvidenceFailClosedError) as exc_info:
            check_evidence_fail_closed(d, config)
        # Error message must NOT contain API key material.
        assert "api_key" not in str(exc_info.value).lower()
        assert "mock" in str(exc_info.value).lower()

    def test_evidence_real_fake_raises(self):
        """§6.4: requested=real + effective=fake → raise."""
        d = self._make_real_decision(
            effective_backend="fake",
            attempts=(
                _make_attempt(effective_backend="fake"),
            ),
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        with pytest.raises(EvidenceFailClosedError):
            check_evidence_fail_closed(d, config)

    def test_evidence_real_with_auth_error_raises(self):
        """§6.4: missing key (LLMAuthError) cannot silently degrade to mock."""
        d = self._make_real_decision(
            effective_backend="real",
            attempts=(
                _make_attempt(error_category="LLMAuthError"),
            ),
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        with pytest.raises(EvidenceFailClosedError) as exc_info:
            check_evidence_fail_closed(d, config)
        # Error message must NOT contain API key value.
        assert "Authorization" not in str(exc_info.value)
        assert "Bearer" not in str(exc_info.value)

    def test_evidence_fallback_when_not_allowed_raises(self):
        """§6.3: EVIDENCE tier + fallback not allowed → raise."""
        d = self._make_real_decision(
            effective_backend="real",
            final_origin="fallback",
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        with pytest.raises(EvidenceFailClosedError):
            check_evidence_fail_closed(d, config)


# ---------------------------------------------------------------------------
# 7. Batch fail-closed (fallback_origin_rate threshold)
# ---------------------------------------------------------------------------


class TestBatchFailClosed:
    def test_empty_batch_no_raise(self):
        config = default_run_level_config(RunTier.EVIDENCE)
        check_evidence_fail_closed_batch((), config)

    def test_evidence_batch_within_rate_passes(self):
        decisions = tuple(
            InferenceDecisionEvent(
                decision_id=f"d{i}",
                tick=i,
                agent_id="a0",
                final_origin="llm",
                requested_backend="real",
                effective_backend="real",
                mock_calls=0,
                run_tier=RunTier.EVIDENCE.value,
            )
            for i in range(10)
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        check_evidence_fail_closed_batch(decisions, config)  # no raise

    def test_smoke_batch_exceeding_rate_raises(self):
        """SMOKE tier: fallback allowed but rate capped at 0.5.

        If fallback rate exceeds threshold, batch raises. SMOKE is the
        right tier for this test because EVIDENCE forbids any fallback
        per-decision (so per-decision check fires before rate check).
        """
        decisions = tuple(
            InferenceDecisionEvent(
                decision_id=f"d{i}",
                tick=i,
                agent_id="a0",
                # 6/10 = 0.6 > 0.5 threshold.
                final_origin="fallback" if i < 6 else "llm",
                requested_backend="real",
                effective_backend="real",
                mock_calls=0,
                run_tier=RunTier.SMOKE.value,
            )
            for i in range(10)
        )
        config = default_run_level_config(RunTier.SMOKE)
        with pytest.raises(EvidenceFailClosedError) as exc_info:
            check_evidence_fail_closed_batch(decisions, config)
        assert "fallback_origin_rate" in str(exc_info.value)

    def test_smoke_batch_within_rate_passes(self):
        """SMOKE tier with fallback rate at exactly the cap (0.5) passes."""
        decisions = tuple(
            InferenceDecisionEvent(
                decision_id=f"d{i}",
                tick=i,
                agent_id="a0",
                # 5/10 = 0.5 == threshold, not > threshold.
                final_origin="fallback" if i < 5 else "llm",
                requested_backend="real",
                effective_backend="real",
                mock_calls=0,
                run_tier=RunTier.SMOKE.value,
            )
            for i in range(10)
        )
        config = default_run_level_config(RunTier.SMOKE)
        check_evidence_fail_closed_batch(decisions, config)  # no raise

    def test_dev_batch_no_rate_check(self):
        """DEV tier: rate_max=1.0, no rate check ever fires."""
        decisions = tuple(
            InferenceDecisionEvent(
                decision_id=f"d{i}",
                tick=i,
                agent_id="a0",
                final_origin="fallback",  # 100% fallback rate
                requested_backend="real",
                effective_backend="real",
                mock_calls=0,
                run_tier=RunTier.DEV.value,
            )
            for i in range(10)
        )
        config = default_run_level_config(RunTier.DEV)
        check_evidence_fail_closed_batch(decisions, config)  # no raise


# ---------------------------------------------------------------------------
# 8. Hash helpers
# ---------------------------------------------------------------------------


class TestHashHelpers:
    def test_hash_legal_action_space_stable(self):
        from worldloop_kernel.protocol import LegalAction

        legal = (LegalAction(action_type="rest"), LegalAction(action_type="forage"))
        h1 = hash_legal_action_space(legal)
        h2 = hash_legal_action_space(legal)
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_hash_legal_action_space_order_sensitive(self):
        from worldloop_kernel.protocol import LegalAction

        a = (LegalAction(action_type="rest"), LegalAction(action_type="forage"))
        b = (LegalAction(action_type="forage"), LegalAction(action_type="rest"))
        assert hash_legal_action_space(a) != hash_legal_action_space(b)

    def test_hash_inference_config_stable(self):
        h1 = hash_inference_config(
            "test-model", 0.0, 256, 2, "decline"
        )
        h2 = hash_inference_config(
            "test-model", 0.0, 256, 2, "decline"
        )
        assert h1 == h2
        assert h1.startswith("sha256:")

    def test_hash_inference_config_param_sensitive(self):
        h1 = hash_inference_config("model-a", 0.0, 256, 2, "decline")
        h2 = hash_inference_config("model-b", 0.0, 256, 2, "decline")
        assert h1 != h2

    def test_hash_prompt_template_stable(self):
        h1 = hash_prompt_template()
        h2 = hash_prompt_template()
        assert h1 == h2
        assert h1.startswith("sha256:prompt_template:")


# ---------------------------------------------------------------------------
# 9. latency_split aggregator (§6.2)
# ---------------------------------------------------------------------------


class TestLatencySplit:
    def test_empty_returns_zeros(self):
        result = latency_split(())
        assert result["provider_per_attempt_ms"] == 0.0
        assert result["retry_backoff_total_ms"] == 0.0
        assert result["decision_end_to_end_ms"] == 0.0
        assert result["fallback_decision_ms"] == 0.0
        assert result["successful_llm_origin_ms"] == 0.0

    def test_split_separates_llm_origin_and_fallback(self):
        llm_decision = build_decision_from_attempts(
            (
                _make_attempt(
                    input_tokens=10,
                    output_tokens=5,
                    latency_ms=100.0,
                    started_at=0.0,
                    finished_at=0.1,
                ),
            ),
            decision_id="d_llm",
            agent_id="a0",
            tick=0,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            fallback_origin_validation=None,
            requested_backend="real",
            run_tier=RunTier.SMOKE,
        )
        fallback_decision = build_decision_from_attempts(
            (
                _make_attempt(
                    attempt_index=1,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=20.0,
                    started_at=0.0,
                    finished_at=0.02,
                    retry_disposition="fallback",
                ),
            ),
            decision_id="d_fb",
            agent_id="a0",
            tick=1,
            prompt_hashes=_make_prompt_hashes(),
            final_origin="fallback",
            fallback_mode="first_legal",
            llm_origin_validation=None,
            fallback_origin_validation=ValidationSummary(parse_ok=True, candidate_ok=True),
            requested_backend="real",
            run_tier=RunTier.SMOKE,
        )
        result = latency_split((llm_decision, fallback_decision))
        # Mean across both decisions.
        assert result["decision_end_to_end_ms"] == (100.0 + 20.0) / 2
        # LLM-origin mean.
        assert result["successful_llm_origin_ms"] == 100.0
        # Fallback mean.
        assert result["fallback_decision_ms"] == 20.0
        # Provider per attempt (one attempt each, both included).
        assert result["provider_per_attempt_ms"] == (100.0 + 20.0) / 2


# ---------------------------------------------------------------------------
# 10. Re-exports
# ---------------------------------------------------------------------------


class TestReExports:
    def test_telemetry_module_exports_all_symbols(self):
        from worldloop_data import telemetry

        for symbol in (
            "TELEMETRY_SCHEMA_VERSION",
            "RunTier",
            "ValidationSummary",
            "InferenceAttemptEvent",
            "InferenceDecisionEvent",
            "RunLevelConfig",
            "default_run_level_config",
            "accumulate_attempt_tokens",
            "build_decision_from_attempts",
            "EvidenceFailClosedError",
            "check_evidence_fail_closed",
            "check_evidence_fail_closed_batch",
            "hash_legal_action_space",
            "hash_inference_config",
            "hash_prompt_template",
            "latency_split",
            "derive_effective_backend",
        ):
            assert hasattr(telemetry, symbol), f"missing export: {symbol}"
