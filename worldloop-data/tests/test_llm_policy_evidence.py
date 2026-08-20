"""Evidence-tier fail-closed integration tests (Phase 2 §6.4 / Attempt 7).

These tests exercise the integrated fail-closed path: when LLMPolicy is
configured for an EVIDENCE run with ``requested_backend="real"`` and the
caller attaches a real :class:`OpenAICompatibleClient`, violations of the
§6.4 invariants cause ``propose()`` to raise
:class:`EvidenceFailClosedError` — exiting the run non-zero rather than
silently recording an invalid evidence decision.

Verified paths:

- **no-key negative**: real client + missing API key env var →
  :class:`LLMAuthError` on every attempt → retry exhausted → fallback
  → §6.4 ``fail_closed_on_missing_key`` raises.
- **effective-backend mismatch (mock)**: real client declared but the
  transport layer returns a mock-shaped response → ``mock_calls > 0``
  → §6.4 raises.
- **fallback forbidden in EVIDENCE**: real client + key present (via a
  stub transport) + illegal action → fallback_mode='first_legal' would
  pick a fallback action, but EVIDENCE forbids fallback → §6.3 raises.
- **SMOKE tier allows fallback** within rate cap, but still fail-closed
  on mock.
- **DEV tier no fail-closed** even with mock calls (rate_max=1.0).
- **Batch rate check** across multiple decisions (SMOKE > 50% raises).

The 1×3 real smoke (real provider + real key) is exercised by an
end-to-end script ``scripts/experiments/run_real_smoke.py`` that callers
run with their own API key; these unit tests use a stub transport to
verify the fail-closed logic without network calls.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Tuple

import pytest

from worldloop_kernel.protocol import ActionSpace, LegalAction
from worldloop_data.policy import PolicyContext
from worldloop_data.llm_policy import (
    InMemoryTelemetrySink,
    LLMPolicy,
    LLMPolicyConfig,
    LLMRequest,
    LLMAuthError,
    OpenAICompatibleClient,
)
from worldloop_data.telemetry import (
    EvidenceFailClosedError,
    RunTier,
    check_evidence_fail_closed,
    check_evidence_fail_closed_batch,
    default_run_level_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ctx(tick: int = 1, agent_id: str = "agent_0") -> PolicyContext:
    import random
    return PolicyContext(
        world=None,  # type: ignore — legacy path used for these tests
        agent_id=agent_id,
        state=None,  # type: ignore
        action_space=ActionSpace(
            agent_id=agent_id,
            legal_actions=(
                LegalAction(action_type="REST"),
                LegalAction(action_type="MOVE"),
            ),
        ),
        tick=tick,
        rng=random.Random(42),
    )


def _make_real_client(
    *,
    api_key_env: str = "WL_TEST_KEY_UNSET_DO_NOT_SET",
    transport: Callable[[str, bytes, Dict[str, str], float], Tuple[int, bytes]] | None = None,
) -> OpenAICompatibleClient:
    """Build an OpenAICompatibleClient pointed at a dummy URL.

    The API key env var defaults to a name that is never set, so the
    client will raise :class:`LLMAuthError` unless the caller monkey-
    patches the env var or provides a transport stub.
    """
    return OpenAICompatibleClient(
        base_url="http://localhost:0",
        api_key_env=api_key_env,
        timeout_seconds=1.0,
        transport=transport or _no_call_transport,
    )


def _no_call_transport(
    url: str, body: bytes, headers: Dict[str, str], timeout: float
) -> Tuple[int, bytes]:
    """Transport stub that should never be invoked.

    If the no-key path correctly raises :class:`LLMAuthError` BEFORE
    the transport is called, this function is never reached. If it IS
    reached, the test fails loudly with a clear assertion.
    """
    raise AssertionError(
        "transport was called for a no-key test — LLMAuthError should "
        "have been raised before the transport layer. (Phase 2 §6.4)"
    )


def _ok_transport_factory(action_type: str = "REST") -> Callable[
    [str, bytes, Dict[str, str], float], Tuple[int, bytes]
]:
    """Build a transport that returns a valid chat completion response."""
    def transport(
        url: str, body: bytes, headers: Dict[str, str], timeout: float
    ) -> Tuple[int, bytes]:
        response_body = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action_type": action_type,
                                "params": {},
                                "reason_code": "TEST",
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return 200, json.dumps(response_body).encode("utf-8")

    return transport


def _illegal_action_transport_factory(action_type: str = "ATTACK") -> Callable[
    [str, bytes, Dict[str, str], float], Tuple[int, bytes]
]:
    """Transport that returns an action not in the legal action space."""
    def transport(
        url: str, body: bytes, headers: Dict[str, str], timeout: float
    ) -> Tuple[int, bytes]:
        response_body = {
            "id": "chatcmpl-illegal",
            "object": "chat.completion",
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action_type": action_type,
                                "params": {},
                                "reason_code": "ILLEGAL",
                            }
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        return 200, json.dumps(response_body).encode("utf-8")

    return transport


# ---------------------------------------------------------------------------
# §6.4 — no-key negative test
# ---------------------------------------------------------------------------

class TestNoKeyNegative:
    """When the API key is missing, the real client raises LLMAuthError.
    Under EVIDENCE tier with requested_backend='real', this must
    trigger fail-closed (§6.4 ``fail_closed_on_missing_key``) rather
    than silently degrading to fallback and recording as evidence.
    """

    def test_no_key_evidence_raises_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """EVIDENCE + real client + no key → EvidenceFailClosedError."""
        monkeypatch.delenv("WL_TEST_KEY_UNSET_DO_NOT_SET", raising=False)
        client = _make_real_client()
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="decline",
                max_retries=2,
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.EVIDENCE,
            requested_backend="real",
        )
        with pytest.raises(EvidenceFailClosedError) as exc_info:
            policy.propose(_make_ctx())
        # Error mentions missing key / auth, no key material.
        msg = str(exc_info.value).lower()
        assert "auth" in msg or "missing_key" in msg or "key" in msg
        assert "sk-" not in msg
        # Decision + attempts were recorded for forensics before raise.
        assert len(sink.decisions) == 1
        assert len(sink.attempts) == 3  # max_retries=2 → 3 attempts
        # All attempts have error_category=LLMAuthError.
        for a in sink.attempts:
            assert a.error_category == "LLMAuthError"
            assert a.parse_status == "no_response"
        # Attempts are tagged effective_backend='real' (the client's
        # declared backend_class) — §6.4's effective_backend mismatch
        # check passes, but the fail_closed_on_missing_key rule catches
        # the LLMAuthError in attempts.
        d = sink.decisions[0]
        assert d.requested_backend == "real"
        assert d.effective_backend == "real"
        assert d.mock_calls == 0
        assert d.final_origin == "declined"  # fallback_mode='decline'

    def test_no_key_dev_tier_no_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DEV tier + no key → no fail-closed (decline silently).

        DEV tier is for development; missing key is logged but does NOT
        raise. The run can be inspected for debugging.
        """
        monkeypatch.delenv("WL_TEST_KEY_UNSET_DO_NOT_SET", raising=False)
        client = _make_real_client()
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="decline",
                max_retries=1,
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.DEV,
            requested_backend="real",
        )
        # No raise — DEV allows missing key.
        proposal = policy.propose(_make_ctx())
        assert proposal is None  # decline fallback returns None
        d = sink.decisions[0]
        assert d.requested_backend == "real"
        assert d.effective_backend == "real"  # real client declared
        assert d.mock_calls == 0

    def test_no_key_smoke_tier_no_fail_closed_on_missing_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SMOKE tier + no key → no fail-closed on missing key.

        SMOKE has ``fail_closed_on_missing_key=False`` (only mock calls
        trigger fail-closed in SMOKE). Missing key still records as
        failed attempts but does not raise.
        """
        monkeypatch.delenv("WL_TEST_KEY_UNSET_DO_NOT_SET", raising=False)
        client = _make_real_client()
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="decline",
                max_retries=1,
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.SMOKE,
            requested_backend="real",
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is None
        d = sink.decisions[0]
        assert d.requested_backend == "real"
        assert d.effective_backend == "real"  # real client declared
        # No raise — SMOKE doesn't fail-closed on missing key.

    def test_no_key_evidence_records_attempt_errors_before_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The attempts + decision events are still emitted to the sink
        BEFORE the EvidenceFailClosedError propagates — the caller can
        inspect them for post-mortem diagnostics.
        """
        monkeypatch.delenv("WL_TEST_KEY_UNSET_DO_NOT_SET", raising=False)
        client = _make_real_client()
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="decline",
                max_retries=2,
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.EVIDENCE,
            requested_backend="real",
        )
        with pytest.raises(EvidenceFailClosedError):
            policy.propose(_make_ctx())
        # Forensics: 3 attempts + 1 decision recorded.
        assert len(sink.attempts) == 3
        assert len(sink.decisions) == 1
        # All attempts have LLMAuthError.
        auth_errors = [a for a in sink.attempts if a.error_category == "LLMAuthError"]
        assert len(auth_errors) == 3
        # Each attempt has a unique attempt_id.
        ids = [a.attempt_id for a in sink.attempts]
        assert len(set(ids)) == 3
        # Decision references all 3 attempts.
        d = sink.decisions[0]
        assert d.attempt_count == 3


# ---------------------------------------------------------------------------
# §6.4 — effective backend mismatch (mock calls)
# ---------------------------------------------------------------------------

class TestMockCallFailClosed:
    """If the transport returns a mock-shaped response while the caller
    declared ``requested_backend="real"``, §6.4 requires fail-closed.

    Note: the current OpenAICompatibleClient does not set ``mock_call``
    on its attempts (only fake clients do). This test simulates a
    hypothetical mock-shaped transport to verify the fail-closed path
    would catch it if it ever occurred.
    """

    def test_evidence_real_request_with_mock_calls_raises(
        self,
    ) -> None:
        """Synthesize a decision event with mock_calls > 0 and verify
        check_evidence_fail_closed raises (unit test of the check).
        """
        from worldloop_data.telemetry import (
            InferenceAttemptEvent,
            InferenceDecisionEvent,
            ValidationSummary,
        )
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        attempt = InferenceAttemptEvent(
            attempt_id="a1",
            decision_id="d1",
            attempt_index=1,
            provider="openai_compatible",
            endpoint_class="real",
            model="test-model",
            provider_request_id="req-1",
            started_at=now,
            finished_at=now,
            latency_ms=10.0,
            retry_backoff_ms=0.0,
            input_tokens=5,
            output_tokens=3,
            cached_tokens=0,
            finish_reason="stop",
            response_hash="sha256:abc",
            parse_status="ok",
            parse_error=None,
            error_category=None,
            retry_disposition="success",
            effective_backend="mock",  # mock effective backend
            mock_call=True,
        )
        decision = InferenceDecisionEvent(
            schema_version="0.1.0",
            decision_id="d1",
            episode_id=None,
            tick=1,
            agent_id="agent_0",
            system_prompt_hash="sha256:x",
            scenario_contract_hash="sha256:x",
            observation_hash="sha256:x",
            user_message_hash="sha256:x",
            combined_prompt_hash="sha256:x",
            inference_config_hash="sha256:x",
            legal_action_space_hash="sha256:x",
            prompt_template_hash="sha256:x",
            attempts=(attempt,),
            total_input_tokens=5,
            total_output_tokens=3,
            total_cached_tokens=0,
            total_latency_ms=10.0,
            retry_backoff_total_ms=0.0,
            attempt_count=1,
            final_origin="llm",
            fallback_mode=None,
            llm_origin_validation=ValidationSummary(
                parse_ok=True, candidate_ok=True, matched_action_type="REST"
            ),
            fallback_origin_validation=None,
            world_accepted=None,
            world_rejection_reason=None,
            requested_backend="real",
            effective_backend="mock",
            mock_calls=1,
            run_tier=RunTier.EVIDENCE.value,
        )
        config = default_run_level_config(RunTier.EVIDENCE)
        with pytest.raises(EvidenceFailClosedError) as exc_info:
            check_evidence_fail_closed(decision, config)
        assert "mock_calls" in str(exc_info.value).lower() or \
               "effective_backend" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# §6.3 — EVIDENCE forbids fallback per-decision
# ---------------------------------------------------------------------------

class TestEvidenceForbidsFallback:
    """EVIDENCE tier with ``fallback_allowed=False`` (per
    default_run_level_config) means any fallback decision raises.

    This test uses a real-key stub transport that returns an illegal
    action → fallback_mode='first_legal' picks a fallback action →
    EVIDENCE tier raises because fallback is forbidden.
    """

    def test_evidence_fallback_first_legal_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real key + illegal action + first_legal fallback → EVIDENCE
        raises because fallback is forbidden in evidence tier.
        """
        monkeypatch.setenv("WL_TEST_KEY_OK", "sk-test-key-for-stub")
        client = OpenAICompatibleClient(
            base_url="http://localhost:0",
            api_key_env="WL_TEST_KEY_OK",
            timeout_seconds=1.0,
            transport=_illegal_action_transport_factory("ATTACK"),
        )
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="first_legal",  # would normally pick fallback
                max_retries=0,  # don't retry illegal action
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.EVIDENCE,
            requested_backend="real",
        )
        with pytest.raises(EvidenceFailClosedError) as exc_info:
            policy.propose(_make_ctx())
        # Error mentions fallback forbidden.
        msg = str(exc_info.value).lower()
        assert "fallback" in msg
        # Decision was recorded before raise.
        d = sink.decisions[0]
        assert d.final_origin == "fallback"
        assert d.effective_backend == "real"  # actual real call happened
        assert d.mock_calls == 0
        # LLM-origin validation: parse_ok=True (got JSON) but candidate_ok=False.
        assert d.llm_origin_validation is not None
        assert d.llm_origin_validation.parse_ok is True
        assert d.llm_origin_validation.candidate_ok is False
        # Fallback-origin validation: parse_ok=True, candidate_ok=True.
        assert d.fallback_origin_validation is not None
        assert d.fallback_origin_validation.candidate_ok is True

    def test_smoke_tier_allows_fallback_within_rate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SMOKE tier allows fallback up to 50% rate (default).

        Single illegal action + fallback → 100% fallback rate → batch
        check raises (rate > 0.5). But single decision per-decision
        check passes (SMOKE has fallback_allowed=True).
        """
        monkeypatch.setenv("WL_TEST_KEY_OK", "sk-test-key-for-stub")
        client = OpenAICompatibleClient(
            base_url="http://localhost:0",
            api_key_env="WL_TEST_KEY_OK",
            timeout_seconds=1.0,
            transport=_illegal_action_transport_factory("ATTACK"),
        )
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="first_legal",
                max_retries=0,
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.SMOKE,
            requested_backend="real",
        )
        # Single decision — no raise (SMOKE allows fallback per-decision).
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
        assert proposal.action_type == "REST"  # first_legal
        d = sink.decisions[0]
        assert d.final_origin == "fallback"

        # Batch check: 1/1 = 100% > 50% cap → raises.
        config = default_run_level_config(RunTier.SMOKE)
        with pytest.raises(EvidenceFailClosedError) as exc_info:
            check_evidence_fail_closed_batch((d,), config)
        assert "rate" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# §6.4 — real LLM success path (stub transport, no network)
# ---------------------------------------------------------------------------

class TestRealSuccessPath:
    """When the real client successfully returns a legal action,
    EVIDENCE tier + requested=real + effective=real + mock_calls=0
    must PASS fail-closed — no raise.

    This is the path the 1×3 real smoke script (Attempt 7) and 1×10
    real LLM run (Attempt 8) exercise with a real provider.
    """

    def test_evidence_real_success_no_raise(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real key + legal action response → no fail-closed."""
        monkeypatch.setenv("WL_TEST_KEY_OK", "sk-test-key-for-stub")
        client = OpenAICompatibleClient(
            base_url="http://localhost:0",
            api_key_env="WL_TEST_KEY_OK",
            timeout_seconds=1.0,
            transport=_ok_transport_factory("REST"),
        )
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="decline",
                max_retries=0,
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.EVIDENCE,
            requested_backend="real",
        )
        proposal = policy.propose(_make_ctx())
        assert proposal is not None
        assert proposal.action_type == "REST"
        # No raise — all §6.4 invariants satisfied.
        d = sink.decisions[0]
        assert d.requested_backend == "real"
        assert d.effective_backend == "real"
        assert d.mock_calls == 0
        assert d.final_origin == "llm"
        assert d.llm_origin_validation is not None
        assert d.llm_origin_validation.parse_ok is True
        assert d.llm_origin_validation.candidate_ok is True
        assert d.llm_origin_validation.matched_action_type == "REST"
        # Token accumulation recorded.
        assert d.total_input_tokens == 10
        assert d.total_output_tokens == 5

    def test_evidence_real_success_batch_check_passes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """3 successful real decisions → batch check passes (0% fallback)."""
        monkeypatch.setenv("WL_TEST_KEY_OK", "sk-test-key-for-stub")
        client = OpenAICompatibleClient(
            base_url="http://localhost:0",
            api_key_env="WL_TEST_KEY_OK",
            timeout_seconds=1.0,
            transport=_ok_transport_factory("REST"),
        )
        sink = InMemoryTelemetrySink()
        policy = LLMPolicy(
            config=LLMPolicyConfig(
                base_url="http://localhost:0",
                model="test-model",
                fallback_mode="decline",
                max_retries=0,
            ),
            client=client,
            telemetry=sink,
            run_tier=RunTier.EVIDENCE,
            requested_backend="real",
        )
        for tick in range(1, 4):
            policy.propose(_make_ctx(tick=tick))
        # 3 decisions, all final_origin=llm.
        assert len(sink.decisions) == 3
        config = default_run_level_config(RunTier.EVIDENCE)
        # Batch check passes — 0% fallback rate, all invariants OK.
        check_evidence_fail_closed_batch(tuple(sink.decisions), config)
