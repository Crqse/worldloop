"""Phase 2 / Beta correction §6 — Telemetry event schema.

Splits the previous single :class:`InferenceEvent` (which overwrote
per-attempt fields across retries) into two frozen dataclasses:

- :class:`InferenceAttemptEvent` — one record per LLM API call within a
  decision. Carries per-call latency, tokens, response hash, parse
  status, error category, retry disposition, and effective backend.
- :class:`InferenceDecisionEvent` — one record per (agent, tick)
  decision. Aggregates all attempts (token sum, not last-attempt
  overwrite), carries prompt component hashes from Phase 1, the
  ``final_origin`` (llm / fallback / declined), and the
  requested/effective backend pair for evidence fail-closed (§6.4).

Design contract (§6.1-6.4 of the Beta correction plan):

- Retry tokens accumulate across all attempts (§6.2). The decision's
  ``total_input_tokens`` / ``total_output_tokens`` / ``total_cached_tokens``
  are the SUM of per-attempt values, not the last attempt's value.
- LLM-origin and fallback-origin validation are tracked separately
  (§6.3). A decision's ``final_origin`` determines which validation
  summary is populated; the other is ``None``.
- Evidence tier fail-closed (§6.4): if ``requested_backend == "real"``
  but ``effective_backend != "real"`` or ``mock_calls > 0``, the
  decision is invalid for evidence and :func:`check_evidence_fail_closed`
  raises :class:`EvidenceFailClosedError`.

This module is intentionally schema-only. The :class:`LLMPolicy`
refactor that emits these events lives in :mod:`worldloop_data.llm_policy`
and is updated in a follow-up attempt. Fake and real clients share the
same event schema (§6.5 — Phase 2 task 6).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping, Optional, Tuple


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

TELEMETRY_SCHEMA_VERSION: str = "0.1.0"
"""Version of the InferenceAttemptEvent / InferenceDecisionEvent schema.

Bumped on any backward-incompatible field change. Recorded on every
:class:`InferenceDecisionEvent` so downstream consumers can dispatch
by schema version."""


# ---------------------------------------------------------------------------
# Run tier (§6.3 fallback discipline)
# ---------------------------------------------------------------------------


class RunTier(str, Enum):
    """Run tier governing fallback discipline and fail-closed behaviour.

    Per §6.3 of the Beta correction plan:

    - ``DEV``: fallback allowed (first_legal / random_legal), but must
      be explicitly marked. No fail-closed on mock calls.
    - ``SMOKE``: fallback allowed with a configured upper bound;
      fallback_origin_rate is reported separately from LLM-origin rate.
    - ``EVIDENCE``: fallback defaults to ``decline`` or fails the whole
      run; mock calls are forbidden when ``requested_backend == "real"``.
      LLM-quality metrics MUST NOT include fallback-origin decisions.
    - ``SAFETY_DEMO``: fallback allowed but the run only proves system
      availability, NOT LLM behaviour quality.
    """

    DEV = "dev"
    SMOKE = "smoke"
    EVIDENCE = "evidence"
    SAFETY_DEMO = "safety_demo"


# ---------------------------------------------------------------------------
# Validation summary (§6.3 — split LLM-origin vs fallback-origin)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationSummary:
    """Validation outcome for one decision origin (LLM or fallback).

    A decision's ``final_origin`` determines which of
    ``llm_origin_validation`` / ``fallback_origin_validation`` is
    populated on :class:`InferenceDecisionEvent`; the other is ``None``.

    Attributes
    ----------
    parse_ok:
        Whether the LLM response parsed as a JSON object. Always
        ``True`` for fallback-origin decisions (fallback builds the
        action directly, no parse step).
    candidate_ok:
        Whether the chosen ``action_type`` matched a legal action. For
        fallback decisions this is always ``True`` (fallback picks from
        ``legal_actions`` directly).
    matched_action_type:
        The action_type that was matched, if any. ``None`` if no match.
    world_accepted:
        Whether the world accepted the proposed action via
        ``validate_action``. Filled in Phase 3 (Rollout); left ``None``
        by the policy itself in Phase 2.
    rejection_reason:
        If ``world_accepted is False``, the world's rejection reason.
        ``None`` otherwise.
    """

    parse_ok: bool
    candidate_ok: bool
    matched_action_type: Optional[str] = None
    world_accepted: Optional[bool] = None
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Per-attempt event (§6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceAttemptEvent:
    """One LLM API call within a decision.

    A decision may issue multiple attempts (retries on transient
    errors, parse failures, or illegal actions). Each attempt gets its
    own :class:`InferenceAttemptEvent` so token usage and latency can
    be summed across all attempts (§6.2 — do not overwrite).

    Attributes
    ----------
    attempt_id:
        Unique identifier for this attempt (UUID).
    decision_id:
        Identifier of the parent :class:`InferenceDecisionEvent`.
    attempt_index:
        1-based index within the decision. First attempt = 1.
    provider:
        Provider name (e.g. ``"openai_compatible"``).
    endpoint_class:
        Class name of the client (``"OpenAICompatibleClient"`` /
        ``"FakeLLMClient"`` / ``"EchoLLMClient"``). Used to distinguish
        real calls from fake ones without inspecting the URL.
    model:
        Model identifier sent in the request.
    provider_request_id:
        Provider-side request identifier (from response headers when
        available). ``""`` if not provided.
    started_at:
        ``time.perf_counter()`` value at attempt start.
    finished_at:
        ``time.perf_counter()`` value at attempt end.
    latency_ms:
        ``finished_at - started_at`` in milliseconds.
    retry_backoff_ms:
        Backoff sleep duration BEFORE this attempt (in milliseconds).
        Zero for the first attempt; non-zero for retries.
    input_tokens:
        Provider-reported input (prompt) token count for this attempt.
    output_tokens:
        Provider-reported output (completion) token count.
    cached_tokens:
        Prompt-caching hit token count (when the provider reports it).
        Zero if the provider does not report cached tokens.
    finish_reason:
        Provider-reported finish reason (``"stop"`` / ``"length"`` /
        ``"content_filter"`` etc.). ``None`` if the call errored before
        returning a finish reason.
    response_hash:
        ``"sha256:"`` + SHA-256 of the raw response text. Empty if the
        call errored before producing a response.
    parse_status:
        One of ``"ok"`` / ``"json_decode_error"`` / ``"no_content"`` /
        ``"not_dict"`` / ``"no_response"`` (when the call errored).
    parse_error:
        Detailed parse error message. ``None`` if ``parse_status == "ok"``.
    error_category:
        Exception class name if the call raised (e.g.
        ``"LLMTimeoutError"`` / ``"LLMRateLimitError"`` /
        ``"LLMAuthError"``). ``None`` if the call succeeded.
    retry_disposition:
        What happened after this attempt: ``"success"`` (decision
        resolved), ``"retry"`` (another attempt will follow),
        ``"give_up"`` (no more retries, fallback will be used),
        ``"fallback"`` (immediate fallback). For the last successful
        attempt of a decision, this is ``"success"``.
    effective_backend:
        ``"real"`` if the call hit a real provider endpoint,
        ``"fake"`` if it used a deterministic test client,
        ``"mock"`` if it was short-circuited without calling any client.
    mock_call:
        ``True`` if this attempt did not invoke any client (e.g.
        config forced mock). Always ``False`` for real/fake backends.
    """

    # Identity
    attempt_id: str
    decision_id: str
    attempt_index: int

    # Provider / endpoint
    provider: str
    endpoint_class: str
    model: str
    provider_request_id: str = ""

    # Timing (§6.2)
    started_at: float = 0.0
    finished_at: float = 0.0
    latency_ms: float = 0.0
    retry_backoff_ms: float = 0.0

    # Token usage (§6.2 — sum all attempts, never overwrite)
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0

    # Response
    finish_reason: Optional[str] = None
    response_hash: str = ""
    parse_status: str = "no_response"
    parse_error: Optional[str] = None
    error_category: Optional[str] = None
    retry_disposition: str = "give_up"

    # Effective backend (§6.4 — fail-closed check)
    effective_backend: str = "real"
    mock_call: bool = False


# ---------------------------------------------------------------------------
# Per-decision event (§6.1)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class InferenceDecisionEvent:
    """One decision = one (agent, tick) inference resolution.

    Aggregates all :class:`InferenceAttemptEvent` records issued during
    this decision. Token usage is the SUM across all attempts (§6.2),
    NOT the last attempt's value.

    Attributes
    ----------
    schema_version:
        :data:`TELEMETRY_SCHEMA_VERSION`. Bumped on backward-incompatible
        schema change.
    decision_id:
        Unique identifier (UUID).
    episode_id:
        Episode identifier if the policy runs inside an episode runner.
        ``None`` for unit tests.
    tick:
        World tick at which the decision was made.
    agent_id:
        Identifier of the agent whose action was proposed.

    Prompt component hashes (Phase 1, per-decision — NOT per-attempt,
    because the prompt is built once per decision and the same
    ``LLMRequest`` is reused across retries):

    observation_hash:
        SHA-256 of the projected :class:`AgentObservationView`.
    legal_action_space_hash:
        SHA-256 of the legal action space tuple.
    system_prompt_hash:
        SHA-256 of :data:`STABLE_SYSTEM_PROMPT`. Cross-tick stable (P-G3).
    scenario_contract_hash:
        SHA-256 of the :class:`ScenarioContract`. Cross-tick stable
        within the same scenario.
    user_message_hash:
        SHA-256 of the per-tick user message JSON.
    combined_prompt_hash:
        SHA-256 of the concatenation of all four component hashes.
    prompt_template_hash:
        Hash of the prompt builder version (Phase 2 task 6 — prompt
        version/hash into the event). Currently a constant derived from
        :data:`PROMPT_CONTRACT_SCHEMA_VERSION`.
    inference_config_hash:
        SHA-256 of the :class:`LLMPolicyConfig` (model, temperature,
        max_tokens, max_retries, fallback_mode). Records the exact
        inference configuration under which the decision was made.

    Attempts aggregation (§6.2 — sum, not overwrite):

    attempts:
        Tuple of :class:`InferenceAttemptEvent`, one per LLM call.
        Empty tuple if the decision short-circuited (no legal actions).
    total_input_tokens:
        Sum of ``input_tokens`` across all attempts.
    total_output_tokens:
        Sum of ``output_tokens`` across all attempts.
    total_cached_tokens:
        Sum of ``cached_tokens`` across all attempts.
    total_latency_ms:
        End-to-end decision latency: ``finished_at - started_at`` of
        the entire attempt loop. Includes retry backoffs.
    retry_backoff_total_ms:
        Sum of ``retry_backoff_ms`` across all attempts.
    attempt_count:
        ``len(attempts)``. Equal to the number of LLM calls made.

    Final origin (§6.1):

    final_origin:
        ``"llm"`` if a valid LLM-origin action was proposed.
        ``"fallback"`` if the fallback policy was used (decline /
        first_legal / random_legal).
        ``"declined"`` if no proposal was produced (no legal actions,
        or fallback_mode=decline and all retries failed).
    fallback_mode:
        The fallback mode string if ``final_origin == "fallback"``;
        ``None`` otherwise.

    Validation split (§6.3 — LLM-origin vs fallback-origin):

    llm_origin_validation:
        :class:`ValidationSummary` if ``final_origin == "llm"``;
        ``None`` otherwise.
    fallback_origin_validation:
        :class:`ValidationSummary` if ``final_origin == "fallback"``;
        ``None`` otherwise.

    World execution linkage:

    world_accepted:
        Filled in Phase 3 by the rollout layer after the world
        validates/executes the proposed action. ``None`` until then.
    world_rejection_reason:
        If ``world_accepted is False``, the world's rejection reason.

    Effective backend check (§6.4 — evidence fail-closed):

    requested_backend:
        ``"real"`` if the policy was configured to use a real client;
        ``"fake"`` if configured to use a fake/mock client.
    effective_backend:
        ``"real"`` if ALL attempts hit a real provider endpoint;
        ``"fake"`` if any attempt used a fake client;
        ``"mock"`` if any attempt was short-circuited without a client.
    mock_calls:
        Count of attempts where ``mock_call is True``. Must be zero
        for evidence-tier runs that requested real backend.

    Run tier context:

    run_tier:
        :class:`RunTier` of the run that produced this decision.
    """

    # Schema
    schema_version: str = TELEMETRY_SCHEMA_VERSION

    # Identity
    decision_id: str = ""
    episode_id: Optional[str] = None
    tick: int = 0
    agent_id: str = ""

    # Prompt component hashes (Phase 1, per-decision)
    observation_hash: str = ""
    legal_action_space_hash: str = ""
    system_prompt_hash: str = ""
    scenario_contract_hash: str = ""
    user_message_hash: str = ""
    combined_prompt_hash: str = ""
    prompt_template_hash: str = ""
    inference_config_hash: str = ""

    # Attempts aggregation (§6.2 — sum, not overwrite)
    attempts: Tuple[InferenceAttemptEvent, ...] = field(default_factory=tuple)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cached_tokens: int = 0
    total_latency_ms: float = 0.0
    retry_backoff_total_ms: float = 0.0
    attempt_count: int = 0

    # Final origin (§6.1)
    final_origin: str = "declined"
    fallback_mode: Optional[str] = None

    # Validation split (§6.3)
    llm_origin_validation: Optional[ValidationSummary] = None
    fallback_origin_validation: Optional[ValidationSummary] = None

    # World execution linkage
    world_accepted: Optional[bool] = None
    world_rejection_reason: Optional[str] = None

    # Effective backend check (§6.4)
    requested_backend: str = "fake"
    effective_backend: str = "fake"
    mock_calls: int = 0

    # Run tier context
    run_tier: str = RunTier.DEV.value


# ---------------------------------------------------------------------------
# Run-level configuration (§6.3 + §6.4)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunLevelConfig:
    """Run-level configuration governing fallback and fail-closed rules.

    Per §6.3 + §6.4 of the Beta correction plan. The run tier
    determines whether fallback is allowed, what fallback_origin_rate
    threshold applies, and whether mock calls / missing keys trigger
    fail-closed.

    Attributes
    ----------
    tier:
        :class:`RunTier` for this run.
    fallback_allowed:
        Whether any fallback mode other than ``"decline"`` is
        permitted. ``False`` for EVIDENCE (default).
    fallback_origin_rate_max:
        Maximum allowed ``fallback_origin_rate`` (fallback decisions /
        total decisions). ``0.0`` for EVIDENCE; ``1.0`` for DEV/SAFETY_DEMO;
        configurable for SMOKE (default ``0.5``).
    fail_closed_on_mock:
        If ``True``, any ``mock_call=True`` attempt in a
        ``requested_backend="real"`` decision raises
        :class:`EvidenceFailClosedError`.
    fail_closed_on_missing_key:
        If ``True``, a missing API key when ``requested_backend="real"``
        raises :class:`EvidenceFailClosedError` instead of silently
        falling back.
    """

    tier: RunTier = RunTier.DEV
    fallback_allowed: bool = True
    fallback_origin_rate_max: float = 1.0
    fail_closed_on_mock: bool = False
    fail_closed_on_missing_key: bool = False


def default_run_level_config(tier: RunTier) -> RunLevelConfig:
    """Return the default :class:`RunLevelConfig` for a tier.

    Per §6.3 fallback discipline table:

    - DEV: fallback allowed, no fail-closed on mock/missing key.
    - SMOKE: fallback allowed with 50% rate cap, fail-closed on mock
      when real was requested.
    - EVIDENCE: fallback NOT allowed (decline-only), 0% fallback rate,
      fail-closed on mock AND on missing key.
    - SAFETY_DEMO: fallback allowed, no fail-closed (proves availability
      only, not LLM quality).
    """
    if tier is RunTier.DEV:
        return RunLevelConfig(
            tier=tier,
            fallback_allowed=True,
            fallback_origin_rate_max=1.0,
            fail_closed_on_mock=False,
            fail_closed_on_missing_key=False,
        )
    if tier is RunTier.SMOKE:
        return RunLevelConfig(
            tier=tier,
            fallback_allowed=True,
            fallback_origin_rate_max=0.5,
            fail_closed_on_mock=True,
            fail_closed_on_missing_key=False,
        )
    if tier is RunTier.EVIDENCE:
        return RunLevelConfig(
            tier=tier,
            fallback_allowed=False,
            fallback_origin_rate_max=0.0,
            fail_closed_on_mock=True,
            fail_closed_on_missing_key=True,
        )
    if tier is RunTier.SAFETY_DEMO:
        return RunLevelConfig(
            tier=tier,
            fallback_allowed=True,
            fallback_origin_rate_max=1.0,
            fail_closed_on_mock=False,
            fail_closed_on_missing_key=False,
        )
    raise ValueError(f"unsupported RunTier: {tier!r}")


# ---------------------------------------------------------------------------
# Token accumulation (§6.2 — sum across all attempts, not last attempt)
# ---------------------------------------------------------------------------


def accumulate_attempt_tokens(
    attempts: Tuple[InferenceAttemptEvent, ...],
) -> Tuple[int, int, int, float, float]:
    """Sum token usage and latency across all attempts.

    Returns ``(total_input, total_output, total_cached,
    retry_backoff_total, total_latency)``.

    Per §6.2: total token MUST be the sum across all attempts, not just
    the last attempt. Retry backoff is also summed to report the time
    spent waiting between retries.

    ``total_latency`` is the end-to-end decision latency: from the
    first attempt's ``started_at`` to the last attempt's
    ``finished_at``. This includes retry backoffs but excludes any
    pre-decision work (prompt building, observation projection).
    """
    if not attempts:
        return (0, 0, 0, 0.0, 0.0)
    total_input = sum(a.input_tokens for a in attempts)
    total_output = sum(a.output_tokens for a in attempts)
    total_cached = sum(a.cached_tokens for a in attempts)
    retry_backoff_total = sum(a.retry_backoff_ms for a in attempts)
    # End-to-end: first started_at → last finished_at. Use perf_counter
    # values directly; if attempts were issued sequentially, this equals
    # sum(latency_ms) + sum(retry_backoff_ms). If attempts overlapped
    # (not currently supported), this still bounds the wall-clock span.
    first_started = attempts[0].started_at
    last_finished = max(a.finished_at for a in attempts)
    total_latency = max(0.0, last_finished - first_started) * 1000.0
    return (
        total_input,
        total_output,
        total_cached,
        retry_backoff_total,
        total_latency,
    )


def build_decision_from_attempts(
    attempts: Tuple[InferenceAttemptEvent, ...],
    *,
    decision_id: str,
    agent_id: str,
    tick: int,
    prompt_hashes: Mapping[str, str],
    final_origin: str,
    fallback_mode: Optional[str],
    llm_origin_validation: Optional[ValidationSummary],
    fallback_origin_validation: Optional[ValidationSummary],
    requested_backend: str,
    run_tier: RunTier,
    episode_id: Optional[str] = None,
    inference_config_hash: str = "",
) -> InferenceDecisionEvent:
    """Build a :class:`InferenceDecisionEvent` from its attempts.

    Token usage and latency are accumulated across all attempts
    (§6.2 — sum, not overwrite). The effective_backend is derived from
    the attempts: if any attempt is a mock, the decision is mock; if
    any attempt is fake, the decision is fake; only if ALL attempts are
    real is the decision real.
    """
    total_input, total_output, total_cached, retry_total, total_latency = (
        accumulate_attempt_tokens(attempts)
    )

    # Derive effective_backend from attempts.
    if not attempts:
        # No attempts → decision short-circuited. effective_backend is
        # whatever was requested (no opportunity to deviate).
        effective = requested_backend
        mock_calls = 0
    else:
        backends = {a.effective_backend for a in attempts}
        if "mock" in backends:
            effective = "mock"
        elif "fake" in backends:
            effective = "fake"
        else:
            effective = "real"
        mock_calls = sum(1 for a in attempts if a.mock_call)

    return InferenceDecisionEvent(
        schema_version=TELEMETRY_SCHEMA_VERSION,
        decision_id=decision_id,
        episode_id=episode_id,
        tick=tick,
        agent_id=agent_id,
        observation_hash=prompt_hashes.get("observation_hash", ""),
        legal_action_space_hash=prompt_hashes.get("legal_action_space_hash", ""),
        system_prompt_hash=prompt_hashes.get("system_prompt_hash", ""),
        scenario_contract_hash=prompt_hashes.get("scenario_contract_hash", ""),
        user_message_hash=prompt_hashes.get("user_message_hash", ""),
        combined_prompt_hash=prompt_hashes.get("combined_prompt_hash", ""),
        prompt_template_hash=prompt_hashes.get("prompt_template_hash", ""),
        inference_config_hash=inference_config_hash,
        attempts=attempts,
        total_input_tokens=total_input,
        total_output_tokens=total_output,
        total_cached_tokens=total_cached,
        total_latency_ms=total_latency,
        retry_backoff_total_ms=retry_total,
        attempt_count=len(attempts),
        final_origin=final_origin,
        fallback_mode=fallback_mode,
        llm_origin_validation=llm_origin_validation,
        fallback_origin_validation=fallback_origin_validation,
        requested_backend=requested_backend,
        effective_backend=effective,
        mock_calls=mock_calls,
        run_tier=run_tier.value,
    )


# ---------------------------------------------------------------------------
# Evidence fail-closed (§6.4)
# ---------------------------------------------------------------------------


class EvidenceFailClosedError(RuntimeError):
    """Raised when an evidence-tier decision violates fail-closed rules.

    Per §6.4: when ``requested_backend == "real"`` but the effective
    backend is not real, or mock calls occurred, the decision is
    invalid for evidence and the run MUST exit non-zero.

    The error message NEVER includes API key material — at most the
    env var NAME (cf. :class:`LLMAuthError`).
    """


def check_evidence_fail_closed(
    decision: InferenceDecisionEvent,
    config: RunLevelConfig,
) -> None:
    """Enforce evidence-tier fail-closed rules on a decision.

    Per §6.4, an evidence-tier run with ``requested_backend == "real"``
    MUST satisfy:

    - ``effective_backend == "real"`` (no mock, no fake)
    - ``mock_calls == 0``
    - If ``config.fail_closed_on_missing_key`` and the decision has
      any attempt with ``error_category == "LLMAuthError"``, raise.
    - If ``not config.fallback_allowed`` and ``final_origin ==
      "fallback"``, raise.

    Raises :class:`EvidenceFailClosedError` on violation. Returns
    ``None`` on success. Non-evidence tiers always return ``None``
    (no fail-closed check).
    """
    if config.tier is not RunTier.EVIDENCE:
        return  # fail-closed only applies to EVIDENCE tier
    if decision.requested_backend != "real":
        return  # only enforced when real was requested

    if decision.effective_backend != "real":
        raise EvidenceFailClosedError(
            f"evidence fail-closed: decision {decision.decision_id!r} "
            f"requested_backend=real but effective_backend="
            f"{decision.effective_backend!r}; refusing to record as "
            "evidence. (Phase 2 §6.4)"
        )
    if decision.mock_calls > 0:
        raise EvidenceFailClosedError(
            f"evidence fail-closed: decision {decision.decision_id!r} "
            f"had mock_calls={decision.mock_calls} > 0 while "
            "requested_backend=real; refusing to record as evidence. "
            "(Phase 2 §6.4)"
        )
    if (
        config.fail_closed_on_missing_key
        and any(a.error_category == "LLMAuthError" for a in decision.attempts)
    ):
        raise EvidenceFailClosedError(
            f"evidence fail-closed: decision {decision.decision_id!r} "
            "had an LLMAuthError (missing/invalid API key) while "
            "requested_backend=real; missing key cannot silently "
            "degrade to mock. (Phase 2 §6.4)"
        )
    if (
        not config.fallback_allowed
        and decision.final_origin == "fallback"
    ):
        raise EvidenceFailClosedError(
            f"evidence fail-closed: decision {decision.decision_id!r} "
            "used fallback while fallback_allowed=False in EVIDENCE "
            "tier; fallback must not enter LLM-quality metrics. "
            "(Phase 2 §6.3)"
        )


def check_evidence_fail_closed_batch(
    decisions: Tuple[InferenceDecisionEvent, ...],
    config: RunLevelConfig,
) -> None:
    """Enforce fail-closed across a batch of decisions.

    For EVIDENCE tier: checks each decision via :func:`check_evidence_fail_closed`
    (raises on first per-decision violation), then checks the aggregate
    fallback_origin_rate against ``config.fallback_origin_rate_max``.

    For SMOKE tier: per-decision fail-closed is enforced only on
    ``mock_call`` / missing-key (via :func:`check_evidence_fail_closed`
    which already short-circuits non-EVIDENCE per-decision fallback
    checks), plus the aggregate rate check.

    For DEV / SAFETY_DEMO: no rate check (rate_max == 1.0).

    Raises :class:`EvidenceFailClosedError` on violation.
    """
    # Per-decision checks (EVIDENCE only enforces fallback-per-decision;
    # SMOKE / DEV / SAFETY_DEMO still enforce mock + missing-key when
    # requested real via check_evidence_fail_closed's own gating).
    for d in decisions:
        check_evidence_fail_closed(d, config)
    # Aggregate rate check applies to any tier with rate_max < 1.0
    # (i.e., SMOKE and EVIDENCE). DEV / SAFETY_DEMO have rate_max=1.0
    # so the check trivially passes.
    if config.fallback_origin_rate_max >= 1.0:
        return
    if not decisions:
        return
    fallback_count = sum(1 for d in decisions if d.final_origin == "fallback")
    rate = fallback_count / len(decisions)
    if rate > config.fallback_origin_rate_max:
        raise EvidenceFailClosedError(
            f"evidence fail-closed: batch fallback_origin_rate={rate:.4f} "
            f"exceeds max={config.fallback_origin_rate_max:.4f} "
            f"({fallback_count}/{len(decisions)} decisions used fallback). "
            "(Phase 2 §6.3)"
        )


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def hash_legal_action_space(legal_actions: Tuple[Any, ...]) -> str:
    """Hash the legal action space for a decision.

    Used to populate ``InferenceDecisionEvent.legal_action_space_hash``.
    The hash is stable across runs for the same legal action tuple.
    """
    payload = json.dumps(
        [
            {
                "action_type": getattr(a, "action_type", str(a)),
                "params": dict(getattr(a, "params", {})),
            }
            for a in legal_actions
        ],
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_inference_config(
    model: str,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    fallback_mode: str,
) -> str:
    """Hash the :class:`LLMPolicyConfig` fields that affect inference.

    Used to populate ``InferenceDecisionEvent.inference_config_hash``.
    Records the exact inference configuration under which the decision
    was made, so downstream analysis can group decisions by config.
    """
    payload = json.dumps(
        {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "max_retries": max_retries,
            "fallback_mode": fallback_mode,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_prompt_template() -> str:
    """Hash the prompt template version.

    Phase 2 task 6 requires prompt version/hash to enter the event.
    Currently a constant derived from the prompt contract schema
    version; will become a real template hash when the prompt builder
    is versioned separately.
    """
    # Imported lazily to avoid a circular import (prompt_contract imports
    # nothing from telemetry, but telemetry importing prompt_contract at
    # module load time would couple the two modules unnecessarily).
    from worldloop_data.prompt_contract import PROMPT_CONTRACT_SCHEMA_VERSION

    return f"sha256:prompt_template:{PROMPT_CONTRACT_SCHEMA_VERSION}"


# ---------------------------------------------------------------------------
# Aggregators for analysis (§6.2 latency split)
# ---------------------------------------------------------------------------


def latency_split(
    decisions: Tuple[InferenceDecisionEvent, ...],
) -> Mapping[str, float]:
    """Split decision latency by origin (§6.2).

    Returns a mapping with:

    - ``provider_per_attempt_ms``: mean latency per attempt across
      all decisions.
    - ``retry_backoff_total_ms``: sum of retry backoffs across all
      decisions.
    - ``decision_end_to_end_ms``: mean end-to-end decision latency.
    - ``fallback_decision_ms``: mean latency of fallback-origin
      decisions (excludes LLM call time).
    - ``successful_llm_origin_ms``: mean latency of successful
      LLM-origin decisions (excludes fallback decisions).
    """
    if not decisions:
        return {
            "provider_per_attempt_ms": 0.0,
            "retry_backoff_total_ms": 0.0,
            "decision_end_to_end_ms": 0.0,
            "fallback_decision_ms": 0.0,
            "successful_llm_origin_ms": 0.0,
        }
    all_attempts = [a for d in decisions for a in d.attempts]
    provider_per_attempt = (
        sum(a.latency_ms for a in all_attempts) / len(all_attempts)
        if all_attempts
        else 0.0
    )
    retry_total = sum(d.retry_backoff_total_ms for d in decisions)
    decision_e2e = sum(d.total_latency_ms for d in decisions) / len(decisions)
    fallback_decisions = [d for d in decisions if d.final_origin == "fallback"]
    fallback_ms = (
        sum(d.total_latency_ms for d in fallback_decisions) / len(fallback_decisions)
        if fallback_decisions
        else 0.0
    )
    llm_origin_decisions = [
        d for d in decisions if d.final_origin == "llm"
    ]
    llm_origin_ms = (
        sum(d.total_latency_ms for d in llm_origin_decisions) / len(llm_origin_decisions)
        if llm_origin_decisions
        else 0.0
    )
    return {
        "provider_per_attempt_ms": provider_per_attempt,
        "retry_backoff_total_ms": retry_total,
        "decision_end_to_end_ms": decision_e2e,
        "fallback_decision_ms": fallback_ms,
        "successful_llm_origin_ms": llm_origin_ms,
    }


# ---------------------------------------------------------------------------
# Convenience: derive effective_backend from a sequence of attempts
# ---------------------------------------------------------------------------


def derive_effective_backend(
    attempts: Tuple[InferenceAttemptEvent, ...],
    requested_backend: str,
) -> Tuple[str, int]:
    """Derive ``(effective_backend, mock_calls)`` from attempts.

    - If any attempt is ``mock_call=True``, effective is ``"mock"``.
    - Else if any attempt has ``effective_backend="fake"``, effective
      is ``"fake"``.
    - Else if all attempts have ``effective_backend="real"``, effective
      is ``"real"``.
    - Else (no attempts), effective is the ``requested_backend``.

    Returns the effective backend string and the count of mock calls.
    """
    if not attempts:
        return (requested_backend, 0)
    mock_calls = sum(1 for a in attempts if a.mock_call)
    if mock_calls > 0:
        return ("mock", mock_calls)
    backends = {a.effective_backend for a in attempts}
    if "fake" in backends:
        return ("fake", 0)
    if backends == {"real"}:
        return ("real", 0)
    # Mixed or unknown — fall back to requested for safety.
    return (requested_backend, mock_calls)


__all__ = [
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
]
