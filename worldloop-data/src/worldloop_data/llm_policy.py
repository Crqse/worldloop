"""LLM-driven policy for WorldLoop v2.

Provides a real LLM policy that wraps an OpenAI-compatible API endpoint,
producing :class:`ActionProposal` objects from structured JSON output
while observing the hard boundary: LLM proposes, world decides.

Design contract (§4.2-4.4 of Beta plan):
- LLM produces candidate actions; ``world.validate_action()`` is the
  sole authority gate.
- ``LLMPolicy`` never calls ``validate_action()`` or ``step()`` directly.
- Failures are observable through telemetry, never silently masked.
- API keys come from environment variables only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from worldloop_kernel.action import ActionProposal
from worldloop_kernel.observation import is_observation_projector
from worldloop_kernel.protocol import ActionSpace, LegalAction, WorldProtocol
from worldloop_data.policy import PolicyContext
from worldloop_data.prompt_contract import (
    STABLE_SYSTEM_PROMPT,
    build_scenario_contract,
    build_user_message,
    compute_prompt_hashes,
)
from worldloop_data.telemetry import (
    InferenceAttemptEvent,
    InferenceDecisionEvent,
    RunTier,
    ValidationSummary,
    build_decision_from_attempts,
    check_evidence_fail_closed,
    default_run_level_config,
    hash_inference_config,
    hash_legal_action_space,
    hash_prompt_template,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMRequest:
    """Serializable request to an LLM endpoint."""

    prompt: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    system_prompt: Optional[str] = None


@dataclass(frozen=True)
class LLMResponse:
    """Parsed LLM response with raw provenance."""

    raw_text: str
    json_body: Optional[Dict[str, Any]] = None
    parse_error: Optional[str] = None
    finish_reason: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    """OpenAI-compatible completion endpoint.

    Implementations include ``OpenAICompatibleClient`` (httpx-based)
    and ``FakeLLMClient`` (deterministic mock for testing).
    """

    def complete(self, request: LLMRequest) -> LLMResponse:
        ...


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class TelemetrySink(Protocol):
    """Legacy observer that receives one :class:`InferenceEvent` per
    decision (Phase 0 governance freeze — preserves the original API
    surface so existing consumers keep working).

    Phase 2 (Beta correction §6) splits this into per-attempt +
    per-decision events via :class:`TelemetrySinkV2`. Sinks that
    implement BOTH protocols get both event streams; sinks that only
    implement this protocol get only the legacy stream.
    """

    def record(self, event: "InferenceEvent") -> None:
        ...


class TelemetrySinkV2(Protocol):
    """Phase 2 observer (Beta correction §6).

    Receives:

    - :meth:`record_attempt` — one :class:`InferenceAttemptEvent` per
      LLM API call (including retries). Emitted in real-time as each
      attempt completes, enabling live observability of retry loops.
    - :meth:`record_decision` — one :class:`InferenceDecisionEvent`
      per (agent, tick) decision, aggregating all attempts. Emitted
      once at the end of the decision with token sums (not last
      attempt overwrite), validation split, and effective backend.
    """

    def record_attempt(self, event: InferenceAttemptEvent) -> None:
        ...

    def record_decision(self, event: InferenceDecisionEvent) -> None:
        ...


@dataclass
class InferenceEvent:
    """Single LLM inference attempt with full observability.

    .. deprecated:: Phase 2 (Beta correction §6)
        This legacy event captures only the LAST attempt's token/latency
        fields, overwriting retry consumption. It is preserved for
        backward compatibility with Phase 0 governance freeze consumers.
        New consumers should read :class:`InferenceAttemptEvent` (per
        call) and :class:`InferenceDecisionEvent` (per decision, with
        proper token accumulation across all attempts).
    """

    inference_id: str
    episode_id: Optional[str] = None
    tick: int = 0
    agent_id: Optional[str] = None
    provider: str = "openai_compatible"
    model: str = ""
    prompt_hash: str = ""
    response_hash: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    attempt_count: int = 1
    parse_ok: bool = False
    candidate_ok: bool = False
    world_accepted: Optional[bool] = None
    fallback_used: bool = False
    error_type: Optional[str] = None
    # Phase 1 / Beta correction §6 — prompt component hashes (additive).
    # Populated when the policy uses the state-aware prompt path
    # (world implements ObservationProjector). Empty for the legacy
    # build_llm_prompt path (preserved for non-projector worlds per
    # Phase 0 governance freeze).
    system_prompt_hash: str = ""
    scenario_contract_hash: str = ""
    observation_hash: str = ""
    user_message_hash: str = ""
    combined_prompt_hash: str = ""
    prompt_path: str = "legacy"


class InMemoryTelemetrySink:
    """In-memory sink implementing BOTH legacy and V2 protocols.

    - ``events``: legacy :class:`InferenceEvent` list (one per decision,
      last attempt's fields — buggy but preserved for compat).
    - ``attempts``: :class:`InferenceAttemptEvent` list (one per LLM
      call, including retries — proper per-attempt granularity).
    - ``decisions``: :class:`InferenceDecisionEvent` list (one per
      decision, with token sums across all attempts per §6.2).

    All three lists are populated when the sink is attached to an
    :class:`LLMPolicy` that has been upgraded to emit V2 events.
    Legacy-only consumers can read ``events``; V2 consumers should
    read ``attempts`` and ``decisions``.
    """

    def __init__(self) -> None:
        self.events: List[InferenceEvent] = []
        self.attempts: List[InferenceAttemptEvent] = []
        self.decisions: List[InferenceDecisionEvent] = []

    # Legacy TelemetrySink protocol
    def record(self, event: InferenceEvent) -> None:
        self.events.append(event)

    # TelemetrySinkV2 protocol
    def record_attempt(self, event: InferenceAttemptEvent) -> None:
        self.attempts.append(event)

    def record_decision(self, event: InferenceDecisionEvent) -> None:
        self.decisions.append(event)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LLMPolicyConfig:
    """Immutable configuration for an LLM policy instance.

    ``fallback_mode`` controls behaviour when the API call or JSON parse
    fails. Allowed values:

    * ``"decline"`` — return ``None`` (skip tick).
    * ``"first_legal"`` — pick the first legal action and mark fallback.
    * ``"random_legal"`` — pick a random legal action (seeded RNG) and
      mark fallback.
    """

    base_url: str
    model: str
    api_key_env: str = "WORLDLOOP_LLM_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 256
    timeout_seconds: float = 30.0
    max_retries: int = 2
    fallback_mode: str = "decline"

    def __post_init__(self) -> None:
        allowed = {"decline", "first_legal", "random_legal"}
        if self.fallback_mode not in allowed:
            raise ValueError(
                f"fallback_mode must be one of {allowed}, "
                f"got {self.fallback_mode!r}"
            )

    @property
    def api_key(self) -> Optional[str]:
        return os.environ.get(self.api_key_env)


# ---------------------------------------------------------------------------
# Fake client (for offline / smoke testing)
# ---------------------------------------------------------------------------

class FakeLLMClient:
    """Deterministic mock that always returns the first legal action.

    Used for offline tests where no real API key is needed.

    ``backend_class`` identifies this as a fake client for telemetry
    (Phase 2 §6.4 — effective backend derivation).
    """

    backend_class: str = "fake"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            raw_text='{"action_type": "REST", "params": {}, "reason_code": "FAKE"}',
            json_body={"action_type": "REST", "params": {}, "reason_code": "FAKE"},
            finish_reason="stop",
            input_tokens=len(request.prompt) // 4,
            output_tokens=20,
        )


class EchoLLMClient:
    """Returns a fixed action_type for every call (useful for
    deterministic debugging).

    ``backend_class`` identifies this as a fake client for telemetry.
    """

    backend_class: str = "fake"

    def __init__(self, action_type: str = "REST") -> None:
        self._action_type = action_type

    def complete(self, request: LLMRequest) -> LLMResponse:
        body = {"action_type": self._action_type, "params": {}, "reason_code": "ECHO"}
        return LLMResponse(
            raw_text=json.dumps(body),
            json_body=body,
            finish_reason="stop",
            input_tokens=len(request.prompt) // 4,
            output_tokens=20,
        )


# ---------------------------------------------------------------------------
# Real client (OpenAI-compatible /chat/completions over stdlib urllib)
# ---------------------------------------------------------------------------

class LLMClientError(RuntimeError):
    """Base class for transport/protocol failures of the real client.

    Messages NEVER contain the API key value — at most the env var NAME.
    ``LLMPolicy`` catches these in its attempt loop, records
    ``error_type = type(exc).__name__`` and retries / falls back per
    ``LLMPolicyConfig``.
    """


class LLMTimeoutError(LLMClientError):
    """Request exceeded ``timeout_seconds``."""


class LLMRateLimitError(LLMClientError):
    """HTTP 429 from the provider."""


class LLMServerError(LLMClientError):
    """HTTP 5xx from the provider."""


class LLMAuthError(LLMClientError):
    """Missing/empty key env var, or HTTP 401/403."""


class LLMProtocolError(LLMClientError):
    """HTTP 200 but the response envelope is not a valid
    chat-completions JSON document.
    """


def _first_api_key(api_key_env: str) -> str:
    """Resolve the API key from the environment at call time.

    The env var value may hold several comma-separated keys; the first
    non-empty entry wins. Raises :class:`LLMAuthError` naming only the
    env VAR, never any key material.
    """
    raw = os.environ.get(api_key_env) or ""
    for part in raw.split(","):
        candidate = part.strip()
        if candidate:
            return candidate
    raise LLMAuthError(
        f"API key env var {api_key_env!r} is unset or empty; "
        "refusing to call the endpoint."
    )


def _extract_json_object(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse assistant content as a JSON object, tolerating markdown
    code fences. Returns ``(json_body, parse_error)``.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop the opening fence (with optional language tag) and any
        # trailing fence.
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[: -len("```")]
        stripped = stripped.strip()
    try:
        body = json.loads(stripped)
    except (json.JSONDecodeError, ValueError) as exc:
        return None, f"json_decode_error: {exc}"
    if not isinstance(body, dict):
        return None, f"json_not_object: got {type(body).__name__}"
    return body, None


# transport(url, body_bytes, headers, timeout_seconds) -> (status, raw_bytes)
Transport = Callable[[str, bytes, Dict[str, str], float], Tuple[int, bytes]]


def _urllib_transport(
    url: str,
    body: bytes,
    headers: Dict[str, str],
    timeout_seconds: float,
) -> Tuple[int, bytes]:
    """Default synchronous transport using stdlib urllib (no extra
    dependency for the data package). Non-2xx responses are returned as
    ``(status, body)`` so status mapping stays in one place
    (:meth:`OpenAICompatibleClient.complete`).
    """
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        # 4xx/5xx: normalise to (status, body); mapping happens upstream.
        return exc.code, exc.read()


class OpenAICompatibleClient:
    """Synchronous client for an OpenAI-compatible ``/chat/completions``
    endpoint, implementing the :class:`LLMClient` protocol.

    Contract:
    - The API key is read from ``api_key_env`` at call time (first
      non-empty entry of a comma-separated value) and only ever placed
      in the Authorization header — never in artifacts, logs, or
      exception messages.
    - Transport/HTTP failures raise :class:`LLMClientError` subclasses
      (429 → :class:`LLMRateLimitError`, 5xx → :class:`LLMServerError`,
      timeout → :class:`LLMTimeoutError`), which ``LLMPolicy`` maps to
      its retry/fallback semantics.
    - Content-level JSON failures do NOT raise: they return an
      :class:`LLMResponse` with ``json_body=None`` and ``parse_error``
      set, matching the existing response semantics.
    - Token usage is back-filled from ``response.usage``.

    ``backend_class`` identifies this as a real client for telemetry
    (Phase 2 §6.4 — effective backend derivation). Custom client
    implementations should set this attribute so the policy can
    classify attempts without isinstance checks.
    """

    backend_class: str = "real"

    def __init__(
        self,
        base_url: str,
        api_key_env: str = "WORLDLOOP_LLM_API_KEY",
        timeout_seconds: float = 30.0,
        transport: Optional[Transport] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout_seconds = timeout_seconds
        self._transport: Transport = transport or _urllib_transport

    def complete(self, request: LLMRequest) -> LLMResponse:
        api_key = _first_api_key(self._api_key_env)
        url = f"{self._base_url}/chat/completions"
        messages: List[Dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload = {
            "model": request.model,
            "messages": messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

        try:
            status, raw = self._transport(
                url, json.dumps(payload).encode("utf-8"), headers, self._timeout_seconds
            )
        except LLMClientError:
            raise
        except TimeoutError as exc:  # socket.timeout is an alias since 3.10
            raise LLMTimeoutError(
                f"request timed out after {self._timeout_seconds}s"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            if isinstance(reason, TimeoutError):
                raise LLMTimeoutError(
                    f"request timed out after {self._timeout_seconds}s"
                ) from exc
            # type name only: never echo bodies/headers into messages.
            raise LLMClientError(
                f"network error: {type(reason).__name__}"
            ) from exc

        if status == 429:
            raise LLMRateLimitError("rate limited (HTTP 429)")
        if status in (401, 403):
            raise LLMAuthError(
                f"authentication rejected (HTTP {status}); "
                f"check env var {self._api_key_env!r}"
            )
        if status >= 500:
            raise LLMServerError(f"server error (HTTP {status})")
        if status != 200:
            raise LLMClientError(f"unexpected HTTP status {status}")

        try:
            envelope = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LLMProtocolError(
                f"response envelope is not valid JSON: {type(exc).__name__}"
            ) from exc

        choices = envelope.get("choices") or []
        if not choices or not isinstance(choices, list):
            raise LLMProtocolError("response envelope has no choices")
        first = choices[0] or {}
        content = ((first.get("message") or {}).get("content")) or ""
        finish_reason = first.get("finish_reason")

        usage = envelope.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens") or 0)
        output_tokens = int(usage.get("completion_tokens") or 0)

        json_body, parse_error = _extract_json_object(content)
        return LLMResponse(
            raw_text=content,
            json_body=json_body,
            parse_error=parse_error,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_llm_prompt(
    ctx: PolicyContext,
    system_prompt: Optional[str] = None,
) -> str:
    """Serialise a minimal prompt for structured action selection.

    Only includes information the model genuinely needs: tick, agent id,
    observation summary, and the list of legal actions. Does NOT leak
    internal world state that the model should not see.

    Returns a stable JSON string whose hash can be used for caching.
    """
    legal = [
        {
            "action_type": a.action_type,
            "params": _sanitize_params(dict(a.params)),
        }
        for a in ctx.action_space.legal_actions
    ]
    payload = {
        "tick": ctx.tick,
        "agent_id": str(ctx.agent_id),
        "legal_actions": legal,
    }
    body = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    if system_prompt:
        return f"{system_prompt}\n\n{body}"
    return body


def _sanitize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only JSON-serialisable scalar values."""
    clean: Dict[str, Any] = {}
    for k, v in params.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            clean[k] = v
        else:
            clean[k] = str(v)
    return clean


# ---------------------------------------------------------------------------
# LLM Policy
# ---------------------------------------------------------------------------

class LLMPolicy:
    """A real LLM-backed policy.

    The policy:
    1. Serialises the current world view and legal actions.
    2. Calls the configured LLM endpoint.
    3. Parses the response as strict JSON.
    4. Matches the chosen ``action_type`` against legal actions.
    5. Returns an ``ActionProposal`` labelled with the policy metadata.

    It NEVER calls ``world.validate_action()`` or ``world.step()``.
    Fallback behaviour is governed by ``LLMPolicyConfig.fallback_mode``.

    Phase 2 (Beta correction §6) adds:

    - :class:`InferenceAttemptEvent` per LLM call (including retries)
      emitted to sinks implementing :class:`TelemetrySinkV2`.
    - :class:`InferenceDecisionEvent` per (agent, tick) decision,
      aggregating all attempts (token SUM per §6.2 — not last-attempt
      overwrite), with validation split and effective backend.
    - ``run_tier`` context (DEV / SMOKE / EVIDENCE / SAFETY_DEMO) for
      fail-closed enforcement by :func:`check_evidence_fail_closed`.
    - ``requested_backend`` (the caller's assertion of real vs fake)
      recorded on every decision for evidence audit.
    """

    policy_id: str
    policy_version: str
    _config: LLMPolicyConfig
    _client: LLMClient
    _telemetry: Optional[TelemetrySink]
    _telemetry_v2: Optional[TelemetrySinkV2]
    _system_prompt: Optional[str]
    _run_tier: RunTier
    _episode_id: Optional[str]
    _requested_backend: str
    _enforce_fail_closed: bool

    def __init__(
        self,
        config: LLMPolicyConfig,
        client: LLMClient,
        telemetry: Optional[TelemetrySink] = None,
        system_prompt: Optional[str] = None,
        policy_id: str = "llm",
        policy_version: str = "0.2.0",
        *,
        run_tier: RunTier = RunTier.DEV,
        episode_id: Optional[str] = None,
        requested_backend: Optional[str] = None,
        enforce_fail_closed: bool = True,
    ) -> None:
        self.policy_id = policy_id
        self.policy_version = policy_version
        self._config = config
        self._client = client
        self._telemetry = telemetry
        # Auto-detect V2 support: if the legacy telemetry sink also
        # implements record_attempt + record_decision, use it as the V2
        # sink (avoids requiring callers to pass the same sink twice).
        self._telemetry_v2 = (
            telemetry  # type: ignore[assignment]
            if telemetry is not None
            and hasattr(telemetry, "record_attempt")
            and hasattr(telemetry, "record_decision")
            else None
        )
        self._system_prompt = system_prompt
        self._run_tier = run_tier
        self._episode_id = episode_id
        # Default requested_backend from the client's backend_class
        # attribute; callers can override (e.g. set "real" when wiring
        # an OpenAICompatibleClient for evidence runs).
        self._requested_backend = requested_backend or getattr(
            client, "backend_class", "fake"
        )
        # When True, the policy enforces §6.4 evidence fail-closed at
        # the end of each decision (raises EvidenceFailClosedError on
        # violation, propagating out of propose() to exit non-zero).
        # Set False only for isolated unit tests that need to inspect a
        # decision event without enforcing the tier contract; production
        # runs MUST leave this True.
        self._enforce_fail_closed = enforce_fail_closed

    # ------------------------------------------------------------------
    # Policy Protocol
    # ------------------------------------------------------------------

    def propose(self, ctx: PolicyContext) -> Optional[ActionProposal]:
        """Propose a candidate action by calling the LLM endpoint.

        Two prompt paths (Phase 1 / Beta correction §5.4-5.6):

        - **State-aware path** (new): if ``ctx.world`` implements
          :class:`ObservationProjector`, project a per-agent
          :class:`AgentObservationView`, build a :class:`ScenarioContract`
          + user message, and call the LLM with separated
          ``system_prompt`` (STABLE_SYSTEM_PROMPT) and ``prompt`` (user
          message) roles. Every prompt component hash is recorded on
          :class:`InferenceEvent` for P-G1~P-G6 auditability.

        - **Legacy path** (preserved): if ``ctx.world`` does NOT
          implement :class:`ObservationProjector`, fall back to
          :func:`build_llm_prompt` (tick + agent_id + legal_actions
          only). This preserves the Phase 0 governance freeze — old
          worlds continue to work without modification. The behavior
          validity gap (BEHAVIOR_NOT_VALIDATED) is documented in
          CURRENT.md / CLAIMS.md and is closed only when the world
          is upgraded to a projector.

        Phase 2 (Beta correction §6) telemetry emission:

        - One :class:`InferenceAttemptEvent` per LLM call (including
          retries) is emitted to V2 sinks in real time, capturing
          per-attempt tokens, latency, parse status, error category,
          and effective backend.
        - One :class:`InferenceDecisionEvent` per (agent, tick) decision
          is emitted at the end, aggregating all attempts (token SUM
          per §6.2 — NOT last-attempt overwrite), with validation
          split (LLM-origin vs fallback-origin) and effective backend
          for evidence fail-closed (§6.4).
        - The legacy :class:`InferenceEvent` is still emitted (one per
          decision, last attempt's fields) for backward compatibility
          with Phase 0 governance freeze consumers.
        """
        decision_id = str(uuid.uuid4())
        event = InferenceEvent(
            inference_id=decision_id,
            tick=ctx.tick,
            agent_id=str(ctx.agent_id),
            provider="openai_compatible",
            model=self._config.model,
        )
        attempts: List[InferenceAttemptEvent] = []

        legal = ctx.action_space.legal_actions
        prompt_hashes, llm_request = self._build_prompt(ctx, event, legal)

        if not legal:
            event.error_type = "no_legal_actions"
            # No attempts made; decision short-circuited.
            self._finalize_decision(
                event=event,
                attempts=attempts,
                ctx=ctx,
                final_origin="declined",
                fallback_mode=None,
                llm_origin_validation=None,
                fallback_origin_validation=None,
                prompt_hashes=prompt_hashes,
            )
            return None

        backend = self._classify_backend()
        backoff_seconds = 0.0

        for attempt_index in range(1, self._config.max_retries + 2):
            event.attempt_count = attempt_index
            attempt_id = str(uuid.uuid4())
            t0 = time.perf_counter()

            try:
                response = self._client.complete(llm_request)
            except Exception as exc:
                t1 = time.perf_counter()
                event.latency_ms = (t1 - t0) * 1000.0
                event.error_type = type(exc).__name__
                # Build + emit attempt event for the failed call.
                attempt_ev = self._build_attempt_event(
                    attempt_id=attempt_id,
                    decision_id=decision_id,
                    attempt_index=attempt_index,
                    t0=t0,
                    t1=t1,
                    backoff_seconds=backoff_seconds,
                    backend=backend,
                    parse_status="no_response",
                    parse_error=None,
                    error_category=type(exc).__name__,
                    retry_disposition=(
                        "retry"
                        if attempt_index <= self._config.max_retries
                        else "give_up"
                    ),
                )
                attempts.append(attempt_ev)
                self._emit_attempt(attempt_ev)

                if attempt_index <= self._config.max_retries:
                    backoff_seconds = 0.5 * attempt_index
                    time.sleep(backoff_seconds)
                    continue

                # Exhausted retries → fallback
                event.fallback_used = True
                return self._finalize_with_fallback(
                    event=event,
                    attempts=attempts,
                    ctx=ctx,
                    prompt_hashes=prompt_hashes,
                    llm_validation=ValidationSummary(
                        parse_ok=False, candidate_ok=False
                    ),
                )

            t1 = time.perf_counter()
            response_hash_full = (
                "sha256:"
                + hashlib.sha256(
                    response.raw_text.encode("utf-8")
                ).hexdigest()
            )
            parse_status, parse_error = self._classify_parse_status(response)
            chosen = (
                response.json_body.get("action_type")
                if response.json_body
                else None
            )
            matched = _find_legal(legal, chosen)

            if matched is not None:
                retry_disposition = "success"
            elif attempt_index <= self._config.max_retries:
                retry_disposition = "retry"
            else:
                retry_disposition = "give_up"

            attempt_ev = self._build_attempt_event(
                attempt_id=attempt_id,
                decision_id=decision_id,
                attempt_index=attempt_index,
                t0=t0,
                t1=t1,
                backoff_seconds=backoff_seconds,
                backend=backend,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                cached_tokens=0,
                finish_reason=response.finish_reason,
                response_hash=response_hash_full,
                parse_status=parse_status,
                parse_error=parse_error,
                error_category=None,
                retry_disposition=retry_disposition,
            )
            attempts.append(attempt_ev)
            self._emit_attempt(attempt_ev)

            # Update legacy event (last attempt's fields — buggy but
            # preserved for Phase 0 governance freeze compat).
            event.latency_ms = (t1 - t0) * 1000.0
            event.input_tokens = response.input_tokens
            event.output_tokens = response.output_tokens
            event.response_hash = hashlib.sha256(
                response.raw_text.encode("utf-8")
            ).hexdigest()[:16]
            event.parse_ok = response.json_body is not None
            event.error_type = response.parse_error

            if not response.json_body:
                if attempt_index <= self._config.max_retries:
                    backoff_seconds = 0.5 * attempt_index
                    time.sleep(backoff_seconds)
                    continue
                event.fallback_used = True
                return self._finalize_with_fallback(
                    event=event,
                    attempts=attempts,
                    ctx=ctx,
                    prompt_hashes=prompt_hashes,
                    llm_validation=ValidationSummary(
                        parse_ok=False, candidate_ok=False
                    ),
                )

            if matched is None:
                event.error_type = "illegal_action"
                event.candidate_ok = False
                if attempt_index <= self._config.max_retries:
                    backoff_seconds = 0.5 * attempt_index
                    time.sleep(backoff_seconds)
                    continue
                event.fallback_used = True
                return self._finalize_with_fallback(
                    event=event,
                    attempts=attempts,
                    ctx=ctx,
                    prompt_hashes=prompt_hashes,
                    llm_validation=ValidationSummary(
                        parse_ok=True, candidate_ok=False
                    ),
                )

            # Success
            event.candidate_ok = True
            event.fallback_used = False
            proposal = ActionProposal(
                agent_id=ctx.agent_id,
                action_type=matched.action_type,
                params={
                    **dict(response.json_body.get("params", {})),
                    "_inference_id": event.inference_id,
                    "_reason_code": response.json_body.get(
                        "reason_code", "UNKNOWN"
                    ),
                    "_prompt_hash": event.prompt_hash,
                },
                proposed_at_tick=ctx.tick,
                proposer=self.policy_id,
            )
            self._finalize_decision(
                event=event,
                attempts=attempts,
                ctx=ctx,
                final_origin="llm",
                fallback_mode=None,
                llm_origin_validation=ValidationSummary(
                    parse_ok=True,
                    candidate_ok=True,
                    matched_action_type=matched.action_type,
                ),
                fallback_origin_validation=None,
                prompt_hashes=prompt_hashes,
            )
            return proposal

        # Defensive — loop exited without return
        event.fallback_used = True
        return self._finalize_with_fallback(
            event=event,
            attempts=attempts,
            ctx=ctx,
            prompt_hashes=prompt_hashes,
            llm_validation=None,
        )

    # ------------------------------------------------------------------
    # Prompt construction (Phase 1)
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        ctx: PolicyContext,
        event: InferenceEvent,
        legal: Sequence[LegalAction],
    ) -> Tuple[Dict[str, str], LLMRequest]:
        """Build the prompt + LLM request, populate prompt hashes on
        ``event``, and return ``(prompt_hashes, llm_request)``.

        Two paths:

        - State-aware (Phase 1): when ``ctx.world`` implements
          :class:`ObservationProjector`, project observation + build
          scenario contract + user message with full hash bundle.
        - Legacy (Phase 0 governance freeze): build_llm_prompt with
          tick + agent_id + legal_actions only.
        """
        # Compute the shared config-derived hashes once.
        legal_hash = hash_legal_action_space(tuple(legal))
        template_hash = hash_prompt_template()
        config_hash = hash_inference_config(
            self._config.model,
            self._config.temperature,
            self._config.max_tokens,
            self._config.max_retries,
            self._config.fallback_mode,
        )

        world = ctx.world
        use_state_aware = world is not None and is_observation_projector(world)

        if use_state_aware:
            observation = world.observe_agent(ctx.agent_id)  # type: ignore[union-attr]
            scenario_contract = build_scenario_contract(observation)
            user_message = build_user_message(observation, scenario_contract)
            bundle = compute_prompt_hashes(
                observation, scenario_contract, user_message
            )
            event.prompt_path = "state_aware"
            event.prompt_hash = bundle.combined_prompt_hash
            event.system_prompt_hash = bundle.system_prompt_hash
            event.scenario_contract_hash = bundle.scenario_contract_hash
            event.observation_hash = bundle.observation_hash
            event.user_message_hash = bundle.user_message_hash
            event.combined_prompt_hash = bundle.combined_prompt_hash
            prompt_hashes: Dict[str, str] = {
                "observation_hash": bundle.observation_hash,
                "legal_action_space_hash": legal_hash,
                "system_prompt_hash": bundle.system_prompt_hash,
                "scenario_contract_hash": bundle.scenario_contract_hash,
                "user_message_hash": bundle.user_message_hash,
                "combined_prompt_hash": bundle.combined_prompt_hash,
                "prompt_template_hash": template_hash,
                "inference_config_hash": config_hash,
            }
            llm_request = LLMRequest(
                prompt=user_message,
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                system_prompt=STABLE_SYSTEM_PROMPT,
            )
        else:
            prompt_text = build_llm_prompt(ctx, self._system_prompt)
            event.prompt_path = "legacy"
            event.prompt_hash = hashlib.sha256(
                prompt_text.encode("utf-8")
            ).hexdigest()[:16]
            prompt_hashes = {
                "observation_hash": "",
                "legal_action_space_hash": legal_hash,
                "system_prompt_hash": "",
                "scenario_contract_hash": "",
                "user_message_hash": "",
                "combined_prompt_hash": event.prompt_hash,
                "prompt_template_hash": template_hash,
                "inference_config_hash": config_hash,
            }
            llm_request = LLMRequest(
                prompt=prompt_text,
                model=self._config.model,
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                system_prompt=self._system_prompt,  # may be None
            )
        return prompt_hashes, llm_request

    # ------------------------------------------------------------------
    # Attempt event construction (Phase 2 §6.1)
    # ------------------------------------------------------------------

    def _build_attempt_event(
        self,
        *,
        attempt_id: str,
        decision_id: str,
        attempt_index: int,
        t0: float,
        t1: float,
        backoff_seconds: float,
        backend: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_tokens: int = 0,
        finish_reason: Optional[str] = None,
        response_hash: str = "",
        parse_status: str = "no_response",
        parse_error: Optional[str] = None,
        error_category: Optional[str] = None,
        retry_disposition: str = "give_up",
    ) -> InferenceAttemptEvent:
        """Build a single :class:`InferenceAttemptEvent`.

        ``mock_call`` is always ``False`` here because this helper is
        only called when ``self._client.complete()`` was actually
        invoked (either returned or raised). Mock short-circuits (no
        client call) are not currently supported by the policy but
        would set ``mock_call=True`` and ``effective_backend="mock"``.
        """
        return InferenceAttemptEvent(
            attempt_id=attempt_id,
            decision_id=decision_id,
            attempt_index=attempt_index,
            provider="openai_compatible",
            endpoint_class=type(self._client).__name__,
            model=self._config.model,
            started_at=t0,
            finished_at=t1,
            latency_ms=(t1 - t0) * 1000.0,
            retry_backoff_ms=backoff_seconds * 1000.0,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            finish_reason=finish_reason,
            response_hash=response_hash,
            parse_status=parse_status,
            parse_error=parse_error,
            error_category=error_category,
            retry_disposition=retry_disposition,
            effective_backend=backend,
            mock_call=False,
        )

    # ------------------------------------------------------------------
    # Decision finalization (Phase 2 §6.1-6.4)
    # ------------------------------------------------------------------

    def _finalize_with_fallback(
        self,
        *,
        event: InferenceEvent,
        attempts: List[InferenceAttemptEvent],
        ctx: PolicyContext,
        prompt_hashes: Dict[str, str],
        llm_validation: Optional[ValidationSummary],
    ) -> Optional[ActionProposal]:
        """Call fallback, then finalize the decision with the fallback
        proposal (or ``None`` if declined).

        Builds the fallback-origin validation summary from the proposal
        and emits the legacy event + V2 decision event.
        """
        proposal = self._fallback(ctx)
        if proposal is not None:
            final_origin = "fallback"
            fallback_mode = self._config.fallback_mode
            fallback_validation = ValidationSummary(
                parse_ok=True,
                candidate_ok=True,
                matched_action_type=proposal.action_type,
            )
        else:
            final_origin = "declined"
            fallback_mode = None
            fallback_validation = None
        self._finalize_decision(
            event=event,
            attempts=attempts,
            ctx=ctx,
            final_origin=final_origin,
            fallback_mode=fallback_mode,
            llm_origin_validation=llm_validation,
            fallback_origin_validation=fallback_validation,
            prompt_hashes=prompt_hashes,
        )
        return proposal

    def _finalize_decision(
        self,
        *,
        event: InferenceEvent,
        attempts: List[InferenceAttemptEvent],
        ctx: PolicyContext,
        final_origin: str,
        fallback_mode: Optional[str],
        llm_origin_validation: Optional[ValidationSummary],
        fallback_origin_validation: Optional[ValidationSummary],
        prompt_hashes: Dict[str, str],
    ) -> None:
        """Build + emit the :class:`InferenceDecisionEvent` (V2) and
        emit the legacy :class:`InferenceEvent`.

        The decision event is built via :func:`build_decision_from_attempts`
        which sums tokens across ALL attempts (§6.2 — not last-attempt
        overwrite). The legacy event is emitted unchanged (it carries
        the last attempt's fields — buggy but preserved for Phase 0
        governance freeze compat).
        """
        # Emit legacy event first (it's already been mutated in-place).
        self._emit(event)

        if self._telemetry_v2 is None:
            return  # no V2 sink attached

        decision_event = build_decision_from_attempts(
            tuple(attempts),
            decision_id=event.inference_id,
            agent_id=str(ctx.agent_id),
            tick=ctx.tick,
            prompt_hashes=prompt_hashes,
            final_origin=final_origin,
            fallback_mode=fallback_mode,
            llm_origin_validation=llm_origin_validation,
            fallback_origin_validation=fallback_origin_validation,
            requested_backend=self._requested_backend,
            run_tier=self._run_tier,
            episode_id=self._episode_id,
            inference_config_hash=prompt_hashes.get("inference_config_hash", ""),
        )
        try:
            self._telemetry_v2.record_decision(decision_event)
        except Exception:
            logger.exception(
                "telemetry V2 sink failed for decision %s",
                event.inference_id,
            )

        # Phase 2 §6.4 — evidence fail-closed enforcement. When the run
        # is EVIDENCE tier and the caller requested a real backend, the
        # decision MUST satisfy §6.4 invariants: effective_backend=real,
        # mock_calls=0, no LLMAuthError, no fallback (EVIDENCE forbids
        # fallback per default_run_level_config). Violations raise
        # EvidenceFailClosedError, which propagates out of propose() and
        # exits the run non-zero — refusing to record as evidence.
        #
        # Non-EVIDENCE tiers (DEV / SMOKE / SAFETY_DEMO) skip this check
        # (check_evidence_fail_closed short-circuits on tier != EVIDENCE).
        # SMOKE batch rate check is enforced separately by the caller via
        # check_evidence_fail_closed_batch at run end.
        if self._enforce_fail_closed:
            try:
                check_evidence_fail_closed(
                    decision_event,
                    default_run_level_config(self._run_tier),
                )
            except Exception:
                # EvidenceFailClosedError must propagate — it signals the
                # run is invalid for evidence and MUST exit non-zero.
                # Log only the decision_id + reason, never API key.
                logger.error(
                    "evidence fail-closed violation on decision %s "
                    "(run_tier=%s, requested=%s, effective=%s, "
                    "mock_calls=%d, final_origin=%s)",
                    event.inference_id,
                    self._run_tier.name,
                    decision_event.requested_backend,
                    decision_event.effective_backend,
                    decision_event.mock_calls,
                    decision_event.final_origin,
                )
                raise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fallback(self, ctx: PolicyContext) -> Optional[ActionProposal]:
        mode = self._config.fallback_mode
        if mode == "decline":
            return None
        legal = list(ctx.action_space.legal_actions)
        if not legal:
            return None
        chosen: LegalAction
        if mode == "first_legal":
            chosen = legal[0]
        elif mode == "random_legal":
            chosen = ctx.rng.choice(legal) if hasattr(ctx, 'rng') else legal[0]
        else:
            return None
        return ActionProposal(
            agent_id=ctx.agent_id,
            action_type=chosen.action_type,
            params={**dict(chosen.params), "_fallback_mode": mode},
            proposed_at_tick=ctx.tick,
            proposer=f"{self.policy_id}_fallback",
        )

    def _classify_backend(self) -> str:
        """Classify the effective backend for attempts.

        Returns the client's ``backend_class`` attribute (``"real"`` /
        ``"fake"``). Defaults to ``"fake"`` for clients that don't
        declare it (safe default — real clients should always declare).
        """
        return getattr(self._client, "backend_class", "fake")

    @staticmethod
    def _classify_parse_status(
        response: LLMResponse,
    ) -> Tuple[str, Optional[str]]:
        """Map :class:`LLMResponse.parse_error` to the schema's
        ``parse_status`` enum + detailed ``parse_error`` message.

        Returns ``(parse_status, parse_error)`` where ``parse_error`` is
        ``None`` when ``parse_status == "ok"``.
        """
        if response.json_body is not None:
            return ("ok", None)
        err = response.parse_error or ""
        if err.startswith("json_decode_error"):
            return ("json_decode_error", err)
        if err.startswith("json_not_object"):
            return ("not_dict", err)
        if err:
            return ("no_content", err)
        return ("no_content", None)

    def _emit(self, event: InferenceEvent) -> None:
        """Emit a legacy :class:`InferenceEvent` to the legacy sink."""
        if self._telemetry is not None:
            try:
                self._telemetry.record(event)
            except Exception:
                logger.exception(
                    "telemetry sink failed for %s", event.inference_id
                )

    def _emit_attempt(self, event: InferenceAttemptEvent) -> None:
        """Emit an :class:`InferenceAttemptEvent` to the V2 sink."""
        if self._telemetry_v2 is not None:
            try:
                self._telemetry_v2.record_attempt(event)
            except Exception:
                logger.exception(
                    "telemetry V2 sink failed for attempt %s",
                    event.attempt_id,
                )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_legal(
    legal_actions: Sequence[LegalAction],
    action_type: Optional[str],
) -> Optional[LegalAction]:
    if action_type is None:
        return None
    for a in legal_actions:
        if a.action_type == action_type:
            return a
    return None
