"""Prompt contract: stable system prompt + scenario contract + per-tick observation.

This module implements Phase 1 / Beta correction §5.4-5.6 of
``docs/07.advice/2026-07-30_WorldLoop主线实验有效性与Beta发布优化实施方案.md``.

The LLM prompt is composed of THREE distinct, versioned, hashable layers:

1. **Stable system prompt** (:data:`STABLE_SYSTEM_PROMPT`) —
   cross-environment rules that NEVER change between scenarios. Sent
   via :class:`LLMRequest.system_prompt`, NOT concatenated into user
   content. This is the only way the model receives its
   action-selection identity and output schema contract.

2. **Scenario contract** (:class:`ScenarioContract`) — derived from a
   frozen :class:`AgentObservationView`. Declares the scenario id /
   version, public objective, action schema (one entry per
   ``legal_actions`` action_type), observation schema summary, and
   the omission disclaimer (which capabilities were intentionally
   hidden). Cross-tick stable for the same scenario spec.

3. **Per-tick user message** (:func:`build_user_message`) — the
   scenario contract PLUS the per-tick :class:`AgentObservationView`
   serialized as canonical JSON. This is the only state-bearing
   component; it MUST be the only place hidden-state leakage could
   occur, and the schema design (no ``rng_state`` / ``private_`` /
   ``cache`` fields on :class:`AgentObservationView`) structurally
   prevents leakage.

All three layers carry their own version + SHA-256 hash
(:class:`PromptHashBundle`).

Prompt Gate verification (:func:`validate_prompt_components`) enforces:

- **P-G1**: different observation -> different prompt hash
  (computed by :func:`compute_prompt_hashes`).
- **P-G2**: same observation, different hidden state -> same prompt
  hash (structural: hidden state is not on the schema).
- **P-G3**: system role separated from user role in transport payload
  (enforced by :func:`build_llm_request` placing the system prompt
  ONLY in :attr:`LLMRequest.system_prompt`).
- **P-G4**: no unauthorized global fields in the user message
  (enforced by :func:`scan_for_forbidden_fields` /
  :data:`FORBIDDEN_GLOBAL_FIELDS`).
- **P-G5**: scenario contract action schema mechanically matches
  ``observation.legal_actions`` (enforced by
  :func:`validate_prompt_components`).
- **P-G6**: every component carries its own version + hash (returned
  by :func:`compute_prompt_hashes`).

Out of scope:

- LLM retry / telemetry event splitting (Phase 2).
- Scenario spec-level visibility policy extension (Phase 1 v0.1 uses
  column-name conventions inside each projector).
- Counterfactual observation projection (M4+).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from worldloop_kernel.observation import (
    OBSERVATION_SCHEMA_VERSION,
    AgentObservationView,
    hash_observation,
)

__all__ = [
    "STABLE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT_VERSION",
    "PROMPT_CONTRACT_SCHEMA_VERSION",
    "USER_MESSAGE_SCHEMA_VERSION",
    "FORBIDDEN_GLOBAL_FIELDS",
    "ScenarioContract",
    "PromptHashBundle",
    "build_scenario_contract",
    "build_user_message",
    "build_llm_request",
    "hash_system_prompt",
    "hash_scenario_contract",
    "hash_user_message",
    "compute_prompt_hashes",
    "scan_for_forbidden_fields",
    "validate_prompt_components",
]


# ---------------------------------------------------------------------------
# Versions
# ---------------------------------------------------------------------------

#: Schema version of the prompt contract module. Bumped whenever the
#: structure of :class:`ScenarioContract` or :func:`build_user_message`
#: output changes. Producers and consumers MUST agree on this version
#: before exchanging prompt artifacts.
PROMPT_CONTRACT_SCHEMA_VERSION: str = "0.1.0"

#: Schema version of the stable system prompt text. Bumped whenever
#: :data:`STABLE_SYSTEM_PROMPT` is edited (even by one character).
SYSTEM_PROMPT_VERSION: str = "0.1.0"

#: Schema version of the user-message JSON payload produced by
#: :func:`build_user_message`. Bumped whenever the JSON shape changes.
USER_MESSAGE_SCHEMA_VERSION: str = "0.1.0"


# ---------------------------------------------------------------------------
# Stable system prompt (P-G3, P-G6)
# ---------------------------------------------------------------------------

#: The stable, cross-environment system prompt. Sent ONLY via
#: :attr:`LLMRequest.system_prompt` — NEVER concatenated into user
#: content. Cross-scenario stable; identical for every tick, every agent,
#: every scenario.
#:
#: This prompt establishes:
#: - the model's identity (action-selection policy for ONE simulated agent)
#: - the input contract (use ONLY the supplied observation and legal actions)
#: - the output contract (exactly one JSON object matching the schema)
#: - the boundary (no hidden global state, no invented values)
STABLE_SYSTEM_PROMPT: str = (
    "You are an action-selection policy for one simulated agent.\n"
    "Use only the supplied observation and legal action contract.\n"
    "Choose at most one legal action.\n"
    "Return exactly one JSON object matching the output schema:\n"
    '{"action_type": "<legal_action_type>", "params": {...}, "reason_code": "<short>"}\n'
    "Do not assume hidden global state or invent missing values.\n"
    "Do not propose multiple actions; choose exactly one or decline.\n"
)


# ---------------------------------------------------------------------------
# Forbidden global fields (P-G4)
# ---------------------------------------------------------------------------

#: Field-name substrings that MUST NEVER appear in the per-tick user
#: message. Their presence indicates hidden-state leakage from the
#: world. The check is substring-based on the canonical JSON keys.
#:
#: Members:
#: - ``rng_state`` / ``random_state``: kernel RNG (must be hidden)
#: - ``private_``: any private column of OTHER agents (focal agent's
#:   ``self_visible_attributes`` is allowed because it is the focal's OWN
#:   private state, but other agents' private columns must not appear)
#: - ``cache`` / ``_cache``: world internal caches
#: - ``world_`` / ``internal_``: world-internal bookkeeping
#: - ``global_`` / ``total_``: global counters / accumulators
#: - ``checkpoint``: replay internals
#: - ``parent_episode_id`` / ``fork_group_id`` / ``branch_id`` /
#:   ``branch_group_id``: counterfactual fork internals
#: - ``source_commit`` / ``source_dirty``: provenance metadata
#: - ``api_key`` / ``secret`` / ``token``: secrets (defensive)
#:
#: Note: ``self_visible_attributes`` is allowed because it carries the
#: FOCAL agent's private state to the FOCAL policy — that is the whole
#: point of self-visibility. The substring ``private_`` would catch
#: a hypothetical ``private_energy`` field name, not the structural
#: ``self_visible_attributes`` mapping.
FORBIDDEN_GLOBAL_FIELDS: frozenset[str] = frozenset(
    {
        "rng_state",
        "random_state",
        "_rng",
        "private_",
        "cache",
        "_cache",
        "world_",
        "internal_",
        "global_",
        "total_",
        "checkpoint",
        "parent_episode_id",
        "fork_group_id",
        "branch_id",
        "branch_group_id",
        "source_commit",
        "source_dirty",
        "api_key",
        "secret",
        "token",
    }
)


# ---------------------------------------------------------------------------
# Scenario contract (P-G5, P-G6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionSchemaEntry:
    """One entry in the scenario contract's action schema.

    Attributes
    ----------
    action_type:
        The action type string (matches ``LegalAction.action_type``).
    params_schema:
        Mapping of parameter name to JSON-schema-like type string
        (e.g., ``{"target_node": "string"}``). Empty if the action
        takes no parameters.
    """

    action_type: str
    params_schema: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ScenarioContract:
    """Cross-tick-stable scenario contract derived from the observation.

    Built from a :class:`AgentObservationView` by :func:`build_scenario_contract`.
    The contract is "cross-tick stable" in the sense that for the same
    scenario spec, the ``scenario_id`` / ``scenario_version`` /
    ``public_objective`` / ``action_schema`` / ``observation_schema_summary``
    do not change between ticks — only the per-tick observation changes.

    Attributes
    ----------
    schema_version:
        :data:`PROMPT_CONTRACT_SCHEMA_VERSION`.
    scenario_id:
        From :attr:`AgentObservationView.scenario_id`.
    scenario_version:
        From :attr:`AgentObservationView.scenario_version`.
    public_objective:
        Generic public objective derived from the scenario_id. This
        is intentionally generic (e.g., "select legal actions to
        maximize survival / efficiency per scenario spec") —
        scenario-specific objectives are part of the scenario spec,
        not the prompt contract. Phase 1 v0.1 does not parse spec.yaml
        to extract scenario-specific objectives.
    action_schema:
        Tuple of :class:`ActionSchemaEntry`, one per distinct
        ``action_type`` in :attr:`AgentObservationView.legal_actions`.
        Mechanically derived from the observation (P-G5).
    observation_schema_summary:
        Mapping describing the observation shape
        (``schema_version`` + the visible-slot list). Used by the
        model to anticipate what fields will appear in the per-tick
        observation block.
    omission_disclaimer:
        Human-readable string listing the capabilities omitted by
        :attr:`AgentObservationView.omission_policy`. Sent to the
        model so it knows what NOT to expect.
    """

    schema_version: str
    scenario_id: str
    scenario_version: str
    public_objective: str
    action_schema: tuple[ActionSchemaEntry, ...]
    observation_schema_summary: Mapping[str, Any]
    omission_disclaimer: str


# ---------------------------------------------------------------------------
# Prompt hash bundle (P-G6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptHashBundle:
    """Five SHA-256 hashes covering every prompt component.

    Attributes
    ----------
    system_prompt_hash:
        SHA-256 of :data:`STABLE_SYSTEM_PROMPT`.
    scenario_contract_hash:
        SHA-256 of the canonical JSON of the scenario contract.
    observation_hash:
        SHA-256 of the observation (delegated to
        :func:`hash_observation`).
    user_message_hash:
        SHA-256 of the user-message JSON string
        (:func:`build_user_message` output).
    combined_prompt_hash:
        SHA-256 of ``system_prompt_hash + scenario_contract_hash +
        observation_hash + user_message_hash`` concatenated. This is
        the single hash that uniquely identifies the full prompt
        fed to the LLM (P-G1 / P-G2).
    """

    system_prompt_hash: str
    scenario_contract_hash: str
    observation_hash: str
    user_message_hash: str
    combined_prompt_hash: str


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_scenario_contract(
    observation: AgentObservationView,
) -> ScenarioContract:
    """Build a :class:`ScenarioContract` from an observation.

    The contract is mechanically derived from the observation — no
    spec.yaml parsing, no scenario-specific logic. The
    :attr:`ScenarioContract.action_schema` has one entry per distinct
    ``action_type`` in :attr:`observation.legal_actions`, with the
    parameter schema inferred from the first occurrence of each
    action_type (parameter types are stringified JSON-schema-like).

    Parameters
    ----------
    observation:
        The per-agent observation to derive the contract from.

    Returns
    -------
    ScenarioContract
        A frozen, hashable contract.
    """
    # Build action schema: one entry per distinct action_type.
    seen_types: set[str] = set()
    action_schema: list[ActionSchemaEntry] = []
    for la in observation.legal_actions:
        if la.action_type in seen_types:
            continue
        seen_types.add(la.action_type)
        params_schema: dict[str, str] = {}
        for k, v in (la.params or {}).items():
            if isinstance(v, bool):
                params_schema[k] = "boolean"
            elif isinstance(v, int):
                params_schema[k] = "integer"
            elif isinstance(v, float):
                params_schema[k] = "number"
            elif isinstance(v, str):
                params_schema[k] = "string"
            elif v is None:
                params_schema[k] = "null"
            else:
                params_schema[k] = "string"
        action_schema.append(
            ActionSchemaEntry(
                action_type=la.action_type,
                params_schema=params_schema,
            )
        )

    # Observation schema summary: which slots are present / non-empty.
    visible_slots: list[str] = []
    if observation.visible_fields:
        visible_slots.append("visible_fields")
    if observation.visible_entities:
        visible_slots.append("visible_entities")
    if observation.visible_relations:
        visible_slots.append("visible_relations")
    if observation.visible_events:
        visible_slots.append("visible_events")
    visible_slots.append("legal_actions")  # always present
    visible_slots.append("focal_agent")  # always present

    observation_schema_summary: dict[str, Any] = {
        "schema_version": observation.schema_version,
        "visible_slots": tuple(visible_slots),
        "focal_agent_id": observation.focal_agent.agent_id,
        "tick": observation.tick,
    }

    # Omission disclaimer: list unsupported capabilities.
    omitted = observation.omission_policy.unsupported_capabilities
    if omitted:
        omission_disclaimer = (
            "Capabilities omitted from this observation (do not expect them): "
            + ", ".join(omitted)
            + "."
        )
    else:
        omission_disclaimer = (
            "All capabilities declared by the scenario are present in the "
            "observation."
        )

    return ScenarioContract(
        schema_version=PROMPT_CONTRACT_SCHEMA_VERSION,
        scenario_id=observation.scenario_id,
        scenario_version=observation.scenario_version,
        public_objective=(
            "Select legal actions to maximize scenario-defined survival "
            "and efficiency outcomes; only the supplied observation and "
            "legal action contract are authorized inputs."
        ),
        action_schema=tuple(action_schema),
        observation_schema_summary=observation_schema_summary,
        omission_disclaimer=omission_disclaimer,
    )


def build_user_message(
    observation: AgentObservationView,
    scenario_contract: ScenarioContract | None = None,
) -> str:
    """Build the per-tick user message JSON string.

    The message contains:

    - ``schema_version``: :data:`USER_MESSAGE_SCHEMA_VERSION`
    - ``scenario_contract``: the contract dict (or built on-the-fly
      if ``scenario_contract`` is None)
    - ``observation``: the canonical observation dict

    The output is JSON-serializable, sort_keys=True, ensure_ascii=False
    — stable across runs for the same observation (P-G1 / P-G2).

    The system prompt is NOT included here; it travels via
    :attr:`LLMRequest.system_prompt` (P-G3).
    """
    if scenario_contract is None:
        scenario_contract = build_scenario_contract(observation)

    contract_dict = _scenario_contract_to_dict(scenario_contract)
    observation_dict = _observation_to_dict(observation)

    payload: dict[str, Any] = {
        "schema_version": USER_MESSAGE_SCHEMA_VERSION,
        "scenario_contract": contract_dict,
        "observation": observation_dict,
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)


def build_llm_request(
    observation: AgentObservationView,
    *,
    model: str,
    temperature: float = 0.0,
    max_tokens: int = 256,
    scenario_contract: ScenarioContract | None = None,
) -> "LLMRequestLike":
    """Build an :class:`LLMRequest` with separated system and user roles.

    The system prompt is placed ONLY in ``system_prompt``; the user
    message (scenario contract + observation) is placed ONLY in
    ``prompt``. The two are NEVER concatenated (P-G3).

    Returns a lightweight dataclass that mirrors
    :class:`worldloop_data.llm_policy.LLMRequest`; callers convert as
    needed. We avoid importing :class:`LLMRequest` here to keep this
    module dependency-free of the LLM policy module (the dependency
    goes the other way: :mod:`llm_policy` imports this module).
    """
    if scenario_contract is None:
        scenario_contract = build_scenario_contract(observation)
    user_message = build_user_message(observation, scenario_contract)
    return LLMRequestLike(
        prompt=user_message,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        system_prompt=STABLE_SYSTEM_PROMPT,
    )


@dataclass(frozen=True)
class LLMRequestLike:
    """Local mirror of :class:`LLMRequest` to avoid a circular import.

    :mod:`llm_policy` imports :func:`build_llm_request` and converts
    this to its own :class:`LLMRequest` before sending to the client.
    """

    prompt: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 256
    system_prompt: str | None = None


# ---------------------------------------------------------------------------
# Hashes (P-G6)
# ---------------------------------------------------------------------------


def _sha256_hex(text: str) -> str:
    """Return ``"sha256:<hex>"`` for canonical hashing."""
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"sha256:{h}"


def hash_system_prompt(system_prompt: str = STABLE_SYSTEM_PROMPT) -> str:
    """SHA-256 of the system prompt text."""
    return _sha256_hex(system_prompt)


def hash_scenario_contract(contract: ScenarioContract) -> str:
    """SHA-256 of the canonical JSON encoding of ``contract``."""
    payload = json.dumps(
        _scenario_contract_to_dict(contract),
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return _sha256_hex(payload)


def hash_user_message(user_message: str) -> str:
    """SHA-256 of the user-message JSON string."""
    return _sha256_hex(user_message)


def compute_prompt_hashes(
    observation: AgentObservationView,
    scenario_contract: ScenarioContract | None = None,
    user_message: str | None = None,
) -> PromptHashBundle:
    """Compute all five prompt component hashes.

    If ``scenario_contract`` is None, it is built from ``observation``.
    If ``user_message`` is None, it is built from ``observation`` and
    ``scenario_contract``.
    """
    if scenario_contract is None:
        scenario_contract = build_scenario_contract(observation)
    if user_message is None:
        user_message = build_user_message(observation, scenario_contract)

    system_hash = hash_system_prompt()
    contract_hash = hash_scenario_contract(scenario_contract)
    obs_hash = hash_observation(observation)
    user_hash = hash_user_message(user_message)
    combined = _sha256_hex(
        system_hash + contract_hash + obs_hash + user_hash
    )
    return PromptHashBundle(
        system_prompt_hash=system_hash,
        scenario_contract_hash=contract_hash,
        observation_hash=obs_hash,
        user_message_hash=user_hash,
        combined_prompt_hash=combined,
    )


# ---------------------------------------------------------------------------
# P-G4: forbidden field scan
# ---------------------------------------------------------------------------


def scan_for_forbidden_fields(user_message: str) -> list[str]:
    """Scan the user-message JSON string for forbidden field-name substrings.

    Returns a list of forbidden substrings that appear as JSON keys
    (or substrings thereof) in the message. Empty list = clean.

    The scan is conservative: it operates on the canonical JSON STRING,
    so it will catch forbidden substrings whether they appear as keys
    or as string values. The caller should treat any non-empty result
    as a P-G4 violation.
    """
    violations: list[str] = []
    for forbidden in FORBIDDEN_GLOBAL_FIELDS:
        if forbidden in user_message:
            violations.append(forbidden)
    return violations


# ---------------------------------------------------------------------------
# P-G5: validation (scenario contract mechanically matches observation)
# ---------------------------------------------------------------------------


def validate_prompt_components(
    observation: AgentObservationView,
    scenario_contract: ScenarioContract | None = None,
    user_message: str | None = None,
) -> list[str]:
    """Validate prompt components against the Prompt Gate spec.

    Returns a list of human-readable error strings. Empty list = PASS.

    Checks:

    - P-G3: scenario_contract.schema_version matches
      :data:`PROMPT_CONTRACT_SCHEMA_VERSION`.
    - P-G4: no forbidden field substrings in ``user_message``.
    - P-G5: scenario_contract.action_schema has one entry per distinct
      ``action_type`` in ``observation.legal_actions``; every
      ``action_type`` in ``observation.legal_actions`` appears in the
      schema; scenario_id / scenario_version match between observation
      and contract.
    - P-G6: every hash in :func:`compute_prompt_hashes` is non-empty
      and starts with ``"sha256:"``.
    """
    if scenario_contract is None:
        scenario_contract = build_scenario_contract(observation)
    if user_message is None:
        user_message = build_user_message(observation, scenario_contract)

    errors: list[str] = []

    # P-G3: schema version match
    if scenario_contract.schema_version != PROMPT_CONTRACT_SCHEMA_VERSION:
        errors.append(
            f"scenario_contract.schema_version={scenario_contract.schema_version!r} "
            f"does not match PROMPT_CONTRACT_SCHEMA_VERSION="
            f"{PROMPT_CONTRACT_SCHEMA_VERSION!r}"
        )

    # P-G4: forbidden field scan
    forbidden_hits = scan_for_forbidden_fields(user_message)
    if forbidden_hits:
        errors.append(
            f"forbidden global fields leaked into user_message: {forbidden_hits}"
        )

    # P-G5: action schema mechanically matches observation.legal_actions
    obs_action_types = {la.action_type for la in observation.legal_actions}
    contract_action_types = {e.action_type for e in scenario_contract.action_schema}
    if obs_action_types != contract_action_types:
        missing = obs_action_types - contract_action_types
        extra = contract_action_types - obs_action_types
        if missing:
            errors.append(
                f"action_schema missing action_types present in legal_actions: "
                f"{sorted(missing)}"
            )
        if extra:
            errors.append(
                f"action_schema has action_types not in legal_actions: "
                f"{sorted(extra)}"
            )
    if scenario_contract.scenario_id != observation.scenario_id:
        errors.append(
            f"scenario_contract.scenario_id={scenario_contract.scenario_id!r} "
            f"!= observation.scenario_id={observation.scenario_id!r}"
        )
    if scenario_contract.scenario_version != observation.scenario_version:
        errors.append(
            f"scenario_contract.scenario_version="
            f"{scenario_contract.scenario_version!r} != "
            f"observation.scenario_version="
            f"{observation.scenario_version!r}"
        )

    # P-G6: every hash is well-formed
    bundle = compute_prompt_hashes(observation, scenario_contract, user_message)
    for name, h in (
        ("system_prompt_hash", bundle.system_prompt_hash),
        ("scenario_contract_hash", bundle.scenario_contract_hash),
        ("observation_hash", bundle.observation_hash),
        ("user_message_hash", bundle.user_message_hash),
        ("combined_prompt_hash", bundle.combined_prompt_hash),
    ):
        if not h or not h.startswith("sha256:"):
            errors.append(f"{name} is empty or not a sha256: hash: {h!r}")
        elif len(h) != len("sha256:") + 64:
            errors.append(
                f"{name} has wrong length {len(h)} (expected {len('sha256:') + 64})"
            )

    return errors


# ---------------------------------------------------------------------------
# Internal: canonical dict serialization (for hashing)
# ---------------------------------------------------------------------------


def _scenario_contract_to_dict(contract: ScenarioContract) -> dict[str, Any]:
    """Canonical dict encoding of a :class:`ScenarioContract`."""
    return {
        "schema_version": contract.schema_version,
        "scenario_id": contract.scenario_id,
        "scenario_version": contract.scenario_version,
        "public_objective": contract.public_objective,
        "action_schema": [
            {
                "action_type": e.action_type,
                "params_schema": dict(e.params_schema),
            }
            for e in contract.action_schema
        ],
        "observation_schema_summary": dict(contract.observation_schema_summary),
        "omission_disclaimer": contract.omission_disclaimer,
    }


def _observation_to_dict(observation: AgentObservationView) -> dict[str, Any]:
    """Canonical dict encoding of an :class:`AgentObservationView`.

    Used by :func:`build_user_message` to produce the per-tick
    observation block. We avoid :func:`dataclasses.asdict` because it
    recurses into nested dataclasses and the result is not stable for
    frozen mappings (``Mapping`` proxies become dicts, which is fine,
    but the order is not guaranteed for nested structures). Here we
    build the dict explicitly with stable ordering.
    """
    return {
        "schema_version": observation.schema_version,
        "scenario_id": observation.scenario_id,
        "scenario_version": observation.scenario_version,
        "tick": observation.tick,
        "focal_agent": {
            "agent_id": observation.focal_agent.agent_id,
            "public_attributes": dict(observation.focal_agent.public_attributes),
            "self_visible_attributes": dict(
                observation.focal_agent.self_visible_attributes
            ),
        },
        "previous_action": {
            "action_type": observation.previous_action.action_type,
            "success": observation.previous_action.success,
            "outcome_code": observation.previous_action.outcome_code,
            "visible_effect_summary": dict(
                observation.previous_action.visible_effect_summary
            ),
        },
        "visible_fields": dict(observation.visible_fields),
        "visible_entities": [
            {
                "entity_id": e.entity_id,
                "columns": dict(e.columns),
            }
            for e in observation.visible_entities
        ],
        "visible_relations": [
            {
                "src": r.src,
                "dst": r.dst,
                "edge_type": r.edge_type,
                "weight": r.weight,
            }
            for r in observation.visible_relations
        ],
        "visible_events": [
            {
                "kind": ev.kind,
                "tick": ev.tick,
                "payload": dict(ev.payload),
            }
            for ev in observation.visible_events
        ],
        "legal_actions": [
            {
                "action_type": la.action_type,
                "params": dict(la.params or {}),
            }
            for la in observation.legal_actions
        ],
        "omission_policy": {
            "omitted_slots": list(observation.omission_policy.omitted_slots),
            "reason": observation.omission_policy.reason,
            "unsupported_capabilities": list(
                observation.omission_policy.unsupported_capabilities
            ),
        },
    }
