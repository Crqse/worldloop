"""Agent-local observation view (Phase 1 / WL-P0-01 + WL-P0-02).

Defines the per-agent observation contract. :class:`AgentObservationView`
is the AUTHORIZED, AUDITABLE state slice that a policy is allowed to
consume when selecting an action. It is intentionally NOT a dump of
the global :class:`worldloop_kernel.state.StateView` — hidden state,
private fields, and unshared internals MUST NOT appear here.

Design rules (per main plan §5.2 / §5.3 of
``docs/07.advice/2026-07-30_WorldLoop主线实验有效性与Beta发布优化实施方案.md``):

- The kernel defines the schema; each world implementation owns the
  projection (``ParameterizedWorld.observe_agent`` /
  ``PettingZooParallelAdapter.observe_agent``).
- Same state + agent + visibility config => canonical identical
  observation (deterministic hash).
- Hidden state changes (RNG draws, internal caches, other agents'
  private fields) MUST NOT change the observation hash.
- Unsupported capabilities MUST be omitted, NOT filled with fake zeros.
  ``OmissionPolicy.unsupported_capabilities`` lists them.
- The observation hash is SHA-256 of the canonical encoding, computed
  via :func:`worldloop_kernel.canonical.hash_state`. Producers compute
  it on demand; consumers verify by recomputing.

Out of scope:

- LLM prompt assembly (lives in ``worldloop_data``).
- Specific visibility policies (each world owns its own).
- Per-agent observation caching (world implementation concern).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TYPE_CHECKING, runtime_checkable

if TYPE_CHECKING:
    from worldloop_kernel.canonical import hash_state as _hash_state  # noqa: F401
    from worldloop_kernel.protocol import LegalAction
    from worldloop_kernel.state import StateView

__all__ = [
    "OBSERVATION_SCHEMA_VERSION",
    "OmissionReason",
    "OmissionPolicy",
    "VisibleEntity",
    "VisibleRelation",
    "VisibleEvent",
    "FocalAgentAttributes",
    "PreviousActionSummary",
    "AgentObservationView",
    "ObservationProjector",
    "hash_observation",
    "is_observation_projector",
]

#: Schema version of the :class:`AgentObservationView` dataclass shape.
#: Bumped whenever the dataclass field set changes. Producers and
#: consumers MUST agree on this version before serialization.
OBSERVATION_SCHEMA_VERSION: str = "0.1.0"

#: Reasons an :class:`OmissionPolicy` may record. Stable string codes so
#: downstream tooling (prompt builder, leakage checker) can branch on
#: them without parsing free text.
OMISSION_REASON_CAPABILITY_UNAVAILABLE = "capability_unavailable"
OMISSION_REASON_VISIBILITY_FILTER = "visibility_filter"
OMISSION_REASON_NOT_FOCAL_RELEVANT = "not_focal_relevant"
OMISSION_REASON_HIDDEN_BY_DESIGN = "hidden_by_design"

OmissionReason = str


# ---------------------------------------------------------------------------
# Omission policy — what was hidden and why
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OmissionPolicy:
    """Documents which :class:`StateView` slots were intentionally
    omitted from this observation and why.

    This is metadata about the *projection*, NOT about the world state.
    It exists so downstream consumers (prompt builder, leakage checker,
    Prompt Gate P-G4) can verify no hidden field was accidentally leaked
    via ``legal_actions`` description or diagnostics.

    Attributes
    ----------
    omitted_slots:
        Tuple of :class:`StateView` slot names that were omitted
        (e.g., ``("registries", "population")``). Empty if nothing
        was omitted.
    reason:
        One of ``OMISSION_REASON_*`` constants. ``capability_unavailable``
        means the world declares the capability as ``False``;
        ``visibility_filter`` means the projection intentionally hides
        it; ``not_focal_relevant`` means the slot exists but is not
        relevant to the focal agent; ``hidden_by_design`` means the
        slot is private world-internal (RNG, caches).
    unsupported_capabilities:
        Tuple of capability slot names that are ``False`` in the
        world's :class:`CapabilityProfile`. These MUST NOT appear in
        ``visible_fields`` / ``visible_entities`` / ``visible_relations``
        / ``visible_events``.
    """

    omitted_slots: tuple[str, ...] = field(default_factory=tuple)
    reason: OmissionReason = OMISSION_REASON_CAPABILITY_UNAVAILABLE
    unsupported_capabilities: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Visible projection types — only what the focal agent is allowed to see
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VisibleEntity:
    """An entity visible to the focal agent.

    Attributes
    ----------
    entity_id:
        ID of the visible entity.
    columns:
        Mapping from column name to value. Only columns the focal
        agent is allowed to see appear here. Hidden columns (e.g.,
        private energy of other agents) MUST be omitted.
    """

    entity_id: str | int
    columns: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VisibleRelation:
    """A relation edge visible to the focal agent.

    Only the ``src`` / ``dst`` / ``edge_type`` / ``weight`` the focal
    agent is allowed to observe appear. Hidden edge metadata (e.g.,
    private trust scores) MUST be omitted by the projector.
    """

    src: str | int
    dst: str | int
    edge_type: str
    weight: float = 1.0


@dataclass(frozen=True)
class VisibleEvent:
    """An event visible to the focal agent.

    Attributes
    ----------
    kind:
        Event kind string (e.g., ``"resource_depleted"``).
    tick:
        Tick when the event occurred.
    payload:
        Visible event payload. Hidden payload fields MUST be omitted
        by the projector.
    """

    kind: str
    tick: int
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FocalAgentAttributes:
    """Public / self-visible attributes of the focal agent.

    Attributes
    ----------
    agent_id:
        ID of the focal agent.
    public_attributes:
        Attributes the focal agent and other agents can both observe
        (e.g., public position, public role).
    self_visible_attributes:
        Attributes only the focal agent can observe about itself
        (e.g., private energy, internal cooldowns). These are NOT
        visible to other agents but ARE visible to the focal agent's
        policy. This is the only place "private" focal state may
        appear; it MUST NOT leak into other agents' observations.
    """

    agent_id: str | int
    public_attributes: Mapping[str, Any] = field(default_factory=dict)
    self_visible_attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PreviousActionSummary:
    """Summary of the focal agent's previous action and its receipt.

    This is a SUMMARY, NOT a full :class:`ActionReceipt` — internal
    outcome codes and hidden validation diagnostics are NOT included.
    Only the parts the focal agent is allowed to remember.

    Attributes
    ----------
    action_type:
        The ``action_type`` of the previous proposal, or ``""`` if no
        previous action.
    success:
        Whether the previous action was accepted by the world.
    outcome_code:
        The kernel outcome code (e.g., ``"ok"``,
        ``"outcome_illegal_target"``). ``""`` if no previous action.
    visible_effect_summary:
        Mapping of visible effect fields (e.g.,
        ``{"energy_delta": -5}``). Hidden effects are NOT included.
    """

    action_type: str = ""
    success: bool = False
    outcome_code: str = ""
    visible_effect_summary: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AgentObservationView — the only state slice a policy may consume
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentObservationView:
    """The authorized, auditable per-agent observation.

    This is the ONLY state slice a policy is allowed to consume when
    selecting an action. Producers (world implementations) MUST project
    from :class:`StateView` to :class:`AgentObservationView` via an
    :class:`ObservationProjector`; consumers (LLM prompt builders,
    scripted policies) MUST NOT bypass this contract by reading
    :class:`StateView` directly.

    Design invariants:

    1. **Deterministic hash**: Same state + agent + visibility config
       produces the same :func:`hash_observation` result. Hidden state
       changes (RNG draws, internal caches, other agents' private
       fields) MUST NOT change the hash.
    2. **No hidden leakage**: Anything not in this view is hidden by
       design. The :class:`OmissionPolicy` documents what was omitted
       and why. The prompt builder MUST NOT read from
       :class:`StateView` directly.
    3. **No fake zeros**: Unsupported capabilities MUST be omitted, NOT
       filled with placeholder zeros. ``omission_policy.unsupported_capabilities``
       lists them.
    4. **Schema versioned**: ``schema_version`` lets consumers verify
       compatibility before parsing.
    5. **Scenario-attributed**: ``scenario_id`` / ``scenario_version``
       identify which scenario contract the observation belongs to.

    The ``observation_hash`` is intentionally NOT stored on the
    dataclass — it is computed on demand via :func:`hash_observation`
    so it always reflects the current visible content. Producers that
    need to persist the hash alongside the observation SHOULD write
    ``hash_observation(view)`` into their artifact manifest.
    """

    schema_version: str
    scenario_id: str
    scenario_version: str
    tick: int
    focal_agent: FocalAgentAttributes
    previous_action: PreviousActionSummary = field(default_factory=PreviousActionSummary)
    visible_fields: Mapping[str, Any] = field(default_factory=dict)
    visible_entities: tuple[VisibleEntity, ...] = field(default_factory=tuple)
    visible_relations: tuple[VisibleRelation, ...] = field(default_factory=tuple)
    visible_events: tuple[VisibleEvent, ...] = field(default_factory=tuple)
    legal_actions: tuple["LegalAction", ...] = field(default_factory=tuple)
    omission_policy: OmissionPolicy = field(default_factory=OmissionPolicy)


# ---------------------------------------------------------------------------
# ObservationProjector — interface each world implements
# ---------------------------------------------------------------------------


@runtime_checkable
class ObservationProjector(Protocol):
    """Interface each world implements to project per-agent observation.

    Worlds that support per-agent observation MUST implement this
    Protocol in addition to :class:`worldloop_kernel.protocol.WorldProtocol`.
    The data layer checks ``isinstance(world, ObservationProjector)``
    before consuming observations; worlds that do not implement it
    fall back to the legacy "no observation" path (Phase 0 governance
    freeze).

    Implementations MUST satisfy:

    - Same ``state`` + ``agent_id`` + visibility config => identical
      :class:`AgentObservationView` (same :func:`hash_observation`).
    - Hidden state changes (RNG, internal caches, other agents' private
      fields) MUST NOT change the resulting observation.
    - Unsupported capabilities MUST be omitted, NOT filled with zeros.
    """

    def observe_agent(
        self,
        agent_id: str | int,
        *,
        state: "StateView | None" = None,
    ) -> AgentObservationView:
        """Project the per-agent observation for ``agent_id``.

        Parameters
        ----------
        agent_id:
            ID of the focal agent.
        state:
            Optional state to project from. If ``None``, the world's
            current state is used. Worlds that do not support
            counterfactual observation MAY raise ``NotImplementedError``
            if ``state`` is not ``None``.

        Returns
        -------
        AgentObservationView
            The authorized observation for the focal agent. The
            ``omission_policy`` field MUST accurately reflect what was
            omitted.
        """
        ...


# ---------------------------------------------------------------------------
# Hashing — computed on demand, not stored
# ---------------------------------------------------------------------------


def hash_observation(observation: AgentObservationView) -> str:
    """Compute the canonical SHA-256 hash of an observation.

    Returns a string of the form ``"sha256:<hex_digest>"``. The hash
    covers every visible field of :class:`AgentObservationView`
    (including ``omission_policy``) but NOT any hidden world state —
    because the dataclass only contains visible content by construction.

    Producers call this to record the observation hash in their
    artifact manifest. Consumers call this to verify that a recorded
    observation matches its declared hash.

    This is a thin wrapper around
    :func:`worldloop_kernel.canonical.hash_state` — it exists as a
    named entry point so call sites document their intent.
    """
    from worldloop_kernel.canonical import hash_state

    return hash_state(observation)


def is_observation_projector(world: Any) -> bool:
    """Return ``True`` if ``world`` implements :class:`ObservationProjector`.

    Convenience wrapper around ``isinstance(world, ObservationProjector)``
    that does NOT raise on non-Protocol objects. Use this in the data
    layer to decide whether to call ``world.observe_agent`` or fall
    back to the legacy "no observation" path.
    """
    return isinstance(world, ObservationProjector)
