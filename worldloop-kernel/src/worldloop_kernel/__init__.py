"""worldloop-kernel: WorldLoop v2 independent transition micro-kernel.

Package version: 0.1.3 (K-03 scaffold + K-04 minimal ten public types
+ K-05 canonical/hash/diff/apply + K-06 validation/recorder
+ K-07 checkpoint codec/replay/branch + K-08 toy world engine
+ Phase 1 observation contract + Phase 5 joint action submission).

The package is structured as:

- :mod:`worldloop_kernel.capability` — :class:`CapabilityProfile` (K-04)
- :mod:`worldloop_kernel.state`      — :class:`StateView` + 7 component types (K-04)
- :mod:`worldloop_kernel.action`     — :class:`ActionProposal` / :class:`ExecutedAction` /
  :class:`ExogenousInput` / :class:`ActionReceipt` + outcome codes (K-04)
- :mod:`worldloop_kernel.transition` — :class:`StateDelta` / :class:`TransitionRecord` /
  :class:`Checkpoint` + per-slot change types (K-04, extended in K-05)
- :mod:`worldloop_kernel.protocol`   — :class:`WorldProtocol` + :class:`ActionSpace` (K-04)
- :mod:`worldloop_kernel.canonical`  — :func:`canonical_encode` / :func:`hash_state` (K-05)
- :mod:`worldloop_kernel.diff_apply` — :func:`diff_state` / :func:`apply_delta` (K-05)
- :mod:`worldloop_kernel.validation` — :func:`validate_transition` + 7 invariants (K-06)
- :mod:`worldloop_kernel.recorder`   — append-only :class:`TransitionRecorder` (K-06)
- :mod:`worldloop_kernel.replay`     — :func:`replay` / :func:`branch` (K-07)
- :mod:`worldloop_kernel.engine`     — :class:`ToyWorld` engine for M0 Gate validation (K-08)
- :mod:`worldloop_kernel.observation` — :class:`AgentObservationView` + :class:`ObservationProjector`
  (Phase 1 / Beta correction: per-agent observation contract)

See ``README.md`` for scope, non-goals, and milestones. Architecture
decisions are documented in the repository ``docs/CLAIMS.md`` (public
repo) and each package's ``README.md``.

The minimal ten public types (per main plan §4.5) are re-exported from
this top-level package so that consumers can write::

    from worldloop_kernel import (
        CapabilityProfile,
        StateView,
        ActionProposal,
        ExecutedAction,
        ExogenousInput,
        ActionReceipt,
        StateDelta,
        TransitionRecord,
        Checkpoint,
        WorldProtocol,
    )

K-05 also re-exports the canonical/hash/diff/apply helpers::

    from worldloop_kernel import (
        canonical_encode,
        hash_state,
        diff_state,
        apply_delta,
    )

K-06 re-exports the validator and recorder::

    from worldloop_kernel import (
        validate_transition,
        ValidationReport,
        InvariantResult,
        TransitionRecorder,
        RecorderManifest,
    )
"""

from __future__ import annotations

__version__ = "0.1.3"

# ---------------------------------------------------------------------------
# K-04 re-exports — the minimal ten public types + auxiliary types.
# ---------------------------------------------------------------------------
# Import order matters: capability / state / action / transition form a
# dependency chain via TYPE_CHECKING (runtime imports are zero thanks to
# ``from __future__ import annotations``). protocol imports none of them
# at runtime either. K-05 adds canonical (no deps) and diff_apply (deps
# on canonical + state + transition, all already imported above).
# K-06 adds validation (deps on action/canonical/diff_apply/state/transition)
# and recorder (deps on transition + validation).

from worldloop_kernel.capability import (
    CapabilityProfile,
    CapabilityError,
    Authority,
    TransitionMode,
    CAPABILITY_SLOTS,
)
from worldloop_kernel.state import (
    StateMeta,
    FieldState,
    EntityTable,
    RelationEdge,
    RelationGraph,
    RegistryEntry,
    RegistrySnapshot,
    BirthRecord,
    DeathRecord,
    PopulationState,
    EventRecord,
    EventContext,
    StateView,
    StateError,
)
from worldloop_kernel.action import (
    ActionProposal,
    ExecutedAction,
    ExogenousInput,
    ActionReceipt,
    ActionError,
    OutcomeCode,
    OUTCOME_OK,
    OUTCOME_DISABLED_BY_ABLATION,
    OUTCOME_FEATURE_DISABLED,
    OUTCOME_UNRECOGNIZED_INTENT,
    OUTCOME_ILLEGAL_TARGET,
    OUTCOME_ILLEGAL_ACTION,
    OUTCOME_INSUFFICIENT_ENERGY,
    OUTCOME_UNKNOWN_FAILURE,
    KERNEL_OUTCOME_CODES,
)
from worldloop_kernel.transition import (
    FieldChange,
    EntityChange,
    EntityChanges,
    RelationChange,
    RelationChanges,
    RegistryChange,
    RegistryChanges,
    PopulationChange,
    PopulationChanges,
    StateDelta,
    TransitionRecord,
    Checkpoint,
    TransitionError,
    PROTOCOL_SCHEMA_VERSION,
)
from worldloop_kernel.protocol import (
    WorldProtocol,
    ActionSpace,
    LegalAction,
)
# K-05: canonical encoding + hashing + diff/apply
from worldloop_kernel.canonical import (
    CanonicalError,
    canonical_encode,
    hash_state,
    HASH_PREFIX,
)
from worldloop_kernel.diff_apply import (
    DiffApplyError,
    diff_state,
    apply_delta,
)
# K-06: validation + recorder
from worldloop_kernel.validation import (
    InvariantResult,
    ValidationReport,
    ValidationError,
    validate_transition,
    INVARIANT_NAMES,
)
from worldloop_kernel.recorder import (
    RecorderManifest,
    RecorderError,
    TransitionRecorder,
)
# K-07: checkpoint codec + replay + branch
from worldloop_kernel.replay import (
    CheckpointCodec,
    ReplayReport,
    BranchResult,
    ReplayError,
    compute_checkpoint_checksum,
    verify_checkpoint_restoration,
    replay,
    branch,
)
# K-08: toy world engine for M0 Gate validation
from worldloop_kernel.engine import (
    ToyWorld,
    TOY_WORLD_ID,
    TOY_WORLD_VERSION,
    TOY_WORLD_PAYLOAD_CODEC,
    DEFAULT_GRID_LENGTH,
    DEFAULT_INITIAL_ENERGY,
    make_toy_capability,
)
# Phase 5: joint action submission (multi-agent same-tick)
from worldloop_kernel.joint import (
    JointAction,
    JointReceipt,
    JointActionError,
    JointActionWorld,
    supports_joint_actions,
    MISSING_AGENT_NOOP,
    MISSING_AGENT_STAY,
    MISSING_AGENT_ERROR,
    MISSING_AGENT_POLICIES,
)
# Phase 1 (Beta correction): agent-local observation contract
from worldloop_kernel.observation import (
    OBSERVATION_SCHEMA_VERSION,
    OmissionPolicy,
    VisibleEntity,
    VisibleRelation,
    VisibleEvent,
    FocalAgentAttributes,
    PreviousActionSummary,
    AgentObservationView,
    ObservationProjector,
    hash_observation,
    is_observation_projector,
)

__all__ = [
    # Version
    "__version__",
    # Capability (K-04)
    "CapabilityProfile",
    "CapabilityError",
    "Authority",
    "TransitionMode",
    "CAPABILITY_SLOTS",
    # State (K-04)
    "StateMeta",
    "FieldState",
    "EntityTable",
    "RelationEdge",
    "RelationGraph",
    "RegistryEntry",
    "RegistrySnapshot",
    "BirthRecord",
    "DeathRecord",
    "PopulationState",
    "EventRecord",
    "EventContext",
    "StateView",
    "StateError",
    # Action (K-04)
    "ActionProposal",
    "ExecutedAction",
    "ExogenousInput",
    "ActionReceipt",
    "ActionError",
    "OutcomeCode",
    "OUTCOME_OK",
    "OUTCOME_DISABLED_BY_ABLATION",
    "OUTCOME_FEATURE_DISABLED",
    "OUTCOME_UNRECOGNIZED_INTENT",
    "OUTCOME_ILLEGAL_TARGET",
    "OUTCOME_ILLEGAL_ACTION",
    "OUTCOME_INSUFFICIENT_ENERGY",
    "OUTCOME_UNKNOWN_FAILURE",
    "KERNEL_OUTCOME_CODES",
    # Transition (K-04, extended in K-05)
    "FieldChange",
    "EntityChange",
    "EntityChanges",
    "RelationChange",
    "RelationChanges",
    "RegistryChange",
    "RegistryChanges",
    "PopulationChange",
    "PopulationChanges",
    "StateDelta",
    "TransitionRecord",
    "Checkpoint",
    "TransitionError",
    "PROTOCOL_SCHEMA_VERSION",
    # Protocol (K-04)
    "WorldProtocol",
    "ActionSpace",
    "LegalAction",
    # Canonical + diff/apply (K-05)
    "CanonicalError",
    "canonical_encode",
    "hash_state",
    "HASH_PREFIX",
    "DiffApplyError",
    "diff_state",
    "apply_delta",
    # Validation + recorder (K-06)
    "InvariantResult",
    "ValidationReport",
    "ValidationError",
    "validate_transition",
    "INVARIANT_NAMES",
    "RecorderManifest",
    "RecorderError",
    "TransitionRecorder",
    # Checkpoint codec + replay + branch (K-07)
    "CheckpointCodec",
    "ReplayReport",
    "BranchResult",
    "ReplayError",
    "compute_checkpoint_checksum",
    "verify_checkpoint_restoration",
    "replay",
    "branch",
    # Toy world engine (K-08)
    "ToyWorld",
    "TOY_WORLD_ID",
    "TOY_WORLD_VERSION",
    "TOY_WORLD_PAYLOAD_CODEC",
    "DEFAULT_GRID_LENGTH",
    "DEFAULT_INITIAL_ENERGY",
    "make_toy_capability",
    # Phase 5: joint action submission (multi-agent same-tick)
    "JointAction",
    "JointReceipt",
    "JointActionError",
    "JointActionWorld",
    "supports_joint_actions",
    "MISSING_AGENT_NOOP",
    "MISSING_AGENT_STAY",
    "MISSING_AGENT_ERROR",
    "MISSING_AGENT_POLICIES",
    # Phase 1 (Beta correction): agent-local observation contract
    "OBSERVATION_SCHEMA_VERSION",
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
