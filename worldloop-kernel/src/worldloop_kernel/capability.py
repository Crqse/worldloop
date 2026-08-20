"""Capability profile (K-04).

Declares which world capabilities a given world implementation provides.
Used by :mod:`worldloop_kernel.state` and :mod:`worldloop_kernel.protocol`
to gate which StateView fields are populated vs. ``missing_mask=True``.

Design rules (per ADR §3 and main plan §4.2 / §4.4):
- ``authority="learned"`` MUST be paired with ``ground_truth=False``.
- ``exact_restore=True`` REQUIRES a CheckpointCodec implementation
  (K-07). The kernel validates the pairing at construction time.
- Capability flags MUST NOT be mixed with ``missing_mask``: capability
  says "this world has this field at all"; missing_mask says "this world
  has the capability but this particular record lacks the value".
- ``transition_mode`` declares whether the world is deterministic given
  a seed+checkpoint. Stochastic worlds MAY still support replay via
  frozen RNG state in :class:`Checkpoint`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

__all__ = [
    "CapabilityProfile",
    "CapabilityError",
    "Authority",
    "TransitionMode",
    "CAPABILITY_SLOTS",
]

#: Authority literal type — who is the source of truth for state transitions.
Authority = Literal["rule", "learned", "hybrid"]

#: Transition mode literal type.
TransitionMode = Literal["deterministic", "stochastic"]

#: Canonical set of capability slot names. Each MUST appear as a boolean
#: field on :class:`CapabilityProfile`. ``missing_mask`` keys on
#: :class:`StateView` MUST be a subset of these names.
CAPABILITY_SLOTS: tuple[str, ...] = (
    "fields",
    "entities",
    "relations",
    "registries",
    "population",
    "events",
)


class CapabilityError(ValueError):
    """Raised when a :class:`CapabilityProfile` is internally inconsistent."""


@dataclass(frozen=True)
class CapabilityProfile:
    """Static capability declaration for a world implementation.

    All fields are booleans or string literals; the profile is immutable
    and hashable. Worlds publish one profile per ``world_id + world_version``
    and the kernel caches it on :class:`StateView` and :class:`Checkpoint`.
    """

    # --- State slots (must align with CAPABILITY_SLOTS) ---
    fields: bool
    entities: bool
    relations: bool
    registries: bool
    population: bool
    events: bool

    # --- Restoration / replay capabilities ---
    exact_restore: bool
    executable_deterministic_replay: bool

    # --- Authority / ground truth ---
    authority: Authority
    ground_truth: bool
    transition_mode: TransitionMode

    def __post_init__(self) -> None:
        # Rule: learned authority cannot claim ground truth.
        if self.authority == "learned" and self.ground_truth:
            raise CapabilityError(
                "authority='learned' MUST be paired with ground_truth=False; "
                "a learned simulator is not a rule-level ground truth."
            )
        # Rule: exact_restore=True REQUIRES deterministic replay capability.
        # A world that cannot replay deterministically cannot restore exactly.
        if self.exact_restore and not self.executable_deterministic_replay:
            raise CapabilityError(
                "exact_restore=True REQUIRES "
                "executable_deterministic_replay=True; without deterministic "
                "replay, restore cannot be exact."
            )
        # Rule: entities is mandatory in the main plan §4.2 ("a world without
        # entities is not a world"). The kernel enforces this at the
        # capability level — worlds that genuinely have no entities must
        # still declare a degenerate EntityTable, not flip this flag.
        if not self.entities:
            raise CapabilityError(
                "entities=False is not allowed; a world without entities is "
                "not a world (main plan §4.2). Use an empty EntityTable if "
                "the world is temporarily unpopulated."
            )

    def slot_flags(self) -> dict[str, bool]:
        """Return the per-slot capability flags as a dict.

        Used by :mod:`worldloop_kernel.state` to validate ``missing_mask``
        keys against the declared capability set.
        """
        return {slot: bool(getattr(self, slot)) for slot in CAPABILITY_SLOTS}
