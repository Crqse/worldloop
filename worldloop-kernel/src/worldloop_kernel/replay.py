"""Checkpoint codec, replay, and counterfactual branch (K-07).

Provides:
- :class:`CheckpointCodec` — Protocol for world-owned checkpoint
  encode/decode. The kernel dispatches on :attr:`Checkpoint.payload_codec`
  but does NOT implement the codecs itself.
- :func:`compute_checkpoint_checksum` — stable checksum of a
  :class:`Checkpoint`'s ``opaque_payload`` + ``state_view`` hash +
  ``rng_bundle`` (if any). Used by ``replay`` to verify restoration.
- :class:`ReplayReport` — frozen dataclass summarizing a replay run.
- :class:`BranchResult` — frozen dataclass summarizing one
  counterfactual branch.
- :func:`replay` — re-run a world from a checkpoint with a frozen
  action sequence. Must produce bit-identical state hashes when the
  world is deterministic.
- :func:`branch` — spawn one or more counterfactual branches from a
  checkpoint. Branches MUST NOT pollute the parent world's state.

Design rules (per main plan §4.6, §4.7 and ADR §3):
- ``replay`` MUST NOT make LLM or network calls. Frozen actions are
  injected directly via ``world.step``; the policy / LLM path is
  skipped because the actions are already executed actions.
- ``branch`` isolates each branch by saving the parent world's current
  state via ``world.checkpoint()``, then restoring to the fork
  checkpoint before each branch. After all branches, the parent is
  restored to its original state.
- For worlds with ``capability.exact_restore=True``, replay MUST be
  bit-identical (same per-tick state hashes as the original run). For
  worlds with ``exact_restore=False``, replay reports
  ``replay_consistent=False`` and the report is informational only.
- Counterfactual branches are recorded with explicit ``branch_id`` and
  ``fork_tick`` so downstream consumers can distinguish parent vs.
  branch records.

Provenance: extracted from ``current/worldloop/core/runtime/
replay_checkpoint.py`` and ``executable_replay.py`` (v1.0.0 tag);
re-implemented in kernel for the v2 minimal protocol. The v1
``ExecutableReplayer`` couples to ``Simulation.step_replay`` (a v1
runtime-specific entry point that skips L3/LLM); the kernel ``replay``
function instead expects ``world.step(ExecutedAction)`` to be the
single execution entry point, with the world responsible for skipping
LLM calls when the action is already an :class:`ExecutedAction`.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from worldloop_kernel.action import ExecutedAction
from worldloop_kernel.canonical import hash_state
from worldloop_kernel.transition import Checkpoint, TransitionRecord
from worldloop_kernel.validation import (
    ValidationReport,
    validate_transition,
)
from worldloop_kernel.protocol import WorldProtocol

__all__ = [
    "CheckpointCodec",
    "ReplayReport",
    "BranchResult",
    "ReplayError",
    "compute_checkpoint_checksum",
    "verify_checkpoint_restoration",
    "replay",
    "branch",
]


class ReplayError(RuntimeError):
    """Raised when replay/branch cannot proceed (e.g., world raises,
    checkpoint checksum mismatch on restore).

    Note that an inconsistent replay (state hash mismatch) is NOT a
    :class:`ReplayError` — it's reported via
    :attr:`ReplayReport.replay_consistent` = ``False``. This exception
    is reserved for programmer / protocol errors.
    """


# ---------------------------------------------------------------------------
# Checksum + restoration verification
# ---------------------------------------------------------------------------


def compute_checkpoint_checksum(checkpoint: Checkpoint) -> str:
    """Compute a stable SHA-256 checksum over the checkpoint's payload.

    The checksum covers:
    - ``schema_version`` (stability across protocol versions)
    - ``world_id`` / ``world_version`` (producer identity)
    - ``tick``
    - ``state_view`` canonical hash (via :func:`hash_state`)
    - ``opaque_payload`` bytes (world-owned, the kernel hashes as-is)
    - ``payload_codec`` (so the same payload under different codecs
      produces different checksums)
    - ``rng_bundle`` (if present; sorted by key)

    The checksum is prefixed with ``"sha256:"`` to match the
    :func:`hash_state` convention.

    Args:
        checkpoint: The :class:`Checkpoint` to checksum.

    Returns:
        ``"sha256:<64 hex chars>"``.
    """
    h = hashlib.sha256()
    h.update(checkpoint.schema_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(checkpoint.world_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(checkpoint.world_version.encode("utf-8"))
    h.update(b"\x00")
    h.update(str(checkpoint.tick).encode("ascii"))
    h.update(b"\x00")
    h.update(hash_state(checkpoint.state_view).encode("ascii"))
    h.update(b"\x00")
    h.update(checkpoint.opaque_payload)
    h.update(b"\x00")
    h.update(checkpoint.payload_codec.encode("utf-8"))
    if checkpoint.rng_bundle:
        # Sort keys for determinism.
        for key in sorted(checkpoint.rng_bundle):
            h.update(b"\x00")
            h.update(key.encode("utf-8"))
            h.update(b"\x01")
            h.update(str(checkpoint.rng_bundle[key]).encode("utf-8"))
    return "sha256:" + h.hexdigest()


def verify_checkpoint_restoration(
    world: WorldProtocol,
    checkpoint: Checkpoint,
) -> tuple[bool, str]:
    """Restore ``world`` from ``checkpoint`` and verify the restoration.

    Verification:
    1. Call ``world.restore(checkpoint)``.
    2. Call ``world.observe()`` and compute its canonical hash.
    3. Compare with ``hash_state(checkpoint.state_view)``.
    4. If ``checkpoint.checksum`` is non-empty, recompute
       :func:`compute_checkpoint_checksum` and compare.

    Args:
        world: The world to restore into.
        checkpoint: The checkpoint to restore from.

    Returns:
        Tuple of ``(ok, message)``. ``ok`` is True iff every check
        passed. ``message`` is empty on success or describes the first
        failure.
    """
    try:
        world.restore(checkpoint)
    except Exception as exc:  # noqa: BLE001 — protocol-level guard
        return False, f"world.restore raised: {exc!r}"

    try:
        observed = world.observe()
    except Exception as exc:  # noqa: BLE001
        return False, f"world.observe raised: {exc!r}"

    observed_hash = hash_state(observed)
    expected_hash = hash_state(checkpoint.state_view)
    if observed_hash != expected_hash:
        return (
            False,
            (
                f"state hash mismatch after restore: observed "
                f"{observed_hash} != checkpoint.state_view {expected_hash}"
            ),
        )

    if checkpoint.checksum:
        recomputed = compute_checkpoint_checksum(checkpoint)
        if recomputed != checkpoint.checksum:
            return (
                False,
                (
                    f"checkpoint checksum mismatch: stored "
                    f"{checkpoint.checksum} != recomputed {recomputed}"
                ),
            )

    return True, ""


# ---------------------------------------------------------------------------
# CheckpointCodec Protocol
# ---------------------------------------------------------------------------


class CheckpointCodec(Protocol):
    """Protocol for world-owned checkpoint encode/decode.

    The kernel does NOT implement codecs; each world provides its own
    codec matching the ``payload_codec`` string it writes into
    :class:`Checkpoint` instances. The kernel uses ``payload_codec`` to
    dispatch to the right codec when interoperating with external
    tools (e.g., dataset exporters that need to interpret payloads).

    A codec is a pair of functions:

    - ``encode(state) -> bytes``: serialize world-internal state to
      ``opaque_payload``.
    - ``decode(payload) -> state``: deserialize ``opaque_payload`` back
      to world-internal state.

    The kernel never calls these directly during ``replay`` / ``branch``
    — the world's ``checkpoint()`` / ``restore()`` methods own the
    encoding/decoding internally. This Protocol exists to document the
    contract and to allow external tools to register codecs for
    interop.
    """

    payload_codec: str

    def encode(self, state: Any) -> bytes: ...

    def decode(self, payload: bytes) -> Any: ...


# ---------------------------------------------------------------------------
# ReplayReport + BranchResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayReport:
    """Summary of a :func:`replay` run.

    Attributes
    ----------
    checkpoint_hash:
        Hash of the input checkpoint (via
        :func:`compute_checkpoint_checksum`).
    actions_hash:
        Stable hash of the frozen action sequence (SHA-256 over the
        canonical encoding of each :class:`ExecutedAction`).
    final_state_hash:
        Canonical hash (via :func:`hash_state`) of the world's state
        after the last action. ``None`` if the action sequence was
        empty (the world's state remains at the checkpoint).
    per_tick_hashes:
        Tuple of canonical state hashes, one per tick executed. The
        first element is the state after the first action; the last
        element equals ``final_state_hash``. Empty if no actions.
    replay_consistent:
        True iff (a) the world declares ``exact_restore=True`` AND
        ``executable_deterministic_replay=True``, (b) every
        :func:`validate_transition` on the produced
        :class:`TransitionRecord` passed, and (c) the restoration
        verification passed. False otherwise. When False, the report
        is informational only — consumers MUST NOT treat the replay
        as ground truth.
    invariant_violations:
        Tuple of human-readable strings, one per invariant violation
        observed during replay. Empty if all invariants held.
    restoration_ok:
        True iff :func:`verify_checkpoint_restoration` passed.
    restoration_message:
        Empty if ``restoration_ok``; otherwise the failure message.
    cap_exact_restore:
        Value of ``world.capabilities.exact_restore`` at replay start.
        Recorded so consumers can tell whether
        ``replay_consistent=False`` is due to capability declaration
        or actual inconsistency.
    cap_deterministic_replay:
        Value of ``world.capabilities.executable_deterministic_replay``.
    n_actions:
        Number of actions in the input sequence.
    """

    checkpoint_hash: str
    actions_hash: str
    final_state_hash: str | None
    per_tick_hashes: tuple[str, ...]
    replay_consistent: bool
    invariant_violations: tuple[str, ...]
    restoration_ok: bool
    restoration_message: str
    cap_exact_restore: bool
    cap_deterministic_replay: bool
    n_actions: int


@dataclass(frozen=True)
class BranchResult:
    """Summary of one counterfactual branch.

    Attributes
    ----------
    branch_id:
        Stable identifier for the branch. Format: ``"b{index}"`` where
        index is the 0-based position in ``alternatives``.
    fork_tick:
        Tick at which the branch forked from the parent (i.e., the
        checkpoint's tick).
    actions:
        Tuple of :class:`ExecutedAction` objects executed on this
        branch.
    final_state_hash:
        Canonical hash of the branch's final state. ``None`` if the
        action sequence was empty.
    per_tick_hashes:
        Tuple of canonical state hashes, one per tick executed on the
        branch.
    diverged_at_tick:
        Tick at which the branch's state hash first differed from the
        parent's state hash at the same tick. ``None`` if (a) the
        branch matches the parent at every tick, or (b) no parent
        per-tick hashes were supplied to :func:`branch` for comparison.
    restoration_ok:
        True iff the parent world was successfully restored to its
        pre-branch state after this branch ran. False indicates the
        parent world is polluted; downstream branches may be invalid.
    error:
        ``None`` if the branch ran without exceptions. Otherwise the
        stringified exception. Branches with ``error`` is not None have
        ``final_state_hash=None`` and empty ``per_tick_hashes``.
    """

    branch_id: str
    fork_tick: int
    actions: tuple[ExecutedAction, ...]
    final_state_hash: str | None
    per_tick_hashes: tuple[str, ...]
    diverged_at_tick: int | None
    restoration_ok: bool
    error: str | None


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------


def _hash_action_sequence(actions: Sequence[ExecutedAction]) -> str:
    """Stable SHA-256 hash over a sequence of ExecutedActions.

    Uses :func:`canonical_encode` from K-05 to encode each action;
    the hash is over the concatenation with a separator.
    """
    from worldloop_kernel.canonical import canonical_encode

    h = hashlib.sha256()
    for action in actions:
        h.update(canonical_encode(action))
        h.update(b"\x00")  # separator to prevent prefix collisions
    return "sha256:" + h.hexdigest()


def replay(
    world: WorldProtocol,
    checkpoint: Checkpoint,
    actions: Sequence[ExecutedAction],
) -> ReplayReport:
    """Re-run a world from a checkpoint with a frozen action sequence.

    The replay proceeds as follows:

    1. Record ``world.capabilities`` (``exact_restore`` and
       ``executable_deterministic_replay``).
    2. Restore the world from ``checkpoint`` and verify restoration
       via :func:`verify_checkpoint_restoration`.
    3. For each action in ``actions``:
       a. Call ``world.step(action)`` to produce a
          :class:`TransitionRecord`.
       b. Validate the record via :func:`validate_transition` (with
          ``before`` = the pre-step state, observed via
          ``world.observe()`` before the step).
       c. Record the post-step state hash via ``world.observe()``.
    4. Compute ``replay_consistent`` = restoration_ok AND no invariant
       violations AND cap_exact_restore AND cap_deterministic_replay.

    The replay does NOT make LLM or network calls. ``world.step`` is
    expected to accept an already-executed action and skip the LLM
    path; this is the world's responsibility per the protocol contract.

    Args:
        world: The world to replay. Must implement :class:`WorldProtocol`.
        checkpoint: The checkpoint to restore from.
        actions: Frozen sequence of :class:`ExecutedAction` objects to
            apply. May be empty (the replay just verifies restoration).

    Returns:
        A :class:`ReplayReport`.

    Raises:
        ReplayError: If ``world.restore`` or ``world.observe`` raises
            an unexpected exception (programmer / protocol error).
            Inconsistent state hashes do NOT raise — they are reported
            via ``ReplayReport.replay_consistent=False``.
    """
    cap = world.capabilities
    checkpoint_hash = compute_checkpoint_checksum(checkpoint)
    actions_hash = _hash_action_sequence(actions)

    restoration_ok, restoration_message = verify_checkpoint_restoration(
        world, checkpoint
    )
    invariant_violations: list[str] = []
    per_tick_hashes: list[str] = []

    for action in actions:
        # Observe pre-step state for validation.
        try:
            before = world.observe()
        except Exception as exc:  # noqa: BLE001
            raise ReplayError(
                f"world.observe raised before step: {exc!r}"
            ) from exc

        # Step the world.
        try:
            record: TransitionRecord = world.step(action)
        except Exception as exc:  # noqa: BLE001
            raise ReplayError(
                f"world.step raised: {exc!r}"
            ) from exc

        # Observe post-step state.
        try:
            after = world.observe()
        except Exception as exc:  # noqa: BLE001
            raise ReplayError(
                f"world.observe raised after step: {exc!r}"
            ) from exc

        # Validate the transition. We use the report's invariant_results
        # to populate invariant_violations; we do NOT raise on failure.
        try:
            report: ValidationReport = validate_transition(
                record, before=before, after=after
            )
        except Exception as exc:  # noqa: BLE001
            invariant_violations.append(
                f"tick={record.tick}: validate_transition raised: {exc!r}"
            )
        else:
            for name, result in report.invariant_results.items():
                if result.passed is False:
                    invariant_violations.append(
                        f"tick={record.tick}: invariant {name} failed: "
                        f"{result.message}"
                    )

        per_tick_hashes.append(hash_state(after))

    final_state_hash = per_tick_hashes[-1] if per_tick_hashes else None

    replay_consistent = (
        restoration_ok
        and not invariant_violations
        and cap.exact_restore
        and cap.executable_deterministic_replay
    )

    return ReplayReport(
        checkpoint_hash=checkpoint_hash,
        actions_hash=actions_hash,
        final_state_hash=final_state_hash,
        per_tick_hashes=tuple(per_tick_hashes),
        replay_consistent=replay_consistent,
        invariant_violations=tuple(invariant_violations),
        restoration_ok=restoration_ok,
        restoration_message=restoration_message,
        cap_exact_restore=cap.exact_restore,
        cap_deterministic_replay=cap.executable_deterministic_replay,
        n_actions=len(actions),
    )


# ---------------------------------------------------------------------------
# branch
# ---------------------------------------------------------------------------


def branch(
    world: WorldProtocol,
    checkpoint: Checkpoint,
    alternatives: Sequence[Sequence[ExecutedAction]],
    *,
    parent_per_tick_hashes: Sequence[str] | None = None,
) -> list[BranchResult]:
    """Spawn one or more counterfactual branches from a checkpoint.

    Each alternative is an independent sequence of actions executed
    from the same fork checkpoint. Branches MUST NOT pollute the
    parent world's state; isolation is achieved by:

    1. Saving the parent world's current state via
       ``world.checkpoint()`` BEFORE any branch runs.
    2. Before each branch: ``world.restore(fork_checkpoint)``.
    3. After all branches: ``world.restore(parent_saved_checkpoint)``.

    If any branch raises an exception, the exception is captured in
    :attr:`BranchResult.error`, and the branch's ``final_state_hash``
    is set to ``None``. The next branch still runs (after restoring to
    the fork checkpoint).

    Args:
        world: The world to branch from. Must implement
            :class:`WorldProtocol`.
        checkpoint: The fork checkpoint. Each branch starts from this
            state.
        alternatives: Sequence of action sequences, one per branch.
            May be empty (returns an empty list).
        parent_per_tick_hashes: Optional sequence of state hashes from
            the parent run, one per tick. If supplied, each branch's
            :attr:`BranchResult.diverged_at_tick` is computed by
            comparing the branch's per-tick hashes to this sequence.
            If ``None`` (default), ``diverged_at_tick`` is always
            ``None``.

    Returns:
        List of :class:`BranchResult`, one per alternative. The list
        order matches ``alternatives``.

    Raises:
        ReplayError: If saving the parent checkpoint fails, or if the
            final parent restoration fails. Branch-level exceptions are
            captured in :attr:`BranchResult.error` and do NOT raise.
    """
    if not alternatives:
        return []

    # Step 1: save the parent world's current state.
    try:
        parent_saved = world.checkpoint()
    except Exception as exc:  # noqa: BLE001
        raise ReplayError(
            f"failed to save parent state before branching: {exc!r}"
        ) from exc

    results: list[BranchResult] = []

    try:
        for index, actions in enumerate(alternatives):
            branch_id = f"b{index}"
            actions_tuple = tuple(actions)

            # Step 2: restore to fork checkpoint before this branch.
            try:
                world.restore(checkpoint)
            except Exception as exc:  # noqa: BLE001
                # Cannot restore — record error and skip this branch.
                results.append(
                    BranchResult(
                        branch_id=branch_id,
                        fork_tick=checkpoint.tick,
                        actions=actions_tuple,
                        final_state_hash=None,
                        per_tick_hashes=(),
                        diverged_at_tick=None,
                        restoration_ok=False,
                        error=f"world.restore(fork) raised: {exc!r}",
                    )
                )
                continue

            # Step 3: run the branch's action sequence.
            per_tick_hashes: list[str] = []
            error: str | None = None
            try:
                for action in actions_tuple:
                    world.step(action)
                    per_tick_hashes.append(hash_state(world.observe()))
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
                # per_tick_hashes has whatever was collected before the
                # exception; final_state_hash is None on error.

            final_hash = per_tick_hashes[-1] if per_tick_hashes else None

            # Step 4: compute diverged_at_tick if parent hashes supplied.
            diverged_at_tick: int | None = None
            if parent_per_tick_hashes is not None and error is None:
                for i, branch_hash in enumerate(per_tick_hashes):
                    if i >= len(parent_per_tick_hashes):
                        # Branch ran longer than parent; treat the
                        # first extra tick as divergence.
                        diverged_at_tick = checkpoint.tick + 1 + i
                        break
                    if branch_hash != parent_per_tick_hashes[i]:
                        # +1 because parent_per_tick_hashes[i] is the
                        # state AFTER tick (checkpoint.tick + 1 + i).
                        diverged_at_tick = checkpoint.tick + 1 + i
                        break

            results.append(
                BranchResult(
                    branch_id=branch_id,
                    fork_tick=checkpoint.tick,
                    actions=actions_tuple,
                    final_state_hash=final_hash,
                    per_tick_hashes=tuple(per_tick_hashes),
                    diverged_at_tick=diverged_at_tick,
                    restoration_ok=True,  # branch ran without restore error
                    error=error,
                )
            )
    finally:
        # Step 5: restore the parent world's state, regardless of
        # whether any branch raised. If THIS restore fails, the parent
        # is polluted — that's a hard error.
        try:
            world.restore(parent_saved)
        except Exception as exc:  # noqa: BLE001
            raise ReplayError(
                f"failed to restore parent state after branching: {exc!r}"
            ) from exc

    return results
