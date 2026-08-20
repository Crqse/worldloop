"""M8 held-out candidate-action ranking task.

Task definition (frozen in the M8 preregistration):

    Input:  S_t (implicitly, via the fork point) + one candidate action.
    Output: predicted energy outcome of executing that candidate.
    A *ranking group* is one fork point — the factual transition plus
    its counterfactual branch siblings (same seed, same tick, same
    ``state_before_hash``). The model scores every candidate in the
    group; the PRIMARY metric is top-1 ranking accuracy over groups.

Outcome variable ("energy_outcome"): the focal agent's REALIZED energy
change extracted from ``state_delta.entity_changes`` (column ``energy``,
kind ``update``, after - before; 0.0 when the action does not touch
energy). This is state-dependent (energy floor at 0, preconditions gate
candidates), unlike ``receipt.energy_delta`` which is a constant
``-cost`` per action type in the ParameterizedWorld.

Metrics:
    - ranking_accuracy (PRIMARY): fraction of groups where the
      predicted-best candidate's TRUE outcome equals the group's best
      TRUE outcome (tie-tolerant: predicting any co-optimal candidate
      counts as a hit).
    - energy_mae (secondary): MAE of predicted vs true outcome over all
      group members.
    - top1_regret (secondary): mean over groups of
      (best true outcome - true outcome of the predicted-best candidate).
    - direction_consistency (secondary): over all ordered candidate
      pairs within groups with a true outcome difference > eps, the
      fraction where the predicted difference has the same sign.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from worldloop_data.evaluation.data_loader import TransitionSample

__all__ = [
    "RankingGroup",
    "RankingMetrics",
    "load_energy_outcomes",
    "load_multistep_energy_outcomes",
    "attach_energy_outcomes",
    "build_ranking_groups",
    "evaluate_ranking",
]

#: True-outcome differences at or below this epsilon are treated as ties.
_TIE_EPS = 1e-9


@dataclass(frozen=True)
class RankingGroup:
    """One fork-point ranking group.

    Attributes
    ----------
    key:
        ``(seed, tick, state_before_hash)`` fork-point key.
    indices:
        Row indices (into the sample list / X matrix) of the group's
        candidates: the factual transition + branch siblings.
    """

    key: tuple[str, int, str]
    indices: tuple[int, ...]


@dataclass(frozen=True)
class RankingMetrics:
    """Ranking evaluation result.

    ``float("nan")`` marks metrics that could not be computed (e.g.,
    no multi-candidate groups). NaN is preserved — never silently
    replaced (数字诚实).
    """

    ranking_accuracy: float
    energy_mae: float
    top1_regret: float
    direction_consistency: float
    n_groups: int
    n_candidates: int

    def to_dict(self) -> dict:
        return {
            "ranking_accuracy": self.ranking_accuracy,
            "energy_mae": self.energy_mae,
            "top1_regret": self.top1_regret,
            "direction_consistency": self.direction_consistency,
            "n_groups": self.n_groups,
            "n_candidates": self.n_candidates,
        }


def load_energy_outcomes(
    transitions_path: str | Path,
) -> dict[tuple[str, int], float]:
    """Scan ``transitions.jsonl`` and extract realized energy outcomes.

    Returns ``{(episode_id, tick): energy_outcome}`` where the outcome
    is the executed (focal) agent's ``energy`` column change
    (after - before) from ``state_delta.entity_changes``; 0.0 when the
    transition did not change the agent's energy.
    """
    outcomes: dict[tuple[str, int], float] = {}
    path = Path(transitions_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            executed = record.get("executed_actions", {}) or {}
            if not executed:
                continue
            agent_id = sorted(executed.keys())[0]
            episode_id = (record.get("provenance") or {}).get("episode_id", "")
            tick = int(record.get("tick", 0))
            delta = 0.0
            changes = (
                (record.get("state_delta") or {})
                .get("entity_changes", {})
                .get("changes", [])
                or []
            )
            for ch in changes:
                if (
                    ch.get("entity_id") == agent_id
                    and ch.get("column") == "energy"
                    and ch.get("kind") == "update"
                ):
                    try:
                        delta = float(ch.get("after", 0.0)) - float(
                            ch.get("before", 0.0)
                        )
                    except (TypeError, ValueError):
                        delta = 0.0
                    break
            outcomes[(episode_id, tick)] = delta
    return outcomes


def load_multistep_energy_outcomes(
    transitions_path: str | Path,
) -> dict[tuple[str, int], float]:
    """Scan ``transitions.jsonl`` and extract MULTI-STEP energy outcomes.

    For branch episodes (episode_id contains ``_cf_``), the outcome is
    the cumulative ``receipt.energy_delta`` over ALL ticks of the branch
    episode (the branch action + its k-step continuation). For
    conventional episodes, the outcome is the single-step realized
    energy change from ``state_delta.entity_changes`` (same as
    :func:`load_energy_outcomes`).

    Returns ``{(episode_id, first_tick): cumulative_energy_outcome}``.

    Rationale: single-step realized energy is often a deterministic
    function of action_type (ceiling effect). The multi-step cumulative
    outcome depends on the branch action's effect on subsequent state,
    making the ranking task non-trivial.
    """
    # Accumulate receipt.energy_delta per (episode_id, tick).
    receipt_deltas: dict[str, list[tuple[int, float]]] = {}
    # Also extract entity_changes energy for conventional episodes.
    entity_outcomes: dict[tuple[str, int], float] = {}

    path = Path(transitions_path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            executed = record.get("executed_actions", {}) or {}
            if not executed:
                continue
            agent_id = sorted(executed.keys())[0]
            episode_id = (record.get("provenance") or {}).get("episode_id", "")
            tick = int(record.get("tick", 0))

            # Receipt energy delta (cost).
            receipt = record.get("receipts", {}).get(agent_id, {})
            ed = float(receipt.get("energy_delta", 0.0))
            receipt_deltas.setdefault(episode_id, []).append((tick, ed))

            # Entity-changes energy (for conventional episodes).
            delta = 0.0
            changes = (
                (record.get("state_delta") or {})
                .get("entity_changes", {})
                .get("changes", [])
                or []
            )
            for ch in changes:
                if (
                    ch.get("entity_id") == agent_id
                    and ch.get("column") == "energy"
                    and ch.get("kind") == "update"
                ):
                    try:
                        delta = float(ch.get("after", 0.0)) - float(
                            ch.get("before", 0.0)
                        )
                    except (TypeError, ValueError):
                        delta = 0.0
                    break
            entity_outcomes[(episode_id, tick)] = delta

    # Build the combined outcome map.
    outcomes: dict[tuple[str, int], float] = {}
    for episode_id, tick_deltas in receipt_deltas.items():
        tick_deltas.sort()
        first_tick = tick_deltas[0][0]
        if "_cf_" in episode_id:
            # Branch episode: cumulative receipt.energy_delta.
            cumulative = sum(ed for _, ed in tick_deltas)
            outcomes[(episode_id, first_tick)] = cumulative
        else:
            # Conventional episode: per-tick entity_changes outcome.
            for tick, _ in tick_deltas:
                outcomes[(episode_id, tick)] = entity_outcomes.get(
                    (episode_id, tick), 0.0
                )
    return outcomes


def attach_energy_outcomes(
    samples: Sequence[TransitionSample],
    outcomes: Mapping[tuple[str, int], float],
) -> np.ndarray:
    """Return a (N,) array of energy outcomes aligned with ``samples``.

    Samples missing from ``outcomes`` get 0.0 (a transition with no
    energy change legitimately has no entity_changes entry).
    """
    return np.array(
        [outcomes.get((s.episode_id, s.tick), 0.0) for s in samples],
        dtype=np.float64,
    )


def build_ranking_groups(
    samples: Sequence[TransitionSample],
    *,
    min_group_size: int = 2,
) -> list[RankingGroup]:
    """Group samples into fork-point ranking groups.

    Group key: ``(seed, tick, state_before_hash)``. Branch records
    share the parent's ``state_before_hash`` at the fork tick (the
    kernel restores the world to the fork checkpoint before stepping),
    so the factual transition and its counterfactual siblings collapse
    into one group. Groups smaller than ``min_group_size`` are dropped
    (a single candidate cannot be ranked).

    Determinism: groups are ordered by key; indices within a group are
    ordered by episode_id (factual ``seedX_runY`` sorts before its
    ``..._cf_*`` siblings only lexicographically by chance — do not rely
    on position 0 being factual).
    """
    buckets: dict[tuple[str, int, str], list[int]] = {}
    for i, s in enumerate(samples):
        key = (str(s.seed), int(s.tick), s.state_before_hash)
        buckets.setdefault(key, []).append(i)
    groups: list[RankingGroup] = []
    for key in sorted(buckets.keys()):
        idxs = sorted(buckets[key], key=lambda i: samples[i].episode_id)
        if len(idxs) >= min_group_size:
            groups.append(RankingGroup(key=key, indices=tuple(idxs)))
    return groups


def evaluate_ranking(
    groups: Sequence[RankingGroup],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> RankingMetrics:
    """Compute ranking metrics over fork-point groups.

    Parameters
    ----------
    groups:
        Fork-point groups (indices into ``y_true`` / ``y_pred``).
    y_true:
        (N,) true energy outcomes.
    y_pred:
        (N,) predicted energy outcomes (higher = better).
    """
    if not groups:
        return RankingMetrics(
            ranking_accuracy=float("nan"),
            energy_mae=float("nan"),
            top1_regret=float("nan"),
            direction_consistency=float("nan"),
            n_groups=0,
            n_candidates=0,
        )

    hits = 0
    regrets: list[float] = []
    pair_total = 0
    pair_agree = 0
    member_indices: list[int] = []

    for g in groups:
        idxs = np.array(g.indices, dtype=int)
        member_indices.extend(g.indices)
        true_vals = y_true[idxs]
        pred_vals = y_pred[idxs]
        best_true = float(np.max(true_vals))
        pred_top = int(np.argmax(pred_vals))
        top_true = float(true_vals[pred_top])
        # Tie-tolerant hit: predicted-best candidate is co-optimal.
        if best_true - top_true <= _TIE_EPS:
            hits += 1
        regrets.append(best_true - top_true)
        # Pairwise direction consistency (unordered pairs, non-tied).
        n = len(idxs)
        for a in range(n):
            for b in range(a + 1, n):
                true_diff = true_vals[a] - true_vals[b]
                if abs(true_diff) <= _TIE_EPS:
                    continue
                pair_total += 1
                pred_diff = pred_vals[a] - pred_vals[b]
                if true_diff * pred_diff > 0:
                    pair_agree += 1

    member_idx_arr = np.array(member_indices, dtype=int)
    mae = float(np.mean(np.abs(y_true[member_idx_arr] - y_pred[member_idx_arr])))
    return RankingMetrics(
        ranking_accuracy=hits / len(groups),
        energy_mae=mae,
        top1_regret=float(np.mean(regrets)),
        direction_consistency=(
            pair_agree / pair_total if pair_total > 0 else float("nan")
        ),
        n_groups=len(groups),
        n_candidates=len(member_indices),
    )
