"""S-12 Leakage Checker — prevent cross-split contamination.

Verifies that no episode, seed, scenario, or branch group appears in
more than one split of the published dataset. Random row-level splits
are the textbook cause of leakage; this checker catches them.

M4 stub checks four leakage kinds (per main plan §14.7 Q5):
- ``seed``: same RNG seed in multiple splits.
- ``scenario``: same ``scenario_id`` in multiple splits.
- ``world_param``: same ``world_parameters_hash`` in multiple splits.
- ``branch_group``: branches from the same fork point in different splits.

The checker is read-only: it consumes the exporter's
:class:`~worldloop_data.exporter.ExportResult` and produces a
:class:`LeakageReport`. It does NOT modify the dataset.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Protocol, Sequence

from worldloop_data.config import LeakageConfig
from worldloop_data.exporter import EpisodeRecords, ExportResult

__all__ = [
    "LeakageChecker",
    "LeakageReport",
    "LeakageViolation",
    "TrivialLeakageChecker",
]


# ---------------------------------------------------------------------------
# LeakageViolation / LeakageReport
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakageViolation:
    """One leakage violation.

    Attributes
    ----------
    kind:
        ``"seed"`` / ``"scenario"`` / ``"world_param"`` / ``"branch_group"``.
    key:
        The leaked key (e.g., the seed value as a string).
    splits:
        Tuple of split names where the key appears.
    description:
        Human-readable description.
    """

    kind: str
    key: str
    splits: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class LeakageReport:
    """Result of :meth:`LeakageChecker.check`.

    Attributes
    ----------
    violations:
        Tuple of :class:`LeakageViolation`. Empty if no leakage.
    by_kind:
        Mapping ``kind -> count``.
    ok:
        ``True`` iff ``violations`` is empty.
    checked_kinds:
        Tuple of kinds actually checked (per :class:`LeakageConfig`).
    """

    violations: tuple[LeakageViolation, ...]
    by_kind: Mapping[str, int]
    ok: bool
    checked_kinds: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "checked_kinds": list(self.checked_kinds),
            "violation_count": len(self.violations),
            "by_kind": dict(self.by_kind),
            "violations": [
                {
                    "kind": v.kind,
                    "key": v.key,
                    "splits": list(v.splits),
                    "description": v.description,
                }
                for v in self.violations
            ],
        }


# ---------------------------------------------------------------------------
# LeakageChecker Protocol
# ---------------------------------------------------------------------------


class LeakageChecker(Protocol):
    """Check the published dataset for cross-split leakage."""

    def check(
        self,
        export_result: ExportResult,
        episodes: Sequence[EpisodeRecords],
    ) -> LeakageReport:
        ...


# ---------------------------------------------------------------------------
# TrivialLeakageChecker — reference stub
# ---------------------------------------------------------------------------


class TrivialLeakageChecker:
    """Reference checker that scans split assignments for overlap.

    The checker builds a ``key -> set(split_names)`` map for each
    configured kind and flags any key that appears in more than one
    split. It does NOT inspect record contents — it operates on the
    episode-level metadata, which is sufficient for the four kinds M4
    checks.
    """

    def __init__(self, *, config: LeakageConfig | None = None) -> None:
        self.config = config or LeakageConfig()

    def check(
        self,
        export_result: ExportResult,
        episodes: Sequence[EpisodeRecords],
    ) -> LeakageReport:
        # Build episode_id -> metadata lookup.
        meta_by_ep: dict[str, EpisodeRecords] = {e.episode_id: e for e in episodes}

        # Build kind -> key -> set(splits).
        key_to_splits: dict[str, dict[str, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for split in export_result.splits:
            for ep_id in split.episode_ids:
                ep = meta_by_ep.get(ep_id)
                if ep is None:
                    continue
                if self.config.check_seed:
                    key_to_splits["seed"][str(ep.seed)].add(split.name)
                if self.config.check_scenario:
                    key_to_splits["scenario"][ep.scenario_id].add(split.name)
                if self.config.check_world_param:
                    key_to_splits["world_param"][ep.world_parameters_hash].add(
                        split.name
                    )
                if (
                    self.config.check_branch_group
                    and ep.branch_group_id is not None
                ):
                    key_to_splits["branch_group"][ep.branch_group_id].add(
                        split.name
                    )

        violations: list[LeakageViolation] = []
        for kind, key_map in key_to_splits.items():
            for key, splits in key_map.items():
                if len(splits) > 1:
                    splits_tuple = tuple(sorted(splits))
                    violations.append(
                        LeakageViolation(
                            kind=kind,
                            key=key,
                            splits=splits_tuple,
                            description=(
                                f"{kind} {key!r} appears in splits "
                                f"{splits_tuple}"
                            ),
                        )
                    )

        by_kind: dict[str, int] = defaultdict(int)
        for v in violations:
            by_kind[v.kind] += 1

        checked = tuple(
            kind
            for kind, enabled in [
                ("seed", self.config.check_seed),
                ("scenario", self.config.check_scenario),
                ("world_param", self.config.check_world_param),
                ("branch_group", self.config.check_branch_group),
            ]
            if enabled
        )

        return LeakageReport(
            violations=tuple(violations),
            by_kind=dict(by_kind),
            ok=(len(violations) == 0),
            checked_kinds=checked,
        )
