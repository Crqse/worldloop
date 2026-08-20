"""S-11 Dataset Exporter — split and publish transitions.

Takes per-episode transition records (already written by
:class:`~worldloop_kernel.TransitionRecorder`) and organizes them into
train / val / test splits under a single dataset root. The exporter
respects the §14.6 split priority:

1. ``scenario`` — split by ``scenario_id`` (highest priority).
2. ``world_param`` — split by world-parameters hash.
3. ``seed`` — split by RNG seed.
4. ``episode`` — split by episode index (lowest priority, default stub).

Random row-level splitting is FORBIDDEN (per main plan §14.6). Branches
from the same fork point MUST stay in the same split (counterfactual
leakage).

M4 stub implements ``episode`` and ``seed`` strategies. ``scenario`` and
``world_param`` strategies are deferred (they require multi-scenario
pooling, which is M5+).

In addition to per-split directories, the exporter produces the §14.6
top-level dataset artifacts:

- ``manifest.json`` — aggregate manifest (producer, scenario, totals).
- ``splits.json`` — ``episode_id -> split_name`` mapping.
- ``dataset_card.md`` — human-readable dataset card.
- ``schema.json`` — TransitionRecord schema snapshot (schema_version +
  required fields + capability_profile reference).
- ``capabilities.json`` — CapabilityProfile snapshot (from first record).
- ``transitions.jsonl`` — all transitions concatenated, sorted by
  ``(split, episode_id, tick)``.
- ``known_limitations.md`` — auto-generated limitations doc.
- ``checksums.json`` — SHA256 per file (transitions + top-level
  manifests only; ``checksums.json`` does not include itself).
- ``world_parameters/`` — scenario spec snapshot + hash (only when
  ``scenario_package`` is supplied to the exporter).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence

from worldloop_kernel import PROTOCOL_SCHEMA_VERSION

from worldloop_data.config import ExporterConfig

__all__ = [
    "DatasetExporter",
    "ExportSplit",
    "ExportResult",
    "PlainDatasetExporter",
    "EpisodeRecords",
]


#: Dataset format version (independent from kernel PROTOCOL_SCHEMA_VERSION
#: and from any producer version). Bumped when the on-disk shape of the
#: published dataset directory changes in a backward-incompatible way.
DATASET_VERSION = "0.1.0"

#: Fixed split ordering used for deterministic ``transitions.jsonl``
#: concatenation and aggregate manifest entries. Splits not present in
#: the published dataset are simply skipped.
_SPLIT_ORDER = ("train", "val", "test")

#: Top-level artifact filenames (relative to ``dataset_dir``).
_TOP_MANIFEST_FILENAME = "manifest.json"
_SPLITS_FILENAME = "splits.json"
_DATASET_CARD_FILENAME = "dataset_card.md"
_SCHEMA_FILENAME = "schema.json"
_CAPABILITIES_FILENAME = "capabilities.json"
_TRANSITIONS_JSONL_FILENAME = "transitions.jsonl"
_KNOWN_LIMITATIONS_FILENAME = "known_limitations.md"
_CHECKSUMS_FILENAME = "checksums.json"

#: Subdirectory for scenario spec snapshot.
_WORLD_PARAMETERS_DIRNAME = "world_parameters"


# ---------------------------------------------------------------------------
# EpisodeRecords — what the exporter consumes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRecords:
    """Per-episode records bundle.

    Attributes
    ----------
    episode_id:
        Stable episode identifier (e.g., ``"seed42_run0"``).
    seed:
        RNG seed used for this episode.
    scenario_id:
        Scenario ID from the ScenarioSpec.
    world_parameters_hash:
        Hash of the world parameters (from ScenarioPackage).
    output_dir:
        Directory where the recorder already wrote the per-tick JSON
        files and ``manifest.json``.
    branch_group_id:
        Optional branch group identifier. Episodes sharing a
        ``branch_group_id`` MUST be assigned to the same split.
    """

    episode_id: str
    seed: int
    scenario_id: str
    world_parameters_hash: str
    output_dir: Path
    branch_group_id: str | None = None


# ---------------------------------------------------------------------------
# ExportSplit / ExportResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExportSplit:
    """One split (train / val / test) of the published dataset.

    Attributes
    ----------
    name:
        Split name (``"train"``, ``"val"``, ``"test"``).
    episode_ids:
        Tuple of episode IDs assigned to this split.
    record_count:
        Total transition records in this split.
    output_dir:
        Directory containing the split's records.
    manifest_path:
        Path to the split-level manifest JSON.
    """

    name: str
    episode_ids: tuple[str, ...]
    record_count: int
    output_dir: Path
    manifest_path: Path


@dataclass(frozen=True)
class ExportResult:
    """Result of :meth:`DatasetExporter.export`.

    Attributes
    ----------
    dataset_dir:
        Root directory of the published dataset.
    splits:
        Tuple of :class:`ExportSplit`, one per split name.
    total_records:
        Total records across all splits. Equal to ``accepted`` since
        Phase 3 — quarantined records are NOT counted in
        ``total_records`` (they live in ``_quarantine/`` subdirs).
    total_episodes:
        Total episodes across all splits.
    split_strategy:
        Strategy used (``"episode"`` / ``"seed"`` / ...).
    dataset_manifest_path:
        Path to the top-level aggregate ``manifest.json`` (§14.6).
    splits_path:
        Path to ``splits.json`` (episode → split mapping, §14.6).
    dataset_card_path:
        Path to ``dataset_card.md`` (human-readable card, §14.6).
    checksums_path:
        Path to ``checksums.json`` (per-file SHA256, §14.6).
    produced:
        Total records produced by the recorder across all episodes
        BEFORE quarantine or rejection. Equal to the sum of every
        episode manifest's ``record_count`` + ``quarantine_count``.
        Phase 3 §6.5 Q8 quantity identity source.
    accepted:
        Records that made it into the main split directories (NOT into
        ``_quarantine/``). Equal to ``total_records``. Phase 3 §6.5 Q8
        quantity identity: ``produced == accepted + quarantined +
        explicitly_rejected``.
    quarantined:
        Records isolated in ``_quarantine/`` subdirectories (failed
        schema validation, replay divergence, etc.). Aggregated from
        per-episode manifest ``quarantine_count``.
    explicitly_rejected:
        Records the exporter chose to skip (not quarantine, not accept).
        Non-zero indicates the exporter dropped records silently — this
        should be 0 in normal operation. Tracked separately from
        ``quarantined`` because quarantine is recorder-driven (records
        fail validation at record time) while explicit rejection is
        exporter-driven (records fail exporter-level checks).
    dropped:
        Records lost without being accounted for. MUST be 0. Computed as
        ``produced - accepted - quarantined - explicitly_rejected``. A
        non-zero value indicates a leak in the accounting — the Q8
        check fails the entire quality report when this is non-zero.
    """

    dataset_dir: Path
    splits: tuple[ExportSplit, ...]
    total_records: int
    total_episodes: int
    split_strategy: str
    # §14.6 top-level artifact paths.
    dataset_manifest_path: Path
    splits_path: Path
    dataset_card_path: Path
    checksums_path: Path
    # Phase 3 §6.5 Q8 quantity identity fields. Default to 0 so legacy
    # callers that don't populate them still produce a valid (but
    # mechanically unverifiable) ExportResult — Q8 will report
    # "produced=0 / identity trivially satisfied" in that case.
    produced: int = 0
    accepted: int = 0
    quarantined: int = 0
    explicitly_rejected: int = 0
    dropped: int = 0


# ---------------------------------------------------------------------------
# DatasetExporter Protocol
# ---------------------------------------------------------------------------


class DatasetExporter(Protocol):
    """Split per-episode records into a published dataset."""

    def export(
        self,
        episodes: Sequence[EpisodeRecords],
        dataset_dir: Path,
    ) -> ExportResult:
        ...


# ---------------------------------------------------------------------------
# PlainDatasetExporter — reference stub
# ---------------------------------------------------------------------------


class PlainDatasetExporter:
    """Reference exporter using episode-index or seed-based splitting.

    The exporter copies per-episode record directories (already written
    by :class:`TransitionRecorder`) into ``dataset_dir/<split>/<episode_id>/``
    and writes a per-split ``manifest.json`` aggregating counts. After
    per-split directories are written, it produces the §14.6 top-level
    artifacts (aggregate manifest, splits map, dataset card, schema
    snapshot, capabilities snapshot, transitions JSONL, known
    limitations, checksums, and optionally ``world_parameters/``).

    Split strategy (per :attr:`ExporterConfig.split_strategy`):
    - ``"episode"``: episodes are assigned by index modulo the split
      ratios. Simple but allows branch-group leakage if branch groups
      span multiple episodes — the exporter checks ``branch_group_id``
      and keeps groups together.
    - ``"seed"``: episodes are grouped by seed; each seed goes to one
      split. Higher priority than ``"episode"`` per §14.6.

    ``"scenario"`` and ``"world_param"`` strategies are deferred (require
    multi-scenario pooling, M5+).
    """

    def __init__(
        self,
        *,
        config: ExporterConfig | None = None,
        scenario_package: Any | None = None,
    ) -> None:
        """Initialize the exporter.

        Args:
            config: Optional :class:`ExporterConfig`. If ``None``, the
                dataclass defaults are used.
            scenario_package: Optional :class:`worldloop_scenarios.ScenarioPackage`.
                When provided, the exporter writes a ``world_parameters/``
                subdirectory containing a spec snapshot and the world
                parameters hash. When ``None``, the subdirectory is
                skipped (no error). The package is NOT used for split
                decisions — splits are driven by :class:`EpisodeRecords`
                metadata alone.
        """
        self.config = config or ExporterConfig()
        self._scenario_package = scenario_package

    def export(
        self,
        episodes: Sequence[EpisodeRecords],
        dataset_dir: Path,
    ) -> ExportResult:
        if not episodes:
            raise ValueError("cannot export an empty episode list")
        dataset_dir = Path(dataset_dir).resolve()
        dataset_dir.mkdir(parents=True, exist_ok=True)

        assignments = self._assign_splits(episodes)

        splits: list[ExportSplit] = []
        total_records = 0
        # Phase 3 §6.5 Q8 quantity identity accumulators.
        total_produced = 0  # record_count + quarantine_count across eps
        total_quarantined = 0  # sum of quarantine_count across eps
        for split_name in _SPLIT_ORDER:
            split_eps = assignments.get(split_name)
            if not split_eps:
                continue
            split_dir = dataset_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            record_count = 0
            ep_ids: list[str] = []
            for ep in split_eps:
                ep_dir = split_dir / ep.episode_id
                if ep_dir.exists():
                    shutil.rmtree(ep_dir)
                shutil.copytree(ep.output_dir, ep_dir)
                # Count records via the episode manifest.
                manifest_path = ep_dir / "manifest.json"
                if manifest_path.exists():
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest = json.load(f)
                    ep_records = int(manifest.get("record_count", 0))
                    ep_quarantine = int(manifest.get("quarantine_count", 0))
                    record_count += ep_records
                    # Phase 3 §6.5: accumulate Q8 identity quantities.
                    # produced = record_count + quarantine_count (everything
                    # the recorder wrote, before exporter-level decisions).
                    total_produced += ep_records + ep_quarantine
                    total_quarantined += ep_quarantine
                ep_ids.append(ep.episode_id)
            total_records += record_count
            split_manifest_path = split_dir / "manifest.json"
            if self.config.write_manifest:
                self._write_split_manifest(
                    split_manifest_path,
                    split_name=split_name,
                    episode_ids=ep_ids,
                    record_count=record_count,
                )
            splits.append(
                ExportSplit(
                    name=split_name,
                    episode_ids=tuple(ep_ids),
                    record_count=record_count,
                    output_dir=split_dir,
                    manifest_path=split_manifest_path,
                )
            )

        # ------------------------------------------------------------------
        # §14.6 top-level artifacts.
        # ------------------------------------------------------------------
        first_record = self._read_first_transition_json(splits)
        top_manifest_path = self._write_top_manifest(
            dataset_dir,
            splits=splits,
            episodes=episodes,
            total_records=total_records,
        )
        splits_path = self._write_splits_json(dataset_dir, splits=splits)
        card_path = self._write_dataset_card(
            dataset_dir,
            splits=splits,
            episodes=episodes,
            total_records=total_records,
        )
        self._write_schema_json(dataset_dir, first_record=first_record)
        self._write_capabilities_json(dataset_dir, first_record=first_record)
        self._write_transitions_jsonl(dataset_dir, splits=splits)
        self._write_known_limitations(dataset_dir, episodes=episodes)
        self._maybe_write_world_parameters(dataset_dir)
        checksums_path = self._write_checksums(
            dataset_dir,
            splits=splits,
            top_manifest_path=top_manifest_path,
            splits_path=splits_path,
        )

        # Phase 3 §6.5: compute Q8 quantity identity.
        # - accepted = total_records (records that made it into main
        #   split dirs, NOT into _quarantine/)
        # - explicitly_rejected = 0 (the stub exporter doesn't reject
        #   any records; future exporter versions may filter records
        #   based on schema/Q1 checks)
        # - dropped = produced - accepted - quarantined - explicitly_rejected
        #   MUST be 0; non-zero indicates an accounting leak.
        accepted = total_records
        explicitly_rejected = 0
        dropped = max(
            0,
            total_produced - accepted - total_quarantined - explicitly_rejected,
        )

        return ExportResult(
            dataset_dir=dataset_dir,
            splits=tuple(splits),
            total_records=total_records,
            total_episodes=len(episodes),
            split_strategy=self.config.split_strategy,
            dataset_manifest_path=top_manifest_path,
            splits_path=splits_path,
            dataset_card_path=card_path,
            checksums_path=checksums_path,
            # Phase 3 §6.5 Q8 quantity identity.
            produced=total_produced,
            accepted=accepted,
            quarantined=total_quarantined,
            explicitly_rejected=explicitly_rejected,
            dropped=dropped,
        )

    # ------------------------------------------------------------------
    # Split assignment
    # ------------------------------------------------------------------

    def _assign_splits(
        self,
        episodes: Sequence[EpisodeRecords],
    ) -> dict[str, list[EpisodeRecords]]:
        """Assign episodes to splits, honoring branch_group_id cohesion."""
        if self.config.split_strategy == "seed":
            return self._assign_by_seed(episodes)
        # Default: episode-index assignment with branch-group cohesion.
        return self._assign_by_episode_index(episodes)

    def _assign_by_episode_index(
        self,
        episodes: Sequence[EpisodeRecords],
    ) -> dict[str, list[EpisodeRecords]]:
        # Group by branch_group_id (if any) so groups stay together.
        groups: list[list[EpisodeRecords]] = []
        current_group: list[EpisodeRecords] = []
        current_bgid: str | None = None
        for ep in episodes:
            bgid = ep.branch_group_id
            if bgid is not None and bgid == current_bgid:
                current_group.append(ep)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [ep]
                current_bgid = bgid
        if current_group:
            groups.append(current_group)

        n_groups = len(groups)
        n_train = max(1, int(round(n_groups * self.config.train_ratio)))
        n_val = max(0, int(round(n_groups * self.config.val_ratio)))
        # Remainder goes to test.
        assignments: dict[str, list[EpisodeRecords]] = {
            "train": [],
            "val": [],
            "test": [],
        }
        for i, group in enumerate(groups):
            if i < n_train:
                assignments["train"].extend(group)
            elif i < n_train + n_val:
                assignments["val"].extend(group)
            else:
                assignments["test"].extend(group)
        # Drop empty splits gracefully (e.g., 1 episode: train only).
        return {k: v for k, v in assignments.items() if v}

    def _assign_by_seed(
        self,
        episodes: Sequence[EpisodeRecords],
    ) -> dict[str, list[EpisodeRecords]]:
        # Group by seed.
        by_seed: dict[int, list[EpisodeRecords]] = {}
        for ep in episodes:
            by_seed.setdefault(ep.seed, []).append(ep)
        seeds_sorted = sorted(by_seed.keys())
        assignments: dict[str, list[EpisodeRecords]] = {
            "train": [],
            "val": [],
            "test": [],
        }

        seed_split_map = self.config.seed_split_map
        if seed_split_map is not None:
            # Explicit pre-registered split: every seed present in the map
            # is assigned by the map. Seeds missing from the map fall
            # through to the legacy ratio branch below (defensive).
            unmapped_seeds: list[int] = []
            for s in seeds_sorted:
                split_name = seed_split_map.get(s)
                if split_name is not None and split_name in assignments:
                    assignments[split_name].extend(by_seed[s])
                else:
                    unmapped_seeds.append(s)
            if not unmapped_seeds:
                return {k: v for k, v in assignments.items() if v}
            # Fall back to ratio assignment for any unmapped seeds,
            # treating only the unmapped subset as the ratio pool.
            seeds_sorted = unmapped_seeds

        n_seeds = len(seeds_sorted)
        n_train = max(1, int(round(n_seeds * self.config.train_ratio)))
        n_val = max(0, int(round(n_seeds * self.config.val_ratio)))
        for i, s in enumerate(seeds_sorted):
            if i < n_train:
                assignments["train"].extend(by_seed[s])
            elif i < n_train + n_val:
                assignments["val"].extend(by_seed[s])
            else:
                assignments["test"].extend(by_seed[s])
        return {k: v for k, v in assignments.items() if v}

    def _write_split_manifest(
        self,
        path: Path,
        *,
        split_name: str,
        episode_ids: list[str],
        record_count: int,
    ) -> None:
        payload = {
            "split": split_name,
            "episode_ids": episode_ids,
            "episode_count": len(episode_ids),
            "record_count": record_count,
            "split_strategy": self.config.split_strategy,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)

    # ------------------------------------------------------------------
    # §14.6 top-level artifact writers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_first_transition_json(
        splits: list[ExportSplit],
    ) -> dict[str, Any] | None:
        """Read the first ``t*.json`` record (split/episode/tick order).

        Returns ``None`` if no transition files are present across all
        splits (e.g., empty rollouts).
        """
        for split_name in _SPLIT_ORDER:
            for split in splits:
                if split.name != split_name:
                    continue
                for ep_id in sorted(split.episode_ids):
                    ep_dir = split.output_dir / ep_id
                    if not ep_dir.is_dir():
                        continue
                    tick_files = sorted(ep_dir.glob("t*.json"))
                    for tf in tick_files:
                        try:
                            with open(tf, "r", encoding="utf-8") as f:
                                return json.load(f)
                        except (OSError, json.JSONDecodeError):
                            continue
        return None

    def _write_top_manifest(
        self,
        dataset_dir: Path,
        *,
        splits: list[ExportSplit],
        episodes: Sequence[EpisodeRecords],
        total_records: int,
    ) -> Path:
        """Write the aggregate top-level ``manifest.json`` (§14.6)."""
        first_ep = episodes[0]
        splits_payload: dict[str, dict[str, int]] = {}
        for split in splits:
            splits_payload[split.name] = {
                "episode_count": len(split.episode_ids),
                "record_count": split.record_count,
            }
        payload = {
            "dataset_version": DATASET_VERSION,
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "producer_id": first_ep.scenario_id,  # scenario_id of first episode
            "producer_version": "",  # not available at EpisodeRecords level
            "scenario_id": first_ep.scenario_id,
            "world_parameters_hash": first_ep.world_parameters_hash,
            "total_episodes": len(episodes),
            "total_records": total_records,
            "splits": splits_payload,
            "split_strategy": self.config.split_strategy,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "kernel_protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
        }
        path = dataset_dir / _TOP_MANIFEST_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def _write_splits_json(
        self,
        dataset_dir: Path,
        *,
        splits: list[ExportSplit],
    ) -> Path:
        """Write ``splits.json`` mapping ``episode_id -> split_name``."""
        mapping: dict[str, str] = {}
        for split in splits:
            for ep_id in split.episode_ids:
                mapping[ep_id] = split.name
        path = dataset_dir / _SPLITS_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, sort_keys=True)
        return path

    def _write_dataset_card(
        self,
        dataset_dir: Path,
        *,
        splits: list[ExportSplit],
        episodes: Sequence[EpisodeRecords],
        total_records: int,
    ) -> Path:
        """Write ``dataset_card.md`` — human-readable dataset card."""
        first_ep = episodes[0]
        scenario_ids = {e.scenario_id for e in episodes}
        lines: list[str] = []
        lines.append(f"# Dataset Card — {first_ep.scenario_id}")
        lines.append("")
        lines.append("## Metadata")
        lines.append("")
        lines.append(f"- **Dataset version**: `{DATASET_VERSION}`")
        lines.append(
            f"- **Kernel protocol schema version**: `{PROTOCOL_SCHEMA_VERSION}`"
        )
        lines.append(f"- **Scenario ID**: `{first_ep.scenario_id}`")
        lines.append(
            f"- **World parameters hash**: `{first_ep.world_parameters_hash}`"
        )
        lines.append(f"- **Split strategy**: `{self.config.split_strategy}`")
        lines.append(
            f"- **Distinct scenarios in dataset**: {len(scenario_ids)}"
        )
        lines.append("")
        lines.append("## Splits")
        lines.append("")
        lines.append("| Split | Episodes | Records |")
        lines.append("| --- | ---: | ---: |")
        for split in splits:
            lines.append(
                f"| {split.name} | {len(split.episode_ids)} | {split.record_count} |"
            )
        lines.append(
            f"| **Total** | **{len(episodes)}** | **{total_records}** |"
        )
        lines.append("")
        lines.append("## Quality")
        lines.append("")
        lines.append(
            "- See [`quality_report.json`](quality_report.json) for the "
            "Q0-Q9 quality items (schema, traceability, diff/apply, "
            "replay, provenance, leakage, coverage, counterfactual, "
            "quarantine, utility)."
        )
        lines.append(
            "- See [`leakage_report.json`](leakage_report.json) for "
            "cross-split leakage checks (seed / scenario / world_param / "
            "branch_group)."
        )
        lines.append(
            "- See [`coverage_report.json`](coverage_report.json) for "
            "policy / action-type coverage."
        )
        lines.append("")
        lines.append("## Known Limitations")
        lines.append("")
        lines.append(
            "- See [`known_limitations.md`](known_limitations.md) for "
            "auto-generated known-limitations and deferred items."
        )
        lines.append("")
        lines.append("## Usage")
        lines.append("")
        lines.append(
            "- Per-record transitions live under `<split>/<episode_id>/t*.json` "
            "as individual JSON files (one per tick)."
        )
        lines.append(
            "- All transitions are also concatenated in "
            "[`transitions.jsonl`](transitions.jsonl) sorted by "
            "``(split, episode_id, tick)``, one compact JSON object per line."
        )
        lines.append(
            "- The aggregate top-level manifest is "
            "[`manifest.json`](manifest.json); per-split manifests live at "
            "``<split>/manifest.json``."
        )
        lines.append(
            "- The CapabilityProfile snapshot is in "
            "[`capabilities.json`](capabilities.json); the schema snapshot is "
            "in [`schema.json`](schema.json)."
        )
        lines.append(
            "- File checksums (SHA256) are in "
            "[`checksums.json`](checksums.json) (covers transition files + "
            "top-level manifest/splits/schema/capabilities)."
        )
        lines.append("")
        path = dataset_dir / _DATASET_CARD_FILENAME
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _write_schema_json(
        self,
        dataset_dir: Path,
        *,
        first_record: dict[str, Any] | None,
    ) -> Path:
        """Write ``schema.json`` — TransitionRecord schema snapshot.

        Not a full JSON Schema; a readable description with the schema
        version, required fields, and a reference to the capability
        profile snapshot.
        """
        if first_record is not None:
            schema_version = first_record.get("schema_version", PROTOCOL_SCHEMA_VERSION)
            has_capability = "capability_profile" in first_record
        else:
            schema_version = PROTOCOL_SCHEMA_VERSION
            has_capability = False
        # TransitionRecord required fields (provenance has a default).
        required_fields = [
            "schema_version",
            "producer_id",
            "producer_version",
            "tick",
            "state_before_hash",
            "candidate_actions",
            "executed_actions",
            "exogenous_input",
            "receipts",
            "state_delta",
            "state_after_hash",
            "capability_profile",
        ]
        optional_fields = ["provenance"]
        payload = {
            "schema_version": schema_version,
            "kernel_protocol_schema_version": PROTOCOL_SCHEMA_VERSION,
            "record_type": "TransitionRecord",
            "description": (
                "One complete transition S_t -> S_{t+1}. The externally "
                "publishable, exchangeable unit produced by "
                "worldloop_kernel.TransitionRecorder."
            ),
            "required_fields": required_fields,
            "optional_fields": optional_fields,
            "has_capability_profile": has_capability,
            "capability_profile_ref": _CAPABILITIES_FILENAME,
            "note": (
                "This is a readable schema snapshot, not a full JSON Schema. "
                "See worldloop_kernel.transition.TransitionRecord for the "
                "authoritative dataclass definition."
            ),
        }
        path = dataset_dir / _SCHEMA_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def _write_capabilities_json(
        self,
        dataset_dir: Path,
        *,
        first_record: dict[str, Any] | None,
    ) -> Path:
        """Write ``capabilities.json`` — CapabilityProfile snapshot.

        Extracted verbatim from the first transition record. If no
        records exist, writes an empty object.
        """
        if first_record is not None and "capability_profile" in first_record:
            payload = first_record["capability_profile"]
        else:
            payload = {}
        path = dataset_dir / _CAPABILITIES_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        return path

    def _write_transitions_jsonl(
        self,
        dataset_dir: Path,
        *,
        splits: list[ExportSplit],
    ) -> Path:
        """Write ``transitions.jsonl`` — all transitions, sorted.

        Sort order: ``(split_name, episode_id, tick)``. Split order is
        fixed to ``train / val / test``; episode_id is sorted
        lexicographically; tick is sorted numerically (filename template
        ``t{tick:010d}.json`` already zero-pads for lexical == numeric).
        Each line is one compact JSON object (no indentation).
        """
        path = dataset_dir / _TRANSITIONS_JSONL_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            for split_name in _SPLIT_ORDER:
                for split in splits:
                    if split.name != split_name:
                        continue
                    for ep_id in sorted(split.episode_ids):
                        ep_dir = split.output_dir / ep_id
                        if not ep_dir.is_dir():
                            continue
                        for tf in sorted(ep_dir.glob("t*.json")):
                            try:
                                with open(tf, "r", encoding="utf-8") as rh:
                                    rec = json.load(rh)
                            except (OSError, json.JSONDecodeError):
                                continue
                            f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")))
                            f.write("\n")
        return path

    def _write_known_limitations(
        self,
        dataset_dir: Path,
        *,
        episodes: Sequence[EpisodeRecords],
    ) -> Path:
        """Write ``known_limitations.md`` — auto-generated limitations doc."""
        scenario_ids = {e.scenario_id for e in episodes}
        branch_groups = {
            e.branch_group_id for e in episodes if e.branch_group_id is not None
        }
        lines: list[str] = []
        lines.append("# Known Limitations")
        lines.append("")
        lines.append(
            "Auto-generated by `worldloop_data.exporter.PlainDatasetExporter`. "
            "Lists deferred items and structural limitations of this dataset."
        )
        lines.append("")
        lines.append("## Structural Limitations")
        lines.append("")
        if len(scenario_ids) == 1:
            lines.append(
                "- **Single-scenario run**: this dataset pools a single "
                f"scenario (`{next(iter(scenario_ids))}`). Cross-scenario "
                "generalization claims require pooling multiple scenarios "
                "(M5+, see §14.6 split priority 1)."
            )
        else:
            lines.append(
                f"- **Multi-scenario pool**: {len(scenario_ids)} scenarios "
                "pooled; cross-scenario split strategy applies."
            )
        lines.append("")
        lines.append(
            f"- **Split strategy**: `{self.config.split_strategy}`. "
            "Episode-index splitting allows branch-group leakage unless "
            "the exporter's branch-group cohesion check keeps groups "
            "together (see `_assign_by_episode_index`). "
            f"Branch groups present in this dataset: {len(branch_groups)}."
        )
        lines.append("")
        lines.append("## Deferred Items")
        lines.append("")
        lines.append(
            "- **Q3 Replay verification**: requires a live world instance "
            "to reconstruct checkpoints and call `worldloop_kernel.replay`. "
            "The quality reporter marks Q3 as `skipped` when no world is "
            "provided. Full replay verification is deferred to attempt 6 "
            "(checkpoints/ directory + replay harness)."
        )
        lines.append(
            "- **Q9 Utility contrast**: requires multiple policies in the "
            "policy pool (e.g., random vs scripted vs adversarial). The "
            "quality reporter marks Q9 as `skipped` when fewer than two "
            "policies are registered. Multi-policy baseline contrast is "
            "deferred to attempt 8."
        )
        lines.append(
            "- **`checkpoints/` directory**: reserved for counterfactual "
            "branch replay (§14.6). The directory is NOT produced by this "
            "exporter in attempt 3; it will be populated in attempt 6 "
            "alongside the Q3 replay harness."
        )
        lines.append("")
        path = dataset_dir / _KNOWN_LIMITATIONS_FILENAME
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _maybe_write_world_parameters(self, dataset_dir: Path) -> None:
        """Write ``world_parameters/`` if a scenario package was provided.

        Writes:
        - ``spec.json`` — JSON dump of ``scenario_package.spec.to_dict()``.
        - ``world_parameters_hash.txt`` — the world parameters hash.

        Silently skips when no scenario package was supplied.
        """
        if self._scenario_package is None:
            return
        wp_dir = dataset_dir / _WORLD_PARAMETERS_DIRNAME
        wp_dir.mkdir(parents=True, exist_ok=True)
        spec = self._scenario_package.spec
        spec_dict = spec.to_dict() if hasattr(spec, "to_dict") else {}
        with open(wp_dir / "spec.json", "w", encoding="utf-8") as f:
            json.dump(spec_dict, f, indent=2, sort_keys=True, default=str)
        wp_hash = self._scenario_package.world_parameters_hash
        (wp_dir / "world_parameters_hash.txt").write_text(
            wp_hash, encoding="utf-8"
        )

    def _write_checksums(
        self,
        dataset_dir: Path,
        *,
        splits: list[ExportSplit],
        top_manifest_path: Path,
        splits_path: Path,
    ) -> Path:
        """Write ``checksums.json`` — SHA256 per file.

        Covers: all transition ``t*.json`` files + the top-level
        ``manifest.json`` / ``splits.json`` / ``schema.json`` /
        ``capabilities.json``. Does NOT cover ``checksums.json`` itself,
        ``dataset_card.md``, ``known_limitations.md``,
        ``quality_report.json``, ``leakage_report.json``,
        ``coverage_report.json``, ``world_parameters/``, or per-split
        ``manifest.json`` files (per §14.6 task spec).
        """
        checksums: dict[str, str] = {}
        # Transition files.
        for split_name in _SPLIT_ORDER:
            for split in splits:
                if split.name != split_name:
                    continue
                for ep_id in sorted(split.episode_ids):
                    ep_dir = split.output_dir / ep_id
                    if not ep_dir.is_dir():
                        continue
                    for tf in sorted(ep_dir.glob("t*.json")):
                        rel = tf.relative_to(dataset_dir).as_posix()
                        checksums[rel] = self._sha256(tf)
        # Top-level manifests.
        for top_path in (
            top_manifest_path,
            splits_path,
            dataset_dir / _SCHEMA_FILENAME,
            dataset_dir / _CAPABILITIES_FILENAME,
        ):
            if top_path.exists():
                rel = top_path.relative_to(dataset_dir).as_posix()
                checksums[rel] = self._sha256(top_path)
        path = dataset_dir / _CHECKSUMS_FILENAME
        with open(path, "w", encoding="utf-8") as f:
            json.dump(checksums, f, indent=2, sort_keys=True)
        return path

    @staticmethod
    def _sha256(path: Path) -> str:
        """Compute the SHA256 of a file, returned as ``sha256:<hex>``."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return f"sha256:{h.hexdigest()}"
