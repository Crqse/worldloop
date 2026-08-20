"""Tests for :attr:`ExporterConfig.seed_split_map` — single source of truth
for pre-registered splits (audit F-03 fix).

Background
----------
``runs/2026-07-27/m6_market_v1_5seed/`` shipped a contradictory pair:
``manifest.json`` declared 3/1/1 (train seeds 42-44 / val 45 / test 46)
while ``splits.json`` actually contained 4/0/1 (seeds 42-45 all in train,
seed 46 in test). Root cause: the data-production script set
``train_ratio=0.8 / val_ratio=0.1 / test_ratio=0.1`` on the exporter, so
5 seeds went through ``round(5×0.8)=4 / round(5×0.1)=0 / remainder=1``;
the script then hand-wrote a 3/1/1 description into the manifest without
updating the exporter's splits.json.

Fix
---
:attr:`ExporterConfig.seed_split_map` is now an explicit ``{seed: split}``
map. When provided, the exporter assigns episodes directly from the map,
bypassing ratio rounding. The data-production script derives its
``manifest.split_assignment`` and ``summary.gate_16_6`` counts from
``ExportResult.splits`` (the same tuple that produced ``splits.json``),
so all three artifacts can never disagree.

These tests assert:
  - 5-seed explicit map → 3 train / 1 val / 1 test, with the exact
    pre-registered seed→split assignment.
  - 10-seed explicit map → 8 train / 1 val / 1 test, with the exact
    pre-registered seed→split assignment.
  - The manifest split_assignment derived from ``result.splits`` matches
    ``splits.json`` exactly (set equality on train / val / test).
  - Backward compatibility: ``seed_split_map=None`` preserves the legacy
    ratio-based round() behavior.
  - Validation: an unknown split name in the map raises ``ValueError``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldloop_data.config import ExporterConfig
from worldloop_data.exporter import EpisodeRecords, PlainDatasetExporter
from worldloop_data.policy import PolicyPool, RandomPolicy
from worldloop_data.coverage import UniformCoverageScheduler
from worldloop_data.rollout import run_rollout
from worldloop_data.config import RolloutConfig

# Resolve the discrete_grid.yaml example from the sibling scenarios package.
_SCENARIOS_ROOT = Path(__file__).resolve().parents[2] / "worldloop-scenarios"
_DISCRETE_GRID_YAML = _SCENARIOS_ROOT / "examples" / "discrete_grid.yaml"


def _load_discrete_grid_package():
    """Compile discrete_grid.yaml into a ScenarioPackage."""
    if not _DISCRETE_GRID_YAML.exists():
        pytest.skip(
            f"discrete_grid.yaml not found at {_DISCRETE_GRID_YAML}; "
            "worldloop-scenarios must be installed alongside worldloop-data."
        )
    from worldloop_scenarios.compiler import compile_file

    return compile_file(_DISCRETE_GRID_YAML)


def _run_one_episode(package, seed, episode_id, output_dir, num_ticks=2):
    """Run a single rollout and return the episode output dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    world = package.world_factory(seed)
    pool = PolicyPool([RandomPolicy()])
    cov = UniformCoverageScheduler()
    run_rollout(
        world=world,
        seed=seed,
        episode_id=episode_id,
        output_dir=output_dir,
        policy_pool=pool,
        coverage=cov,
        config=RolloutConfig(num_ticks=num_ticks, record=True),
    )
    return output_dir


def _make_episodes(tmp_path, package, seeds, num_ticks=2):
    """Build a list of :class:`EpisodeRecords` for the given seed tuple.

    Episode IDs follow the ``seed{S}_run{I}`` convention used by the
    data-production scripts (rollout index N corresponds to seeds[N]).
    """
    episodes = []
    rollouts_root = tmp_path / "_episodes"
    for idx, seed in enumerate(seeds):
        ep_id = f"seed{seed}_run{idx}"
        ep_dir = rollouts_root / ep_id
        _run_one_episode(package, seed, ep_id, ep_dir, num_ticks=num_ticks)
        episodes.append(
            EpisodeRecords(
                episode_id=ep_id,
                seed=seed,
                scenario_id=package.spec.scenario.scenario_id,
                world_parameters_hash=package.world_parameters_hash,
                output_dir=ep_dir,
                branch_group_id=None,
            )
        )
    return episodes


def _build_manifest_split_assignment(result) -> dict[str, list[str]]:
    """Mirror the data-production script logic: derive split_assignment
    from ``ExportResult.splits``.

    This is the SAME construction the fixed script uses, so any drift
    between this and ``splits.json`` indicates a script bug.
    """
    split_assignment: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for split in result.splits:
        split_assignment[split.name] = list(split.episode_ids)
    return split_assignment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def package():
    return _load_discrete_grid_package()


# ---------------------------------------------------------------------------
# Pre-registered split layouts (audit F-03 fix)
# ---------------------------------------------------------------------------


def test_seed_split_map_5seed_yields_3_1_1(tmp_path, package):
    """5-seed explicit map → 3 train / 1 val / 1 test, exact seeds.

    Pre-registration: train=seeds 42,43,44 / val=seed 45 / test=seed 46.
    """
    seeds = (42, 43, 44, 45, 46)
    seed_split_map = {
        42: "train",
        43: "train",
        44: "train",
        45: "val",
        46: "test",
    }
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_5seed"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    split_counts = {s.name: len(s.episode_ids) for s in result.splits}
    assert split_counts.get("train", 0) == 3
    assert split_counts.get("val", 0) == 1
    assert split_counts.get("test", 0) == 1


def test_seed_split_map_5seed_specific_seeds(tmp_path, package):
    """5-seed explicit map assigns the exact pre-registered seeds."""
    seeds = (42, 43, 44, 45, 46)
    seed_split_map = {42: "train", 43: "train", 44: "train", 45: "val", 46: "test"}
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_5seed_seeds"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    by_split = {s.name: set(s.episode_ids) for s in result.splits}
    assert by_split.get("train", set()) == {"seed42_run0", "seed43_run1", "seed44_run2"}
    assert by_split.get("val", set()) == {"seed45_run3"}
    assert by_split.get("test", set()) == {"seed46_run4"}


def test_seed_split_map_10seed_yields_8_1_1(tmp_path, package):
    """10-seed explicit map → 8 train / 1 val / 1 test, exact seeds.

    Pre-registration: train=seeds 42..49 / val=seed 50 / test=seed 51.
    """
    seeds = tuple(range(42, 52))  # 42..51
    seed_split_map = {s: "train" for s in range(42, 50)}
    seed_split_map[50] = "val"
    seed_split_map[51] = "test"
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_10seed"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    split_counts = {s.name: len(s.episode_ids) for s in result.splits}
    assert split_counts.get("train", 0) == 8
    assert split_counts.get("val", 0) == 1
    assert split_counts.get("test", 0) == 1


def test_seed_split_map_10seed_specific_seeds(tmp_path, package):
    """10-seed explicit map assigns the exact pre-registered seeds."""
    seeds = tuple(range(42, 52))
    seed_split_map = {s: "train" for s in range(42, 50)}
    seed_split_map[50] = "val"
    seed_split_map[51] = "test"
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_10seed_seeds"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    by_split = {s.name: set(s.episode_ids) for s in result.splits}
    expected_train = {f"seed{s}_run{s - 42}" for s in range(42, 50)}
    assert by_split.get("train", set()) == expected_train
    assert by_split.get("val", set()) == {"seed50_run8"}
    assert by_split.get("test", set()) == {"seed51_run9"}


# ---------------------------------------------------------------------------
# manifest.split_assignment == splits.json consistency (audit F-03 core)
# ---------------------------------------------------------------------------


def test_5seed_manifest_split_assignment_matches_splits_json(tmp_path, package):
    """For the 5-seed scenario, the manifest split_assignment derived from
    ``result.splits`` MUST equal the ``splits.json`` mapping.

    Asserts (per audit F-03 acceptance criteria):
        set(manifest.train) == set(episode_id for ep, split in splits.json
                                   if split == "train")
        (val / test analogous)
    """
    seeds = (42, 43, 44, 45, 46)
    seed_split_map = {42: "train", 43: "train", 44: "train", 45: "val", 46: "test"}
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_5seed_consistency"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    # Read splits.json (the exporter's authoritative episode→split map).
    with open(dataset_dir / "splits.json", "r", encoding="utf-8") as f:
        splits_mapping = json.load(f)

    # Build the manifest split_assignment EXACTLY as the fixed script does.
    manifest_split_assignment = _build_manifest_split_assignment(result)

    # Per-split set equality.
    for split_name in ("train", "val", "test"):
        manifest_set = set(manifest_split_assignment.get(split_name, []))
        splits_set = {
            ep_id for ep_id, sp in splits_mapping.items() if sp == split_name
        }
        assert manifest_set == splits_set, (
            f"5-seed {split_name} mismatch: manifest={manifest_set}, "
            f"splits.json={splits_set}"
        )

    # Sanity: total counts.
    assert len(manifest_split_assignment["train"]) == 3
    assert len(manifest_split_assignment["val"]) == 1
    assert len(manifest_split_assignment["test"]) == 1


def test_10seed_manifest_split_assignment_matches_splits_json(tmp_path, package):
    """For the 10-seed scenario, manifest split_assignment == splits.json."""
    seeds = tuple(range(42, 52))
    seed_split_map = {s: "train" for s in range(42, 50)}
    seed_split_map[50] = "val"
    seed_split_map[51] = "test"
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_10seed_consistency"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    with open(dataset_dir / "splits.json", "r", encoding="utf-8") as f:
        splits_mapping = json.load(f)

    manifest_split_assignment = _build_manifest_split_assignment(result)

    for split_name in ("train", "val", "test"):
        manifest_set = set(manifest_split_assignment.get(split_name, []))
        splits_set = {
            ep_id for ep_id, sp in splits_mapping.items() if sp == split_name
        }
        assert manifest_set == splits_set, (
            f"10-seed {split_name} mismatch: manifest={manifest_set}, "
            f"splits.json={splits_set}"
        )

    assert len(manifest_split_assignment["train"]) == 8
    assert len(manifest_split_assignment["val"]) == 1
    assert len(manifest_split_assignment["test"]) == 1


def test_5seed_summary_counts_match_splits_json(tmp_path, package):
    """summary.gate_16_6.{train,val,test}_seeds MUST equal the count of
    episodes in each split in ``splits.json`` (single source of truth)."""
    seeds = (42, 43, 44, 45, 46)
    seed_split_map = {42: "train", 43: "train", 44: "train", 45: "val", 46: "test"}
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_5seed_summary"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    with open(dataset_dir / "splits.json", "r", encoding="utf-8") as f:
        splits_mapping = json.load(f)

    # Simulate the fixed script's summary.gate_16_6 derivation.
    split_assignment = _build_manifest_split_assignment(result)
    train_count = len(split_assignment["train"])
    val_count = len(split_assignment["val"])
    test_count = len(split_assignment["test"])

    # Assert summary counts equal splits.json per-split counts.
    assert train_count == sum(1 for sp in splits_mapping.values() if sp == "train")
    assert val_count == sum(1 for sp in splits_mapping.values() if sp == "val")
    assert test_count == sum(1 for sp in splits_mapping.values() if sp == "test")
    # 5-seed pre-registration: 3/1/1.
    assert (train_count, val_count, test_count) == (3, 1, 1)


# ---------------------------------------------------------------------------
# Backward compatibility & validation
# ---------------------------------------------------------------------------


def test_seed_split_map_none_preserves_ratio_behavior(tmp_path, package):
    """When seed_split_map is None, the legacy ratio-based round() behavior
    is preserved (backward compatibility).

    With 5 seeds and default ratios 0.6/0.2/0.2:
        n_train = round(5*0.6) = 3
        n_val   = round(5*0.2) = 1
        remainder = 1 → test
    → 3/1/1 (matches the legacy test_exporter.py expectations).
    """
    seeds = (42, 43, 44, 45, 46)
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(split_strategy="seed"),  # seed_split_map=None
    )
    dataset_dir = tmp_path / "dataset_legacy"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    split_counts = {s.name: len(s.episode_ids) for s in result.splits}
    # Legacy ratio behavior on 5 seeds with 0.6/0.2/0.2 → 3/1/1.
    assert split_counts.get("train", 0) == 3
    assert split_counts.get("val", 0) == 1
    assert split_counts.get("test", 0) == 1


def test_seed_split_map_rejects_unknown_split_name():
    """An unknown split name in seed_split_map raises ValueError at
    config construction time (fail-fast)."""
    with pytest.raises(ValueError, match="unknown split name"):
        ExporterConfig(
            split_strategy="seed",
            seed_split_map={42: "TRAIN"},  # wrong case / unknown
        )


def test_seed_split_map_overrides_ratio_no_rounding_drift(tmp_path, package):
    """Reproduce the F-03 failure mode (5 seeds with 0.8/0.1/0.1 ratios
    would yield 4/0/1 via round()) and verify seed_split_map fixes it.

    Without the map: round(5*0.8)=4 / round(5*0.1)=0 / remainder=1 → 4/0/1.
    With the map (3/1/1): the map wins; ratios are ignored.
    """
    seeds = (42, 43, 44, 45, 46)
    seed_split_map = {42: "train", 43: "train", 44: "train", 45: "val", 46: "test"}
    episodes = _make_episodes(tmp_path, package, seeds, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(
            split_strategy="seed",
            train_ratio=0.8,  # would yield 4 train on 5 seeds without map
            val_ratio=0.1,
            test_ratio=0.1,
            seed_split_map=seed_split_map,
        ),
    )
    dataset_dir = tmp_path / "dataset_override"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    split_counts = {s.name: len(s.episode_ids) for s in result.splits}
    # The map wins; no 4/0/1 drift.
    assert split_counts.get("train", 0) == 3
    assert split_counts.get("val", 0) == 1
    assert split_counts.get("test", 0) == 1
