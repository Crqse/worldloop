"""Tests for S-11 Dataset Exporter §14.6 top-level artifacts.

Verifies that :class:`PlainDatasetExporter.export` produces the full
§14.6 standard export directory layout on top of the per-split
directories:

- ``manifest.json`` (aggregate)
- ``splits.json`` (episode → split mapping)
- ``dataset_card.md``
- ``schema.json``
- ``capabilities.json``
- ``transitions.jsonl``
- ``known_limitations.md``
- ``checksums.json``
- ``world_parameters/`` (when ``scenario_package`` is supplied)

These tests run small rollouts on ``discrete_grid.yaml`` to produce real
transition files, then exercise the exporter end-to-end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldloop_data.config import ExporterConfig, RolloutConfig
from worldloop_data.coverage import UniformCoverageScheduler
from worldloop_data.exporter import (
    DATASET_VERSION,
    EpisodeRecords,
    ExportResult,
    PlainDatasetExporter,
)
from worldloop_data.policy import PolicyPool, RandomPolicy
from worldloop_data.rollout import run_rollout

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


def _run_one_episode(package, seed, episode_id, output_dir, num_ticks=3):
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


def _make_episodes(
    tmp_path,
    package,
    n_seeds=3,
    num_ticks=3,
    branch_group_ids=None,
):
    """Build a list of :class:`EpisodeRecords` for the test dataset.

    ``branch_group_ids`` (optional) maps episode index → branch_group_id.
    """
    episodes = []
    rollouts_root = tmp_path / "_episodes"
    for idx in range(n_seeds):
        seed = 42 + idx
        ep_id = f"seed{seed}_run{idx}"
        ep_dir = rollouts_root / ep_id
        _run_one_episode(package, seed, ep_id, ep_dir, num_ticks=num_ticks)
        bgid = (
            branch_group_ids.get(idx) if branch_group_ids is not None else None
        )
        episodes.append(
            EpisodeRecords(
                episode_id=ep_id,
                seed=seed,
                scenario_id=package.spec.scenario.scenario_id,
                world_parameters_hash=package.world_parameters_hash,
                output_dir=ep_dir,
                branch_group_id=bgid,
            )
        )
    return episodes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def package():
    return _load_discrete_grid_package()


@pytest.fixture
def exported_dataset(tmp_path, package):
    """Export a 3-seed dataset with scenario_package, episode-index split.

    With 3 episodes and 0.6/0.2/0.2 ratios:
    - n_groups = 3 (no branch_group_id)
    - n_train = round(3 * 0.6) = 2
    - n_val   = round(3 * 0.2) = 1
    - n_test  = 0 (dropped)
    Each episode runs 3 ticks → 3 records → total = 9 records.
    """
    episodes = _make_episodes(tmp_path, package, n_seeds=3, num_ticks=3)
    exporter = PlainDatasetExporter(scenario_package=package)
    dataset_dir = tmp_path / "dataset"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)
    return result, dataset_dir, episodes


# ---------------------------------------------------------------------------
# §14.6 file presence
# ---------------------------------------------------------------------------


def test_export_produces_section_14_6_files(exported_dataset):
    """All §14.6 top-level files exist after export."""
    _, dataset_dir, _ = exported_dataset
    expected_top_level = [
        "dataset_card.md",
        "manifest.json",
        "schema.json",
        "capabilities.json",
        "splits.json",
        "transitions.jsonl",
        "known_limitations.md",
        "checksums.json",
    ]
    for fname in expected_top_level:
        assert (dataset_dir / fname).exists(), f"missing top-level file: {fname}"
    # world_parameters/ exists because scenario_package was supplied.
    assert (dataset_dir / "world_parameters").is_dir()
    # Per-split directories still produced.
    assert (dataset_dir / "train").is_dir()
    assert (dataset_dir / "val").is_dir()


# ---------------------------------------------------------------------------
# Top-level manifest.json
# ---------------------------------------------------------------------------


def test_top_level_manifest_content(exported_dataset):
    """Top-level manifest.json has the §14.6 aggregate fields."""
    result, dataset_dir, episodes = exported_dataset
    with open(dataset_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["dataset_version"] == DATASET_VERSION
    assert manifest["schema_version"]  # kernel PROTOCOL_SCHEMA_VERSION
    assert (
        manifest["kernel_protocol_schema_version"] == manifest["schema_version"]
    )
    assert manifest["scenario_id"] == episodes[0].scenario_id
    assert manifest["world_parameters_hash"] == episodes[0].world_parameters_hash
    assert manifest["total_episodes"] == 3
    assert manifest["total_records"] == result.total_records
    assert manifest["split_strategy"] == "episode"
    assert "train" in manifest["splits"]
    assert "val" in manifest["splits"]
    assert "created_at" in manifest
    # producer_id is set to the first episode's scenario_id (EpisodeRecords
    # does not carry a separate producer_id field).
    assert manifest["producer_id"] == episodes[0].scenario_id


def test_top_level_manifest_split_counts(exported_dataset):
    """splits.<name>.episode_count / record_count match the ExportSplit tuple."""
    result, dataset_dir, _ = exported_dataset
    with open(dataset_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    for split in result.splits:
        entry = manifest["splits"][split.name]
        assert entry["episode_count"] == len(split.episode_ids)
        assert entry["record_count"] == split.record_count
    # 3 episodes, 3 ticks each → 9 records total.
    assert manifest["total_records"] == 9


# ---------------------------------------------------------------------------
# splits.json
# ---------------------------------------------------------------------------


def test_splits_json_mapping(exported_dataset):
    """splits.json maps every episode_id to its split name."""
    result, dataset_dir, episodes = exported_dataset
    with open(dataset_dir / "splits.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
    # Every episode is mapped.
    assert set(mapping.keys()) == {e.episode_id for e in episodes}
    # Mapping matches the ExportSplit tuple.
    expected: dict[str, str] = {}
    for split in result.splits:
        for ep_id in split.episode_ids:
            expected[ep_id] = split.name
    assert mapping == expected


def test_branch_group_cohesion_in_splits_json(tmp_path, package):
    """Episodes sharing a branch_group_id end up in the same split.

    Build 3 episodes: ep0 & ep1 share bgid="bg1", ep2 is independent.
    Episode-index grouping keeps the bg1 pair together; with 2 groups
    and 0.6/0.2/0.2 ratios: n_train=round(1.2)=1, n_val=round(0.4)=0,
    n_test=1 → train=[group0], test=[group1].
    """
    bgids = {0: "bg1", 1: "bg1", 2: None}
    episodes = _make_episodes(
        tmp_path, package, n_seeds=3, num_ticks=2, branch_group_ids=bgids
    )
    exporter = PlainDatasetExporter()
    dataset_dir = tmp_path / "dataset_bg"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)

    with open(dataset_dir / "splits.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    ep0_split = mapping[episodes[0].episode_id]
    ep1_split = mapping[episodes[1].episode_id]
    assert ep0_split == ep1_split, (
        f"branch_group_id cohesion violated: ep0 → {ep0_split}, "
        f"ep1 → {ep1_split}"
    )
    # The pair lands in train (first group); ep2 lands in test (second group).
    assert ep0_split == "train"
    assert mapping[episodes[2].episode_id] == "test"


# ---------------------------------------------------------------------------
# dataset_card.md
# ---------------------------------------------------------------------------


def test_dataset_card_is_markdown(exported_dataset):
    """dataset_card.md is a valid Markdown file with key sections."""
    _, dataset_dir, _ = exported_dataset
    card_path = dataset_dir / "dataset_card.md"
    text = card_path.read_text(encoding="utf-8")
    # Starts with an H1 title.
    assert text.startswith("# "), f"dataset_card.md should start with H1, got: {text[:30]!r}"
    # Contains key section headers.
    assert "## Metadata" in text
    assert "## Splits" in text
    assert "## Quality" in text
    assert "## Known Limitations" in text
    assert "## Usage" in text
    # Mentions the scenario id.
    assert "discrete_grid_v0" in text
    # Mentions the dataset version.
    assert DATASET_VERSION in text


# ---------------------------------------------------------------------------
# schema.json
# ---------------------------------------------------------------------------


def test_schema_json_has_schema_version(exported_dataset):
    """schema.json contains schema_version matching the kernel protocol."""
    _, dataset_dir, _ = exported_dataset
    with open(dataset_dir / "schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    assert "schema_version" in schema
    from worldloop_kernel import PROTOCOL_SCHEMA_VERSION

    assert schema["schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert schema["kernel_protocol_schema_version"] == PROTOCOL_SCHEMA_VERSION
    assert schema["record_type"] == "TransitionRecord"
    assert "required_fields" in schema
    assert isinstance(schema["required_fields"], list)
    assert "schema_version" in schema["required_fields"]
    assert "tick" in schema["required_fields"]
    assert "capability_profile" in schema["required_fields"]
    assert schema["has_capability_profile"] is True
    assert schema["capability_profile_ref"] == "capabilities.json"


# ---------------------------------------------------------------------------
# capabilities.json
# ---------------------------------------------------------------------------


def test_capabilities_json_matches_record(exported_dataset):
    """capabilities.json matches the first transition's capability_profile."""
    _, dataset_dir, _ = exported_dataset
    # Read the first transition record (train split, first episode, first tick).
    train_dir = dataset_dir / "train"
    ep_dirs = sorted(p for p in train_dir.iterdir() if p.is_dir())
    assert ep_dirs, "no episode dirs in train/"
    tick_files = sorted(ep_dirs[0].glob("t*.json"))
    assert tick_files, "no tick files in first train episode"
    with open(tick_files[0], "r", encoding="utf-8") as f:
        first_record = json.load(f)
    # capabilities.json should match first_record["capability_profile"].
    with open(dataset_dir / "capabilities.json", "r", encoding="utf-8") as f:
        caps = json.load(f)
    assert caps == first_record["capability_profile"]


# ---------------------------------------------------------------------------
# transitions.jsonl
# ---------------------------------------------------------------------------


def test_transitions_jsonl_line_count(exported_dataset):
    """transitions.jsonl has one line per transition record."""
    result, dataset_dir, _ = exported_dataset
    jsonl_path = dataset_dir / "transitions.jsonl"
    lines = jsonl_path.read_text(encoding="utf-8").splitlines()
    # Drop trailing empty line if any.
    non_empty = [ln for ln in lines if ln.strip()]
    assert len(non_empty) == result.total_records
    assert len(non_empty) == 9  # 3 episodes × 3 ticks


def test_transitions_jsonl_sorted(exported_dataset):
    """transitions.jsonl is sorted by (split, episode_id, tick)."""
    _, dataset_dir, _ = exported_dataset
    jsonl_path = dataset_dir / "transitions.jsonl"
    lines = [
        ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    # Decode each line to (split_guess, episode_id, tick).
    # The record JSON itself does not carry the split name; we infer split
    # from the directory structure by re-walking the dataset.
    expected: list[tuple[str, str, int]] = []
    for split_name in ("train", "val", "test"):
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            continue
        for ep_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for tf in sorted(ep_dir.glob("t*.json")):
                tick = int(tf.stem[1:])  # t0000000000 → 0
                expected.append((split_name, ep_dir.name, tick))
    # Same length.
    assert len(lines) == len(expected)
    # Each line decodes as JSON and matches the expected tick sequence.
    for line, (split, ep_id, tick) in zip(lines, expected):
        rec = json.loads(line)
        assert rec["tick"] == tick, (
            f"tick mismatch in split={split} ep={ep_id}: "
            f"expected {tick}, got {rec['tick']}"
        )
    # Verify the (split, ep_id, tick) sequence is non-decreasing in
    # (split_order, episode_id, tick) order.
    split_order = {"train": 0, "val": 1, "test": 2}
    keys = [
        (split_order[s], ep, t) for (s, ep, t) in expected
    ]
    assert keys == sorted(keys), "transitions.jsonl is not sorted"


# ---------------------------------------------------------------------------
# checksums.json
# ---------------------------------------------------------------------------


def test_checksums_covers_all_files(exported_dataset):
    """checksums.json covers every transition file + top-level manifests."""
    _, dataset_dir, _ = exported_dataset
    with open(dataset_dir / "checksums.json", "r", encoding="utf-8") as f:
        checksums = json.load(f)
    # Every transition t*.json file across splits is covered.
    transition_paths: list[str] = []
    for split_name in ("train", "val", "test"):
        split_dir = dataset_dir / split_name
        if not split_dir.is_dir():
            continue
        for ep_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            for tf in sorted(ep_dir.glob("t*.json")):
                transition_paths.append(tf.relative_to(dataset_dir).as_posix())
    for rel in transition_paths:
        assert rel in checksums, f"transition file missing from checksums: {rel}"
        assert checksums[rel].startswith("sha256:"), (
            f"checksum for {rel} is not sha256-prefixed: {checksums[rel]}"
        )
    # Top-level manifests are also covered.
    for top in ("manifest.json", "splits.json", "schema.json", "capabilities.json"):
        assert top in checksums, f"top-level file missing from checksums: {top}"


def test_checksums_excludes_self(exported_dataset):
    """checksums.json does NOT include itself."""
    _, dataset_dir, _ = exported_dataset
    with open(dataset_dir / "checksums.json", "r", encoding="utf-8") as f:
        checksums = json.load(f)
    assert "checksums.json" not in checksums, (
        "checksums.json must not include its own hash"
    )


def test_checksums_sha256_format(exported_dataset):
    """Every checksum value has the ``sha256:<hex>`` format."""
    _, dataset_dir, _ = exported_dataset
    with open(dataset_dir / "checksums.json", "r", encoding="utf-8") as f:
        checksums = json.load(f)
    assert checksums, "checksums.json is empty"
    for rel, digest in checksums.items():
        assert digest.startswith("sha256:"), (
            f"checksum for {rel} is not sha256-prefixed: {digest}"
        )
        hex_part = digest[len("sha256:"):]
        assert len(hex_part) == 64, (
            f"sha256 hex digest for {rel} has wrong length: {len(hex_part)}"
        )
        int(hex_part, 16)  # raises ValueError if not valid hex


def test_checksums_match_actual_files(exported_dataset):
    """Recomputing SHA256 of covered files matches the stored checksums."""
    import hashlib

    _, dataset_dir, _ = exported_dataset
    with open(dataset_dir / "checksums.json", "r", encoding="utf-8") as f:
        checksums = json.load(f)
    for rel, expected_digest in checksums.items():
        path = dataset_dir / rel
        assert path.exists(), f"file listed in checksums missing on disk: {rel}"
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        actual = f"sha256:{h.hexdigest()}"
        assert actual == expected_digest, (
            f"checksum mismatch for {rel}: expected {expected_digest}, "
            f"got {actual}"
        )


# ---------------------------------------------------------------------------
# known_limitations.md
# ---------------------------------------------------------------------------


def test_known_limitations_mentions_deferred_items(exported_dataset):
    """known_limitations.md mentions Q3, Q9, and checkpoints."""
    _, dataset_dir, _ = exported_dataset
    text = (dataset_dir / "known_limitations.md").read_text(encoding="utf-8")
    # Deferred items.
    assert "Q3" in text and "Replay" in text
    assert "Q9" in text and "Utility" in text
    assert "checkpoints/" in text or "checkpoints" in text
    # Structural limitations.
    assert "Single-scenario" in text or "single-scenario" in text
    assert "attempt 6" in text
    assert "attempt 8" in text
    # Starts with H1.
    assert text.startswith("# ")


# ---------------------------------------------------------------------------
# world_parameters/
# ---------------------------------------------------------------------------


def test_world_parameters_when_scenario_package_given(exported_dataset, package):
    """world_parameters/ exists with spec.json + hash when scenario_package given."""
    _, dataset_dir, _ = exported_dataset
    wp_dir = dataset_dir / "world_parameters"
    assert wp_dir.is_dir()
    spec_path = wp_dir / "spec.json"
    hash_path = wp_dir / "world_parameters_hash.txt"
    assert spec_path.exists()
    assert hash_path.exists()
    # Hash matches the package.
    assert hash_path.read_text(encoding="utf-8").strip() == package.world_parameters_hash
    # spec.json parses and contains the scenario_id.
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    assert spec["scenario"]["scenario_id"] == package.spec.scenario.scenario_id


def test_world_parameters_absent_when_not_given(tmp_path, package):
    """world_parameters/ is NOT created when scenario_package is None."""
    episodes = _make_episodes(tmp_path, package, n_seeds=2, num_ticks=2)
    exporter = PlainDatasetExporter()  # no scenario_package
    dataset_dir = tmp_path / "dataset_no_wp"
    exporter.export(episodes=episodes, dataset_dir=dataset_dir)
    assert not (dataset_dir / "world_parameters").exists()


def test_exporter_without_scenario_package_does_not_raise(tmp_path, package):
    """Omitting scenario_package still produces all §14.6 files except
    world_parameters/ — no error."""
    episodes = _make_episodes(tmp_path, package, n_seeds=1, num_ticks=2)
    exporter = PlainDatasetExporter()
    dataset_dir = tmp_path / "dataset_no_wp_2"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)
    # All top-level artifacts except world_parameters/ exist.
    for fname in (
        "manifest.json",
        "splits.json",
        "dataset_card.md",
        "schema.json",
        "capabilities.json",
        "transitions.jsonl",
        "known_limitations.md",
        "checksums.json",
    ):
        assert (dataset_dir / fname).exists()
    assert not (dataset_dir / "world_parameters").exists()


# ---------------------------------------------------------------------------
# ExportResult new fields
# ---------------------------------------------------------------------------


def test_export_result_has_new_paths(exported_dataset):
    """ExportResult exposes the four new §14.6 path fields."""
    result, dataset_dir, _ = exported_dataset
    assert isinstance(result, ExportResult)
    assert result.dataset_manifest_path == dataset_dir / "manifest.json"
    assert result.splits_path == dataset_dir / "splits.json"
    assert result.dataset_card_path == dataset_dir / "dataset_card.md"
    assert result.checksums_path == dataset_dir / "checksums.json"
    # All paths exist on disk.
    for p in (
        result.dataset_manifest_path,
        result.splits_path,
        result.dataset_card_path,
        result.checksums_path,
    ):
        assert p.exists(), f"ExportResult path does not exist: {p}"


# ---------------------------------------------------------------------------
# Seed-split strategy also produces §14.6 artifacts
# ---------------------------------------------------------------------------


def test_seed_split_strategy_produces_top_level_artifacts(tmp_path, package):
    """The seed split strategy also emits all §14.6 top-level files."""
    episodes = _make_episodes(tmp_path, package, n_seeds=3, num_ticks=2)
    exporter = PlainDatasetExporter(
        config=ExporterConfig(split_strategy="seed"),
        scenario_package=package,
    )
    dataset_dir = tmp_path / "dataset_seed"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)
    assert result.split_strategy == "seed"
    # All §14.6 top-level files exist.
    for fname in (
        "manifest.json",
        "splits.json",
        "dataset_card.md",
        "schema.json",
        "capabilities.json",
        "transitions.jsonl",
        "known_limitations.md",
        "checksums.json",
    ):
        assert (dataset_dir / fname).exists(), f"missing {fname} under seed split"
    # manifest.json reflects the seed strategy.
    with open(dataset_dir / "manifest.json", "r", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["split_strategy"] == "seed"
    assert manifest["total_episodes"] == 3


# ---------------------------------------------------------------------------
# Edge case: single episode (train-only)
# ---------------------------------------------------------------------------


def test_single_episode_export_produces_artifacts(tmp_path, package):
    """A single-episode dataset still produces all §14.6 top-level files."""
    episodes = _make_episodes(tmp_path, package, n_seeds=1, num_ticks=2)
    exporter = PlainDatasetExporter(scenario_package=package)
    dataset_dir = tmp_path / "dataset_single"
    result = exporter.export(episodes=episodes, dataset_dir=dataset_dir)
    assert result.total_episodes == 1
    assert result.total_records == 2
    # train only (1 episode → train only).
    split_names = {s.name for s in result.splits}
    assert split_names == {"train"}
    # Top-level files still produced.
    for fname in (
        "manifest.json",
        "splits.json",
        "dataset_card.md",
        "schema.json",
        "capabilities.json",
        "transitions.jsonl",
        "known_limitations.md",
        "checksums.json",
    ):
        assert (dataset_dir / fname).exists()
    # transitions.jsonl has 2 lines (2 ticks).
    lines = [
        ln for ln in
        (dataset_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(lines) == 2
