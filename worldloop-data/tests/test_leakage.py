"""Property tests for S-12 TrivialLeakageChecker.

Exercises the four leakage kinds (seed / scenario / world_param /
branch_group) and the report structure. Uses synthetic
:class:`ExportResult` fixtures — no real rollouts — because the
checker only inspects episode-level metadata.
"""

from __future__ import annotations

from pathlib import Path

from worldloop_data.config import LeakageConfig
from worldloop_data.exporter import EpisodeRecords, ExportResult, ExportSplit
from worldloop_data.leakage import LeakageReport, LeakageViolation, TrivialLeakageChecker

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_episode(episode_id, seed, scenario_id="scen_a", world_parameters_hash="hash_a",
                  branch_group_id=None, output_dir=None):
    return EpisodeRecords(
        episode_id=episode_id, seed=seed, scenario_id=scenario_id,
        world_parameters_hash=world_parameters_hash,
        output_dir=output_dir or Path(f"/tmp/{episode_id}"),
        branch_group_id=branch_group_id,
    )


def _make_split(name, episode_ids):
    return ExportSplit(
        name=name, episode_ids=tuple(episode_ids),
        record_count=len(episode_ids),
        output_dir=Path(f"/tmp/{name}"),
        manifest_path=Path(f"/tmp/{name}/manifest.json"),
    )


def _make_export_result(splits):
    return ExportResult(
        dataset_dir=Path("/tmp/ds"), splits=tuple(splits),
        total_records=sum(s.record_count for s in splits),
        total_episodes=sum(len(s.episode_ids) for s in splits),
        split_strategy="episode",
        dataset_manifest_path=Path("/tmp/ds/manifest.json"),
        splits_path=Path("/tmp/ds/splits.json"),
        dataset_card_path=Path("/tmp/ds/dataset_card.md"),
        checksums_path=Path("/tmp/ds/checksums.json"),
    )


def _check(export_result, episodes, config=None):
    checker = TrivialLeakageChecker(config=config) if config else TrivialLeakageChecker()
    return checker.check(export_result, episodes)


# ---------------------------------------------------------------------------
# Basic structure
# ---------------------------------------------------------------------------


def test_trivial_checker_returns_empty_report_when_no_episodes():
    export_result = _make_export_result((_make_split("train", ["ep_a"]),))
    report = _check(export_result, episodes=[])
    assert report.ok is True
    assert report.violations == ()
    assert dict(report.by_kind) == {}


def test_trivial_checker_returns_empty_report_when_single_split():
    ep1 = _make_episode("ep_a", seed=42)
    ep2 = _make_episode("ep_b", seed=42, scenario_id="scen_a")
    export_result = _make_export_result((_make_split("train", ["ep_a", "ep_b"]),))
    report = _check(export_result, episodes=[ep1, ep2])
    assert report.ok is True
    assert report.violations == ()
    assert dict(report.by_kind) == {}


def test_leakage_report_to_dict_serializes_correctly():
    v = LeakageViolation(kind="seed", key="42", splits=("train", "val"),
                         description="seed '42' appears in splits ('train', 'val')")
    report = LeakageReport(violations=(v,), by_kind={"seed": 1}, ok=False,
                          checked_kinds=("seed", "scenario", "branch_group"))
    d = report.to_dict()
    assert d["ok"] is False
    assert d["violation_count"] == 1
    assert d["checked_kinds"] == ["seed", "scenario", "branch_group"]
    assert d["by_kind"] == {"seed": 1}
    assert len(d["violations"]) == 1
    v0 = d["violations"][0]
    assert v0["kind"] == "seed" and v0["key"] == "42"
    assert v0["splits"] == ["train", "val"]
    assert "description" in v0 and v0["description"]


# ---------------------------------------------------------------------------
# Seed leakage
# ---------------------------------------------------------------------------


def test_seed_leakage_detected_when_same_seed_in_two_splits():
    ep1 = _make_episode("ep_a", seed=42)
    ep2 = _make_episode("ep_b", seed=42, scenario_id="scen_b")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    report = _check(export_result, episodes=[ep1, ep2])
    seed_vs = [v for v in report.violations if v.kind == "seed"]
    assert len(seed_vs) == 1, f"expected one seed violation, got {seed_vs}"
    assert report.ok is False
    assert dict(report.by_kind).get("seed") == 1


def test_seed_no_leakage_when_same_seed_in_same_split():
    ep1 = _make_episode("ep_a", seed=42)
    ep2 = _make_episode("ep_b", seed=42)
    export_result = _make_export_result((_make_split("train", ["ep_a", "ep_b"]),))
    report = _check(export_result, episodes=[ep1, ep2])
    assert report.ok is True
    assert report.violations == ()


def test_seed_check_disabled_when_config_check_seed_false():
    ep1 = _make_episode("ep_a", seed=42)
    ep2 = _make_episode("ep_b", seed=42, scenario_id="scen_b")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=True,
                        check_world_param=False, check_branch_group=False)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    assert report.ok is True, f"expected ok with seed check disabled, got {report.violations}"
    assert "seed" not in report.checked_kinds


def test_seed_leakage_violation_has_correct_key_and_splits():
    ep1 = _make_episode("ep_a", seed=42)
    ep2 = _make_episode("ep_b", seed=42, scenario_id="scen_b")
    export_result = _make_export_result(
        (_make_split("val", ["ep_b"]), _make_split("train", ["ep_a"])))
    cfg = LeakageConfig(check_seed=True, check_scenario=False,
                        check_world_param=False, check_branch_group=False)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    assert len(report.violations) == 1
    v = report.violations[0]
    assert v.kind == "seed"
    assert v.key == "42"
    assert v.splits == ("train", "val"), f"splits should be sorted tuple, got {v.splits}"


# ---------------------------------------------------------------------------
# Scenario leakage
# ---------------------------------------------------------------------------


def test_scenario_leakage_detected_when_same_scenario_in_two_splits():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_x")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=True,
                        check_world_param=False, check_branch_group=False)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    scen_vs = [v for v in report.violations if v.kind == "scenario"]
    assert len(scen_vs) == 1
    assert scen_vs[0].key == "scen_x"
    assert set(scen_vs[0].splits) == {"train", "val"}
    assert report.ok is False


def test_scenario_no_leakage_when_different_scenarios_in_different_splits():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=True,
                        check_world_param=False, check_branch_group=False)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    assert report.ok is True
    assert report.violations == ()


# ---------------------------------------------------------------------------
# World-param leakage
# ---------------------------------------------------------------------------


def test_world_param_leakage_detected_when_enabled():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", world_parameters_hash="hash_z")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y", world_parameters_hash="hash_z")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=False,
                        check_world_param=True, check_branch_group=False)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    wp_vs = [v for v in report.violations if v.kind == "world_param"]
    assert len(wp_vs) == 1
    assert wp_vs[0].key == "hash_z"
    assert set(wp_vs[0].splits) == {"train", "val"}
    assert report.ok is False
    assert "world_param" in report.checked_kinds


def test_world_param_no_leakage_when_disabled_by_default():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", world_parameters_hash="hash_z")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y", world_parameters_hash="hash_z")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    report = _check(export_result, episodes=[ep1, ep2])  # default: check_world_param=False
    wp_vs = [v for v in report.violations if v.kind == "world_param"]
    assert wp_vs == [], f"world_param should not be checked by default, got {wp_vs}"
    assert "world_param" not in report.checked_kinds


def test_world_param_no_leakage_when_different_hashes():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", world_parameters_hash="hash_1")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y", world_parameters_hash="hash_2")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=False,
                        check_world_param=True, check_branch_group=False)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    assert report.ok is True
    assert report.violations == ()


# ---------------------------------------------------------------------------
# Branch-group leakage
# ---------------------------------------------------------------------------


def test_branch_group_leakage_detected_when_same_branch_group_in_two_splits():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", branch_group_id="bg_1")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y", branch_group_id="bg_1")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=False,
                        check_world_param=False, check_branch_group=True)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    bg_vs = [v for v in report.violations if v.kind == "branch_group"]
    assert len(bg_vs) == 1
    assert bg_vs[0].key == "bg_1"
    assert set(bg_vs[0].splits) == {"train", "val"}
    assert report.ok is False


def test_branch_group_no_leakage_when_branch_group_id_is_none():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", branch_group_id=None)
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y", branch_group_id=None)
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=False,
                        check_world_param=False, check_branch_group=True)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    bg_vs = [v for v in report.violations if v.kind == "branch_group"]
    assert bg_vs == [], f"None branch_group_id should not be checked, got {bg_vs}"
    assert report.ok is True


def test_branch_group_no_leakage_when_same_branch_group_in_same_split():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", branch_group_id="bg_1")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_x", branch_group_id="bg_1")
    export_result = _make_export_result((_make_split("train", ["ep_a", "ep_b"]),))
    cfg = LeakageConfig(check_seed=False, check_scenario=False,
                        check_world_param=False, check_branch_group=True)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    bg_vs = [v for v in report.violations if v.kind == "branch_group"]
    assert bg_vs == [], f"same branch_group in same split is not leakage, got {bg_vs}"
    assert report.ok is True


def test_branch_group_check_disabled_when_config_check_branch_group_false():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", branch_group_id="bg_1")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y", branch_group_id="bg_1")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    cfg = LeakageConfig(check_seed=False, check_scenario=False,
                        check_world_param=False, check_branch_group=False)
    report = _check(export_result, episodes=[ep1, ep2], config=cfg)
    bg_vs = [v for v in report.violations if v.kind == "branch_group"]
    assert bg_vs == []
    assert "branch_group" not in report.checked_kinds
    assert report.ok is True


# ---------------------------------------------------------------------------
# Multi-kind / integration
# ---------------------------------------------------------------------------


def test_multiple_leakage_kinds_detected_simultaneously():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x", branch_group_id="bg_1")
    ep2 = _make_episode("ep_b", seed=42, scenario_id="scen_x", branch_group_id="bg_1")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    report = _check(export_result, episodes=[ep1, ep2])  # default config
    kinds = {v.kind for v in report.violations}
    assert kinds == {"seed", "scenario", "branch_group"}, (
        f"expected seed+scenario+branch_group leakage, got {kinds}"
    )
    assert report.ok is False
    assert dict(report.by_kind) == {"seed": 1, "scenario": 1, "branch_group": 1}


def test_checked_kinds_reflects_config():
    splits = (_make_split("train", ["ep_a"]),)
    export_result = _make_export_result(splits)
    cfg_default = LeakageConfig(check_seed=True, check_scenario=True,
                                check_world_param=False, check_branch_group=True)
    assert _check(export_result, episodes=[], config=cfg_default).checked_kinds == (
        "seed", "scenario", "branch_group")

    cfg_all = LeakageConfig(check_seed=True, check_scenario=True,
                           check_world_param=True, check_branch_group=True)
    assert _check(export_result, episodes=[], config=cfg_all).checked_kinds == (
        "seed", "scenario", "world_param", "branch_group")

    cfg_none = LeakageConfig(check_seed=False, check_scenario=False,
                            check_world_param=False, check_branch_group=False)
    assert _check(export_result, episodes=[], config=cfg_none).checked_kinds == ()


def test_ok_true_when_no_violations():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x")
    ep2 = _make_episode("ep_b", seed=43, scenario_id="scen_y")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    report = _check(export_result, episodes=[ep1, ep2])
    assert report.ok is True
    assert report.violations == ()


def test_ok_false_when_any_violation():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x")
    ep2 = _make_episode("ep_b", seed=42, scenario_id="scen_y")
    export_result = _make_export_result(
        (_make_split("train", ["ep_a"]), _make_split("val", ["ep_b"])))
    report = _check(export_result, episodes=[ep1, ep2])
    assert report.ok is False
    assert len(report.violations) >= 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_episodes_missing_from_meta_by_ep_are_skipped():
    ep1 = _make_episode("ep_a", seed=42, scenario_id="scen_x")
    # ep_missing is referenced by a split but absent from episodes list.
    export_result = _make_export_result((
        _make_split("train", ["ep_a", "ep_missing"]),
        _make_split("val", ["ep_missing"]),
    ))
    report = _check(export_result, episodes=[ep1])
    assert report.ok is True, f"missing episode should be skipped, got {report.violations}"
    assert report.violations == ()


def test_empty_splits_returns_ok():
    export_result = _make_export_result(splits=())
    report = _check(export_result, episodes=[])
    assert report.ok is True
    assert report.violations == ()
    assert dict(report.by_kind) == {}
