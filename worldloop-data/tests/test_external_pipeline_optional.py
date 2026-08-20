"""Optional external-environment pipeline test (B4).

Runs the minimal end-to-end pipeline against the PettingZoo Simple Spread
package from ``worldloop-adapters``. This is the OPTIONAL dependency path
(pyproject extra ``external``): when ``worldloop_adapters`` / ``pettingzoo``
/ ``mpe2`` are not installed, every test here is skipped — worldloop_data
core has no hard dependency on them (duck-typed scenario package protocol).
"""
from __future__ import annotations

import pytest

pytest.importorskip(
    "worldloop_adapters",
    reason="worldloop-adapters not installed (optional extra 'external')",
)
pytest.importorskip("pettingzoo", reason="pettingzoo not installed")
pytest.importorskip("mpe2", reason="mpe2 not installed")

from worldloop_adapters.scenario_package import (  # noqa: E402
    SimpleSpreadConfig,
    make_simple_spread_package,
)

from worldloop_data.config import ExporterConfig, LeakageConfig, PipelineConfig  # noqa: E402
from worldloop_data.exporter import PlainDatasetExporter  # noqa: E402
from worldloop_data.leakage import TrivialLeakageChecker  # noqa: E402
from worldloop_data.pipeline import run_pipeline  # noqa: E402
from worldloop_data.policy import RandomPolicy  # noqa: E402


@pytest.fixture(scope="module")
def pipeline_result(tmp_path_factory):
    """One minimal external pipeline run shared by every assertion."""
    out_dir = tmp_path_factory.mktemp("external_simple_spread")
    package = make_simple_spread_package(
        SimpleSpreadConfig(n_agents=2, max_cycles=10)
    )
    config = PipelineConfig(
        seeds=(42,),
        num_ticks=3,
        output_dir=str(out_dir),
        producer_id="worldloop-data-external-test",
    )
    return run_pipeline(
        scenario_package=package,
        policies=[RandomPolicy()],
        config=config,
        exporter=PlainDatasetExporter(
            config=ExporterConfig(split_strategy="seed"),
            scenario_package=package,
        ),
        leakage_checker=TrivialLeakageChecker(
            config=LeakageConfig(
                check_seed=True,
                check_scenario=False,
                check_world_param=False,
                check_branch_group=True,
            )
        ),
    )


class TestExternalPipelineMinimal:
    def test_rollout_produced_transitions(self, pipeline_result):
        assert len(pipeline_result.rollouts) == 1
        assert pipeline_result.rollouts[0].transition_count == 3

    def test_dataset_dir_published(self, pipeline_result):
        assert pipeline_result.dataset_dir.exists()
        assert (pipeline_result.dataset_dir / "quality_report.json").exists()
        assert (pipeline_result.dataset_dir / "leakage_report.json").exists()

    def test_world_parameters_snapshot_written(self, pipeline_result):
        wp = pipeline_result.dataset_dir / "world_parameters"
        assert (wp / "spec.json").exists()
        assert (wp / "world_parameters_hash.txt").exists()

    def test_leakage_clean(self, pipeline_result):
        assert pipeline_result.leakage_report.ok

    def test_episode_metadata_carries_external_scenario(self, pipeline_result):
        ep = pipeline_result.episodes[0]
        assert ep.scenario_id.startswith("external-pettingzoo-simple-spread")
        assert ep.world_parameters_hash.startswith("sha256:")

    def test_q3_replay_not_failed(self, pipeline_result):
        """exact_restore=True world: Q3 replay must not FAIL (pass expected)."""
        q3 = next(
            item
            for item in pipeline_result.quality_report.items
            if item.key == "Q3"
        )
        assert q3.status != "fail", f"Q3 replay failed: {q3.evidence}"
