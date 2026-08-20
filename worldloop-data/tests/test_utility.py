"""Q9 matched policy-outcome utility gate tests."""

from __future__ import annotations

from pathlib import Path

from worldloop_data.policy import AdversarialPolicy, ScriptedPolicy
from worldloop_data.utility import (
    UtilityEvaluationReport,
    evaluate_matched_policy_utility,
)
from worldloop_scenarios.compiler import compile_file


_SCENARIO = (
    Path(__file__).resolve().parents[2]
    / "worldloop-scenarios"
    / "examples"
    / "discrete_grid.yaml"
)


def test_matched_utility_uses_same_state_and_exogenous():
    report = evaluate_matched_policy_utility(
        scenario_package=compile_file(_SCENARIO),
        policies=[
            AdversarialPolicy(),
            ScriptedPolicy(preferred_action_type="forage"),
        ],
        seeds=(42,),
        horizon=3,
        baseline_policy_id="adversarial",
    )

    assert isinstance(report, UtilityEvaluationReport)
    assert report.valid_comparisons
    assert len(report.valid_comparisons) == 3
    assert all(c.matched for c in report.valid_comparisons)
    assert all(c.state_before_hash for c in report.valid_comparisons)
    assert all(c.exogenous_hash == "none" for c in report.valid_comparisons)
    assert report.best_delta == 4.0


def test_matched_utility_tie_does_not_claim_improvement():
    report = evaluate_matched_policy_utility(
        scenario_package=compile_file(_SCENARIO),
        policies=[
            ScriptedPolicy(),
            ScriptedPolicy(preferred_action_type="forage"),
        ],
        seeds=(42,),
        horizon=1,
        baseline_policy_id="scripted:first",
    )

    # Both policies choose the same first action, so a mere difference in
    # policy id cannot manufacture a utility improvement.
    assert report.best_delta == 0.0
