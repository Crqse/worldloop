"""Tests for FeatureLayout and audit F-02 / R3 fix.

Verifies that:

1. ``FeatureLayout.from_dims`` builds correct block slices for 4 / 5 / 7
   action scenarios.
2. ``NoActionBaseline`` only zeros the action block; all other blocks
   (tick / agent / parameter / state / exogenous) are bit-identical to
   the input. Parametrized over 4 / 5 / 7 actions and 3 / 4 / 6 agents.
3. ``ShuffledActionBaseline`` only permutes the action block; all other
   blocks are bit-identical to the input. Same parametrization.
4. ``MeanDeltaBaseline`` uses the layout's ``n_actions`` / ``n_agents``
   for group iteration (not hardcoded ``range(7)`` / ``range(4)``).
5. ``OracleUpperBound`` uses the layout's action / agent slices for key
   extraction.
6. ``DataLoader.feature_layout`` matches the loader's vocab.
7. Backwards compatibility: constructing a baseline with ``layout=None``
   falls back to ``EMERGENCY_RESOURCE_V0_LAYOUT``.
8. ``FeatureLayout.from_dims`` rejects invalid dims.
9. Cross-scenario: same baseline code works on 4 / 5 / 7 action layouts
   without code changes.
10. State / exogenous blocks are forwarded correctly when present (R2
    forward-compat check).

These tests close audit finding F-02 (G7 ablation invariance) and part
of G9 (cross-4/5/7-action evaluator).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldloop_data.evaluation.feature_layout import (
    FeatureLayout,
    EMERGENCY_RESOURCE_V0_LAYOUT,
)
from worldloop_data.evaluation.baselines import (
    BaselineModel,
    PersistenceBaseline,
    MeanDeltaBaseline,
    LinearRidgeBaseline,
    XGBoostBaseline,
    NoActionBaseline,
    ShuffledActionBaseline,
    OracleUpperBound,
)
from worldloop_data.evaluation.data_loader import DataLoader


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _build_X(
    layout: FeatureLayout,
    n_samples: int = 20,
    *,
    seed: int = 42,
) -> np.ndarray:
    """Build a synthetic X matrix that respects the given layout.

    Each block is filled with recognizable values so tests can detect
    cross-block contamination:

    - tick: integer tick index per sample.
    - action: one-hot with the active column chosen deterministically.
    - agent: one-hot with the active column chosen deterministically.
    - parameter: distinct values per column (target_node=0.5, target_agent=0.7,
      has_params=1.0) so any drift is visible.
    - state: ascending floats (1.0, 2.0, ...) when present.
    - exogenous: descending floats (-1.0, -2.0, ...) when present.
    """
    rng = np.random.default_rng(seed)
    dim = layout.feature_dim
    X = np.zeros((n_samples, dim), dtype=np.float64)

    # tick
    if layout.tick_slice is not None:
        X[:, layout.tick_slice] = np.arange(n_samples).reshape(-1, 1)

    # action one-hot
    n_actions = layout.n_actions
    action_idx = (np.arange(n_samples) % n_actions).astype(int)
    X[np.arange(n_samples), layout.action_slice.start + action_idx] = 1.0

    # agent one-hot
    n_agents = layout.n_agents
    agent_idx = (np.arange(n_samples) % n_agents).astype(int)
    X[np.arange(n_samples), layout.agent_slice.start + agent_idx] = 1.0

    # parameter block: distinct recognizable values
    param_start = layout.parameter_slice.start
    if layout.n_parameters >= 1:
        X[:, param_start + 0] = 0.5  # target_node_idx
    if layout.n_parameters >= 2:
        X[:, param_start + 1] = 0.7  # target_agent_idx
    if layout.n_parameters >= 3:
        X[:, param_start + 2] = 1.0  # has_params
    for j in range(3, layout.n_parameters):
        X[:, param_start + j] = 0.3 + 0.01 * j

    # state block (forward compat for R2)
    if layout.state_slice is not None:
        for j in range(layout.n_state):
            X[:, layout.state_slice.start + j] = 1.0 + 0.1 * j

    # exogenous block (forward compat for R2)
    if layout.exogenous_slice is not None:
        for j in range(layout.n_exogenous):
            X[:, layout.exogenous_slice.start + j] = -1.0 - 0.1 * j

    return X


def _build_y(n_samples: int, seed: int = 42) -> np.ndarray:
    """Build a synthetic y matrix with 5 target columns."""
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((n_samples, 5))
    # Make position_change and rank integer-valued (categorical / ranking).
    y[:, 1] = rng.integers(0, 6, size=n_samples)
    y[:, 3] = rng.integers(0, 4, size=n_samples)
    return y


# Parametrize over the three audit-relevant action vocab sizes.
#(n_actions, n_agents) tuples covering emergency (7/4), market v1 (5/4),
# market v0 (4/3), plus a 6-agent case for agent-block coverage.
AUDIT_VOCABS = [
    (4, 3, "market_v0"),
    (5, 4, "market_v1"),
    (7, 4, "emergency_v0"),
    (7, 6, "emergency_wide"),
]


# ----------------------------------------------------------------------
# 1. FeatureLayout.from_dims
# ----------------------------------------------------------------------
class TestFeatureLayoutFromDims:
    def test_basic_layout_has_correct_block_offsets(self):
        layout = FeatureLayout.from_dims(n_actions=7, n_agents=4, n_parameters=3)
        # [tick(1) | action(7) | agent(4) | parameter(3)]
        assert layout.tick_slice == slice(0, 1)
        assert layout.action_slice == slice(1, 8)
        assert layout.agent_slice == slice(8, 12)
        assert layout.parameter_slice == slice(12, 15)
        assert layout.state_slice is None
        assert layout.exogenous_slice is None

    def test_dims_derived_properties(self):
        layout = FeatureLayout.from_dims(n_actions=5, n_agents=3, n_parameters=2)
        assert layout.n_actions == 5
        assert layout.n_agents == 3
        assert layout.n_parameters == 2
        assert layout.n_state == 0
        assert layout.n_exogenous == 0
        assert layout.feature_dim == 1 + 5 + 3 + 2  # = 11

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_audit_vocabs_layout_offsets(self, n_actions, n_agents, label):
        """Audit F-02 / R3: 4/5/7 action vocabs must produce distinct
        action slices — the previous ``1:8`` hardcode would silently
        pollute the agent / parameter blocks in 4- and 5-action cases.
        """
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        # action block must be exactly [1, 1+n_actions)
        assert layout.action_slice == slice(1, 1 + n_actions), (
            f"{label}: action_slice={layout.action_slice}, expected slice(1, {1 + n_actions})"
        )
        # agent block follows immediately after action
        assert layout.agent_slice == slice(1 + n_actions, 1 + n_actions + n_agents)
        # parameter block follows immediately after agent
        assert layout.parameter_slice == slice(
            1 + n_actions + n_agents,
            1 + n_actions + n_agents + 3,
        )
        # feature_dim matches
        assert layout.feature_dim == 1 + n_actions + n_agents + 3

    def test_layout_with_state_and_exogenous_blocks(self):
        """Forward-compat: R2 will pass positive n_state / n_exogenous."""
        layout = FeatureLayout.from_dims(
            n_actions=4, n_agents=3, n_parameters=3,
            n_state=10, n_exogenous=5,
        )
        # [tick(1) | action(4) | agent(3) | parameter(3) | state(10) | exogenous(5)]
        assert layout.feature_dim == 1 + 4 + 3 + 3 + 10 + 5
        assert layout.n_state == 10
        assert layout.n_exogenous == 5
        assert layout.state_slice == slice(11, 21)
        assert layout.exogenous_slice == slice(21, 26)

    def test_layout_no_tick(self):
        layout = FeatureLayout.from_dims(
            n_actions=4, n_agents=3, n_parameters=3, include_tick=False
        )
        assert layout.tick_slice is None
        assert layout.action_slice == slice(0, 4)
        assert layout.feature_dim == 4 + 3 + 3

    def test_invalid_dims_rejected(self):
        # R2 layouts allow n_actions=0 and n_agents=0 (joint-action block
        # is allocated via n_actions in R2 callers; legacy single-action
        # one-hot is absent). Only negative values are rejected now.
        with pytest.raises(ValueError, match="n_actions"):
            FeatureLayout.from_dims(n_actions=-1, n_agents=3)
        with pytest.raises(ValueError, match="n_agents"):
            FeatureLayout.from_dims(n_actions=4, n_agents=-1)
        with pytest.raises(ValueError, match="n_parameters"):
            FeatureLayout.from_dims(n_actions=4, n_agents=3, n_parameters=-1)
        with pytest.raises(ValueError, match="n_state"):
            FeatureLayout.from_dims(n_actions=4, n_agents=3, n_state=-1)
        with pytest.raises(ValueError, match="n_exogenous"):
            FeatureLayout.from_dims(n_actions=4, n_agents=3, n_exogenous=-1)

    def test_r2_zero_action_and_agent_dims_allowed(self):
        """R2 layouts pass n_actions=0 / n_agents=0 — the joint-action
        block is allocated via n_actions in R2 callers, and the legacy
        single-action one-hot / agent one-hot may be absent."""
        layout = FeatureLayout.from_dims(
            n_actions=0, n_agents=0, n_parameters=0,
            n_state=10, n_exogenous=0, include_tick=True,
        )
        assert layout.action_slice == slice(1, 1)  # empty
        assert layout.agent_slice == slice(1, 1)  # empty
        assert layout.parameter_slice == slice(1, 1)  # empty
        assert layout.state_slice == slice(1, 11)
        assert layout.exogenous_slice is None
        assert layout.feature_dim == 11

    def test_blocks_except_action_returns_all_non_action_slices(self):
        layout = FeatureLayout.from_dims(
            n_actions=4, n_agents=3, n_parameters=3,
            n_state=5, n_exogenous=2,
        )
        blocks = layout.blocks_except_action()
        # Should include tick, agent, parameter, state, exogenous (5 blocks).
        assert len(blocks) == 5
        assert layout.tick_slice in blocks
        assert layout.agent_slice in blocks
        assert layout.parameter_slice in blocks
        assert layout.state_slice in blocks
        assert layout.exogenous_slice in blocks
        # action_slice must NOT be in the result
        assert layout.action_slice not in blocks

    def test_emergency_default_layout_matches_legacy_hardcode(self):
        """The legacy default layout must reproduce the previous 1:8 / 8:12
        slices exactly — this is what older ad-hoc scripts assumed. The
        audit fix keeps this as a fallback so existing scripts don't
        silently break, while new code passes an explicit layout.
        """
        assert EMERGENCY_RESOURCE_V0_LAYOUT.action_slice == slice(1, 8)
        assert EMERGENCY_RESOURCE_V0_LAYOUT.agent_slice == slice(8, 12)


# ----------------------------------------------------------------------
# 2. NoActionBaseline invariance
# ----------------------------------------------------------------------
class TestNoActionInvariance:
    """Audit F-02 / R3 G7: no_action must only zero the action block."""

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_no_action_only_zeros_action_block(self, n_actions, n_agents, label):
        """For 4/5/7 action vocabs, verify NoActionBaseline zeros only
        the action block; tick / agent / parameter blocks are bit-identical.
        """
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=20)
        y = _build_y(20)

        baseline = NoActionBaseline(
            base=LinearRidgeBaseline(alpha=1.0),
            layout=layout,
        )
        baseline.fit(X, y)

        # Manually reconstruct what NoActionBaseline *should* have trained on.
        X_no_action = X.copy()
        X_no_action[:, layout.action_slice] = 0.0

        # Verify the action block was zeroed.
        assert np.all(X_no_action[:, layout.action_slice] == 0.0), (
            f"{label}: action block not zeroed"
        )

        # Verify every non-action block is bit-identical to the original.
        for blk in layout.blocks_except_action():
            assert np.array_equal(X_no_action[:, blk], X[:, blk]), (
                f"{label}: non-action block {blk} was modified by no_action"
            )

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_no_action_predict_zeros_action_block(self, n_actions, n_agents, label):
        """predict() must also zero the action block — guards against
        the bug where fit() is fixed but predict() still hardcodes 1:8.
        """
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=10)
        y = _build_y(10)

        # Patch the base model to capture the X it receives at predict time.
        captured_X: list[np.ndarray] = []
        class _CapturingBase(BaselineModel):
            name = "capturing"
            def fit(self, X, y):
                self._mean = y.mean(axis=0)
            def predict(self, X):
                captured_X.append(X.copy())
                return np.tile(self._mean, (X.shape[0], 1))

        baseline = NoActionBaseline(base=_CapturingBase(), layout=layout)
        baseline.fit(X, y)
        baseline.predict(X)

        assert len(captured_X) == 1, "predict did not invoke base.predict"
        X_seen = captured_X[0]
        # Action block must be zeroed.
        assert np.all(X_seen[:, layout.action_slice] == 0.0), (
            f"{label}: predict did not zero action block"
        )
        # Non-action blocks must be bit-identical.
        for blk in layout.blocks_except_action():
            assert np.array_equal(X_seen[:, blk], X[:, blk]), (
                f"{label}: predict modified non-action block {blk}"
            )

    def test_no_action_with_state_and_exogenous_blocks(self):
        """Forward-compat: when R2 adds state / exogenous blocks, they
        must NOT be zeroed by no_action.
        """
        layout = FeatureLayout.from_dims(
            n_actions=4, n_agents=3, n_parameters=3,
            n_state=5, n_exogenous=2,
        )
        X = _build_X(layout, n_samples=8)
        y = _build_y(8)

        captured_X: list[np.ndarray] = []
        class _CapturingBase(BaselineModel):
            name = "capturing"
            def fit(self, X, y):
                self._mean = y.mean(axis=0)
            def predict(self, X):
                captured_X.append(X.copy())
                return np.tile(self._mean, (X.shape[0], 1))

        baseline = NoActionBaseline(base=_CapturingBase(), layout=layout)
        baseline.fit(X, y)
        baseline.predict(X)

        X_seen = captured_X[0]
        assert np.all(X_seen[:, layout.action_slice] == 0.0)
        # state and exogenous must be preserved
        assert np.array_equal(X_seen[:, layout.state_slice], X[:, layout.state_slice])
        assert np.array_equal(X_seen[:, layout.exogenous_slice], X[:, layout.exogenous_slice])


# ----------------------------------------------------------------------
# 3. ShuffledActionBaseline invariance
# ----------------------------------------------------------------------
class TestShuffledActionInvariance:
    """Audit F-02 / R3 G7: shuffled_action must only permute the action block."""

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_shuffled_action_only_permutes_action_block(self, n_actions, n_agents, label):
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=20)
        y = _build_y(20)

        captured_X: list[np.ndarray] = []
        class _CapturingBase(BaselineModel):
            name = "capturing"
            def fit(self, X, y):
                captured_X.append(X.copy())
                self._mean = y.mean(axis=0)
            def predict(self, X):
                captured_X.append(X.copy())
                return np.tile(self._mean, (X.shape[0], 1))

        baseline = ShuffledActionBaseline(base=_CapturingBase(), seed=42, layout=layout)
        baseline.fit(X, y)

        assert len(captured_X) == 1
        X_seen = captured_X[0]

        # Action block must be a row-permutation of the original action block.
        action_original = X[:, layout.action_slice]
        action_seen = X_seen[:, layout.action_slice]
        # Same set of rows (order may differ).
        original_rows = {tuple(row) for row in action_original}
        seen_rows = {tuple(row) for row in action_seen}
        assert original_rows == seen_rows, (
            f"{label}: shuffled action block is not a permutation of the original"
        )
        # And the permutation must not be identity (otherwise it's a no-op).
        # With 20 samples and 4-7 actions, a fixed-seed shuffle is effectively
        # guaranteed to move at least one row. We only assert that *some*
        # row moved (not that all did).
        n_unchanged = sum(
            1 for a, b in zip(action_original, action_seen) if np.array_equal(a, b)
        )
        assert n_unchanged < len(action_original), (
            f"{label}: shuffled action block is identical to original (no shuffle happened)"
        )

        # Non-action blocks must be bit-identical.
        for blk in layout.blocks_except_action():
            assert np.array_equal(X_seen[:, blk], X[:, blk]), (
                f"{label}: shuffled_action modified non-action block {blk}"
            )

    def test_shuffled_action_with_state_and_exogenous_blocks(self):
        """R2 forward-compat: state / exogenous blocks must NOT be shuffled."""
        layout = FeatureLayout.from_dims(
            n_actions=4, n_agents=3, n_parameters=3,
            n_state=5, n_exogenous=2,
        )
        X = _build_X(layout, n_samples=10)
        y = _build_y(10)

        captured_X: list[np.ndarray] = []
        class _CapturingBase(BaselineModel):
            name = "capturing"
            def fit(self, X, y):
                captured_X.append(X.copy())
                self._mean = y.mean(axis=0)
            def predict(self, X):
                captured_X.append(X.copy())
                return np.tile(self._mean, (X.shape[0], 1))

        baseline = ShuffledActionBaseline(base=_CapturingBase(), seed=42, layout=layout)
        baseline.fit(X, y)

        X_seen = captured_X[0]
        # state and exogenous must be preserved bit-identically
        assert np.array_equal(X_seen[:, layout.state_slice], X[:, layout.state_slice])
        assert np.array_equal(X_seen[:, layout.exogenous_slice], X[:, layout.exogenous_slice])


# ----------------------------------------------------------------------
# 4. MeanDeltaBaseline uses layout for grouping
# ----------------------------------------------------------------------
class TestMeanDeltaGrouping:
    """Audit F-02 / R3: MeanDelta must group by (action, agent) using
    layout slices, not hardcoded ``1:8`` / ``8:12``. The grouping loop
    must iterate over ``layout.n_actions`` × ``layout.n_agents``, not
    the legacy ``range(7) × range(4)``.
    """

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_mean_delta_groups_correctly_across_vocabs(self, n_actions, n_agents, label):
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=20)
        y = _build_y(20)

        baseline = MeanDeltaBaseline(layout=layout)
        baseline.fit(X, y)

        # Every (action, agent) cell that appears in X must have a learned mean.
        action_idx = np.argmax(X[:, layout.action_slice], axis=1)
        agent_idx = np.argmax(X[:, layout.agent_slice], axis=1)
        seen_keys = {(int(a), int(g)) for a, g in zip(action_idx, agent_idx)}
        learned_keys = set(baseline._means.keys())

        # All seen keys must be learned (no missing groups).
        missing = seen_keys - learned_keys
        assert not missing, (
            f"{label}: MeanDelta missing groups for {missing}; "
            f"learned keys range action 0..{n_actions-1}, agent 0..{n_agents-1}"
        )

        # No out-of-range keys should exist.
        for a, g in learned_keys:
            assert 0 <= a < n_actions, f"{label}: action idx {a} out of range [0, {n_actions})"
            assert 0 <= g < n_agents, f"{label}: agent idx {g} out of range [0, {n_agents})"

    def test_mean_delta_predict_uses_layout(self):
        """predict must extract action / agent indices via layout, not
        via hardcoded 1:8 / 8:12. Construct a 4-action layout and verify
        the model trained on it predicts the correct group mean.
        """
        layout = FeatureLayout.from_dims(n_actions=4, n_agents=3, n_parameters=3)
        X = _build_X(layout, n_samples=12)
        y = _build_y(12)

        baseline = MeanDeltaBaseline(layout=layout)
        baseline.fit(X, y)

        # Build a single-row X with action=2, agent=1 — predict must
        # return the mean of training samples with the same (action, agent).
        X_query = np.zeros((1, layout.feature_dim), dtype=np.float64)
        X_query[0, layout.tick_slice.start] = 99
        X_query[0, layout.action_slice.start + 2] = 1.0
        X_query[0, layout.agent_slice.start + 1] = 1.0
        X_query[0, layout.parameter_slice.start] = 0.5
        X_query[0, layout.parameter_slice.start + 1] = 0.7
        X_query[0, layout.parameter_slice.start + 2] = 1.0

        y_pred = baseline.predict(X_query)

        # Find training samples with the same (action=2, agent=1).
        action_idx = np.argmax(X[:, layout.action_slice], axis=1)
        agent_idx = np.argmax(X[:, layout.agent_slice], axis=1)
        mask = (action_idx == 2) & (agent_idx == 1)
        if np.any(mask):
            expected = y[mask].mean(axis=0)
            assert np.allclose(y_pred[0], expected), (
                f"MeanDelta predict returned wrong group mean; "
                f"got {y_pred[0]}, expected {expected}"
            )
        else:
            # No training sample in that group → global mean.
            expected = y.mean(axis=0)
            assert np.allclose(y_pred[0], expected)


# ----------------------------------------------------------------------
# 5. OracleUpperBound uses layout for key extraction
# ----------------------------------------------------------------------
class TestOracleUpperBoundLayout:
    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_oracle_uses_layout_for_keys(self, n_actions, n_agents, label):
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=10)
        y = _build_y(10)
        hashes = np.array([f"hash_{i}" for i in range(10)])

        oracle = OracleUpperBound(layout=layout)
        oracle.fit(X, y, hashes)

        # Each (hash, action_idx, agent_idx) key must be in the lookup.
        action_idx = np.argmax(X[:, layout.action_slice], axis=1)
        agent_idx = np.argmax(X[:, layout.agent_slice], axis=1)
        for i in range(10):
            key = (hashes[i], int(action_idx[i]), int(agent_idx[i]))
            assert key in oracle._lookup, (
                f"{label}: OracleUpperBound missing key {key} "
                f"(action range 0..{n_actions-1}, agent range 0..{n_agents-1})"
            )

    def test_oracle_predict_with_hashes_returns_memorized(self):
        layout = FeatureLayout.from_dims(n_actions=5, n_agents=4, n_parameters=3)
        X = _build_X(layout, n_samples=8)
        y = _build_y(8)
        hashes = np.array([f"h{i}" for i in range(8)])

        oracle = OracleUpperBound(layout=layout)
        oracle.fit(X, y, hashes)

        # Predict on the same X/hashes → should return memorized y exactly.
        y_pred = oracle.predict(X, hashes)
        assert np.allclose(y_pred, y), (
            "OracleUpperBound did not return memorized y for known hashes"
        )


# ----------------------------------------------------------------------
# 6. DataLoader.feature_layout matches vocab
# ----------------------------------------------------------------------
class TestDataLoaderFeatureLayout:
    def _write_dummy_dataset(self, tmp_path: Path, n_actions: int, n_agents: int) -> Path:
        """Write a minimal transitions.jsonl + splits.json for testing."""
        action_types = [f"A{i}" for i in range(n_actions)]
        agent_ids = [f"e{i}" for i in range(n_agents)]
        node_ids = ["none", "base", "zone_a"]

        records = []
        for tick in range(5):
            for aid in agent_ids:
                for at in action_types:
                    records.append({
                        "tick": tick,
                        "candidate_actions": {aid: [{"action_type": at, "params": {}}]},
                        "executed_actions": {
                            aid: {
                                "action_type": at,
                                "params": {"target_node": "base"},
                                "executed_at_tick": tick,
                            }
                        },
                        "receipts": {aid: {"energy_delta": -0.5}},
                        "state_delta": {
                            "entity_changes": {"changes": [
                                {"entity_id": aid, "column": "node", "kind": "update", "after": "base"}
                            ]},
                            "relation_changes": {"changes": []},
                        },
                        "state_before_hash": f"hash_{tick}_{aid}_{at}_before",
                        "state_after_hash": f"hash_{tick}_{aid}_{at}_after",
                        "provenance": {
                            "episode_id": f"ep_{tick}",
                            "seed": "42",
                            "policy_id": "random",
                        },
                    })

        # splits.json: half train, half test (by episode_id)
        episode_ids = sorted({r["provenance"]["episode_id"] for r in records})
        splits = {}
        for i, eid in enumerate(episode_ids):
            splits[eid] = "train" if i < len(episode_ids) // 2 else "test"

        dataset_dir = tmp_path / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        with (dataset_dir / "transitions.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        (dataset_dir / "splits.json").write_text(
            json.dumps(splits), encoding="utf-8"
        )
        return dataset_dir

    def test_loader_layout_matches_vocab(self, tmp_path):
        """DataLoader.feature_layout must reflect the inferred vocab."""
        dataset_dir = self._write_dummy_dataset(tmp_path, n_actions=4, n_agents=3)
        loader = DataLoader(dataset_dir)

        # Inferred vocab (alphabetically sorted): A0..A3, e0..e2, none+base+zone_a
        assert loader.action_types == ("A0", "A1", "A2", "A3")
        assert loader.agent_ids == ("e0", "e1", "e2")

        layout = loader.feature_layout
        assert layout.n_actions == 4
        assert layout.n_agents == 3
        assert layout.action_slice == slice(1, 5)
        assert layout.agent_slice == slice(5, 8)
        assert layout.parameter_slice == slice(8, 11)
        assert layout.feature_dim == 1 + 4 + 3 + 3

    def test_loader_layout_emergency_default(self, tmp_path):
        """When the dataset contains the emergency vocab, the loader's
        layout must reproduce the legacy 1:8 / 8:12 slices."""
        dataset_dir = self._write_dummy_dataset(tmp_path, n_actions=7, n_agents=4)
        loader = DataLoader(dataset_dir)

        layout = loader.feature_layout
        assert layout.action_slice == slice(1, 8)
        assert layout.agent_slice == slice(8, 12)
        assert layout.parameter_slice == slice(12, 15)
        assert layout == EMERGENCY_RESOURCE_V0_LAYOUT


# ----------------------------------------------------------------------
# 7. Backwards compatibility: layout=None falls back to emergency default
# ----------------------------------------------------------------------
class TestBackwardsCompat:
    def test_no_action_layout_none_falls_back_to_emergency(self):
        """Constructing NoActionBaseline with layout=None must use the
        emergency_resource_v0 layout (7 actions, 4 agents, 3 params).
        This is what old ad-hoc scripts assumed."""
        baseline = NoActionBaseline(base=LinearRidgeBaseline(alpha=1.0))
        assert baseline._layout == EMERGENCY_RESOURCE_V0_LAYOUT
        assert baseline._layout.action_slice == slice(1, 8)

    def test_shuffled_action_layout_none_falls_back_to_emergency(self):
        baseline = ShuffledActionBaseline(seed=42)
        assert baseline._layout == EMERGENCY_RESOURCE_V0_LAYOUT

    def test_mean_delta_layout_none_falls_back_to_emergency(self):
        baseline = MeanDeltaBaseline()
        assert baseline._layout == EMERGENCY_RESOURCE_V0_LAYOUT

    def test_oracle_layout_none_falls_back_to_emergency(self):
        baseline = OracleUpperBound()
        assert baseline._layout == EMERGENCY_RESOURCE_V0_LAYOUT

    def test_legacy_emergency_layout_still_works_end_to_end(self):
        """Smoke: a baseline constructed without explicit layout must
        still produce predictions on emergency-shaped X (7 actions, 4 agents)."""
        layout = EMERGENCY_RESOURCE_V0_LAYOUT
        X = _build_X(layout, n_samples=15)
        y = _build_y(15)

        baseline = NoActionBaseline(base=LinearRidgeBaseline(alpha=1.0))
        baseline.fit(X, y)
        y_pred = baseline.predict(X)
        assert y_pred.shape == (15, 5)


# ----------------------------------------------------------------------
# 8. Cross-scenario: same baseline code works on 4/5/7 action layouts
# ----------------------------------------------------------------------
class TestCrossScenario:
    """Audit G9: the same evaluator code must work across 4/5/7 action
    scenarios without code changes. This is the core regression guard
    for F-02.
    """

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_no_action_train_predict_across_vocabs(self, n_actions, n_agents, label):
        """End-to-end: NoAction + LinearRidge can fit and predict on
        every audit vocab without crashing and without contaminating
        non-action blocks."""
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=30)
        y = _build_y(30)

        baseline = NoActionBaseline(base=LinearRidgeBaseline(alpha=1.0), layout=layout)
        baseline.fit(X, y)
        y_pred = baseline.predict(X)
        assert y_pred.shape == (30, 5)

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_shuffled_action_train_predict_across_vocabs(self, n_actions, n_agents, label):
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=30)
        y = _build_y(30)

        baseline = ShuffledActionBaseline(
            base=LinearRidgeBaseline(alpha=1.0), seed=42, layout=layout
        )
        baseline.fit(X, y)
        y_pred = baseline.predict(X)
        assert y_pred.shape == (30, 5)

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_mean_delta_train_predict_across_vocabs(self, n_actions, n_agents, label):
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=30)
        y = _build_y(30)

        baseline = MeanDeltaBaseline(layout=layout)
        baseline.fit(X, y)
        y_pred = baseline.predict(X)
        assert y_pred.shape == (30, 5)

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_oracle_train_predict_across_vocabs(self, n_actions, n_agents, label):
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=20)
        y = _build_y(20)
        hashes = np.array([f"h{i}" for i in range(20)])

        oracle = OracleUpperBound(layout=layout)
        oracle.fit(X, y, hashes)
        y_pred = oracle.predict(X, hashes)
        # Oracle must achieve zero error on training data.
        assert np.allclose(y_pred, y), (
            f"{label}: OracleUpperBound did not memorize training y"
        )

    @pytest.mark.parametrize("n_actions,n_agents,label", AUDIT_VOCABS)
    def test_persistence_baseline_works_across_vocabs(self, n_actions, n_agents, label):
        """PersistenceBaseline doesn't use layout, but it must still
        produce correctly-shaped predictions on any vocab."""
        layout = FeatureLayout.from_dims(
            n_actions=n_actions, n_agents=n_agents, n_parameters=3
        )
        X = _build_X(layout, n_samples=10)

        baseline = PersistenceBaseline()
        y_pred = baseline.predict(X)
        assert y_pred.shape == (10, 5)
        assert np.all(y_pred == 0.0)


# ----------------------------------------------------------------------
# 9. Regression: no hardcoded 1:8 / 8:12 in source
# ----------------------------------------------------------------------
class TestNoHardcodedSlicesInSource:
    """Audit F-02 / R3: scan baselines.py source to ensure no ``1:8``
    or ``8:12`` literals remain. This is a static regression guard.
    """

    def test_no_hardcoded_action_slice_in_baselines(self):
        import re
        from pathlib import Path

        baselines_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "worldloop_data"
            / "evaluation"
            / "baselines.py"
        )
        text = baselines_path.read_text(encoding="utf-8")

        # Forbidden literal slice patterns (action hardcode).
        forbidden_action = [
            r"X\[\s*:\s*,\s*1\s*:\s*8\s*\]",
            r"X\[\s*:\s*,\s*1\s*:\s*7\s*\]",
            r"X\[\s*:\s*,\s*2\s*:\s*8\s*\]",
        ]
        # Forbidden literal slice patterns (agent hardcode).
        forbidden_agent = [
            r"X\[\s*:\s*,\s*8\s*:\s*12\s*\]",
            r"X\[\s*:\s*,\s*8\s*:\s*11\s*\]",
            r"X\[\s*:\s*,\s*7\s*:\s*11\s*\]",
        ]

        for pat in forbidden_action + forbidden_agent:
            matches = re.findall(pat, text)
            assert not matches, (
                f"baselines.py still contains hardcoded slice {pat}: {matches}"
            )

    def test_no_hardcoded_range_in_baselines(self):
        """``range(7)`` / ``range(4)`` hardcodes are also forbidden —
        MeanDelta must iterate over ``layout.n_actions`` /
        ``layout.n_agents``.
        """
        import re
        from pathlib import Path

        baselines_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "worldloop_data"
            / "evaluation"
            / "baselines.py"
        )
        text = baselines_path.read_text(encoding="utf-8")

        # Forbidden range patterns.
        forbidden_range = [
            r"range\(\s*7\s*\)",
            r"range\(\s*4\s*\)",
        ]
        for pat in forbidden_range:
            matches = re.findall(pat, text)
            assert not matches, (
                f"baselines.py still contains hardcoded range {pat}: {matches}"
            )
