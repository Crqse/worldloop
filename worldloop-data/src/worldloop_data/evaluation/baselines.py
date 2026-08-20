"""Baseline models for M6 evaluation (§16.4).

7 baseline categories:
    1. PersistenceBaseline: predict delta = 0 (state_after == state_before).
    2. MeanDeltaBaseline: predict delta = training-set mean per (action_type, agent_id).
    3. LinearRidgeBaseline: ridge regression (tick, action_type, agent_id) → targets.
    4. XGBoostBaseline: gradient-boosted trees (optional, falls back to LinearRidge if xgboost missing).
    5. NoActionBaseline: full model but with action features zeroed out.
    6. ShuffledActionBaseline: full model but with action features shuffled across samples.
    7. OracleUpperBound: hash-based lookup (cheating upper bound).

All baselines share the same interface:
    - fit(X_train, y_train)
    - predict(X) -> y_pred
    - name (str property)

Targets (y columns):
    0: energy_delta (float, regression)
    1: position_change_idx (int categorical, classification)
    2: edge_change_count (int, regression)
    3: executed_candidate_rank (int, ranking — 0 if single candidate)
    4: multi_step_energy_delta (float, regression, L-06 short-horizon planning)

Audit F-02 / R3 fix: all column slicing now goes through
:class:`FeatureLayout`. Hardcoded legacy action / agent column slices
have been removed; baselines that need to know the action / agent /
parameter block boundaries accept a ``layout`` argument in ``__init__``.
When ``layout`` is ``None`` the baseline falls back to
:data:`EMERGENCY_RESOURCE_V0_LAYOUT` for backwards compatibility with
ad-hoc scripts that pre-date the audit fix. New code (including the M6
runner in ``scripts/run_m6_evaluation.py``) should pass an explicit
``loader.feature_layout``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from worldloop_data.evaluation.feature_layout import (
    FeatureLayout,
    EMERGENCY_RESOURCE_V0_LAYOUT,
)


class BaselineModel(ABC):
    """Abstract base class for all M6 baselines."""

    name: str = "baseline"
    # Number of target columns (5 since L-06: added multi_step_energy_delta).
    _n_targets: int = 5

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train on (X, y) matrices."""

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return y_pred (same shape as y)."""


class PersistenceBaseline(BaselineModel):
    """Predict delta = 0 for all targets (state_after == state_before)."""

    name = "persistence"

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # No training needed.
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        # y_pred: 5 zeros (energy_delta, position, edge, rank, multi_step_energy).
        y_pred = np.zeros((X.shape[0], self._n_targets), dtype=np.float64)
        return y_pred


class MeanDeltaBaseline(BaselineModel):
    """Predict delta = training-set mean per (action_type, agent_id) group."""

    name = "mean_delta"

    def __init__(self, layout: FeatureLayout | None = None):
        # layout is required to know which columns are action / agent one-hot.
        # Falls back to emergency_resource_v0 layout for backwards compat.
        self._layout = layout or EMERGENCY_RESOURCE_V0_LAYOUT

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Group by (action_type_idx, agent_id_idx) using layout slices.
        layout = self._layout
        self._means: dict[tuple[int, int], np.ndarray] = {}
        self._global_mean = y.mean(axis=0) if y.size else np.zeros(self._n_targets)

        action_idx = np.argmax(X[:, layout.action_slice], axis=1)
        agent_idx = np.argmax(X[:, layout.agent_slice], axis=1)

        n_actions = layout.n_actions
        n_agents = layout.n_agents
        for a_idx in range(n_actions):
            for g_idx in range(n_agents):
                mask = (action_idx == a_idx) & (agent_idx == g_idx)
                if np.any(mask):
                    self._means[(a_idx, g_idx)] = y[mask].mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        layout = self._layout
        action_idx = np.argmax(X[:, layout.action_slice], axis=1)
        agent_idx = np.argmax(X[:, layout.agent_slice], axis=1)
        y_pred = np.zeros((X.shape[0], self._n_targets), dtype=np.float64)
        for i in range(X.shape[0]):
            key = (int(action_idx[i]), int(agent_idx[i]))
            y_pred[i] = self._means.get(key, self._global_mean)
        return y_pred


class LinearRidgeBaseline(BaselineModel):
    """Ridge regression for each target column."""

    name = "linear_ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._weights: np.ndarray | None = None  # (n_features+1, n_targets)
        self._most_common_position = 0
        self._most_common_rank = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Add bias term.
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        # Ridge: w = (X^T X + alpha I)^-1 X^T y
        n_features = X_aug.shape[1]
        A = X_aug.T @ X_aug + self.alpha * np.eye(n_features)
        b = X_aug.T @ y
        self._weights = np.linalg.solve(A, b)

        # For categorical targets (position_change, rank), use most common as fallback.
        self._most_common_position = int(np.bincount(y[:, 1].astype(int) + 1, minlength=7)[1:].argmax()) if y.size else 0
        rank_col = y[:, 3].astype(int)
        rank_nonzero = rank_col[rank_col > 0]
        self._most_common_rank = int(np.bincount(rank_nonzero).argmax()) if rank_nonzero.size else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("LinearRidgeBaseline not fitted")
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        y_pred = X_aug @ self._weights
        # Round categorical targets.
        y_pred[:, 1] = np.clip(np.round(y_pred[:, 1]), 0, 5)
        y_pred[:, 3] = np.clip(np.round(y_pred[:, 3]), 0, 10)
        return y_pred


class XGBoostBaseline(BaselineModel):
    """XGBoost regressor for each target column (falls back to LinearRidge)."""

    name = "xgboost"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 4,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        # M8: seedable for model-seed variance estimation (default keeps
        # the historical value 42, so existing behavior is unchanged).
        self.random_state = random_state
        self._models: list = []
        self._fallback: LinearRidgeBaseline | None = None
        self._most_common_position = 0
        self._most_common_rank = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            import xgboost as xgb
        except ImportError:
            # Fallback to LinearRidge if xgboost not installed.
            self._fallback = LinearRidgeBaseline(alpha=1.0)
            self._fallback.fit(X, y)
            return

        self._models = []
        for col in range(self._n_targets):
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                objective="reg:squarederror",
                random_state=self.random_state,
            )
            model.fit(X, y[:, col])
            self._models.append(model)

        self._most_common_position = int(np.bincount(y[:, 1].astype(int) + 1, minlength=7)[1:].argmax()) if y.size else 0
        rank_col = y[:, 3].astype(int)
        rank_nonzero = rank_col[rank_col > 0]
        self._most_common_rank = int(np.bincount(rank_nonzero).argmax()) if rank_nonzero.size else 0

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._fallback is not None:
            return self._fallback.predict(X)

        y_pred = np.zeros((X.shape[0], self._n_targets), dtype=np.float64)
        for col, model in enumerate(self._models):
            y_pred[:, col] = model.predict(X)
        # Round categorical targets.
        y_pred[:, 1] = np.clip(np.round(y_pred[:, 1]), 0, 5)
        y_pred[:, 3] = np.clip(np.round(y_pred[:, 3]), 0, 10)
        return y_pred


class NoActionBaseline(BaselineModel):
    """Wrap a base model but zero out the action block at fit / predict time.

    Audit F-02 / R3: the action block is now identified by
    ``layout.action_slice`` instead of the hardcoded ``1:8``. The non-action
    blocks (tick / agent / parameter / state / exogenous) are left
    bit-identical — see ``test_feature_layout.py`` for the invariance test.
    """

    name = "no_action"

    def __init__(self, base: BaselineModel | None = None, layout: FeatureLayout | None = None):
        self.base = base or XGBoostBaseline()
        # layout is required to know which columns are the action block.
        # Falls back to emergency_resource_v0 layout for backwards compat.
        self._layout = layout or EMERGENCY_RESOURCE_V0_LAYOUT

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Train on X with action features zeroed.
        X_no_action = X.copy()
        X_no_action[:, self._layout.action_slice] = 0.0
        self.base.fit(X_no_action, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_no_action = X.copy()
        X_no_action[:, self._layout.action_slice] = 0.0
        return self.base.predict(X_no_action)


class ShuffledActionBaseline(BaselineModel):
    """Wrap a base model but shuffle the action block across samples.

    Audit F-02 / R3: only the action block (``layout.action_slice``) is
    shuffled; all other blocks are bit-identical to the input. The previous
    legacy action-slice hardcode would also shuffle agent / parameter
    columns in 4- or 5-action scenarios, contaminating the ablation.
    """

    name = "shuffled_action"

    def __init__(
        self,
        base: BaselineModel | None = None,
        seed: int = 42,
        layout: FeatureLayout | None = None,
    ):
        self.base = base or XGBoostBaseline()
        self.rng = np.random.default_rng(seed)
        # layout is required to know which columns are the action block.
        # Falls back to emergency_resource_v0 layout for backwards compat.
        self._layout = layout or EMERGENCY_RESOURCE_V0_LAYOUT

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        # Train on X with action features shuffled across samples.
        X_shuffled = X.copy()
        perm = self.rng.permutation(X.shape[0])
        X_shuffled[:, self._layout.action_slice] = X[perm, self._layout.action_slice]
        self.base.fit(X_shuffled, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # At predict time, also shuffle (deterministic per-call with same rng state).
        X_shuffled = X.copy()
        perm = self.rng.permutation(X.shape[0])
        X_shuffled[:, self._layout.action_slice] = X[perm, self._layout.action_slice]
        return self.base.predict(X_shuffled)


class OracleUpperBound(BaselineModel):
    """Hash-based lookup (cheating upper bound).

    Memorizes (state_before_hash, action_type, agent_id) → y for training samples.
    At predict time, if the same key exists, return the memorized y; else return
    the global mean. This is an upper bound on what a lookup-based model can do.

    Audit F-02 / R3: action / agent indices are now extracted via
    ``layout.action_slice`` / ``layout.agent_slice`` instead of the
    hardcoded ``1:8`` / ``8:12``.
    """

    name = "oracle_upper_bound"

    def __init__(self, base: BaselineModel | None = None, layout: FeatureLayout | None = None):
        self.base = base or MeanDeltaBaseline(layout=layout)
        self._layout = layout or EMERGENCY_RESOURCE_V0_LAYOUT
        self._lookup: dict[tuple, np.ndarray] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, hashes: np.ndarray | None = None) -> None:
        # Train the fallback base model. Note: MeanDeltaBaseline uses the
        # same layout, so the (action, agent) grouping is consistent.
        self.base.fit(X, y)
        self._lookup = {}
        if hashes is None:
            return
        layout = self._layout
        action_idx = np.argmax(X[:, layout.action_slice], axis=1)
        agent_idx = np.argmax(X[:, layout.agent_slice], axis=1)
        for i in range(X.shape[0]):
            key = (hashes[i], int(action_idx[i]), int(agent_idx[i]))
            self._lookup[key] = y[i]

    def predict(self, X: np.ndarray, hashes: np.ndarray | None = None) -> np.ndarray:
        y_pred = self.base.predict(X)
        if hashes is None:
            return y_pred
        layout = self._layout
        action_idx = np.argmax(X[:, layout.action_slice], axis=1)
        agent_idx = np.argmax(X[:, layout.agent_slice], axis=1)
        for i in range(X.shape[0]):
            key = (hashes[i], int(action_idx[i]), int(agent_idx[i]))
            if key in self._lookup:
                y_pred[i] = self._lookup[key]
        return y_pred

    def fit_predict_with_hashes(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        hashes_train: np.ndarray,
        X_test: np.ndarray,
        hashes_test: np.ndarray,
    ) -> np.ndarray:
        """Convenience: fit on train, predict on test with hash lookup."""
        self.fit(X_train, y_train, hashes_train)
        return self.predict(X_test, hashes_test)
