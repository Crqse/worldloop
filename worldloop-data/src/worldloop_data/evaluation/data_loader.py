"""Data loader for M6 evaluation.

Loads ``transitions.jsonl`` + ``splits.json`` from a dataset directory and
produces featureized :class:`TransitionSample` objects for training/eval.

The featureization is deliberately simple (§16.4 baselines must use the
SAME data budget — no privileged state reconstruction):

Input features X (dynamic dim, depends on scenario vocab):
    - tick (int)
    - action_type one-hot (n_actions dims)
    - agent_id one-hot (n_agents dims)
    - params_bag (3 dims: target_node_idx, target_agent_idx, has_params)

Targets y:
    - energy_delta (float): receipt.energy_delta
    - position_change (int categorical): index into node_ids (0 if no change)
    - edge_change_count (int): number of relation_changes
    - executed_candidate_rank (int): which candidate was executed (0 if single candidate)
    - multi_step_energy_delta (float): cumulative energy_delta over next 3 ticks (L-06 short-horizon planning)

Pre-registered primary metrics (§16.6):
    - energy_delta MAE
    - position_change accuracy
    - edge_change_count MAE
    - executed_candidate_rank NDCG@3
    - multi_step_energy_delta MAE  (L-06 added to break position_change saturation)

Vocab is inferred from the dataset (scans first 100 records) or can be
passed explicitly to override. This lets the same loader work across
scenarios with different action_types / agent_ids / node_ids (e.g.
emergency_resource_v0 vs market_exchange_v0).

R2 integration (audit F-03): when an :class:`EncoderSchema` and per-episode
initial state views are supplied, the loader can also produce
:class:`TrainingTransition` objects via :class:`StateMaterializer` and
build a feature matrix that includes the full ``S_t + U_t`` blocks (field
/ entity / graph / registry / population / exogenous). The legacy
:class:`TransitionSample` path remains the default for backwards
compatibility; the R2 path is opt-in via :meth:`DataLoader.load_training_transitions`
and :meth:`DataLoader.feature_matrix_from_transitions`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from worldloop_data.evaluation.feature_layout import FeatureLayout
from worldloop_data.evaluation.state_materializer import (
    EncoderSchema,
    StateEncoder,
    StateMaterializer,
    TrainingTransition,
    MaterializerError,
)

# Default vocab for emergency_resource_v0 (used by top-level feature_matrix()
# and as fallback when inference fails).
EMERGENCY_ACTION_TYPES = (
    "COLLECT",
    "COMMUNICATE",
    "DELIVER",
    "MOVE",
    "REPAIR",
    "REST",
    "SHARE",
)
EMERGENCY_AGENT_IDS = ("e0", "e1", "e2", "e3")
EMERGENCY_NODE_IDS = ("none", "base", "zone_a", "zone_b", "zone_c", "zone_d")


@dataclass
class TransitionSample:
    """Featureized transition sample for M6 evaluation.

    Stores raw indices into the loader's vocab. The actual feature vector
    is built by :meth:`DataLoader.feature_matrix` so that vocab can vary
    across scenarios without changing this dataclass.
    """

    # Input features (indices into loader vocab)
    tick: int
    action_type_idx: int          # index into loader.action_types
    agent_id_idx: int             # index into loader.agent_ids
    target_node_idx: int          # index into loader.node_ids (0 if no target_node)
    target_agent_idx: int         # index into loader.agent_ids, -1 if no target_agent
    has_params: int               # 0/1

    # Targets
    energy_delta: float           # receipt.energy_delta
    position_change_idx: int      # index into loader.node_ids (0 if no position change)
    edge_change_count: int        # len(relation_changes.changes)
    executed_candidate_rank: int  # 0 if single candidate; rank in candidate_actions
    multi_step_energy_delta: float  # L-06: cumulative energy_delta over next 3 ticks

    # Metadata (not used as features, but for grouping/splitting)
    episode_id: str
    seed: str
    split: str
    policy_id: str
    state_before_hash: str
    state_after_hash: str

    @property
    def target_vector(self) -> np.ndarray:
        """Return the target vector y (length = 5)."""
        return np.array([
            self.energy_delta,
            self.position_change_idx,
            self.edge_change_count,
            self.executed_candidate_rank,
            self.multi_step_energy_delta,
        ], dtype=np.float64)


def _infer_vocab_from_dataset(
    transitions_path: Path,
    scan_limit: int = 100,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Scan first ``scan_limit`` records to infer (action_types, agent_ids, node_ids).

    Vocab is sorted alphabetically for stable order. ``node_ids`` always
    includes "none" as the first element (used for "no position change").
    """
    action_types: set[str] = set()
    agent_ids: set[str] = set()
    node_ids: set[str] = {"none"}
    n_scanned = 0
    with transitions_path.open("r", encoding="utf-8") as f:
        for line in f:
            if n_scanned >= scan_limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            executed = record.get("executed_actions", {}) or {}
            for aid, info in executed.items():
                agent_ids.add(aid)
                at = info.get("action_type", "")
                if at:
                    action_types.add(at)
            # Collect node ids from entity_changes (position updates).
            for ch in (record.get("state_delta") or {}).get("entity_changes", {}).get("changes", []) or []:
                if ch.get("column") == "node" and ch.get("kind") == "update":
                    after = ch.get("after", "none")
                    if after:
                        node_ids.add(after)
            # Also collect from params.target_node (MOVE action).
            for aid, info in executed.items():
                params = info.get("params", {}) or {}
                tn = params.get("target_node")
                if tn:
                    node_ids.add(tn)
            n_scanned += 1
    return (
        tuple(sorted(action_types)),
        tuple(sorted(agent_ids)),
        tuple(sorted(node_ids)),
    )


def _extract_executed_action(record: dict) -> tuple[str, str, dict, str]:
    """Return (agent_id, action_type, params, executed_at_tick) from record.

    If multiple agents executed in one tick, return the first one (alphabetical).
    """
    executed = record.get("executed_actions", {})
    if not executed:
        return "", "", {}, ""
    agent_id = sorted(executed.keys())[0]
    info = executed[agent_id]
    return (
        agent_id,
        info.get("action_type", ""),
        info.get("params", {}) or {},
        info.get("executed_at_tick", 0),
    )


def _extract_receipt(record: dict, agent_id: str) -> dict:
    return record.get("receipts", {}).get(agent_id, {})


def _extract_edge_change_count(record: dict) -> int:
    return len(
        (record.get("state_delta") or {})
        .get("relation_changes", {})
        .get("changes", [])
        or []
    )


def _extract_executed_candidate_rank(record: dict, executed_agent_id: str) -> int:
    """Return the rank of the executed candidate among all candidate_actions.

    Rank is 1-indexed by alphabetical order of candidate agent_ids.
    Returns 0 if only one candidate (no ranking task).
    """
    candidates = record.get("candidate_actions", {}) or {}
    if len(candidates) <= 1:
        return 0
    sorted_ids = sorted(candidates.keys())
    return sorted_ids.index(executed_agent_id) + 1


class DataLoader:
    """Load transitions.jsonl + splits.json into TransitionSample lists.

    Vocab (action_types / agent_ids / node_ids) is inferred from the
    dataset by scanning the first 100 records, or can be passed explicitly
    to override. This lets the same loader work across scenarios.

    Usage:
        loader = DataLoader(dataset_dir)
        train_samples, test_samples = loader.load_splits()
        X_train, y_train = loader.feature_matrix(train_samples)
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        action_types: tuple[str, ...] | None = None,
        agent_ids: tuple[str, ...] | None = None,
        node_ids: tuple[str, ...] | None = None,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.transitions_path = self.dataset_dir / "transitions.jsonl"
        self.splits_path = self.dataset_dir / "splits.json"
        if not self.transitions_path.exists():
            raise FileNotFoundError(f"transitions.jsonl not found at {self.transitions_path}")
        if not self.splits_path.exists():
            raise FileNotFoundError(f"splits.json not found at {self.splits_path}")

        # Infer vocab if not provided.
        if action_types is None or agent_ids is None or node_ids is None:
            inferred_actions, inferred_agents, inferred_nodes = _infer_vocab_from_dataset(
                self.transitions_path
            )
            self.action_types = action_types if action_types is not None else inferred_actions
            self.agent_ids = agent_ids if agent_ids is not None else inferred_agents
            self.node_ids = node_ids if node_ids is not None else inferred_nodes
        else:
            self.action_types = action_types
            self.agent_ids = agent_ids
            self.node_ids = node_ids

        self._action_to_idx = {a: i for i, a in enumerate(self.action_types)}
        self._agent_to_idx = {a: i for i, a in enumerate(self.agent_ids)}
        self._node_to_idx = {n: i for i, n in enumerate(self.node_ids)}

        self._n_actions = len(self.action_types)
        self._n_agents = len(self.agent_ids)
        self._n_nodes = len(self.node_ids)
        # feature_dim = 1 (tick) + n_actions + n_agents + 3 (target_node_idx, target_agent_idx, has_params)
        self._feature_dim = 1 + self._n_actions + self._n_agents + 3
        # FeatureLayout: built once from vocab; baselines use this to slice
        # X without hardcoding 1:8 / 8:12 etc. (audit F-02 / R3 fix).
        # state and exogenous blocks are absent until R2 (TrainingTransition)
        # lands; they remain None here.
        self._feature_layout = FeatureLayout.from_dims(
            n_actions=self._n_actions,
            n_agents=self._n_agents,
            n_parameters=3,
            n_state=0,
            n_exogenous=0,
            include_tick=True,
        )

    @property
    def feature_layout(self) -> FeatureLayout:
        """Block layout of X matrices produced by this loader.

        Baselines should consume this layout instead of hardcoding column
        slices. See :class:`FeatureLayout` for the block order.
        """
        return self._feature_layout

    def _record_to_sample(self, record: dict, split: str) -> TransitionSample | None:
        """Convert a raw transition record dict to a TransitionSample.

        Returns None if the record is malformed or uses unknown vocab.
        """
        agent_id, action_type, params, _ = _extract_executed_action(record)
        if not agent_id or not action_type:
            return None

        receipt = _extract_receipt(record, agent_id)
        energy_delta = float(receipt.get("energy_delta", 0.0))
        edge_change_count = _extract_edge_change_count(record)
        executed_candidate_rank = _extract_executed_candidate_rank(record, agent_id)

        action_type_idx = self._action_to_idx.get(action_type, -1)
        if action_type_idx < 0:
            # Unknown action type for this vocab — skip.
            return None

        agent_id_idx = self._agent_to_idx.get(agent_id, -1)
        if agent_id_idx < 0:
            # Unknown agent — skip.
            return None

        # Position change: look at entity_changes for this agent's "node" column.
        position_change_idx = 0
        changes = (
            (record.get("state_delta") or {})
            .get("entity_changes", {})
            .get("changes", [])
            or []
        )
        for ch in changes:
            if (
                ch.get("entity_id") == agent_id
                and ch.get("column") == "node"
                and ch.get("kind") == "update"
            ):
                after = ch.get("after", "none")
                position_change_idx = self._node_to_idx.get(after, 0)
                break

        # target_node_idx from params.target_node
        target_node = params.get("target_node")
        target_node_idx = self._node_to_idx.get(target_node, 0) if target_node else 0

        # target_agent_idx from params.target_agent
        target_agent = params.get("target_agent")
        target_agent_idx = self._agent_to_idx.get(target_agent, -1) if target_agent else -1

        provenance = record.get("provenance", {}) or {}

        return TransitionSample(
            tick=int(record.get("tick", 0)),
            action_type_idx=action_type_idx,
            agent_id_idx=agent_id_idx,
            target_node_idx=target_node_idx,
            target_agent_idx=target_agent_idx,
            has_params=1 if params else 0,
            energy_delta=energy_delta,
            position_change_idx=position_change_idx,
            edge_change_count=edge_change_count,
            executed_candidate_rank=executed_candidate_rank,
            multi_step_energy_delta=0.0,  # filled by _attach_multi_step_targets
            episode_id=provenance.get("episode_id", ""),
            seed=provenance.get("seed", ""),
            split=split,
            policy_id=provenance.get("policy_id", ""),
            state_before_hash=record.get("state_before_hash", ""),
            state_after_hash=record.get("state_after_hash", ""),
        )

    def _sample_to_feature(self, sample: TransitionSample) -> np.ndarray:
        """Build the feature vector for a single sample using this loader's vocab."""
        v = np.zeros(self._feature_dim, dtype=np.float64)
        v[0] = sample.tick
        # action_type one-hot (dims 1 .. 1+n_actions)
        v[1 + sample.action_type_idx] = 1.0
        # agent_id one-hot (dims 1+n_actions .. 1+n_actions+n_agents)
        v[1 + self._n_actions + sample.agent_id_idx] = 1.0
        # target_node_idx
        v[1 + self._n_actions + self._n_agents + 0] = sample.target_node_idx
        # target_agent_idx — normalize to [0, 1]; -1 (absent) maps to 0
        v[1 + self._n_actions + self._n_agents + 1] = max(0.0, sample.target_agent_idx) / max(1, self._n_agents)
        # has_params
        v[1 + self._n_actions + self._n_agents + 2] = sample.has_params
        return v

    def feature_matrix(
        self,
        samples: Iterable[TransitionSample],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack samples into (X, y) matrices using this loader's vocab.

        X shape: (N, feature_dim), y shape: (N, 5).
        """
        sample_list = list(samples)
        if not sample_list:
            return (
                np.zeros((0, self._feature_dim), dtype=np.float64),
                np.zeros((0, 5), dtype=np.float64),
            )
        X = np.stack([self._sample_to_feature(s) for s in sample_list])
        y = np.stack([s.target_vector for s in sample_list])
        return X, y

    # ------------------------------------------------------------------
    # R2 integration (audit F-03): TrainingTransition path
    # ------------------------------------------------------------------

    def build_materializer(
        self,
        encoder_schema: EncoderSchema,
        *,
        hash_buckets: int = 32,
    ) -> StateMaterializer:
        """Build a :class:`StateMaterializer` wired to this loader's vocab.

        The materializer uses this loader's ``action_types`` and ``agent_ids``
        as the joint-action vocab, so the resulting
        :class:`JointActionFeatures` aligns with the layout exposed by
        :attr:`feature_layout`.

        Parameters
        ----------
        encoder_schema:
            Schema describing the per-block dims the encoder will produce.
            Built once per scenario from the ``CapabilityProfile`` +
            scenario vocab.
        hash_buckets:
            Bucket count for string-value hashing in the encoder.

        Returns
        -------
        StateMaterializer
            A materializer bound to this loader's action / agent vocab and
            the supplied encoder schema.
        """
        encoder = StateEncoder(encoder_schema, hash_buckets=hash_buckets)
        return StateMaterializer(
            encoder=encoder,
            action_types=self.action_types,
            agent_ids=self.agent_ids,
        )

    def load_training_transitions(
        self,
        initial_state_views: Mapping[str, Mapping[str, Any]],
        encoder_schema: EncoderSchema,
        *,
        splits: tuple[str, ...] = ("train", "val", "test"),
        hash_buckets: int = 32,
    ) -> dict[str, list[TrainingTransition]]:
        """Load :class:`TrainingTransition` objects via :class:`StateMaterializer`.

        Audit F-03 / R2 fix. For each episode that has an entry in
        ``initial_state_views``, this method walks the transition
        sequence, applies ``state_delta`` diffs from the initial state,
        and emits one :class:`TrainingTransition` per tick with the full
        ``S_t + A_t + U_t`` blocks (field / entity / graph / registry /
        population + joint action + exogenous).

        Episodes without an initial state view are skipped — the R2
        "no silent degradation" rule applies at the materializer level
        (a single episode with ``None`` raises
        :class:`MaterializerError`), but the loader filters missing
        episodes before invoking the materializer so that callers can
        pass a partial dict.

        Parameters
        ----------
        initial_state_views:
            Mapping ``episode_id -> initial_state_view``. Episodes not
            in this mapping are skipped. The initial state view is the
            StateView dict at tick 0 (before any action).
        encoder_schema:
            Schema for the state encoder.
        splits:
            Which splits to load. Defaults to all three.
        hash_buckets:
            Bucket count for the state encoder.

        Returns
        -------
        dict[str, list[TrainingTransition]]
            Mapping ``split_name -> list_of_training_transitions``.
            Only non-empty splits appear in the dict.
        """
        split_map = json.loads(self.splits_path.read_text(encoding="utf-8"))
        materializer = self.build_materializer(
            encoder_schema, hash_buckets=hash_buckets
        )

        # Group records by episode_id, preserving tick order within each
        # episode. transitions.jsonl is already sorted by
        # (split, episode_id, tick) per the exporter, but we re-sort
        # defensively.
        per_episode: dict[str, list[dict[str, Any]]] = {}
        with self.transitions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                episode_id = (record.get("provenance") or {}).get("episode_id", "")
                if not episode_id:
                    continue
                per_episode.setdefault(episode_id, []).append(record)

        # Sort each episode's records by tick.
        for ep_id in per_episode:
            per_episode[ep_id].sort(key=lambda r: int(r.get("tick", 0)))

        out: dict[str, list[TrainingTransition]] = {}
        for ep_id, records in per_episode.items():
            split = split_map.get(ep_id, "")
            if split not in splits:
                continue
            if ep_id not in initial_state_views:
                # Skip episodes without an initial state view (R2: no
                # silent degradation at the materializer level; the
                # loader filters missing episodes here).
                continue
            samples = materializer.materialize_episode(
                initial_state_view=initial_state_views[ep_id],
                transition_records=records,
                episode_id=ep_id,
                split=split,
            )
            out.setdefault(split, []).extend(samples)
        return out

    def feature_matrix_from_transitions(
        self,
        transitions: Iterable[TrainingTransition],
        *,
        include_state: bool = True,
        include_exogenous: bool = True,
        include_action: bool = True,
        include_tick: bool = True,
        include_agent_id: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build (X, y) matrices from :class:`TrainingTransition` objects.

        This is the R2 path (audit F-03). Unlike the legacy
        :meth:`feature_matrix` which only emits ``tick + action one-hot +
        agent one-hot + params``, this method concatenates the full
        block-wise features from the materializer.

        Block order matches :meth:`r2_feature_layout` so baselines can
        slice the resulting ``X`` using the layout's slices::

            [tick?] + [joint_action?] + [agent_id?] + [state?] + [exogenous?]

        where ``joint_action`` occupies ``action_slice`` (the R2 layout
        puts the joint-action block in the ``action_slice`` position so
        existing baselines like :class:`NoActionBaseline` /
        :class:`ShuffledActionBaseline` work without modification —
        zeroing / permuting the joint-action block is the correct R2
        ablation semantics).

        Targets ``y`` come from
        :attr:`TrainingTransition.target_vector` (5-column legacy layout)
        so downstream metric code is unchanged.

        Parameters
        ----------
        transitions:
            Iterable of :class:`TrainingTransition` (e.g., from
            :meth:`load_training_transitions`).
        include_state:
            Concatenate the state_before features block.
        include_exogenous:
            Concatenate the exogenous features block.
        include_action:
            Concatenate the joint action features block.
        include_tick:
            Prepend the tick scalar.
        include_agent_id:
            Concatenate the agent_id one-hot (using this loader's vocab).
            Off by default — the joint action already encodes per-agent
            rows.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(X, y)`` where ``X.shape == (N, feature_dim)`` and
            ``y.shape == (N, 5)``. Returns zero-row arrays if the input
            is empty.
        """
        sample_list = list(transitions)
        if not sample_list:
            return (
                np.zeros((0, 0), dtype=np.float64),
                np.zeros((0, 5), dtype=np.float64),
            )

        # Compute feature dim by inspecting the first sample.
        first = sample_list[0]
        dim = 0
        if include_tick:
            dim += 1
        if include_action:
            dim += first.joint_action_features.to_vector().shape[0]
        if include_agent_id:
            dim += self._n_agents
        if include_state:
            dim += first.state_before_features.to_vector().shape[0]
        if include_exogenous:
            dim += first.exogenous_features.to_vector().shape[0]

        X = np.zeros((len(sample_list), dim), dtype=np.float64)
        y = np.zeros((len(sample_list), 5), dtype=np.float64)

        for i, t in enumerate(sample_list):
            cursor = 0
            # Block order: [tick?] + [action?] + [agent_id?] + [state?] + [exogenous?]
            if include_tick:
                X[i, cursor] = float(t.provenance.tick)
                cursor += 1
            if include_action:
                v = t.joint_action_features.to_vector()
                end = cursor + v.shape[0]
                X[i, cursor:end] = v
                cursor = end
            if include_agent_id:
                # Use the first agent that actually acted, else 0.
                agent_id = ""
                for aid in t.joint_action_features.agent_ids:
                    if aid in t.receipt_targets:
                        agent_id = aid
                        break
                idx = self._agent_to_idx.get(agent_id, -1)
                if idx >= 0:
                    X[i, cursor + idx] = 1.0
                cursor += self._n_agents
            if include_state:
                v = t.state_before_features.to_vector()
                end = cursor + v.shape[0]
                X[i, cursor:end] = v
                cursor = end
            if include_exogenous:
                v = t.exogenous_features.to_vector()
                end = cursor + v.shape[0]
                X[i, cursor:end] = v
                cursor = end
            y[i] = t.target_vector

        return X, y

    def r2_feature_layout(
        self,
        encoder_schema: EncoderSchema,
        *,
        include_state: bool = True,
        include_exogenous: bool = True,
        include_action: bool = True,
        include_tick: bool = True,
        include_agent_id: bool = False,
    ) -> FeatureLayout:
        """Build a :class:`FeatureLayout` matching :meth:`feature_matrix_from_transitions`.

        This lets baselines slice the R2 feature matrix using
        ``layout.action_slice`` / ``layout.state_slice`` /
        ``layout.exogenous_slice`` instead of hardcoding column indices.

        Block order (matches :meth:`feature_matrix_from_transitions`)::

            [tick?] + [joint_action as action_slice] + [agent_id?] + [state?] + [exogenous?]

        The R2 layout puts the joint-action block into ``action_slice``
        (with ``n_actions = joint_action_dim``) so existing
        :class:`NoActionBaseline` / :class:`ShuffledActionBaseline`
        work without modification: zeroing / permuting ``action_slice``
        is the correct R2 ablation semantics (ablate the joint action,
        not the legacy single-action one-hot).

        ``agent_slice`` is empty unless ``include_agent_id=True``;
        ``parameter_slice`` is always empty in the R2 layout
        (parameters live inside the joint-action block).
        """
        # Joint action dim: n_agents * (n_actions + 3)
        joint_action_dim = (
            len(self.agent_ids) * (len(self.action_types) + 3)
            if include_action
            else 0
        )
        n_state = encoder_schema.encoded_dim() if include_state else 0
        # Exogenous dim is dynamic per-record; the loader cannot know
        # the exact exogenous channel count without scanning the data.
        # We use 0 here — callers that need exogenous in the layout
        # should pass a custom FeatureLayout or extend this method.
        n_exogenous = 0
        n_agents = self._n_agents if include_agent_id else 0
        return FeatureLayout.from_dims(
            n_actions=joint_action_dim,  # R2: action_slice -> joint_action block
            n_agents=n_agents,
            n_parameters=0,  # R2: parameters live inside joint_action
            n_state=n_state,
            n_exogenous=n_exogenous,
            include_tick=include_tick,
        )

    def load_splits(self) -> tuple[list[TransitionSample], list[TransitionSample]]:
        """Return (train_samples, test_samples) parsed from the dataset.

        If no explicit test split exists, returns (train, val).
        """
        splits = json.loads(self.splits_path.read_text(encoding="utf-8"))
        train_samples: list[TransitionSample] = []
        val_samples: list[TransitionSample] = []
        test_samples: list[TransitionSample] = []

        with self.transitions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                episode_id = (record.get("provenance") or {}).get("episode_id", "")
                split = splits.get(episode_id, "")
                if split == "train":
                    target_list = train_samples
                elif split == "val":
                    target_list = val_samples
                elif split == "test":
                    target_list = test_samples
                else:
                    continue
                sample = self._record_to_sample(record, split)
                if sample is not None:
                    target_list.append(sample)

        # L-06: fill multi_step_energy_delta per-split to avoid cross-split leakage.
        _attach_multi_step_targets(train_samples)
        _attach_multi_step_targets(val_samples)
        _attach_multi_step_targets(test_samples)

        if test_samples:
            return train_samples, test_samples
        return train_samples, val_samples

    def load_all(self) -> list[TransitionSample]:
        """Return all samples regardless of split."""
        all_samples: list[TransitionSample] = []
        with self.transitions_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                episode_id = (record.get("provenance") or {}).get("episode_id", "")
                # split is "unknown" for load_all; not used for filtering.
                sample = self._record_to_sample(record, "unknown")
                if sample is not None:
                    all_samples.append(sample)
        _attach_multi_step_targets(all_samples)
        return all_samples


def _attach_multi_step_targets(
    samples: list[TransitionSample],
    horizon: int = 3,
) -> None:
    """L-06: fill ``multi_step_energy_delta`` for each sample in-place.

    For each sample, sum the ``energy_delta`` of the next ``horizon`` samples
    **within the same episode** (sorted by tick). Samples at the end of an
    episode accumulate only the remaining ticks (no cross-episode leakage).

    Rationale: a single-step energy_delta is dominated by the action_type
    (deterministic mapping in many scenarios). The cumulative delta over a
    short horizon is more sensitive to *sequences* of actions, which gives
    the full (S_t, A_t, U_t) model a chance to demonstrate planning value
    beyond what no-action / shuffled-action baselines can capture.
    """
    if not samples:
        return

    by_episode: dict[str, list[TransitionSample]] = {}
    for s in samples:
        by_episode.setdefault(s.episode_id, []).append(s)

    for ep_id, ep_samples in by_episode.items():
        ep_samples.sort(key=lambda s: s.tick)
        n = len(ep_samples)
        for i in range(n):
            total = 0.0
            for j in range(1, horizon + 1):
                if i + j < n:
                    total += ep_samples[i + j].energy_delta
                else:
                    break
            ep_samples[i].multi_step_energy_delta = total


# ----------------------------------------------------------------------------
# Backwards-compatibility: top-level feature_matrix() for emergency vocab.
# Prefer DataLoader.feature_matrix() for cross-scenario support.
# ----------------------------------------------------------------------------

def feature_matrix(samples: Iterable[TransitionSample]) -> tuple[np.ndarray, np.ndarray]:
    """Stack samples into (X, y) matrices using emergency_resource_v0 vocab.

    DEPRECATED for cross-scenario use: this function assumes the emergency
    vocab (7 action types, 4 agents, 6 nodes). For other scenarios, use
    ``DataLoader.feature_matrix()`` instead.

    X shape: (N, 15), y shape: (N, 5).
    """
    loader = _EmergencyCompatLoader()
    return loader.feature_matrix(samples)


class _EmergencyCompatLoader:
    """Minimal loader exposing feature_matrix() with emergency vocab."""

    action_types = EMERGENCY_ACTION_TYPES
    agent_ids = EMERGENCY_AGENT_IDS
    node_ids = EMERGENCY_NODE_IDS
    _n_actions = len(action_types)
    _n_agents = len(agent_ids)
    _n_nodes = len(node_ids)
    _feature_dim = 1 + _n_actions + _n_agents + 3

    def feature_matrix(
        self,
        samples: Iterable[TransitionSample],
    ) -> tuple[np.ndarray, np.ndarray]:
        sample_list = list(samples)
        if not sample_list:
            return (
                np.zeros((0, self._feature_dim), dtype=np.float64),
                np.zeros((0, 5), dtype=np.float64),
            )
        # Reuse DataLoader's _sample_to_feature logic via duck-typing.
        loader = DataLoader.__new__(DataLoader)
        loader.action_types = self.action_types
        loader.agent_ids = self.agent_ids
        loader.node_ids = self.node_ids
        loader._n_actions = self._n_actions
        loader._n_agents = self._n_agents
        loader._n_nodes = self._n_nodes
        loader._feature_dim = self._feature_dim
        return loader.feature_matrix(sample_list)
