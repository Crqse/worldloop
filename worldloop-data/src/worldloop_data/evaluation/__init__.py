"""worldloop_data.evaluation — M6 data value evaluation toolkit.

This subpackage implements the L-01 ~ L-08 evaluation pipeline described in
``docs/07.advice/2026-07-26_WorldLoop_v1封板与v2小核心实施总方案.md`` §16.

Public API:
    - FeatureLayout: block layout of feature matrices (audit F-02 / R3 fix).
    - TransitionSample: featureized transition record for training/eval.
    - DataLoader: load transitions.jsonl + splits.json into samples.
    - Baseline models: Persistence / MeanDelta / LinearRidge / XGBoostBaseline.
    - Metrics: mae, accuracy, ndcg_at_k.

M8 (Beta B5) additions:
    - grouped_split: group-aware split + leakage assertions (seed /
      episode family / counterfactual fork point never cross splits).
    - action_ranking: held-out candidate-action ranking task + metrics.
    - treatment_comparison: A/B/C/D/D-matched treatment construction +
      paired comparison statistics (bootstrap CI, verdicts).

R2 integration (audit F-03):
    - EncoderSchema: declares per-block dims for the state encoder.
    - StateEncoder: converts a StateView dict into StateFeatures.
    - StateMaterializer: materializes TrainingTransition objects from a
      transition sequence + initial state view.
    - TrainingTransition: full ``S_t + A_t + U_t -> S_{t+1}`` training
      sample with mechanical provenance.
    - StateFeatures / JointActionFeatures / ExogenousFeatures: block-wise
      feature containers produced by the materializer.
"""

from worldloop_data.evaluation.feature_layout import (
    FeatureLayout,
    EMERGENCY_RESOURCE_V0_LAYOUT,
)
from worldloop_data.evaluation.data_loader import (
    DataLoader,
    TransitionSample,
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
from worldloop_data.evaluation.metrics import (
    mae,
    accuracy,
    ndcg_at_k,
)
from worldloop_data.evaluation.grouped_split import (
    GroupKey,
    GroupLeakageError,
    episode_family,
    is_branch_episode,
    group_key_for_sample,
    grouped_split,
    assert_no_group_leakage,
    assert_branch_siblings_together,
)
from worldloop_data.evaluation.action_ranking import (
    RankingGroup,
    RankingMetrics,
    load_energy_outcomes,
    attach_energy_outcomes,
    build_ranking_groups,
    evaluate_ranking,
)
from worldloop_data.evaluation.treatment_comparison import (
    TREATMENT_NAMES,
    is_counterfactual_sample,
    prefix_subset,
    downsample_matched,
    build_treatments,
    PairedStats,
    paired_stats,
    verdict_from_stats,
)
from worldloop_data.evaluation.state_materializer import (
    EncoderSchema,
    StateEncoder,
    StateMaterializer,
    StateFeatures,
    JointActionFeatures,
    ExogenousFeatures,
    TrainingProvenance,
    TrainingTransition,
    MaterializerError,
    StateBlockType,
)

__all__ = [
    "FeatureLayout",
    "EMERGENCY_RESOURCE_V0_LAYOUT",
    "DataLoader",
    "TransitionSample",
    "BaselineModel",
    "PersistenceBaseline",
    "MeanDeltaBaseline",
    "LinearRidgeBaseline",
    "XGBoostBaseline",
    "NoActionBaseline",
    "ShuffledActionBaseline",
    "OracleUpperBound",
    "mae",
    "accuracy",
    "ndcg_at_k",
    # M8 (Beta B5)
    "GroupKey",
    "GroupLeakageError",
    "episode_family",
    "is_branch_episode",
    "group_key_for_sample",
    "grouped_split",
    "assert_no_group_leakage",
    "assert_branch_siblings_together",
    "RankingGroup",
    "RankingMetrics",
    "load_energy_outcomes",
    "attach_energy_outcomes",
    "build_ranking_groups",
    "evaluate_ranking",
    "TREATMENT_NAMES",
    "is_counterfactual_sample",
    "prefix_subset",
    "downsample_matched",
    "build_treatments",
    "PairedStats",
    "paired_stats",
    "verdict_from_stats",
    # R2 integration (audit F-03)
    "EncoderSchema",
    "StateEncoder",
    "StateMaterializer",
    "StateFeatures",
    "JointActionFeatures",
    "ExogenousFeatures",
    "TrainingProvenance",
    "TrainingTransition",
    "MaterializerError",
    "StateBlockType",
]
