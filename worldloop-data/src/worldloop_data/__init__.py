"""worldloop-data: WorldLoop v2 Phase G (M4) autonomous trajectory production.

This package is the orchestration layer that composes six components on
top of ``worldloop_kernel`` primitives to turn a
``worldloop_scenarios.ScenarioPackage`` into a published, leak-checked,
quality-scored dataset:

- :mod:`worldloop_data.policy`         — S-08 Policy Pool
- :mod:`worldloop_data.coverage`       — S-09 Coverage Scheduler
- :mod:`worldloop_data.counterfactual` — S-10 Counterfactual Branch Scheduler
- :mod:`worldloop_data.exporter`       — S-11 Dataset Exporter
- :mod:`worldloop_data.leakage`        — S-12 Leakage Checker
- :mod:`worldloop_data.quality`        — S-13 Quality Reporter
- :mod:`worldloop_data.rollout`        — single-episode orchestrator
- :mod:`worldloop_data.pipeline`       — end-to-end pipeline

Design rules (per main plan §14 and ADR §3):
- Depends only on ``worldloop_kernel`` / ``worldloop_scenarios`` /
  ``worldloop_adapters``. Zero ``current/worldloop/core/*`` imports.
- Reuses kernel ``TransitionRecorder`` / ``replay`` / ``branch`` /
  ``validate_transition`` / ``hash_state`` / ``diff_state`` / ``apply_delta``.
- Every component is a ``typing.Protocol``; concrete classes are
  reference stubs, not the only allowed implementations.
"""

from __future__ import annotations

__version__ = "0.1.3"

from worldloop_data.config import (
    PolicyPoolConfig,
    CoverageConfig,
    CounterfactualConfig,
    ExporterConfig,
    LeakageConfig,
    QualityConfig,
    RolloutConfig,
    PipelineConfig,
)
from worldloop_data.policy import (
    Policy,
    PolicyContext,
    PolicyPool,
    RandomPolicy,
    ScriptedPolicy,
    FrozenReplayPolicy,
    LLMPolicyStub,
    AdversarialPolicy,
    PlannerPolicyStub,
)
from worldloop_data.coverage import (
    CoverageScheduler,
    CoverageReport,
    UniformCoverageScheduler,
)
from worldloop_data.counterfactual import (
    CounterfactualBranchScheduler,
    BranchSpec,
    JointBranchSpec,
    NoOpBranchScheduler,
    KernelBranchScheduler,
    JointKernelBranchScheduler,
    HeldFixedVerification,
)
from worldloop_data.exporter import (
    DatasetExporter,
    ExportSplit,
    ExportResult,
    PlainDatasetExporter,
)
from worldloop_data.leakage import (
    LeakageChecker,
    LeakageReport,
    LeakageViolation,
    TrivialLeakageChecker,
)
from worldloop_data.quality import (
    QualityReporter,
    QualityReport,
    MinimalQualityReporter,
)
from worldloop_data.rollout import (
    RolloutResult,
    run_rollout,
    run_joint_rollout,
)
from worldloop_data.pipeline import (
    PipelineResult,
    run_pipeline,
)
from worldloop_data.utility import (
    OutcomeUtility,
    PolicyOutcome,
    UtilityComparison,
    UtilityEvaluationReport,
    evaluate_matched_policy_utility,
)
from worldloop_data.llm_policy import (
    LLMPolicy,
    LLMPolicyConfig,
    LLMClient,
    LLMRequest,
    LLMResponse,
    FakeLLMClient,
    EchoLLMClient,
    build_llm_prompt,
    InMemoryTelemetrySink,
    InferenceEvent,
    TelemetrySink,
    TelemetrySinkV2,
    OpenAICompatibleClient,
    LLMClientError,
    LLMTimeoutError,
    LLMRateLimitError,
    LLMServerError,
    LLMAuthError,
    LLMProtocolError,
)
from worldloop_data.prompt_contract import (
    STABLE_SYSTEM_PROMPT,
    SYSTEM_PROMPT_VERSION,
    PROMPT_CONTRACT_SCHEMA_VERSION,
    USER_MESSAGE_SCHEMA_VERSION,
    FORBIDDEN_GLOBAL_FIELDS,
    ScenarioContract,
    ActionSchemaEntry,
    PromptHashBundle,
    LLMRequestLike,
    build_scenario_contract,
    build_user_message,
    build_llm_request,
    hash_system_prompt,
    hash_scenario_contract,
    hash_user_message,
    compute_prompt_hashes,
    scan_for_forbidden_fields,
    validate_prompt_components,
)
from worldloop_data.telemetry import (
    TELEMETRY_SCHEMA_VERSION,
    RunTier,
    ValidationSummary,
    InferenceAttemptEvent,
    InferenceDecisionEvent,
    RunLevelConfig,
    default_run_level_config,
    accumulate_attempt_tokens,
    build_decision_from_attempts,
    EvidenceFailClosedError,
    check_evidence_fail_closed,
    check_evidence_fail_closed_batch,
    hash_legal_action_space,
    hash_inference_config,
    hash_prompt_template,
    latency_split,
    derive_effective_backend,
)
from worldloop_data.rng_seeds import (
    derive_seed,
    derive_continuation_seed,
    derive_per_episode_seed,
    PROTOCOL_HASH_DEFAULT,
)

__all__ = [
    "__version__",
    # Config
    "PolicyPoolConfig",
    "CoverageConfig",
    "CounterfactualConfig",
    "ExporterConfig",
    "LeakageConfig",
    "QualityConfig",
    "RolloutConfig",
    "PipelineConfig",
    # S-08 Policy pool
    "Policy",
    "PolicyContext",
    "PolicyPool",
    "RandomPolicy",
    "ScriptedPolicy",
    "FrozenReplayPolicy",
    "LLMPolicyStub",
    "AdversarialPolicy",
    "PlannerPolicyStub",
    # S-09 Coverage
    "CoverageScheduler",
    "CoverageReport",
    "UniformCoverageScheduler",
    # S-10 Counterfactual (+ Phase 5 joint branch scheduling)
    "CounterfactualBranchScheduler",
    "BranchSpec",
    "JointBranchSpec",
    "NoOpBranchScheduler",
    "KernelBranchScheduler",
    "JointKernelBranchScheduler",
    "HeldFixedVerification",
    # S-11 Exporter
    "DatasetExporter",
    "ExportSplit",
    "ExportResult",
    "PlainDatasetExporter",
    # S-12 Leakage
    "LeakageChecker",
    "LeakageReport",
    "LeakageViolation",
    "TrivialLeakageChecker",
    # S-13 Quality
    "QualityReporter",
    "QualityReport",
    "MinimalQualityReporter",
    # Rollout + pipeline (+ Phase 5 joint rollout)
    "RolloutResult",
    "run_rollout",
    "run_joint_rollout",
    "PipelineResult",
    "run_pipeline",
    # Q9 matched outcome utility
    "OutcomeUtility",
    "PolicyOutcome",
    "UtilityComparison",
    "UtilityEvaluationReport",
    "evaluate_matched_policy_utility",
    # LLM Policy (B3)
    "LLMPolicy",
    "LLMPolicyConfig",
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "FakeLLMClient",
    "EchoLLMClient",
    "build_llm_prompt",
    "InMemoryTelemetrySink",
    "InferenceEvent",
    "TelemetrySink",
    "TelemetrySinkV2",
    "OpenAICompatibleClient",
    "LLMClientError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMAuthError",
    "LLMProtocolError",
    # Prompt contract (Phase 1 / Beta correction §5.4-5.6)
    "STABLE_SYSTEM_PROMPT",
    "SYSTEM_PROMPT_VERSION",
    "PROMPT_CONTRACT_SCHEMA_VERSION",
    "USER_MESSAGE_SCHEMA_VERSION",
    "FORBIDDEN_GLOBAL_FIELDS",
    "ScenarioContract",
    "ActionSchemaEntry",
    "PromptHashBundle",
    "LLMRequestLike",
    "build_scenario_contract",
    "build_user_message",
    "build_llm_request",
    "hash_system_prompt",
    "hash_scenario_contract",
    "hash_user_message",
    "compute_prompt_hashes",
    "scan_for_forbidden_fields",
    "validate_prompt_components",
    # Telemetry (Phase 2 / Beta correction §6)
    "TELEMETRY_SCHEMA_VERSION",
    "RunTier",
    "ValidationSummary",
    "InferenceAttemptEvent",
    "InferenceDecisionEvent",
    "RunLevelConfig",
    "default_run_level_config",
    "accumulate_attempt_tokens",
    "build_decision_from_attempts",
    "EvidenceFailClosedError",
    "check_evidence_fail_closed",
    "check_evidence_fail_closed_batch",
    "hash_legal_action_space",
    "hash_inference_config",
    "hash_prompt_template",
    "latency_split",
    "derive_effective_backend",
    # RNG seed derivation (Phase 4 / M8 deterministic correction §8.2-§8.4)
    "derive_seed",
    "derive_continuation_seed",
    "derive_per_episode_seed",
    "PROTOCOL_HASH_DEFAULT",
]
