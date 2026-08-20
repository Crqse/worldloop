# worldloop-data

WorldLoop v2 Phase G (M4) — Autonomous trajectory production pipeline.

This package provides the orchestration layer that turns a
`worldloop_scenarios.ScenarioPackage` into a published, leak-checked,
quality-scored dataset. It composes six components on top of the
`worldloop_kernel` primitives (`TransitionRecorder`, `replay`, `branch`,
`validate_transition`, `hash_state`, `diff_state`, `apply_delta`):

| Component | Module | Symbol | M4 checklist |
|---|---|---|---|
| Policy Pool | `worldloop_data.policy` | `Policy` Protocol, `PolicyPool`, `RandomPolicy`, `ScriptedPolicy`, `FrozenReplayPolicy` | S-08 |
| Coverage Scheduler | `worldloop_data.coverage` | `CoverageScheduler` Protocol, `UniformCoverageScheduler` | S-09 |
| Counterfactual Branch Scheduler | `worldloop_data.counterfactual` | `CounterfactualBranchScheduler` Protocol, `NoOpBranchScheduler`, `KernelBranchScheduler` | S-10 |
| Dataset Exporter | `worldloop_data.exporter` | `DatasetExporter` Protocol, `PlainDatasetExporter` | S-11 |
| Leakage Checker | `worldloop_data.leakage` | `LeakageChecker` Protocol, `TrivialLeakageChecker` | S-12 |
| Quality Reporter | `worldloop_data.quality` | `QualityReporter` Protocol, `MinimalQualityReporter` | S-13 |

On top of the M4 base, 0.1.2 (beta candidate) adds:

| Capability | Module | Key symbols |
|---|---|---|
| Real LLM policy | `worldloop_data.llm_policy` | `LLMPolicy`, `LLMPolicyConfig`, `OpenAICompatibleClient` (plus `FakeLLMClient`/`EchoLLMClient` for tests); typed client errors; fail-closed in evidence runs |
| Prompt contract | `worldloop_data.prompt_contract` | versioned neutral system prompt, `ScenarioContract`, `build_user_message`, `compute_prompt_hashes`, forbidden-global-field scan |
| Per-attempt telemetry | `worldloop_data.telemetry` | `InferenceAttemptEvent` / `InferenceDecisionEvent`, run tiers (`RunTier`), `check_evidence_fail_closed*` (EVIDENCE tier forbids fallback origins), token/latency accounting |
| Seed governance | `worldloop_data.rng_seeds` | derived per-component seeds recorded in manifests (no shared global RNG) |
| Joint-action rollout | `worldloop_data.rollout` / `.counterfactual` | `run_joint_rollout`, `JointBranchSpec`, `JointKernelBranchScheduler`, `HeldFixedVerification` — all-agents-per-tick recording with held-fixed counterfactual branches; validated only on the adapter's 2 exact-restore-verified MPE env families |
| Evaluation (M8) | `worldloop_data.evaluation` | counterfactual data-value evaluation harness (fork-group aware splits, Q1/Q7 mechanical gates, paired A–E condition runs); requires the `[evaluation]` extra (numpy); xgboost is optional (lazy import) |

The end-to-end pipeline (`worldloop_data.pipeline.run_pipeline`) runs:

```
scenario → policy pool → coverage-driven run → counterfactual branch
        → dataset export → leakage check → quality report
```

## Design rules (per main plan §14 and ADR §3)

- **No `current/worldloop/core/*` imports.** This package depends only on
  `worldloop_kernel`, `worldloop_scenarios`, and `worldloop_adapters`.
- **Reuse kernel primitives.** Recording, validation, replay, and
  counterfactual branching reuse the kernel implementations; this package
  is the orchestration layer, not a re-implementation.
- **Protocol-first.** Every component is a `typing.Protocol`; concrete
  stubs are minimal references for testing and as a contract template.
- **No hard-coded policy enumeration.** The `Policy` Protocol is open;
  users register policies by passing instances into `PolicyPool`.
- **Evidence-grade honesty.** EVIDENCE-tier runs fail closed: any
  fallback-origin decision aborts the run instead of silently degrading;
  mock/echo backends are labeled as such in telemetry and can never be
  reported as real-model evidence.

## Installation

```powershell
cd current\worldloop-data
python -m pip install -e .
python -m pip install -e ".[dev]"
```

## Smoke test

```powershell
cd current\worldloop-data
python -m pytest tests -q
```

## Status

- **Version**: 0.1.3 (beta candidate, beta.4)
- M4 Phase G complete; 0.1.2 layers on Beta-correction Phases 2–6: real
  `LLMPolicy` + prompt contract + per-attempt telemetry (B-series), M8
  evaluation harness, and Phase 5 joint-action rollout.
- **Boundaries (do not over-claim):** joint mode is engineering-validated
  on 2 MPE env families only (`mpe2/simple_spread_v3`,
  `mpe2/simple_tag_v3`); the M8 fresh confirmatory run on
  `emergency_resource_v1` returned a NEGATIVE data-value result for the
  matched-budget counterfactual condition — this package makes no claim
  of counterfactual training gains.
