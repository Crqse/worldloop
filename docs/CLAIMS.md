# CLAIMS (public evidence)

Every claim below is written so a reader can *verify it from this public
repository alone* (no development-repo-only material, e.g. M-series
protocols, pre-registered experiment scripts, or hand-labelled datasets —
those are intentionally **not** reproduced here).

## 1. Protocol claim

> Every state transition is deterministic, hash-chained, and can be
> replayed, branched, and compared counterfactually.

Evidence:
- `worldloop-kernel/src/worldloop_kernel/recorder/` — transition record +
  hash chain implementation.
- `worldloop-kernel/tests/test_recorder.py` — round-trip write / load /
  replay tests.
- `examples/quickstart.ipynb` — replay & counterfactual branch demo with
  two adjacent seeds side-by-side.

## 2. "Agents propose, world adjudicates" claim

> An LLM or policy only *proposes* candidate actions. Validity checks,
> conflict resolution, cost/resource settlement, and state write-back
> are all performed by deterministic rules.

Evidence:
- `worldloop-kernel/src/worldloop_kernel/runtime/` — the proposal →
  validate → settle → write pipeline.
- `worldloop-scenarios/examples/discrete_grid.yaml` — example rule set
  (forage / move / rest) as a public readable spec.
- `worldloop-scenarios/tests/test_parameterized_world.py` — parametrized
  rule-execution contract on discrete_grid.yaml, continuous_field.yaml,
  graph_registry.yaml.

## 3. Scenario schema claim

> Scenarios are YAML-described, schema-validated, and compiled to a
> `ScenarioPackage` before use; invalid scenarios are rejected at
> compile time, not mid-run.

Evidence:
- `worldloop-scenarios/src/worldloop_scenarios/schema/` — `ScenarioSpec`
  v0 schema + JSON-schema definitions.
- `worldloop-scenarios/tests/test_validator.py` — all three showcase
  scenarios pass semantic validation.
- `worldloop-scenarios/tests/test_compiler.py` — `test_compile_yaml_file`
  parametrized over `discrete_grid`, `continuous_field`,
  `graph_registry`, `emergency_resource`.
- `worldloop-scenarios/examples/invalid_missing_termination.yaml` — a
  deliberately invalid fixture that the validator must reject
  (`test_spec_v0.test_invalid_spec_rejected`).

## 4. Dataset-export claim

> Transitions can be exported to a structured trajectory dataset
> (counterfactual branches retained, leakage checks reported, no
> training-gain claim asserted).

Evidence:
- `worldloop-data/src/worldloop_data/exporter/` — exporter, leakage
  checker, quality reporter.
- `worldloop-data/tests/test_exporter.py` — discrete_grid.yaml →
  trajectory-dataset round-trip with leakage emitted.
- `examples/quickstart.ipynb` — exported dataset sample + counterfactual
  side-by-side.
