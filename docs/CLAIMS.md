# CLAIMS (public evidence)

Every claim below is written so a reader can *verify it from this public
repository alone* (no development-repo-only material, e.g. M-series
protocols, pre-registered experiment scripts, or hand-labelled datasets —
those are intentionally **not** reproduced here).

## 1. Protocol claim

> Every state transition is deterministic, hash-chained, and can be
> replayed, branched, and compared counterfactually.

Evidence:
- `worldloop-kernel/src/worldloop_kernel/recorder.py` — transition record +
  hash chain implementation.
- `worldloop-kernel/tests/test_k06_validation_recorder.py` — recorder
  round-trip, hash-chain closure, invariant quarantine.
- `worldloop-kernel/tests/test_k07_replay_branch.py` — deterministic replay
  and counterfactual branch tests.
- `examples/quickstart.ipynb` — replay & counterfactual branch demo with
  two adjacent seeds side-by-side.

## 2. "Agents propose, world adjudicates" claim

> An LLM or policy only *proposes* candidate actions. Validity checks,
> conflict resolution, cost/resource settlement, and state write-back
> are all performed by deterministic rules.

Evidence:
- `worldloop-kernel/src/worldloop_kernel/protocol.py` — `WorldProtocol`
  (propose → validate → settle → write contract).
- `worldloop-kernel/src/worldloop_kernel/engine.py` — `ToyWorld`
  reference engine used in the quickstart.
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
- `worldloop-scenarios/src/worldloop_scenarios/spec.py` — `ScenarioSpec`
  v0 dataclass.
- `worldloop-scenarios/src/worldloop_scenarios/schemas/spec_v0.schema.json`
  — JSON-schema definitions.
- `worldloop-scenarios/src/worldloop_scenarios/validator.py` — semantic
  validation; all three showcase scenarios pass.
- `worldloop-scenarios/tests/test_compiler.py` — `test_compile_example`
  parametrized over `discrete_grid`, `continuous_field`,
  `graph_registry`, `emergency_resource`.
- `worldloop-scenarios/examples/invalid_missing_termination.yaml` — a
  deliberately invalid fixture that the validator must reject.

## 4. Dataset-export claim

> Transitions can be exported to a structured trajectory dataset
> (counterfactual branches retained, leakage checks reported, no
> training-gain claim asserted).

Evidence:
- `worldloop-data/src/worldloop_data/exporter.py` — dataset exporter.
- `worldloop-data/src/worldloop_data/leakage.py` — leakage checker
  (absolute paths / API keys / cache / PII).
- `worldloop-data/src/worldloop_data/quality.py` — quality reporter.
- `worldloop-data/tests/test_exporter.py` — discrete_grid.yaml →
  trajectory-dataset round-trip with leakage emitted.
- `examples/quickstart.ipynb` — exported dataset sample + counterfactual
  side-by-side.
