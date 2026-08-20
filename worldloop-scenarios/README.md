# worldloop-scenarios

WorldLoop v2 Phase F (M3): ScenarioSpec v0 — schema-driven world definition for the `worldloop-kernel`.

## What this package does

Lets you define a new world by writing a YAML/JSON spec — **no Python code changes required**. The spec is validated by JSON Schema + static semantic checks, then compiled into a `ParameterizedWorld` that implements `worldloop_kernel.WorldProtocol`.

```
spec.yaml  →  schema validation (S-02)  →  semantic validator (S-03)  →  compiler (S-04)  →  ParameterizedWorld  →  kernel
```

## Spec v0 structure (11 sections)

| Section | Required | Description |
|---|---|---|
| `scenario` | yes | Metadata (id, version, description, author, tags) |
| `time` | yes | Time model (dt, max_ticks, deterministic) |
| `space` | yes | Topology: `discrete` / `continuous` / `graph` |
| `fields` | no | WST-compatible field channels |
| `entities` | yes | Entity table schema + spawn template |
| `relations` | no | Relation graph schema (WorldGraph) |
| `registries` | no | Object/tool/artifact registries |
| `actions` | yes | Action definitions (effects, cost, preconditions) |
| `exogenous` | no | Exogenous events applied before actions |
| `termination` | yes | Stop conditions (≥1 required) |
| `data` | no | M4 data production hooks (placeholder in M3) |

See `examples/` for working specs:
- `discrete_grid.yaml` — minimal discrete grid (S-05 template)
- `continuous_field.yaml` — continuous box with resource field (S-06 template)
- `graph_registry.yaml` — graph + registry (S-07 template)
- `invalid_missing_termination.yaml` — invalid spec for testing
- `emergency_demo.yaml` — four-role emergency scheduling showcase (public README demo)

Showcase assets (built from the `emergency_demo` scenario):
- `demo/emergency_demo.py` — four-role policy + ASCII-animated CLI demo
- `make_emergency_gif.py` / `make_emergency_web.py` — build the README GIF
  and the self-contained interactive web demo
- `assets/single_step_chain.svg`, `assets/emergency_scheduling.gif`,
  `assets/emergency_scheduling.html` — generated visual assets
- `quickstart.ipynb` — 5-minute notebook (observe / propose / adjudicate /
  reject / replay / counterfactual branch)

## Install

```powershell
cd current\worldloop-scenarios
python -m pip install -e .
```

## Use

```python
import yaml
from worldloop_scenarios.spec import ScenarioSpec
from worldloop_scenarios.schema_loader import validate_against_schema

with open("examples/discrete_grid.yaml") as f:
    spec_dict = yaml.safe_load(f)

# Layer 1: JSON Schema validation (syntactic)
validate_against_schema(spec_dict)

# Build ScenarioSpec dataclass
spec = ScenarioSpec.from_dict(spec_dict)

# Stable hash of structure + dynamics (excludes metadata)
print(spec.world_parameters_hash())
```

## Status

- **Version**: 0.1.3 (beta candidate, beta.4)
- **Phase**: M3 complete (S-01/02/03/04/05/06/07/14/15 done); compiled
  `ParameterizedWorld` is consumed by `worldloop-data` pipelines (M4+)
- **0.1.2 additions (Beta correction Phase 1)**: compiled worlds expose the
  kernel per-agent observation path (`observe_agent` via the kernel
  `ObservationProjector` / `AgentObservationView`), so scenario-driven runs
  can record per-agent views instead of full state only
- **Out of scope**: LLM auto-fix, codegen; data production lives in
  `worldloop-data`, not here
- **Dependencies**: `worldloop-kernel`, `pyyaml`, `jsonschema`
- **No v1 imports**: this package does NOT import `current/worldloop/core/*`
