# Contributing to WorldLoop

Thanks for considering contributing! WorldLoop is a small, focused project
— the goal is to make **environment-authoritative, deterministic
simulation infrastructure** that researchers and infrastructure engineers
can trust, not to grow into a general agent framework.

## Code of Conduct

This project follows the [Contributor Covenant 2.1](./CODE_OF_CONDUCT.md).
Reported issues go to <1148395497@qq.com>.

## What kind of contributions are most useful right now

Before opening a large PR, it's usually best to open an issue first so we
can align the scope. Good first contributions:

1. **New scenario YAML files.** `worldloop-scenarios/examples/` contains
   the showcase ones. A new YAML that follows the spec in
   `worldloop_scenarios/schema/` and compiles cleanly against
   `worldloop-scenarios/tests/test_compiler.py` is a great addition.
2. **Bug reports that include a reproducible scenario** (a minimal YAML +
   seed + version that fails to build or that produces a suspect state
   transition).
3. **PettingZoo / Gymnasium adapter fixes or new external-environment
   wrappers** in `worldloop-adapters/`.
4. **Dataset-quality regressions** — if `worldloop-data` exports a
   trajectory that looks wrong under the invariants documented in the
   exporter, file an issue with the run config + seed.

Out of scope for this repository: LLM-prompt libraries, opinionated
agentic orchestrators, or product-specific domain rules — those are
downstream consumers of the kernel, not part of WorldLoop's core mandate.

## Dev setup

WorldLoop ships as four sibling Python packages in one flat monorepo.
Install them in dependency order (editable so changes are picked up
immediately):

```bash
python -m pip install -e ./worldloop-kernel
python -m pip install -e ./worldloop-scenarios
python -m pip install -e ./worldloop-data
python -m pip install -e ./worldloop-adapters
python -m pip install pytest
```

Then run the four suites to confirm a green baseline:

```bash
pytest worldloop-kernel/tests -q
pytest worldloop-scenarios/tests -q
pytest worldloop-adapters/tests -q
pytest worldloop-data/tests -q
```

The CI (`.github/workflows/ci.yml`) runs exactly the same four commands on
Ubuntu + Windows × Python 3.10 / 3.12. If those four lines stay green on
your branch, you're almost done.

## PR checklist

- [ ] All four `pytest …/tests -q` suites pass locally on a clean venv.
- [ ] New scenario YAMLs compile via `worldloop-scenarios/tests/test_compiler.py`
      (or explain why a compiler tweak is needed).
- [ ] No new cross-layer imports that break the
      kernel → scenarios → adapters → data direction.
- [ ] Any claim added to the README or docs is reproducible from this
      public repository alone; when in doubt, add a fixture + a test.
