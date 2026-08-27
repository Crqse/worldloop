# worldloop-adapters

WorldLoop v2 external environment adapters: wrap PettingZoo Parallel, Gymnasium, and OpenEnv environments as kernel `WorldProtocol` implementations.

## Scope

This package provides adapters that let the independent `worldloop-kernel` drive external RL ecosystem environments. The kernel records and verifies transitions; the adapter translates between the external environment's API and the kernel's `WorldProtocol`.

**In scope (M2 / Phase E, complete):**
- `PettingZooParallelAdapter` — wrap PettingZoo Parallel API environments (A-01)
- Simple Spread conformance (A-02)
- Simple Tag conformance (A-03)
- At least one non-MPE environment conformance (A-04)
- `GymnasiumAdapter` — wrap Gymnasium single-agent environments (A-05)
- `OpenEnvWorldAdapter` — wrap OpenEnv environments (A-06, A-07)
- Capability reconciliation report (A-08)
- Independent `worldloop-adapters` package (A-09)
- M2 Gate (A-10)

**In scope (Phase 5, 0.1.2):**
- Joint action mode: `validate_joint_action()` / `step_joint()` on the
  PettingZoo adapter — every active agent submits a proposal the same tick;
  per-agent receipts/rewards/termination/truncation flags are recorded.
  The legacy single-focal `step()` remains available as the explicitly
  labeled *sequential compatibility mode* (others forced to STAY).
- Exact-restore capability layering: `EXACT_RESTORE_VERIFIED_ENV_FAMILIES`
  allowlist. **Only two env families are mechanically verified**
  (`mpe2/simple_spread_v3`, `mpe2/simple_tag_v3`, via
  checkpoint→immediate-restore hash equality probes). Generic PettingZoo
  envs get `exact_restore=False` until verified per family — the adapter
  never claims capabilities it has not proven.

**Out of scope:**
- learned/neural world adapters (E5, deferred to v2 follow-up)
- MCP / database tool environments (E4, deferred to M2.5)
- LLM calls (adapters never call LLMs; policies live outside the kernel)
- Generic PettingZoo joint-policy support beyond the verified allowlist
  (joint mode is validated on 2 MPE env families only; no claim of
  cross-domain generalization or external multi-agent training gains)

## Architecture

```
external env (PettingZoo / Gymnasium / OpenEnv)
                │
                ▼
   worldloop_adapters.<env>Adapter  (implements WorldProtocol)
                │
                ▼
        worldloop-kernel          (records + validates transitions)
```

**Dependency direction (hard constraint, per main plan §3.3):**
- `worldloop-adapters` MAY import `worldloop_kernel`.
- `worldloop-adapters` MAY import `pettingzoo` / `gymnasium` / `mpe2` / `numpy` (versioned).
- `worldloop-adapters` MUST NOT import `current.worldloop.core.*` (v1 five-layer).
- `worldloop-adapters` MUST NOT call LLMs.

## Installation

```bash
# Editable install with PettingZoo support
pip install -e ".[pettingzoo,dev]"

# Or with all extras
pip install -e ".[pettingzoo,gymnasium,dev]"
```

## Usage

```python
from worldloop_kernel import ActionProposal
from worldloop_adapters.pettingzoo import PettingZooParallelAdapter, make_simple_spread_env

env = make_simple_spread_env(n_agents=2, n_landmarks=2, max_cycles=25)
adapter = PettingZooParallelAdapter(env=env, env_id="simple_spread_v3")
adapter.reset(seed=42)

state = adapter.observe()
space = adapter.legal_actions(agent_id="agent_0")

# A policy (or LLM) submits a candidate action; the adapter validates it.
proposal = ActionProposal(
    agent_id="agent_0", action_type="move",
    params={"discrete_action": 0},          # one of space.legal_actions
    proposed_at_tick=state.meta.tick, proposer="readme-example",
)
executed, receipt = adapter.validate_action(proposal)
record = adapter.step(executed)          # sequential compatibility mode
```

Joint action mode (Phase 5):

```python
from worldloop_kernel import JointAction

joint = JointAction(tick=state.meta.tick, active_agents=("agent_0", "agent_1"),
                    proposals_by_agent={...}, missing_agent_policy="stay")
executed_joint, joint_receipt = adapter.validate_joint_action(joint)
record = adapter.step_joint(executed_joint)      # all agents same tick
```

## M2 Gate (§12.7)

The M2 Gate validates that adapters correctly implement the kernel `WorldProtocol` across at least 3 environment classes. See `tests/` for conformance test coverage (M2 Gate 10/10 PASS, 158 tests).

## Version

0.1.3 — M2 Phase E complete (A-01..A-10, M2 Gate PASS), Phase 5 joint action + exact-restore verified allowlist.
