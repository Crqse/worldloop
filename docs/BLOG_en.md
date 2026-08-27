# Why LLM agents should propose, not adjudicate

> A short, opinionated note on why we put the **world** back in charge, and
> what that buys for multi-agent simulation & research. Written for people
> building agent frameworks, MARL/simulation tooling, and synthetic-data
> pipelines.

- Repo: **[Crqse/worldloop](https://github.com/Crqse/worldloop)**
- Runnable 40-line gist: **[worldloop_min_demo.py](https://gist.github.com/Crqse/0ad906b3aa956133bd1391b5c7404796)**

## The one-sentence pitch

> **LLMs / policies only *propose*. A deterministic world *adjudicates* every
> consequence — and every change is a verifiable, hash-chained state diff that
> can be replayed, branched, and compared counterfactually.**

## The problem with "LLM as the world"

Most agent frameworks carry state in the chat history. The model is the
proposer **and** the judge **and** the record-keeper at once. That makes three
questions impossible to answer honestly:

1. **What actually happened?** — who proposed what, what actually executed,
   what was the settlement result?
2. **Can I replay it?** — can two runs reproduce tick-for-tick?
3. **Can I trace a number back to a transition?** — or is every metric just a
   vibe from a context window?

When the LLM both proposes and adjudicates, you cannot separate *what the
model wanted* from *what the system actually did*. You also cannot branch a
trajectory deterministically — because the "world" is a non-deterministic
string of tokens.

## WorldLoop's answer

WorldLoop inverts the layering:

```
observe S_t -> propose -> world validates & settles -> write S_{t+1}
-> verify hash & invariants -> record / replay / branch / export
```

- An LLM or policy can only **submit a candidate action**. It cannot write
  energy, position, resource counts, or life/death directly.
- The world performs **legality checks, conflict resolution, cost/resource
  settlement, and state write-back** — all with deterministic rules.
- Every transition is recorded with a state-diff + hash chain.

Because the world is deterministic and authoritative, a trajectory becomes a
**first-class, inspectable artifact**: you can replay it exactly, branch it at
any tick with a different action, and ask "what if agent_0 had moved right
instead of up at tick 4?"

## A 40-line taste

The full runnable gist: https://gist.github.com/Crqse/0ad906b3aa956133bd1391b5c7404796

```python
def run(seed, policy):
    env = make_simple_spread_env(n_agents=2, n_landmarks=2, max_cycles=25)
    adapter = PettingZooParallelAdapter(env=env, env_id="simple_spread_v3")
    adapter.reset(seed=seed)
    chain = []
    for tick in range(N_TICKS):
        proposal = ActionProposal(
            agent_id=AGENT, action_type="move",
            params={"discrete_action": policy(tick)},
            proposed_at_tick=tick, proposer="demo")
        executed, _ = adapter.validate_action(proposal)
        record = adapter.step(executed)   # other agents default to STAY
        chain.append(record.state_after_hash)
    return chain

# determinism
assert run(42, policy_a) == run(42, policy_a)          # replay identical
# counterfactual
assert run(42, policy_a)[4] != run(42, policy_b)[4]     # branch at tick 4
```

Output: `deterministic replay same? True`, `branches diverge by tick: [4,5,6,7]`.

## Why "another agent framework"? No.

WorldLoop is **not** a general agent framework. It is an
**environment-authoritative, verifiable state-transition substrate**. That has
three concrete use-cases:

1. **Deterministic replay & counterfactual studies** — the substrate that makes
   a claim about a *change* reproducible as a diff, not a vibe.
2. **Scenario-as-data** — scenarios are YAML, schema-validated, compiled
   before run (invalid ones rejected at compile time, not mid-run).
3. **Trajectory exports without leakage** — transitions export to structured
   datasets with a leakage report, so you can reason about what you're
   actually handing to a downstream trainer.

## Honest limits (we report negative results)

- This is a **research-grade** beta, not a production battle-tested system.
- The M8 counterfactual data-value result is a **negative result**: at this
  scale, matched counterfactuals did not measurably improve downstream
  trajectory-dataset value. We report it openly rather than spin it.
- Exact state-restore for *arbitrary* environments is only declared where it
  is mechanically verified (currently the MPE2 Simple Spread / Simple Tag
  family). We never claim a capability we have not proven.
- No "training-gain" claim is asserted anywhere. We release the *mechanism*
  (replay, branching, serialization, leakage reporting), not a headline number.

## What's in the box

Four sibling packages, one flat repo, all green on Ubuntu + Windows x Python
3.10 / 3.12 via GitHub Actions `ci.yml`:

| package | role |
|---|---|
| `worldloop-kernel` | protocol, state, action, transition, recorder, hash chain, replay, branch, checkpoint |
| `worldloop-scenarios` | ScenarioSpec v0 YAML schema -> validator -> compiler -> parametrized world |
| `worldloop-adapters` | PettingZoo Parallel / Gymnasium / OpenEnv -> kernel `WorldProtocol` |
| `worldloop-data` | coverage scheduler, counterfactual brancher, exporter, leakage checker, quality reporter |

## Get started

```bash
git clone https://github.com/Crqse/worldloop.git
python -m pip install -e ./worldloop-kernel -e ./worldloop-scenarios \
    -e ./worldloop-data -e "worldloop-adapters[dev]"
pytest worldloop-kernel/tests worldloop-scenarios/tests \
       worldloop-adapters/tests worldloop-data/tests -q
```

---

*If you are building "agents that act", the most underrated engineering
decision is: who gets to be the judge. We think the environment should be.
Open issues, or email 1148395497@qq.com — happy to talk about paid integration
support, technical support subscriptions, or custom development.*
