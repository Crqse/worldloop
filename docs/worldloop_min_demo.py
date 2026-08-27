"""WorldLoop gist demo: agents propose, the world adjudicates (PettingZoo).

Run:
    pip install worldloop-kernel "worldloop-adapters[pettingzoo]"
    python worldloop_min_demo.py
"""
from worldloop_adapters.pettingzoo import (
    PettingZooParallelAdapter,
    make_simple_spread_env,
)
from worldloop_kernel.action import ActionProposal

N_TICKS = 8
SEED = 42
AGENT = "agent_0"


def run(seed, policy):
    """Run N ticks of Simple Spread under WorldLoop; return the per-tick
    state hash chain (the 'trajectory fingerprint')."""
    env = make_simple_spread_env(n_agents=2, n_landmarks=2, max_cycles=25)
    adapter = PettingZooParallelAdapter(
        env=env, env_id="simple_spread_v3", run_id=f"gist-seed{seed}"
    )
    adapter.reset(seed=seed)
    chain = []
    for tick in range(N_TICKS):
        disc = policy(tick)
        proposal = ActionProposal(
            agent_id=AGENT,
            action_type="move",
            params={"discrete_action": disc},
            proposed_at_tick=tick,
            proposer="gist-demo",
        )
        executed, receipt = adapter.validate_action(proposal)
        record = adapter.step(executed)  # other agents default to STAY
        chain.append(record.state_after_hash)
    return chain


def policy_a(tick):
    # cycle STAY(0) / LEFT(1) / RIGHT(2) / DOWN(3) / UP(4)
    return tick % 5


def policy_b(tick):
    # counterfactual: at tick 4 agent_0 turns RIGHT(2) instead of UP(4)
    return 2 if tick == 4 else tick % 5


def main():
    # 1) determinism: same seed + same policy -> identical hash chain
    a1 = run(SEED, policy_a)
    a2 = run(SEED, policy_a)
    print("deterministic replay same? ", a1 == a2)
    print("chain length              ", len(a1))

    # 2) counterfactual: same seed, one different action at tick 4 -> branch
    b = run(SEED, policy_b)
    print("tick 4 diverging?         ", a1[4] != b[4])
    print("final hash (policy A)", a1[-1])
    print("final hash (policy B)", b[-1])
    print("final diverges?       ", a1[-1] != b[-1])
    print("branches diverge by tick:", [t for t in range(N_TICKS) if a1[t] != b[t]])


if __name__ == "__main__":
    main()
