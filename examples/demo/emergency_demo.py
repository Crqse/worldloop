"""Emergency scheduling demo — a four-role policy on the graph world.

This script is a *showcase* for the public README (and doubles as the
"just run it" CLI demo). It drives ``examples/emergency_demo.yaml`` with
four role policies and prints a live, ASCII-animated view of the world:

    leader  -> repairs damaged facilities, pushes hazard down
    gatherer-> collects resources
    comms   -> builds the communication network
    patrol  -> keeps moving through the zones

Determinism: every tick a fixed hazard escalation (+0.2) is injected as an
exogenous input, so two runs with the same seed are identical tick-by-tick
(state hashes match).

Usage:
    python examples/demo/emergency_demo.py [--ticks 40] [--seed 42]
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

from worldloop_kernel.action import ActionProposal, ExogenousInput
from worldloop_scenarios import compile_file

EXAMPLE_DIR = Path(__file__).resolve().parent.parent
SCENARIO = EXAMPLE_DIR / "emergency_demo.yaml"
HAZARD_RATE = 0.25  # deterministic exogenous escalation per tick
PATROL_CYCLE = ["base", "zone_a", "zone_b", "zone_c", "zone_d"]
MAINTENANCE_NODE = "zone_b"  # leader holds this node once facilities are healthy

# role -> (agent slot, label, color hint)
ROLES = {
    "e0": ("leader", "L", "leader"),
    "e1": ("gatherer", "G", "gatherer"),
    "e2": ("comms", "C", "comms"),
    "e3": ("patrol", "P", "patrol"),
}


def build_adjacency(spec) -> dict[str, list[str]]:
    """Adjacency list from the spec's graph topology."""
    adj: dict[str, list[str]] = {}
    for src, dst in spec.space.edges:
        adj.setdefault(src, []).append(dst)
        adj.setdefault(dst, []).append(src)
    return adj


def nearest_step(node: str, target: str, adj: dict[str, list[str]]) -> str:
    """Pick the neighbour of ``node`` that is one BFS step closer to ``target``."""
    if node == target:
        return node
    # BFS from target to compute distances.
    dist = {target: 0}
    queue = deque([target])
    while queue:
        cur = queue.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    # Choose neighbour minimising distance to target.
    best, best_d = node, dist.get(node, 10**9)
    for nxt in adj.get(node, []):
        d = dist.get(nxt, 10**9)
        if d < best_d:
            best, best_d = nxt, d
    return best


class EmergencyDemo:
    """Runs the four-role policy and exposes a per-tick timeline."""

    def __init__(self, seed: int = 42) -> None:
        self.package = compile_file(SCENARIO)
        self.spec = self.package.spec
        self.world = self.package.world_factory(seed=seed)
        self.seed = seed
        self.adj = build_adjacency(self.spec)
        self.timeline: list[dict] = []

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def _act(self, eid: str, role: str, tick: int) -> ActionProposal:
        state = self.world.observe()
        ids = state.entities.ids
        cols = state.entities.columns
        idx = ids.index(eid)
        node = cols["node"][idx]
        energy = float(cols["energy"][idx])
        registry = (
            list(state.registries.entries) if state.registries is not None else []
        )

        def proposal(action_type: str, params: dict) -> ActionProposal:
            return ActionProposal(
                agent_id=eid,
                action_type=action_type,
                params=params,
                proposed_at_tick=tick,
                proposer=f"demo:{role}",
            )

        # Survival first: low energy -> rest and recover.
        if energy < 4.0:
            return proposal("REST", {})

        if role == "leader":
            # 1) Repair any damaged facility (walk to it first).
            damaged = [
                e for e in registry
                if e.registry_type == "facility" and e.state == "damaged"
            ]
            if damaged:
                target_node = str(damaged[0].metadata.get("node", ""))
                if node == target_node:
                    return proposal("REPAIR", {"entry_id": damaged[0].entry_id})
                nxt = nearest_step(node, target_node, self.adj)
                return proposal("MOVE", {"target_node": nxt})
            # 2) Facilities healthy: hold the maintenance node and repair on
            #    a cycle — hazard (+0.25/tick) slightly outpaces repairs
            #    (-0.5 every 4th tick), producing a visible tug-of-war.
            if node != MAINTENANCE_NODE:
                nxt = nearest_step(node, MAINTENANCE_NODE, self.adj)
                return proposal("MOVE", {"target_node": nxt})
            if tick % 4 == 0:
                facility = next(
                    (e for e in registry if e.registry_type == "facility"), None
                )
                if facility is not None:
                    return proposal(
                        "REPAIR", {"entry_id": facility.entry_id}
                    )
            return proposal("REST", {})

        if role == "gatherer":
            # 1) Collect an available resource if standing on its node.
            avail = [
                e for e in registry
                if e.registry_type == "resource" and e.state == "available"
            ]
            if avail:
                res_node = str(avail[0].metadata.get("node", ""))
                if node == res_node:
                    return proposal(
                        "COLLECT",
                        {"entry_id": avail[0].entry_id, "amount": 2.0},
                    )
                nxt = nearest_step(node, res_node, self.adj)
                return proposal("MOVE", {"target_node": nxt})
            # 2) Everything collected: deliver at base, then patrol zones.
            if node != "base":
                nxt = nearest_step(node, "base", self.adj)
                return proposal("MOVE", {"target_node": nxt})
            if node == "base" and tick % 4 == 0:
                return proposal("REST", {})
            target = PATROL_CYCLE[(tick // 2) % len(PATROL_CYCLE)]
            nxt = nearest_step(node, target, self.adj)
            return proposal("MOVE", {"target_node": nxt})

        if role == "comms":
            # Build the communication network — one edge per pair of agents.
            connected = self._connected_pairs()
            others = [
                o for o in ids
                if o != eid
                and cols["alive"][ids.index(o)]
                and frozenset((eid, o)) not in connected
            ]
            if others and energy >= 3.0:
                return proposal("COMMUNICATE", {"target_agent": others[0]})
            # Network complete: patrol and rest on a cycle so energy stays
            # in a healthy band (REST alone would push energy to the ceiling).
            if tick % 4 == 0:
                return proposal("REST", {})
            target = PATROL_CYCLE[(tick // 2) % len(PATROL_CYCLE)]
            nxt = nearest_step(node, target, self.adj)
            return proposal("MOVE", {"target_node": nxt})

        # patrol: follow the node cycle.
        target = PATROL_CYCLE[tick % len(PATROL_CYCLE)]
        nxt = nearest_step(node, target, self.adj)
        return proposal("MOVE", {"target_node": nxt})

    def _connected_pairs(self) -> set[frozenset]:
        """Pairs of agents already connected by a communication edge."""
        state = self.world.observe()
        pairs: set[frozenset] = set()
        if state.relations is not None:
            for e in state.relations.edges:
                if e.edge_type == "communication":
                    pairs.add(frozenset((e.src, e.dst)))
        return pairs

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, max_ticks: int = 40) -> list[dict]:
        world = self.world
        for tick in range(max_ticks):
            state = world.observe()
            alive = [
                eid for eid in state.entities.ids
                if state.entities.columns["alive"][state.entities.ids.index(eid)]
            ]
            actions: list[tuple[str, str, str]] = []  # (agent, action, params)
            for i, eid in enumerate(alive):
                role = ROLES[eid][0] if eid in ROLES else "patrol"
                proposal = self._act(eid, role, tick)
                executed, _receipt = world.validate_action(proposal)
                actions.append(
                    (eid, executed.action_type, str(executed.params))
                )
                # Deterministic exogenous hazard escalation — injected exactly
                # once per tick (on the last agent's step), so the gauge
                # rises by HAZARD_RATE per tick regardless of agent count.
                exogenous = (
                    ExogenousInput(
                        tick=tick,
                        kind="hazard_escalation",
                        payload={"rate": HAZARD_RATE, "field": "hazard_level"},
                    )
                    if i == len(alive) - 1
                    else None
                )
                world.step(executed, exogenous=exogenous)
            # Snapshot the world AFTER this tick settled.
            after = world.observe()
            self.timeline.append(self._snapshot(after, actions, tick))
        return self.timeline

    @staticmethod
    def _snapshot(state, actions: list[tuple[str, str, str]], tick: int):
        cols = state.entities.columns
        ids = state.entities.ids
        registry = (
            list(state.registries.entries) if state.registries is not None else []
        )
        comm_edges = 0
        if state.relations is not None:
            comm_edges = sum(
                1 for e in state.relations.edges if e.edge_type == "communication"
            )
        hazard = None
        if state.fields is not None:
            hazard = state.fields.channels.get("hazard_level")
        return {
            "tick": tick,
            "agents": {
                ids[i]: {
                    "node": cols["node"][i],
                    "energy": float(cols["energy"][i]),
                    "load": float(cols["load"][i]),
                    "alive": bool(cols["alive"][i]),
                }
                for i in range(len(ids))
            },
            "registry": {e.entry_id: e.state for e in registry},
            "hazard": float(hazard) if hazard is not None else 0.0,
            "comm_edges": comm_edges,
            "actions": actions,
            "state_hash": hash_state_hex(state),
        }


def hash_state_hex(state) -> str:
    """Short, stable hash fingerprint of the state for the demo."""
    from worldloop_kernel.canonical import hash_state

    return str(hash_state(state))[:12]


# ---------------------------------------------------------------------------
# CLI (ASCII animation)
# ---------------------------------------------------------------------------


def _energy_bar(energy: float, width: int = 18) -> str:
    filled = max(0, min(width, int(energy / 30.0 * width)))
    return "█" * filled + "░" * (width - filled)


def render_cli(timeline: list[dict], sleep: float = 0.25) -> None:
    for frame in timeline:
        sys.stdout.write("\033c")  # clear screen (ANSI)
        tick = frame["tick"]
        hazard = frame["hazard"]
        print("=" * 70)
        print(f"  WorldLoop · emergency_demo · tick {tick:>3}  "
              f"hazard {hazard:+.2f}")
        print("=" * 70)
        hbar = "█" * max(0, min(50, int(hazard * 10)))
        print(f"  hazard : [{hbar:<50}] {hazard:+.2f}")
        print(f"  comm   : {frame['comm_edges']} edge(s)   "
              f"hash   : {frame['state_hash']}")
        print("-" * 70)
        for eid, role in (("e0", "leader"), ("e1", "gatherer"),
                          ("e2", "comms"), ("e3", "patrol")):
            info = frame["agents"].get(eid)
            if info is None:
                continue
            status = "ALIVE" if info["alive"] else " dead "
            bar = _energy_bar(info["energy"])
            print(f"  {role:<8} {eid} @ {info['node']:<6} "
                  f"E {bar} {info['energy']:5.1f}  {status}")
        reg = frame["registry"]
        fac = {k: v for k, v in reg.items() if k.startswith("fac_")}
        res = {k: v for k, v in reg.items() if k.startswith("res_")}
        print("-" * 70)
        print(f"  facilities: {fac}")
        print(f"  resources : {res}")
        act = "  ".join(f"{a[0]}:{a[1]}" for a in frame["actions"])
        print(f"  actions   : {act}")
        print()
        time.sleep(sleep)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=0.25,
                        help="seconds between frames (0 = print all instantly)")
    args = parser.parse_args()

    demo = EmergencyDemo(seed=args.seed)
    timeline = demo.run(max_ticks=args.ticks)
    if args.sleep > 0:
        render_cli(timeline, sleep=args.sleep)
    else:
        for frame in timeline:
            print(frame)
    # Determinism self-check: hash chain of state after each tick.
    print("\n[determinism] timeline hashes:",
          " -> ".join(f["state_hash"] for f in timeline[:5]), "...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
