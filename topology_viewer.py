import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import Counter

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


# ----------------------------
# Config
# ----------------------------
@dataclass
class TopologyConfig:
    groups: int = 6

    subgroups_per_group: int = 4          # chassis per group
    compute_nodes_per_subgroup: int = 8   # endpoints per subgroup

    switches_per_subgroup: int = 4        # FIXED (keep 4)
    ports_per_switch: int = 28            # configurable “links per switch”

    min_links_per_group_pair: int = 2     # MUST be satisfied for every pair

    seed: int = 42

    # Metrics (optional)
    metric_scope: str = "switches"        # "switches" or "all"
    sample_pairs: int = 4000

    # Draw
    figsize: Tuple[int, int] = (14, 10)
    node_size_switch: int = 120
    node_size_compute: int = 120

    ring_radius: float = 12.0
    group_gap_angle: float = 0.10
    node_radius_offset: float = 1.2

    @property
    def switches_per_group(self) -> int:
        return self.subgroups_per_group * self.switches_per_subgroup


class TopologyConfigError(ValueError):
    pass


# ----------------------------
# Generator
# ----------------------------
def generate_dragonfly(cfg: TopologyConfig) -> nx.MultiGraph:
    rnd = random.Random(cfg.seed)
    G = nx.MultiGraph()

    def ordered_switches(gid: int) -> List[str]:
        sw = [n for n, d in G.nodes(data=True)
              if d.get("kind") == "switch" and d.get("group") == gid]
        sw.sort(key=lambda x: (G.nodes[x]["subgroup"], G.nodes[x]["switch_in_subgroup"]))
        return sw

    def add_k_edges(u: str, v: str, k: int, kind: str):
        for _ in range(k):
            G.add_edge(u, v, kind=kind)

    # ---- build nodes + local structure ----
    for gid in range(cfg.groups):
        for sg in range(cfg.subgroups_per_group):
            # 4 switches per subgroup
            subgroup_switches: List[str] = []
            for s in range(cfg.switches_per_subgroup):
                sid = f"g{gid}_sg{sg}_s{s}"
                G.add_node(
                    sid,
                    kind="switch",
                    group=gid,
                    subgroup=sg,
                    switch_in_subgroup=s,
                )
                subgroup_switches.append(sid)

            # compute nodes per subgroup, each connected to ALL 4 switches
            for n in range(cfg.compute_nodes_per_subgroup):
                nid = f"g{gid}_sg{sg}_n{n}"
                G.add_node(
                    nid,
                    kind="compute",
                    group=gid,
                    subgroup=sg,
                    node_in_subgroup=n,
                )
                for sid in subgroup_switches:
                    G.add_edge(sid, nid, kind="inj")

            # subgroup intra (S1..S4 clique)
            for i in range(len(subgroup_switches)):
                for j in range(i + 1, len(subgroup_switches)):
                    G.add_edge(subgroup_switches[i], subgroup_switches[j], kind="subgroup_intra")

            # extra double links: S1-S2 and S3-S4 (two additional edges each)
            s1, s2, s3, s4 = subgroup_switches
            add_k_edges(s1, s2, 2, kind="local_extra")
            add_k_edges(s3, s4, 2, kind="local_extra")

        # group intra: connect switches from different subgroups (all-to-all across subgroups)
        sw = ordered_switches(gid)
        for i in range(len(sw)):
            for j in range(i + 1, len(sw)):
                a, b = sw[i], sw[j]
                if G.nodes[a]["subgroup"] != G.nodes[b]["subgroup"]:
                    G.add_edge(a, b, kind="group_intra")

    # ---- validation: local wiring must fit in ports_per_switch ----
    over = []
    for n, d in G.nodes(data=True):
        if d.get("kind") == "switch":
            used = G.degree(n)  # counts inj + subgroup_intra + local_extra + group_intra
            if used > cfg.ports_per_switch:
                over.append((n, used))

    if over:
        worst = sorted(over, key=lambda x: x[1], reverse=True)[:8]
        msg = (
            "Config error: local wiring already exceeds ports_per_switch.\n"
            f"ports_per_switch={cfg.ports_per_switch}, "
            f"groups={cfg.groups}, subgroups_per_group={cfg.subgroups_per_group}, "
            f"compute_nodes_per_subgroup={cfg.compute_nodes_per_subgroup}\n"
            "Worst switches:\n" + "\n".join([f"  {n}: used {u}" for n, u in worst])
        )
        raise TopologyConfigError(msg)

    # ---- global links: min 2 per pair + then fill remaining ports round-robin ----
    group_pairs = [(a, b) for a in range(cfg.groups) for b in range(a + 1, cfg.groups)]
    if not group_pairs:
        return G

    rr_idx = {gid: 0 for gid in range(cfg.groups)}

    def remaining_ports(sw: str) -> int:
        return cfg.ports_per_switch - G.degree(sw)

    def pick_switch_with_free_port(gid: int) -> str | None:
        sw = ordered_switches(gid)
        if not sw:
            return None
        start = rr_idx[gid] % len(sw)
        for t in range(len(sw)):
            s = sw[(start + t) % len(sw)]
            if remaining_ports(s) > 0:
                rr_idx[gid] = (start + t + 1) % len(sw)
                return s
        return None

    # 1) allocate minimum links per pair
    for (ga, gb) in group_pairs:
        for _ in range(cfg.min_links_per_group_pair):
            sa = pick_switch_with_free_port(ga)
            sb = pick_switch_with_free_port(gb)
            if sa is None or sb is None:
                freeA = sum(max(0, remaining_ports(s)) for s in ordered_switches(ga))
                freeB = sum(max(0, remaining_ports(s)) for s in ordered_switches(gb))
                raise TopologyConfigError(
                    "Config error: not enough ports to satisfy minimum global connectivity.\n"
                    f"Need >= {cfg.min_links_per_group_pair} links per group pair.\n"
                    f"Failed on pair ({ga}, {gb}). Free ports: group {ga}={freeA}, group {gb}={freeB}.\n"
                    "Fix: increase ports_per_switch OR reduce compute_nodes_per_subgroup OR reduce subgroups_per_group OR reduce groups."
                )
            G.add_edge(sa, sb, kind="global")

    # 2) use remaining ports: keep adding in round-robin over all pairs
    progress = True
    while progress:
        progress = False
        for (ga, gb) in group_pairs:
            sa = pick_switch_with_free_port(ga)
            sb = pick_switch_with_free_port(gb)
            if sa is None or sb is None:
                continue
            G.add_edge(sa, sb, kind="global")
            progress = True

    return G


# ----------------------------
# Ring layout (consistent with subgroup attrs)
# ----------------------------
def dragonfly_ring_positions(G: nx.MultiGraph, cfg: TopologyConfig) -> Dict[str, Tuple[float, float]]:
    switches = [n for n, d in G.nodes(data=True) if d.get("kind") == "switch"]
    switches.sort(key=lambda x: (G.nodes[x]["group"], G.nodes[x]["subgroup"], G.nodes[x]["switch_in_subgroup"]))

    groups = sorted({G.nodes[s]["group"] for s in switches})
    group_to_switches: Dict[int, List[str]] = {g: [] for g in groups}
    for s in switches:
        group_to_switches[G.nodes[s]["group"]].append(s)

    total_groups = len(groups)
    pos: Dict[str, Tuple[float, float]] = {}

    gaps_total = total_groups * cfg.group_gap_angle
    usable_angle = 2 * math.pi - gaps_total
    total_switches = sum(len(group_to_switches[g]) for g in groups)
    angle_per_switch = usable_angle / max(1, total_switches)

    theta = 0.0
    R = cfg.ring_radius

    for g in groups:
        theta += cfg.group_gap_angle / 2
        for s in group_to_switches[g]:
            pos[s] = (R * math.cos(theta), R * math.sin(theta))
            theta += angle_per_switch
        theta += cfg.group_gap_angle / 2

    # compute nodes placed outside near subgroup center
    compute_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "compute"]
    for n in compute_nodes:
        gid = G.nodes[n]["group"]
        sg = G.nodes[n]["subgroup"]

        subgroup_switches = [s for s in group_to_switches[gid] if G.nodes[s]["subgroup"] == sg]
        cx = sum(pos[s][0] for s in subgroup_switches) / len(subgroup_switches)
        cy = sum(pos[s][1] for s in subgroup_switches) / len(subgroup_switches)

        norm = math.hypot(cx, cy) or 1.0
        ux, uy = (cx / norm, cy / norm)

        nz = G.nodes[n].get("node_in_subgroup", 0)
        lateral = (nz - (cfg.compute_nodes_per_subgroup - 1) / 2) * 0.15
        px, py = (-uy, ux)

        pos[n] = (
            cx + ux * cfg.node_radius_offset + px * lateral,
            cy + uy * cfg.node_radius_offset + py * lateral
        )

    return pos


# ----------------------------
# Draw (compute-only)
# ----------------------------
def draw_topology_dragonflyish(G: nx.MultiGraph, pos: Dict[str, Tuple[float, float]], cfg: TopologyConfig) -> None:
    plt.figure(figsize=cfg.figsize)
    ax = plt.gca()
    ax.axis("off")

    switches = [n for n, d in G.nodes(data=True) if d.get("kind") == "switch"]
    compute_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "compute"]

    cx0 = sum(pos[n][0] for n in switches) / max(1, len(switches))
    cy0 = sum(pos[n][1] for n in switches) / max(1, len(switches))

    # outward direction per switch (toward its compute nodes)
    out_vec = {}
    for s in switches:
        nbrs = [nbr for nbr in G.neighbors(s) if G.nodes[nbr].get("kind") == "compute"]
        sx, sy = pos[s]
        if nbrs:
            vx = sum(pos[n][0] - sx for n in nbrs) / len(nbrs)
            vy = sum(pos[n][1] - sy for n in nbrs) / len(nbrs)
        else:
            vx, vy = (sx - cx0, sy - cy0)
        L = math.hypot(vx, vy) or 1.0
        out_vec[s] = (vx / L, vy / L)

    def rad_away_from_compute(u, v, mag: float) -> float:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = (x2 - x1), (y2 - y1)

        # left normal
        nx_, ny_ = (-dy, dx)

        oux, ouy = out_vec.get(u, (0.0, 0.0))
        ovx, ovy = out_vec.get(v, (0.0, 0.0))
        outx, outy = ((oux + ovx) / 2.0, (ouy + ovy) / 2.0)

        # inward is opposite of outward
        inx, iny = (-outx, -outy)
        return mag if (nx_ * inx + ny_ * iny) > 0 else -mag

    # edge lists
    global_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "global"]
    group_intra = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "group_intra"]
    local_extra = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "local_extra"]
    inj_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "inj"]

    # --- global curved (black) ---
    if global_edges:
        pair_counts = Counter()
        for u, v in global_edges:
            a, b = (u, v) if u < v else (v, u)
            pair_counts[(a, b)] += 1

        for (u, v), cnt in pair_counts.items():
            for i in range(cnt):
                mag = 0.22 + 0.07 * min(i, 6)
                rad = rad_away_from_compute(u, v, mag)
                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-",
                    lw=1.6,
                    color="black",
                    alpha=0.65,
                    zorder=0.2
                ))

    # --- group intra (faint red, deeper inward, separated by distance) ---
    if group_intra:
        group_switches_sorted = {}
        switch_order = {}
        for s in switches:
            gid = G.nodes[s]["group"]
            group_switches_sorted.setdefault(gid, []).append(s)

        for gid, sw_list in group_switches_sorted.items():
            def ang(s):
                x, y = pos[s]
                return math.atan2(y - cy0, x - cx0)
            sw_list.sort(key=ang)
            for i, s in enumerate(sw_list):
                switch_order[s] = i

        for (u, v) in group_intra:
            dist = abs(switch_order.get(u, 0) - switch_order.get(v, 0))
            mag = 0.22 + 0.04 * min(dist, 12)  # more curvy, more separated
            rad = rad_away_from_compute(u, v, mag)
            ax.add_patch(FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={-rad}",
                arrowstyle="-",
                lw=0.7,
                color="black",
                alpha=0.65,
                zorder=0.3
            ))

    # --- local extra (blue, two sides) ---
    if local_extra:
        extra_counts = Counter()
        for (u, v) in local_extra:
            a, b = (u, v) if u < v else (v, u)
            extra_counts[(a, b)] += 1

        for (u, v), cnt in extra_counts.items():
            if cnt <= 0:
                continue
            mag = 0.28
            inward = rad_away_from_compute(u, v, mag)
            outward = -inward
            for rad in ([inward, outward] if cnt >= 2 else [inward]):
                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-",
                    lw=2.4,
                    color="black",
                    alpha=0.65,
                    zorder=1.0
                ))

    # --- subgroup internal clique (blue): draw per subgroup with a readable pattern ---
    # We draw S2-S3 as straight line, and the other clique edges as inward curves.
    for gid in range(cfg.groups):
        for sg in range(cfg.subgroups_per_group):
            sw = [n for n, d in G.nodes(data=True)
                  if d.get("kind") == "switch" and d.get("group") == gid and d.get("subgroup") == sg]
            sw.sort(key=lambda n: G.nodes[n].get("switch_in_subgroup", 0))
            if len(sw) != 4:
                continue

            s1, s2, s3, s4 = sw[0], sw[1], sw[2], sw[3]

            # Straight adjacency (your requirement: S2-S3 should be a line)
            x1, y1 = pos[s2];
            x2, y2 = pos[s3]
            ax.plot([x1, x2], [y1, y2],
                    color="black", alpha=0.55, lw=1.2, zorder=1.2)

            # Other clique edges as curvy (inner side)
            for (u, v, mag) in [
                (s1, s2, 0.18),
                (s3, s4, 0.18),
                (s1, s3, 0.22),
                (s2, s4, 0.22),
                (s1, s4, 0.32),
            ]:
                rad = rad_away_from_compute(u, v, mag)
                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    connectionstyle=f"arc3,rad={-rad}",
                    arrowstyle="-",
                    lw=1.2,
                    color="black",
                    alpha=0.50,
                    zorder=1.2
                ))

    # --- injection edges (grey) ---
    if inj_edges:
        nx.draw_networkx_edges(nx.Graph(inj_edges), pos, alpha=0.45, width=1.6, edge_color="black")

    # --- nodes on top ---
    ax.scatter([pos[n][0] for n in switches], [pos[n][1] for n in switches],
               s=cfg.node_size_switch, c="tab:green",
               edgecolors="black", linewidths=0.8, zorder=10, label="Switches")

    ax.scatter([pos[n][0] for n in compute_nodes], [pos[n][1] for n in compute_nodes],
               s=cfg.node_size_compute, c="tab:blue",
               edgecolors="black", linewidths=0.5, zorder=10, label="Compute nodes")

    plt.tight_layout()
    plt.legend(scatterpoints=1, frameon=False, loc="upper left")
    plt.show()


if __name__ == "__main__":
    cfg = TopologyConfig(
        groups=6,
        subgroups_per_group=4,
        compute_nodes_per_subgroup=8,
        ports_per_switch=26,
        min_links_per_group_pair=2,
    )

    G = generate_dragonfly(cfg)
    pos = dragonfly_ring_positions(G, cfg)
    draw_topology_dragonflyish(G, pos, cfg)
