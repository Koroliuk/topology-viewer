from matplotlib.patches import Ellipse, FancyArrowPatch
import numpy as np

import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional

import networkx as nx
import matplotlib.pyplot as plt


@dataclass
class TopologyConfig:
    compute_groups: int = 6
    storage_groups: int = 1
    service_groups: int = 1

    chassis_per_group: int = 8          # 8 subgroups
    switches_per_chassis: int = 4       # 4 switches per subgroup
    compute_nodes_per_chassis: int = 8  # 8 nodes per subgroup (your model)

    global_links_per_group_pair: int = 2

    intra_group: str = "clique"  # keep for metrics, but we won't draw all those edges
    seed: int = 42

    # Draw
    figsize: Tuple[int, int] = (14, 10)
    node_size_switch: int = 90
    node_size_compute: int = 18
    show_labels: bool = False

    # Ring layout knobs
    ring_radius: float = 12.0
    group_gap_angle: float = 0.10  # radians, space between group segments
    node_radius_offset: float = 1.2

    @property
    def switches_per_group(self) -> int:
        return self.chassis_per_group * self.switches_per_chassis


def generate_aurora_like(cfg: TopologyConfig) -> nx.MultiGraph:
    rnd = random.Random(cfg.seed)
    G = nx.MultiGraph()

    # Group types list
    groups: List[Tuple[str, int]] = []
    for gi in range(cfg.compute_groups):
        groups.append(("compute", gi))
    for si in range(cfg.storage_groups):
        groups.append(("storage", cfg.compute_groups + si))
    for vi in range(cfg.service_groups):
        groups.append(("service", cfg.compute_groups + cfg.storage_groups + vi))

    # Create chassis -> switches + compute nodes
    for gtype, gid in groups:
        switches_in_group: List[str] = []

        for c in range(cfg.chassis_per_group):
            chassis_switches: List[str] = []

            # 4 switches per chassis
            for s in range(cfg.switches_per_chassis):
                sid = f"g{gid}_c{c}_s{s}"
                G.add_node(
                    sid,
                    kind="switch",
                    group=gid,
                    group_type=gtype,
                    chassis=c,
                    switch_in_chassis=s,
                    label=sid
                )
                chassis_switches.append(sid)
                switches_in_group.append(sid)

            # 8 compute nodes per chassis
            for n in range(cfg.compute_nodes_per_chassis):
                nid = f"g{gid}_c{c}_n{n}"
                G.add_node(
                    nid,
                    kind="compute",
                    group=gid,
                    group_type=gtype,
                    chassis=c,
                    label=nid
                )
                # Connect node to ALL switches of chassis (clear demo model)
                for sid in chassis_switches:
                    G.add_edge(sid, nid, kind="inj")

            # Optional: local chassis wiring (symbolic, not critical)
            # connect switches in chassis in a small chain/ring
            for i in range(len(chassis_switches) - 1):
                G.add_edge(chassis_switches[i], chassis_switches[i + 1], kind="local_chassis")

        # Intra-group dense wiring (for metrics realism)
        if cfg.intra_group == "clique":
            for i in range(len(switches_in_group)):
                for j in range(i + 1, len(switches_in_group)):
                    G.add_edge(switches_in_group[i], switches_in_group[j], kind="intra")

    # Global links between compute groups (keep your round-robin deterministic mapping)
    compute_group_ids = list(range(cfg.compute_groups))

    def ordered_switches(gid: int) -> List[str]:
        sw = [n for n, d in G.nodes(data=True) if d["kind"] == "switch" and d["group"] == gid]
        # order by chassis then switch index
        sw.sort(key=lambda x: (G.nodes[x].get("chassis", 0), G.nodes[x].get("switch_in_chassis", 0)))
        return sw

    for i in range(len(compute_group_ids)):
        for j in range(i + 1, len(compute_group_ids)):
            ga = compute_group_ids[i]
            gb = compute_group_ids[j]
            sw_a = ordered_switches(ga)
            sw_b = ordered_switches(gb)

            for k in range(cfg.global_links_per_group_pair):
                ia = (ga + gb + k) % len(sw_a)
                ib = (ga * 3 + gb + k) % len(sw_b)
                G.add_edge(sw_a[ia], sw_b[ib], kind="global")

    return G


def hierarchical_positions(G: nx.MultiGraph, cfg: TopologyConfig) -> Dict[str, Tuple[float, float]]:
    # Group centers on a circle; switches around group center; compute nodes near switch
    groups = sorted({d["group"] for _, d in G.nodes(data=True)})
    group_index = {gid: i for i, gid in enumerate(groups)}

    R = 10.0  # big radius for groups
    r_sw = 1.8  # radius for switches around a group center
    r_n = 0.35  # radius for compute nodes around switch

    pos: Dict[str, Tuple[float, float]] = {}

    # Pre-collect switches per group
    switches_by_group: Dict[int, List[str]] = {gid: [] for gid in groups}
    computes_by_switch: Dict[str, List[str]] = {}

    for n, d in G.nodes(data=True):
        if d["kind"] == "switch":
            switches_by_group[d["group"]].append(n)

    for n, d in G.nodes(data=True):
        if d["kind"] == "compute":
            # parent switch name is prefix before "_n"
            parent = n.split("_n")[0]
            computes_by_switch.setdefault(parent, []).append(n)

    for gid in groups:
        gi = group_index[gid]
        angle = 2 * math.pi * gi / max(1, len(groups))
        gx, gy = (R * math.cos(angle), R * math.sin(angle))

        switches = sorted(switches_by_group[gid])
        for si, sw in enumerate(switches):
            a2 = 2 * math.pi * si / max(1, len(switches))
            sx, sy = (gx + r_sw * math.cos(a2), gy + r_sw * math.sin(a2))
            pos[sw] = (sx, sy)

            # Place compute nodes around the switch
            cnodes = sorted(computes_by_switch.get(sw, []))
            for ci, cn in enumerate(cnodes):
                a3 = 2 * math.pi * ci / max(1, len(cnodes))
                pos[cn] = (sx + r_n * math.cos(a3), sy + r_n * math.sin(a3))

    return pos


def compute_metrics(G: nx.MultiGraph, cfg: TopologyConfig) -> Dict[str, float]:
    # Choose node set for metrics
    if cfg.metric_scope == "switches":
        nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "switch"]
        H = nx.Graph(G.subgraph(nodes))  # simple graph for distances
    else:
        H = nx.Graph(G)

    N = H.number_of_nodes()
    E = H.number_of_edges()

    degrees = [deg for _, deg in H.degree()]
    S_max = max(degrees) if degrees else 0
    S_avg = sum(degrees) / len(degrees) if degrees else 0.0

    # Handle disconnected graphs
    if N == 0:
        return {"N": 0, "S_max": 0, "S_avg": 0, "D": 0, "D*": 0, "Q": 0, "C_edges": 0, "C_ports": 0}

    if not nx.is_connected(H):
        # Use largest connected component for distances/diameter
        largest = max(nx.connected_components(H), key=len)
        Hc = H.subgraph(largest).copy()
    else:
        Hc = H

    Nc = Hc.number_of_nodes()

    # Diameter exact only for reasonable sizes
    if Nc <= 800:
        D = nx.diameter(Hc)
    else:
        # crude approximation: two-sweep BFS on a few random starts
        rnd = random.Random(cfg.seed)
        starts = [rnd.choice(list(Hc.nodes())) for _ in range(5)]
        best = 0
        for s in starts:
            lengths = nx.single_source_shortest_path_length(Hc, s)
            far = max(lengths.values())
            best = max(best, far)
        D = best

    # Average path length and Q: exact for small, sampled for large
    if Nc <= 500:
        D_star = nx.average_shortest_path_length(Hc)
        # Q = sum_{u<v} dist(u,v)
        # networkx doesn't provide directly; compute by summing single-source and dividing by 2
        Q = 0
        for u in Hc.nodes():
            lengths = nx.single_source_shortest_path_length(Hc, u)
            Q += sum(lengths.values())
        Q = Q / 2
    else:
        rnd = random.Random(cfg.seed)
        nodes_list = list(Hc.nodes())
        pairs = set()
        target = min(cfg.sample_pairs, Nc * (Nc - 1) // 2)
        while len(pairs) < target:
            u = rnd.choice(nodes_list)
            v = rnd.choice(nodes_list)
            if u != v:
                a, b = (u, v) if u < v else (v, u)
                pairs.add((a, b))

        total = 0
        count = 0
        for (u, v) in pairs:
            try:
                d = nx.shortest_path_length(Hc, u, v)
                total += d
                count += 1
            except nx.NetworkXNoPath:
                pass

        D_star = (total / count) if count else 0.0
        # Q approx: scale average distance by number of pairs in the component
        total_pairs = Nc * (Nc - 1) / 2
        Q = D_star * total_pairs

    C_edges = E
    C_ports = sum(deg for _, deg in H.degree())  # = 2E

    return {
        "N": float(N),
        "S_max": float(S_max),
        "S_avg": float(S_avg),
        "D": float(D),
        "D*": float(D_star),
        "Q": float(Q),
        "C_edges": float(C_edges),
        "C_ports": float(C_ports),
        "connected_component_size": float(Nc),
    }

from matplotlib.patches import FancyArrowPatch

from matplotlib.patches import FancyArrowPatch

def draw_topology_dragonflyish(G: nx.MultiGraph, pos, cfg: TopologyConfig) -> None:
    plt.figure(figsize=cfg.figsize)
    ax = plt.gca()
    ax.axis("off")

    # --- edges by type ---
    global_edges = [(u, v) for u, v, d in G.edges(data=True)
                    if d.get("kind") in ("global", "global_storage", "global_service")]

    inj_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "inj"]

    local_chassis = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get("kind") in ("local_chassis",)]

    # Global: black chords (between groups)
    if global_edges:
        nx.draw_networkx_edges(
            nx.Graph(global_edges),
            pos,
            alpha=0.75,
            width=1.8,
            edge_color="black",
        )

    # Local chassis: blue arcs (symbolic local links)
    for (u, v) in local_chassis:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        patch = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.25",
            arrowstyle="-",
            lw=1.4,
            color="tab:blue",
            alpha=0.70,
            zorder=2,
        )
        ax.add_patch(patch)

    # Injection: thicker and more visible (node <-> switch)
    if inj_edges:
        nx.draw_networkx_edges(
            nx.Graph(inj_edges),
            pos,
            alpha=0.35,     # was ~0.06 before
            width=1.2,      # thicker as requested
            edge_color="gray",
        )

    # --- nodes by type + group_type ---
    switches_compute = [n for n, d in G.nodes(data=True)
                        if d.get("kind") == "switch" and d.get("group_type") == "compute"]
    switches_storage = [n for n, d in G.nodes(data=True)
                        if d.get("kind") == "switch" and d.get("group_type") == "storage"]
    switches_service = [n for n, d in G.nodes(data=True)
                        if d.get("kind") == "switch" and d.get("group_type") == "service"]

    compute_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "compute"]

    # Switches as squares (colored)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=switches_compute,
        node_size=cfg.node_size_switch,
        node_color="tab:green",
        edgecolors="black",
        linewidths=0.8,
        node_shape="s",
        label="Compute switches"
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=switches_storage,
        node_size=cfg.node_size_switch,
        node_color="tab:orange",
        edgecolors="black",
        linewidths=0.8,
        node_shape="s",
        label="Storage switches"
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=switches_service,
        node_size=cfg.node_size_switch,
        node_color="tab:purple",
        edgecolors="black",
        linewidths=0.8,
        node_shape="s",
        label="Service switches"
    )

    # Compute nodes as blue circles
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=compute_nodes,
        node_size=cfg.node_size_compute,
        node_color="tab:blue",
        edgecolors="black",
        linewidths=0.5,
        node_shape="o",
        label="Compute nodes"
    )

    plt.tight_layout()
    plt.legend(scatterpoints=1, frameon=False, loc="upper left")
    plt.show()



def dragonfly_ring_positions(G: nx.MultiGraph, cfg: TopologyConfig) -> Dict[str, Tuple[float, float]]:
    # Order switches by group -> chassis -> switch
    switches = [n for n, d in G.nodes(data=True) if d["kind"] == "switch"]
    switches.sort(key=lambda x: (G.nodes[x]["group"], G.nodes[x].get("chassis", 0), G.nodes[x].get("switch_in_chassis", 0)))

    groups = sorted({G.nodes[s]["group"] for s in switches})
    group_to_switches: Dict[int, List[str]] = {g: [] for g in groups}
    for s in switches:
        group_to_switches[G.nodes[s]["group"]].append(s)

    total_groups = len(groups)
    pos: Dict[str, Tuple[float, float]] = {}

    # total angle budget = 2π minus gaps
    gaps_total = total_groups * cfg.group_gap_angle
    usable_angle = 2 * math.pi - gaps_total
    # each group gets equal angular span proportional to its #switches
    total_switches = sum(len(group_to_switches[g]) for g in groups)
    angle_per_switch = usable_angle / max(1, total_switches)

    theta = 0.0
    R = cfg.ring_radius

    # place switches group by group
    for g in groups:
        theta += cfg.group_gap_angle / 2  # half-gap before group
        sw_list = group_to_switches[g]

        for s in sw_list:
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            pos[s] = (x, y)
            theta += angle_per_switch

        theta += cfg.group_gap_angle / 2  # half-gap after group

    # place compute nodes behind their chassis (slightly outside, near chassis center angle)
    computes = [n for n, d in G.nodes(data=True) if d["kind"] == "compute"]
    for n in computes:
        gid = G.nodes[n]["group"]
        chassis = G.nodes[n].get("chassis", 0)

        # find the 4 switches of this chassis
        chassis_switches = [
            s for s in group_to_switches[gid]
            if G.nodes[s].get("chassis", -1) == chassis
        ]
        if not chassis_switches:
            continue

        # chassis center = average of its 4 switch positions
        cx = sum(pos[s][0] for s in chassis_switches) / len(chassis_switches)
        cy = sum(pos[s][1] for s in chassis_switches) / len(chassis_switches)

        # radial outward direction
        norm = math.hypot(cx, cy) or 1.0
        ux, uy = (cx / norm, cy / norm)

        # small lateral offset so nodes don't overlap completely
        # (use node index from name gX_cY_nZ)
        nz = int(n.split("_n")[1])
        lateral = (nz - (cfg.compute_nodes_per_chassis - 1) / 2) * 0.15
        px, py = (-uy, ux)  # perpendicular

        pos[n] = (
            cx + ux * cfg.node_radius_offset + px * lateral,
            cy + uy * cfg.node_radius_offset + py * lateral
        )

    return pos



if __name__ == "__main__":
    cfg = TopologyConfig(
        compute_groups=6,
        storage_groups=0,
        service_groups=0,

        chassis_per_group=4,
        switches_per_chassis=4,
        compute_nodes_per_chassis=8,

        global_links_per_group_pair=2,
        intra_group="clique",
        seed=42,

        show_labels=False
    )

    # cfg = TopologyConfig(
    #     compute_groups=6,
    #     switches_per_group=8,
    #     compute_nodes_per_switch=1,
    #     global_mode="aurora",
    #     global_mapping="round_robin",
    #     global_links_per_group_pair=2
    # )

    G = generate_aurora_like(cfg)
    pos = dragonfly_ring_positions(G, cfg)
    draw_topology_dragonflyish(G, pos, cfg)

