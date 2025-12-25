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

    switches_per_group: int = 8          # Aurora: 32 (too big for drawing)
    compute_nodes_per_switch: int = 1    # Aurora has many endpoints; keep small for draw

    # In Aurora compute<->compute: 2 global links between each pair of compute groups
    global_links_per_group_pair: int = 2

    # Intra-group wiring: Aurora is all-to-all among 32 switches (clique)
    intra_group: str = "clique"          # "clique" or "dense"
    dense_p: float = 0.6                 # used if intra_group == "dense"

    seed: int = 42

    # Metrics
    metric_scope: str = "switches"       # "switches" or "all"
    sample_pairs: int = 5000             # for D*, Q when graph is large

    # Draw
    figsize: Tuple[int, int] = (14, 10)
    node_size_switch: int = 220
    node_size_compute: int = 60
    show_labels: bool = False

    # Global wiring style
    global_mode: str = "aurora"          # "aurora" (all-to-all) or "sparse"
    sparse_group_degree: int = 3         # used if global_mode == "sparse" (neighbors on ring)

    global_mapping: str = "round_robin"  # "round_robin" or "random"



def generate_aurora_like(cfg: TopologyConfig) -> nx.MultiGraph:
    rnd = random.Random(cfg.seed)
    G = nx.MultiGraph()

    # Group types: compute groups first, then storage, then service
    groups: List[Tuple[str, int]] = []
    for gi in range(cfg.compute_groups):
        groups.append(("compute", gi))
    for si in range(cfg.storage_groups):
        groups.append(("storage", cfg.compute_groups + si))
    for vi in range(cfg.service_groups):
        groups.append(("service", cfg.compute_groups + cfg.storage_groups + vi))

    # Create switches + compute nodes inside each group
    for gtype, gid in groups:
        switch_ids = []
        for s in range(cfg.switches_per_group):
            sid = f"g{gid}_s{s}"
            G.add_node(
                sid,
                kind="switch",
                group=gid,
                group_type=gtype,
                label=sid
            )
            switch_ids.append(sid)

            # attach compute nodes to each switch (small for demo)
            for c in range(cfg.compute_nodes_per_switch):
                nid = f"{sid}_n{c}"
                G.add_node(
                    nid,
                    kind="compute",
                    group=gid,
                    group_type=gtype,
                    label=nid
                )
                G.add_edge(sid, nid, kind="inj")

        # Intra-group switch wiring
        if cfg.intra_group == "clique":
            for i in range(len(switch_ids)):
                for j in range(i + 1, len(switch_ids)):
                    G.add_edge(switch_ids[i], switch_ids[j], kind="intra")
        else:
            # dense random
            for i in range(len(switch_ids)):
                for j in range(i + 1, len(switch_ids)):
                    if rnd.random() < cfg.dense_p:
                        G.add_edge(switch_ids[i], switch_ids[j], kind="intra")

    # --- Global links: compute groups connectivity ---
    compute_group_ids = list(range(cfg.compute_groups))

    def switches_in_group(gid: int) -> List[str]:
        return [n for n, d in G.nodes(data=True) if d["kind"] == "switch" and d["group"] == gid]

    # Precompute deterministic ordering of switches by index in name g{gid}_s{idx}
    def ordered_switches(gid: int) -> List[str]:
        sw = switches_in_group(gid)
        # sort by the integer after "_s"
        sw.sort(key=lambda x: int(x.split("_s")[1]))
        return sw

    # Choose which group pairs to connect depending on mode
    group_pairs: List[Tuple[int, int]] = []

    if cfg.global_mode == "aurora":
        # all-to-all pairs
        for i in range(len(compute_group_ids)):
            for j in range(i + 1, len(compute_group_ids)):
                group_pairs.append((compute_group_ids[i], compute_group_ids[j]))
    else:
        # sparse ring: each group connects to next K groups (wrap-around)
        K = max(1, cfg.sparse_group_degree)
        for i in compute_group_ids:
            for step in range(1, K + 1):
                j = (i + step) % cfg.compute_groups
                if i < j:
                    group_pairs.append((i, j))
                else:
                    # ensure uniqueness
                    group_pairs.append((j, i))
        group_pairs = sorted(set(group_pairs))

    for gid_a, gid_b in group_pairs:
        sw_a = ordered_switches(gid_a)
        sw_b = ordered_switches(gid_b)

        if cfg.global_mapping == "random":
            # old behavior
            for _ in range(cfg.global_links_per_group_pair):
                a = rnd.choice(sw_a)
                b = rnd.choice(sw_b)
                G.add_edge(a, b, kind="global")
        else:
            # round-robin deterministic mapping (reduces crossings)
            # pick switch indices based on group ids and link index
            for k in range(cfg.global_links_per_group_pair):
                ia = (gid_a + gid_b + k) % len(sw_a)
                ib = (gid_a * 3 + gid_b + k) % len(sw_b)
                a = sw_a[ia]
                b = sw_b[ib]
                G.add_edge(a, b, kind="global")

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


def draw_topology(G: nx.MultiGraph, pos, cfg: TopologyConfig) -> None:
    plt.figure(figsize=cfg.figsize)

    # Separate nodes by kind
    switches = [n for n, d in G.nodes(data=True) if d["kind"] == "switch"]
    computes = [n for n, d in G.nodes(data=True) if d["kind"] == "compute"]

    # Separate edges by kind (as stored in generate_aurora_like)
    intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "intra"]
    global_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "global"]
    inj_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "inj"]

    # --- Edges styling by type ---
    # Intra-group: light + thin
    if intra_edges:
        nx.draw_networkx_edges(
            nx.Graph(intra_edges),
            pos,
            alpha=0.12,
            width=0.6,
        )

    # Global: darker + thicker
    if global_edges:
        nx.draw_networkx_edges(
            nx.Graph(global_edges),
            pos,
            alpha=0.45,
            width=1.8,
        )

    # Injection: very faint + very thin
    if inj_edges:
        nx.draw_networkx_edges(
            nx.Graph(inj_edges),
            pos,
            alpha=0.06,
            width=0.4,
        )

    # --- Nodes styling by kind + group_type ---
    switches_compute = [n for n, d in G.nodes(data=True) if d["kind"] == "switch" and d.get("group_type") == "compute"]
    switches_storage = [n for n, d in G.nodes(data=True) if d["kind"] == "switch" and d.get("group_type") == "storage"]
    switches_service = [n for n, d in G.nodes(data=True) if d["kind"] == "switch" and d.get("group_type") == "service"]

    compute_endpoints = [n for n, d in G.nodes(data=True) if d["kind"] == "compute"]

    # Switches by group type
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=switches_compute,
        node_size=cfg.node_size_switch,
        node_color="tab:blue",
        linewidths=0.6,
        edgecolors="white",
        label="Compute switches"
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=switches_storage,
        node_size=cfg.node_size_switch,
        node_color="tab:orange",
        linewidths=0.6,
        edgecolors="white",
        label="Storage switches"
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=switches_service,
        node_size=cfg.node_size_switch,
        node_color="tab:purple",
        linewidths=0.6,
        edgecolors="white",
        label="Service switches"
    )

    # Endpoints (compute nodes)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=compute_endpoints,
        node_size=cfg.node_size_compute,
        node_color="lightgrey",
        linewidths=0.0,
        label="Compute nodes"
    )

    if cfg.show_labels:
        labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)

    plt.axis("off")
    plt.tight_layout()
    plt.legend(scatterpoints=1, frameon=False, loc="upper left")
    plt.show()


if __name__ == "__main__":
    cfg = TopologyConfig(
        compute_groups=6,
        storage_groups=1,
        service_groups=1,
        switches_per_group=32,
        compute_nodes_per_switch=2,
        global_links_per_group_pair=2,
        intra_group="clique",
        metric_scope="switches",
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
    pos = hierarchical_positions(G, cfg)
    metrics = compute_metrics(G, cfg)

    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k:>24}: {v}")

    draw_topology(G, pos, cfg)
