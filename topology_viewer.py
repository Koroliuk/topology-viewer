import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

import networkx as nx


@dataclass
class TopologyConfig:
    compute_groups: int = 6

    chassis_per_group: int = 8          # 8 subgroups
    switches_per_chassis: int = 4       # 4 switches per subgroup
    compute_nodes_per_chassis: int = 8  # 8 nodes per subgroup (your model)

    global_links_per_group_pair: int = 2

    intra_group: str = "clique"  # keep for metrics, but we won't draw all those edges
    seed: int = 42

    # Draw
    figsize: Tuple[int, int] = (14, 10)
    node_size_switch: int = 120
    node_size_compute: int = 120
    show_labels: bool = False

    # Ring layout knobs
    ring_radius: float = 12.0
    group_gap_angle: float = 0.10  # radians, space between group segments
    node_radius_offset: float = 1.2

    @property
    def switches_per_group(self) -> int:
        return self.chassis_per_group * self.switches_per_chassis

    global_mode: str = "round_robin"  # "all_to_all" or "round_robin"
    global_k: int = 2  # number of neighbor groups to connect to (per group)


def generate_aurora_like(cfg: TopologyConfig) -> nx.MultiGraph:
    rnd = random.Random(cfg.seed)
    G = nx.MultiGraph()

    # Group types list
    groups: List[Tuple[str, int]] = []
    for gi in range(cfg.compute_groups):
        groups.append(("compute", gi))

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

            # --- Subgroup (chassis) switch wiring you described ---
            # Switches S1..S4 form a clique: each connected to each other
            for i in range(len(chassis_switches)):
                for j in range(i + 1, len(chassis_switches)):
                    G.add_edge(chassis_switches[i], chassis_switches[j], kind="local_clique")

            # Additional (parallel) connections:
            # extra links between S1-S2 and S3-S4 (two more edges each)
            if len(chassis_switches) >= 4:
                s1, s2, s3, s4 = chassis_switches[0], chassis_switches[1], chassis_switches[2], chassis_switches[3]
                G.add_edge(s1, s2, kind="local_extra")
                G.add_edge(s1, s2, kind="local_extra")
                G.add_edge(s3, s4, kind="local_extra")
                G.add_edge(s3, s4, kind="local_extra")

        # --- Intra-group links across chassis (group-level all-to-all) ---
        # We connect switches from different chassis. Within-chassis all-to-all is already modeled by local_clique.
        group_switches = [n for n, d in G.nodes(data=True)
                          if d.get("kind") == "switch" and d.get("group") == gid and "chassis" in d]

        group_switches.sort(key=lambda n: (G.nodes[n].get("chassis", 0), G.nodes[n].get("switch_in_chassis", 0)))

        for i in range(len(group_switches)):
            for j in range(i + 1, len(group_switches)):
                a, b = group_switches[i], group_switches[j]
                if G.nodes[a].get("chassis") != G.nodes[b].get("chassis"):
                    G.add_edge(a, b, kind="group_intra")


        # Intra-group dense wiring (for metrics realism)
        if cfg.intra_group == "clique":
            for i in range(len(switches_in_group)):
                for j in range(i + 1, len(switches_in_group)):
                    G.add_edge(switches_in_group[i], switches_in_group[j], kind="intra")

    # Global links between compute groups (keep your round-robin deterministic mapping)
    # ---- Global links between compute groups ----
    G_ids = list(range(cfg.compute_groups))

    def ordered_switches(gid: int) -> List[str]:
        sw = [n for n, d in G.nodes(data=True) if d["kind"] == "switch" and d["group"] == gid]
        sw.sort(key=lambda x: (G.nodes[x].get("chassis", 0), G.nodes[x].get("switch_in_chassis", 0)))
        return sw

    added_pairs = set()

    if cfg.global_mode == "all_to_all":
        pairs = [(a, b) for i, a in enumerate(G_ids) for b in G_ids[i + 1:]]
    else:
        # round-robin neighbors on the ring
        k = max(1, min(cfg.global_k, len(G_ids) // 2))
        pairs = []
        for a in G_ids:
            for step in range(1, k + 1):
                b = (a + step) % len(G_ids)
                x, y = (a, b) if a < b else (b, a)
                if (x, y) not in added_pairs:
                    added_pairs.add((x, y))
                    pairs.append((x, y))

    # Add L parallel links per selected group pair, deterministically mapped to switches
    for ga, gb in pairs:
        sw_a = ordered_switches(ga)
        sw_b = ordered_switches(gb)

        for link_i in range(cfg.global_links_per_group_pair):
            ia = (gb + link_i) % len(sw_a)
            ib = (ga + 2 * link_i + 1) % len(sw_b)
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

import math
from collections import Counter
from matplotlib.patches import FancyArrowPatch
import networkx as nx
import matplotlib.pyplot as plt


def draw_topology_dragonflyish(G: nx.MultiGraph, pos, cfg: TopologyConfig) -> None:
    plt.figure(figsize=cfg.figsize)
    ax = plt.gca()
    ax.axis("off")

    # Precompute an "outward" direction for each switch: where its compute nodes are
    switches = [n for n, d in G.nodes(data=True) if d.get("kind") == "switch"]
    computes = [n for n, d in G.nodes(data=True) if d.get("kind") == "compute"]

    # Center of the switch ring (fallback)
    cx0 = sum(pos[n][0] for n in switches) / max(1, len(switches))
    cy0 = sum(pos[n][1] for n in switches) / max(1, len(switches))

    out_vec = {}  # switch -> (vx, vy) pointing toward compute nodes (outward)
    for s in switches:
        # compute neighbors of this switch
        nbrs = []
        for nbr in G.neighbors(s):
            if G.nodes[nbr].get("kind") == "compute":
                nbrs.append(nbr)

        sx, sy = pos[s]

        if nbrs:
            vx = sum(pos[n][0] - sx for n in nbrs) / len(nbrs)
            vy = sum(pos[n][1] - sy for n in nbrs) / len(nbrs)
        else:
            # fallback: outward = from ring center to switch
            vx, vy = (sx - cx0, sy - cy0)

        # normalize
        L = math.hypot(vx, vy) or 1.0
        out_vec[s] = (vx / L, vy / L)

    def rad_away_from_compute(u, v, mag: float) -> float:
        """
        Choose rad sign so the arc bends AWAY from compute nodes.
        That places local switch-switch curves on the empty side of the switch line.
        """
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = (x2 - x1), (y2 - y1)

        # left normal of segment u->v
        nx_, ny_ = (-dy, dx)

        # direction toward compute nodes (outward) ~ average of endpoints' outward vectors
        oux, ouy = out_vec.get(u, (0.0, 0.0))
        ovx, ovy = out_vec.get(v, (0.0, 0.0))
        outx, outy = ((oux + ovx) / 2.0, (ouy + ovy) / 2.0)

        # We want the curve to go to the opposite side => inward = -outward
        inx, iny = (-outx, -outy)

        # pick rad sign so the normal points inward
        return mag if (nx_ * inx + ny_ * iny) > 0 else -mag

    # --- edges by type ---
    global_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "global"]
    inj_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "inj"]

    local_extra = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "local_extra"]
    group_intra = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "group_intra"]

    # --- draw global first ---
    if global_edges:
        nx.draw_networkx_edges(
            nx.Graph(global_edges),
            pos,
            alpha=0.80,
            width=2.0,
            edge_color="black"
        )

    # --- draw subgroup internal edges (per chassis) ---
    chassis_keys = sorted({
        (d.get("group"), d.get("chassis"))
        for _, d in G.nodes(data=True)
        if d.get("kind") == "switch" and d.get("chassis") is not None
    })

    for gid, ch in chassis_keys:
        sw = [n for n, d in G.nodes(data=True)
              if d.get("kind") == "switch" and d.get("group") == gid and d.get("chassis") == ch]
        sw.sort(key=lambda n: G.nodes[n].get("switch_in_chassis", 0))
        if len(sw) < 4:
            continue

        s1, s2, s3, s4 = sw[0], sw[1], sw[2], sw[3]

        # straight adjacency edges, includes S2-S3
        for u, v in [(s2, s3)]:
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2],
                    color="tab:blue", alpha=0.40, lw=1.0, zorder=1)

        # inward curved diagonals
        for (u, v, mag) in [(s1, s3, 0.22), (s2, s4, 0.22), (s1, s4, 0.32)]:
            rad = rad_away_from_compute(u, v, mag)
            patch = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={-rad}",
                arrowstyle="-",
                lw=1.0,
                color="tab:blue",
                alpha=0.40,
                zorder=1
            )
            ax.add_patch(patch)

    # --- double links: two curves on both sides ---
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

        rads = [inward, outward] if cnt >= 2 else [inward]
        for rad in rads:
            patch = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-",
                lw=2.6,
                color="tab:blue",
                alpha=0.85,
                zorder=2
            )
            ax.add_patch(patch)

    # Intra-group all-to-all: very faint curved arcs (inner side, away from compute nodes)
    # Intra-group all-to-all: draw as deeper inward arcs with varied curvature so they separate visually
    if group_intra:
        # Build an index of switch order within each group (along the ring)
        group_switches_sorted = {}
        switch_order = {}  # switch -> index within its group

        # Collect switches by group and sort them in ring order using their angle
        for n, d in G.nodes(data=True):
            if d.get("kind") == "switch":
                gid = d.get("group")
                group_switches_sorted.setdefault(gid, []).append(n)

        for gid, sw_list in group_switches_sorted.items():
            # sort by polar angle around ring center
            def angle(s):
                x, y = pos[s]
                return math.atan2(y - cy0, x - cx0)

            sw_list.sort(key=angle)
            for i, s in enumerate(sw_list):
                switch_order[s] = i

        # Draw arcs with curvature based on distance in ring order
        for (u, v) in group_intra:
            iu = switch_order.get(u, 0)
            iv = switch_order.get(v, 0)
            dist = abs(iu - iv)

            # Map distance to curvature:
            # near neighbors -> small curve, far pairs -> deep curve
            # tweak these constants to taste
            mag = 0.18 + 0.03 * min(dist, 10)  # grows up to ~0.48

            rad = rad_away_from_compute(u, v, mag)

            patch = FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={-rad}",
                arrowstyle="-",
                lw=0.7,
                color="red",
                alpha=0.16,
                zorder=0.6
            )
            ax.add_patch(patch)

    # --- draw injection after local (so local doesn't sit on top of injection) ---
    if inj_edges:
        nx.draw_networkx_edges(
            nx.Graph(inj_edges),
            pos,
            alpha=0.45,
            width=1.8,
            edge_color="gray"
        )

    # --- nodes ---
    switches_compute = [n for n, d in G.nodes(data=True)
                        if d.get("kind") == "switch" and d.get("group_type") == "compute"]
    compute_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "compute"]

    # draw nodes LAST so they are always on top
    def scatter(nodes, color, size, label):
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        ax.scatter(xs, ys, s=size, c=color, edgecolors="black", linewidths=0.8,
                   zorder=10, label=label)

    scatter(switches_compute, "tab:green", cfg.node_size_switch, "Switches")

    ax.scatter([pos[n][0] for n in compute_nodes],
               [pos[n][1] for n in compute_nodes],
               s=cfg.node_size_compute, c="tab:blue",
               edgecolors="black", linewidths=0.5, zorder=10,
               label="Compute nodes")

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
        chassis_per_group=4,
        switches_per_chassis=4,
        compute_nodes_per_chassis=8,

        global_links_per_group_pair=2,
        intra_group="clique",
        seed=42,

        show_labels=False
    )

    G = generate_aurora_like(cfg)
    pos = dragonfly_ring_positions(G, cfg)
    draw_topology_dragonflyish(G, pos, cfg)

