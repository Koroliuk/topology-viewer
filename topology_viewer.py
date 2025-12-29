import math
import zlib
from collections import Counter
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch


# ----------------------------
# Config
# ----------------------------
@dataclass
class TopologyConfig:
    groups: int = 6

    subgroups_per_group: int = 4  # chassis per group
    compute_nodes_per_subgroup: int = 8  # endpoints per subgroup

    switches_per_subgroup: int = 4  # FIXED (keep 4)
    ports_per_switch: int = 28  # configurable “links per switch”

    min_links_per_group_pair: int = 2  # MUST be satisfied for every pair

    seed: int = 42

    # Metrics (optional)
    metric_scope: str = "switches"  # "switches" or "all"
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
# ----------------------------
# Draw (compute-only) - MODIFIED
# ----------------------------
def draw_topology_base(G: nx.MultiGraph, pos: Dict[str, Tuple[float, float]], cfg: TopologyConfig):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    ax.axis("off")

    # --- REGISTRY TO STORE CURVATURE DATA ---
    # Key: (u, v), Value: rad (float)
    edge_registry: Dict[Tuple[str, str], float] = {}

    switches = [n for n, d in G.nodes(data=True) if d.get("kind") == "switch"]
    compute_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "compute"]

    cx0 = sum(pos[n][0] for n in switches) / max(1, len(switches))
    cy0 = sum(pos[n][1] for n in switches) / max(1, len(switches))

    # outward direction per switch
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
        # ... (Same logic as before) ...
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        dx, dy = (x2 - x1), (y2 - y1)
        nx_, ny_ = (-dy, dx)
        oux, ouy = out_vec.get(u, (0.0, 0.0))
        ovx, ovy = out_vec.get(v, (0.0, 0.0))
        outx, outy = ((oux + ovx) / 2.0, (ouy + ovy) / 2.0)
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

                # STORE IN REGISTRY
                edge_registry[(u, v)] = rad

                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-", lw=1.6, color="black", alpha=0.65, zorder=0.2
                ))

    # --- group intra ---
    if group_intra:
        # ... (Sort logic same as before) ...
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
            mag = 0.22 + 0.04 * min(dist, 12)
            rad = rad_away_from_compute(u, v, mag)

            # STORE IN REGISTRY (Note: logic used -rad in original code, so we store -rad)
            edge_registry[(u, v)] = -rad

            ax.add_patch(FancyArrowPatch(
                posA=pos[u], posB=pos[v],
                connectionstyle=f"arc3,rad={-rad}",
                arrowstyle="-", lw=0.7, color="black", alpha=0.65, zorder=0.3
            ))

    # --- local extra ---
    if local_extra:
        extra_counts = Counter()
        for (u, v) in local_extra:
            a, b = (u, v) if u < v else (v, u)
            extra_counts[(a, b)] += 1

        for (u, v), cnt in extra_counts.items():
            if cnt <= 0: continue
            mag = 0.28
            inward = rad_away_from_compute(u, v, mag)
            outward = -inward

            # Note: with multi-edges, we just store the last one for visualization defaults
            # or store both if we want to be fancy. For viewer, picking 'inward' is usually fine.
            edge_registry[(u, v)] = inward

            for rad in ([inward, outward] if cnt >= 2 else [inward]):
                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-", lw=2.4, color="black", alpha=0.65, zorder=1.0
                ))

    # --- subgroup internal clique ---
    for gid in range(cfg.groups):
        for sg in range(cfg.subgroups_per_group):
            sw = [n for n, d in G.nodes(data=True)
                  if d.get("kind") == "switch" and d.get("group") == gid and d.get("subgroup") == sg]
            sw.sort(key=lambda n: G.nodes[n].get("switch_in_subgroup", 0))
            if len(sw) != 4: continue
            s1, s2, s3, s4 = sw[0], sw[1], sw[2], sw[3]

            # Straight S2-S3
            edge_registry[(s2, s3)] = 0.0
            ax.plot([pos[s2][0], pos[s3][0]], [pos[s2][1], pos[s3][1]],
                    color="black", alpha=0.55, lw=1.2, zorder=1.2)

            # Curved others
            for (u, v, mag) in [(s1, s2, 0.18), (s3, s4, 0.18), (s1, s3, 0.22), (s2, s4, 0.22), (s1, s4, 0.32)]:
                rad = rad_away_from_compute(u, v, mag)

                edge_registry[(u, v)] = -rad

                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v],
                    connectionstyle=f"arc3,rad={-rad}",
                    arrowstyle="-", lw=1.2, color="black", alpha=0.50, zorder=1.2
                ))

    # --- injection edges ---
    if inj_edges:
        nx.draw_networkx_edges(nx.Graph(inj_edges), pos, alpha=0.45, width=1.6, edge_color="black")
        # Injection edges are straight, rad = 0.0 (default behavior of getter)

    # --- nodes ---
    ax.scatter([pos[n][0] for n in switches], [pos[n][1] for n in switches],
               s=cfg.node_size_switch, c="tab:green", edgecolors="black", linewidths=0.8, zorder=10, label="Комутатори")
    ax.scatter([pos[n][0] for n in compute_nodes], [pos[n][1] for n in compute_nodes],
               s=cfg.node_size_compute, c="tab:blue", edgecolors="black", linewidths=0.5, zorder=10,
               label="Обчислювальні вузли")

    fig.tight_layout()
    ax.legend(scatterpoints=1, frameon=False, loc="upper left")

    # RETURN REGISTRY TOO
    return fig, ax, edge_registry

import random
import networkx as nx


def compute_metrics_ua(G: nx.MultiGraph, cfg, scope: str = "switches") -> dict:
    """
    scope:
      - "switches": метрики тільки по комутаторах (рекомендовано)
      - "all": по всіх вузлах (комутатори + compute)
    """
    if scope == "switches":
        nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "switch"]
        H = nx.Graph(G.subgraph(nodes))  # простий граф
    else:
        H = nx.Graph(G)

    N = H.number_of_nodes()
    E = H.number_of_edges()

    if N == 0:
        return {
            "N": 0, "S_max": 0, "S_сер": 0,
            "D": 0, "D*": 0, "Q": 0,
            "C_лінки": 0, "C_порти": 0,
            "компонента": 0
        }

    degrees = [deg for _, deg in H.degree()]
    S_max = max(degrees) if degrees else 0
    S_avg = (sum(degrees) / len(degrees)) if degrees else 0.0

    # якщо граф не зв'язний — беремо найбільшу зв'язну компоненту для D, D*, Q
    if not nx.is_connected(H):
        largest = max(nx.connected_components(H), key=len)
        Hc = H.subgraph(largest).copy()
    else:
        Hc = H

    Nc = Hc.number_of_nodes()

    # D
    if Nc <= 800:
        D = nx.diameter(Hc)
    else:
        # наближення (5 стартів BFS)
        rnd = random.Random(getattr(cfg, "seed", 42))
        starts = [rnd.choice(list(Hc.nodes())) for _ in range(5)]
        best = 0
        for s in starts:
            lengths = nx.single_source_shortest_path_length(Hc, s)
            best = max(best, max(lengths.values()))
        D = best

    # D* та Q
    if Nc <= 500:
        D_star = nx.average_shortest_path_length(Hc)
        Q = 0
        for u in Hc.nodes():
            lengths = nx.single_source_shortest_path_length(Hc, u)
            Q += sum(lengths.values())
        Q = Q / 2  # бо порахували кожну пару двічі
    else:
        rnd = random.Random(getattr(cfg, "seed", 42))
        nodes_list = list(Hc.nodes())
        target = min(getattr(cfg, "sample_pairs", 4000), Nc * (Nc - 1) // 2)

        pairs = set()
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
        total_pairs = Nc * (Nc - 1) / 2
        Q = D_star * total_pairs

    C = D * N * S_max

    return {
        "N": int(N),
        "S_max": float(S_max),
        "S_сер": float(S_avg),
        "D": float(D),
        "D*": float(D_star),
        "Q": float(Q),
        "C": int(C)
    }


def print_metrics_ua(m: dict, scope: str):
    print("\n=== МЕТРИКИ ТОПОЛОГІЇ ===")
    print(
        f"Область розрахунку: {'лише комутатори' if scope == 'switches' else 'усі вузли (комутатори+обчислювальні вузли)'}")
    print(f"Кількість вузлів N: {m['N']}")
    print(f"Ступінь топології S: {m['S_max']:.2f}")
    print(f"Діаметр D: {m['D']:.2f}")
    print(f"Середній діаметр D_сер: {m['D*']:.4f}")
    print(f"Топологічний трафік Q: {m['Q']:.2f}")
    print(f"Вартість C: {m['C']}")


class CastViewer:
    def __init__(self, G: nx.MultiGraph, pos: Dict[str, Tuple[float, float]], cfg: TopologyConfig):
        self.G = G
        self.pos = pos
        self.cfg = cfg

        # --- UI state ---
        self.mode = "VIEW"
        self.cast = "UNICAST"
        self.show_discovery = True
        self.src: Optional[str] = None
        self.dst: Optional[str] = None
        self.dsts: Set[str] = set()

        # --- simulation state ---
        self.sim_active = False
        self.sim_phase = "IDLE"
        self.H: Optional[nx.Graph] = None

        # BFS state
        self.bfs_q = deque()
        self.bfs_parent = {}
        self.bfs_visited = set()
        self.bfs_neighbors = {}
        self.bfs_idx = {}

        # STORE NODES INSTEAD OF SEGMENTS FOR CURVATURE LOOKUP
        self.bfs_checked_edges: List[Tuple[str, str]] = []

        self.discovery_targets = None
        self.discovery_targets_found = set()

        # Delivery state
        self.path = []
        self.path_i = 0
        self.tree_children = {}
        self.received = set()
        self.frontier = []

        # STORE NODES INSTEAD OF SEGMENTS
        self.delivered_edges: List[Tuple[str, str]] = []
        self.current_wave_edges: List[Tuple[str, str]] = []

        # --- Base Draw with Registry ---
        self.fig, self.ax, self.edge_registry = draw_topology_base(G, pos, cfg)
        self.fig = self.ax.figure

        # --- Dynamic Patches Storage ---
        # We need to track patches to remove them on updates
        self.active_patches: List[FancyArrowPatch] = []

        self._init_overlays()

        # ... (rest of init for hud/events remains same) ...
        self.hud = self.ax.text(0.02, 0.02, self._hud_text(), transform=self.ax.transAxes, fontsize=10, va="bottom")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    # ... (Helper methods for hud, empty_offsets, nearest_node remain same) ...
    def _empty_offsets(self):
        return np.empty((0, 2))

    def _active_graph(self) -> nx.Graph:
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())
        for u, v, _d in self.G.edges(data=True): H.add_edge(u, v)
        return H

    def _hud_text(self) -> str:
        # (Same as before)
        def short(x): return x if x is not None else "—"

        return (f"Cast: {self.cast} | Phase: {self.sim_phase} | Discovery: {'ON' if self.show_discovery else 'OFF'}\n"
                f"Mode: {self.mode} | src: {short(self.src)} | dst: {short(self.dst)} | m_dsts: {len(self.dsts)}\n"
                "Keys: Space(step) r(reset) ...")

    def _update_hud(self):
        self.hud.set_text(self._hud_text())

    def _set_msg(self, txt: str):
        self.msg.set_text(txt)

    def _nearest_node(self, x, y):
        # (Same as before)
        thr2 = 0.75 ** 2
        best, best_d2 = None, 1e18
        for n, (nx_, ny_) in self.pos.items():
            d2 = (nx_ - x) ** 2 + (ny_ - y) ** 2
            if d2 < best_d2: best, best_d2 = n, d2
        return best if best is not None and best_d2 <= thr2 else None

    # ---------- overlays ----------
    def _init_overlays(self):
        # Nodes / Markers
        self.src_sc = self.ax.scatter([], [], s=240, facecolors="none", edgecolors="black", linewidths=2.2, zorder=40)
        self.dst_sc = self.ax.scatter([], [], s=240, facecolors="none", edgecolors="black", linewidths=2.2, zorder=40)
        self.dsts_sc = self.ax.scatter([], [], s=190, facecolors="none", edgecolors="black", linewidths=1.6, zorder=39)
        self.frontier_sc = self.ax.scatter([], [], s=170, facecolors="none", edgecolors="red", linewidths=2.0,
                                           zorder=35)
        self.visited_sc = self.ax.scatter([], [], s=120, facecolors="none", edgecolors="red", linewidths=1.0,
                                          alpha=0.25, zorder=34)
        self.packet_sc = self.ax.scatter([], [], s=95, c="red", zorder=45)
        self.msg = self.ax.text(0.5, 0.98, "", transform=self.ax.transAxes, fontsize=11, va="top", ha="center")

        # Remove LineCollections! We will use self.active_patches instead.

        self.show_dsts_panel = True
        self.dsts_panel = self.ax.text(0.98, 0.02, "", transform=self.ax.transAxes, fontsize=9, va="bottom", ha="right",
                                       bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.75),
                                       zorder=60)
        self.tooltip = self.ax.text(0, 0, "", fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85), zorder=70,
                                    visible=False)

    # --- KEY FIX: Drawing helper ---
    def _draw_curved_edge(self, u, v, color, lw, alpha, zorder):
        # Determine curvature from registry
        # The registry stores 'rad' for (u, v).
        # If we are drawing u->v, we use rad.
        # If we are drawing v->u (path reverse), we must invert rad relative to posA/posB.

        rad = 0.0
        if (u, v) in self.edge_registry:
            rad = self.edge_registry[(u, v)]
        elif (v, u) in self.edge_registry:
            # If the original edge was v->u with rad R,
            # drawing u->v requires -R to match the same physical arc.
            rad = -self.edge_registry[(v, u)]

        patch = FancyArrowPatch(
            posA=self.pos[u], posB=self.pos[v],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-",
            color=color, lw=lw, alpha=alpha, zorder=zorder
        )
        self.ax.add_patch(patch)
        self.active_patches.append(patch)

    def _redraw_dynamic_edges(self):
        # 1. Clear old dynamic patches
        for p in self.active_patches:
            p.remove()
        self.active_patches.clear()

        # 2. Draw BFS Checked Edges (Pale Red)
        for u, v in self.bfs_checked_edges:
            self._draw_curved_edge(u, v, "red", 1.6, 0.25, 30)

        # 3. Draw Delivery History (Medium Red)
        for u, v in self.delivered_edges:
            self._draw_curved_edge(u, v, "red", 3.0, 0.55, 32)

        # 4. Draw Current Wave / Hop (Bright Red)
        for u, v in self.current_wave_edges:
            self._draw_curved_edge(u, v, "red", 4.2, 0.95, 33)

    # ... (Keep _update_dsts_panel, _show_tooltip, _hide_tooltip, _update_src_dst_markers) ...
    def _update_dsts_panel(self):
        # (Copy your existing code here)
        if not getattr(self, "show_dsts_panel", True): self.dsts_panel.set_visible(False); return
        if self.cast != "MULTICAST": self.dsts_panel.set_visible(False); return
        self.dsts_panel.set_visible(True)
        items = sorted(self.dsts)
        if not items: self.dsts_panel.set_text("Multicast group:\n(empty)"); return
        N = 10;
        head = items[:N];
        more = len(items) - len(head)
        text = "Multicast group:\n" + "\n".join(head)
        if more > 0: text += f"\n… +{more} more"
        self.dsts_panel.set_text(text)

    def _show_tooltip(self, node_id, x, y):
        # (Copy your existing code)
        d = self.G.nodes[node_id];
        self.tooltip.set_text(f"{node_id}\n{d.get('kind', '?')}");
        self.tooltip.set_position((x, y));
        self.tooltip.set_visible(True)

    def _hide_tooltip(self):
        self.tooltip.set_visible(False)

    def _update_src_dst_markers(self):
        # (Copy your existing code)
        self.src_sc.set_offsets([self.pos[self.src]] if self.src else self._empty_offsets())
        if self.cast == "MULTICAST":
            self.dst_sc.set_offsets(self._empty_offsets())
        else:
            self.dst_sc.set_offsets([self.pos[self.dst]] if self.dst else self._empty_offsets())
        self.dsts_sc.set_offsets([self.pos[n] for n in sorted(self.dsts)] if self.dsts else self._empty_offsets())
        self._update_dsts_panel()

    def _reset_overlays(self, keep_selection: bool = True):
        self.sim_active = False
        self.sim_phase = "IDLE"
        self.H = None

        self.bfs_q.clear();
        self.bfs_parent.clear();
        self.bfs_visited.clear()
        self.bfs_neighbors.clear();
        self.bfs_idx.clear()

        self.bfs_checked_edges = []  # Clear edge list

        self.discovery_targets = None
        self.discovery_targets_found = set()

        self.path = [];
        self.path_i = 0
        self.tree_children = {};
        self.received = set();
        self.frontier = []

        self.delivered_edges = []  # Clear edge list
        self.current_wave_edges = []  # Clear edge list

        self.frontier_sc.set_offsets(self._empty_offsets())
        self.visited_sc.set_offsets(self._empty_offsets())
        self.packet_sc.set_offsets(self._empty_offsets())

        self._redraw_dynamic_edges()  # Updates visuals

        self._set_msg("")
        if not keep_selection:
            self.src = None;
            self.dst = None;
            self.dsts.clear()

        self._update_src_dst_markers()
        self._update_hud()
        self.fig.canvas.draw_idle()

    # ... (Event handlers _on_click, _on_key remain exactly the same) ...
    def _on_click(self, event):
        # (Copy your existing code, no changes needed)
        if event.inaxes != self.ax or event.xdata is None: return
        n = self._nearest_node(event.xdata, event.ydata)
        if n is None: return
        if event.button == 3:
            self._show_tooltip(n, event.xdata, event.ydata); self.fig.canvas.draw_idle(); return
        else:
            self._hide_tooltip()
        if self.mode == "SELECT_SOURCE":
            self.src = None if self.src == n else n
            self._set_msg(f"Source: {self.src}")
            self.mode = "VIEW";
            self._update_src_dst_markers();
            self._update_hud();
            self.fig.canvas.draw_idle();
            return
        if self.mode == "SELECT_DEST_ONE":
            self.dst = None if self.dst == n else n
            self._set_msg(f"Dest: {self.dst}")
            self.mode = "VIEW";
            self._update_src_dst_markers();
            self._update_hud();
            self.fig.canvas.draw_idle();
            return
        if self.mode == "SELECT_DEST_MANY":
            if n in self.dsts:
                self.dsts.remove(n)
            else:
                self.dsts.add(n)
            self._update_src_dst_markers();
            self._update_hud();
            self.fig.canvas.draw_idle();
            return

    def _on_key(self, event):
        # (Copy your existing code, no changes needed)
        k = (event.key or "").lower()
        if k == "u":
            self.cast = "UNICAST"; self.mode = "VIEW"; self._set_msg(
                "Unicast"); self._update_hud(); self.fig.canvas.draw_idle()
        elif k == "b":
            self.cast = "BROADCAST"; self.mode = "VIEW"; self._set_msg(
                "Broadcast"); self._update_hud(); self.fig.canvas.draw_idle()
        elif k == "m":
            self.cast = "MULTICAST"; self.mode = "VIEW"; self._set_msg(
                "Multicast"); self._update_hud(); self.fig.canvas.draw_idle()
        elif k == "s":
            self.mode = "SELECT_SOURCE"; self._set_msg("Click Source"); self._update_hud(); self.fig.canvas.draw_idle()
        elif k == "d":
            self.mode = "SELECT_DEST_MANY" if self.cast == "MULTICAST" else "SELECT_DEST_ONE"; self._set_msg(
                "Click Dest"); self._update_hud(); self.fig.canvas.draw_idle()
        elif k == "c":
            self.dsts.clear(); self._update_src_dst_markers(); self.fig.canvas.draw_idle()
        elif k == "w":
            self.show_discovery = not self.show_discovery; self._update_hud(); self.fig.canvas.draw_idle()
        elif k == "r":
            self._reset_overlays(True)
        elif k in ("enter", "return"):
            self.prepare()
        elif k in (" ", "space", "n"):
            self.step_once()
        elif k == "l":
            self.show_dsts_panel = not getattr(self, "show_dsts_panel",
                                               True); self._update_src_dst_markers(); self.fig.canvas.draw_idle()
        elif k == "z":
            self.src = None; self._update_src_dst_markers(); self.fig.canvas.draw_idle()
        elif k == "x":
            self.dst = None; self._update_src_dst_markers(); self.fig.canvas.draw_idle()

    # ---------- prepare ----------
    def prepare(self):
        self._reset_overlays(keep_selection=True)
        if self.src is None: self._set_msg("Set src first."); self.fig.canvas.draw_idle(); return
        self.H = self._active_graph()

        # ... (Rest of prepare logic is the same, just calling init methods) ...
        if self.cast == "UNICAST":
            if self.dst is None: self._set_msg("Set dst first."); return
            if self.show_discovery:
                self._init_bfs_discovery(targets={self.dst})
            else:
                self._init_unicast_delivery_direct()
        elif self.cast == "BROADCAST":
            if self.show_discovery:
                self._init_bfs_discovery(targets=None)
            else:
                self._build_bfs_parents_full(); self._init_tree_delivery_from_parents(True)
        elif self.cast == "MULTICAST":
            if not self.dsts: self._set_msg("Pick multicast group."); return
            if self.show_discovery:
                self._init_bfs_discovery(targets=set(self.dsts))
            else:
                self._build_bfs_parents_until_targets(set(self.dsts)); self._init_multicast_tree_delivery(
                    set(self.dsts))

        self.sim_active = True
        self._update_src_dst_markers();
        self._update_hud();
        self.fig.canvas.draw_idle()

    # ---------- BFS init / stepping ----------
    def _init_bfs_discovery(self, targets):
        self.sim_phase = "DISCOVERY"
        self.discovery_targets = targets
        self.discovery_targets_found = set()
        self.bfs_q = deque([self.src])
        self.bfs_parent = {self.src: None}
        self.bfs_visited = {self.src}
        self.bfs_checked_edges = []  # Reset
        self.current_wave_edges = []

        self.bfs_neighbors = {}
        self.bfs_idx = {}
        for u in self.H.nodes():
            nbrs = sorted(list(self.H.neighbors(u)))
            seed_u = (int(self.cfg.seed) * 1000003) ^ zlib.adler32(str(u).encode("utf-8"))
            rnd_u = random.Random(seed_u)
            rnd_u.shuffle(nbrs)
            self.bfs_neighbors[u] = nbrs
            self.bfs_idx[u] = 0

        self.visited_sc.set_offsets([self.pos[self.src]])
        self.frontier_sc.set_offsets([self.pos[self.src]])
        self._redraw_dynamic_edges()

    def _step_bfs_one_edge(self):
        while self.bfs_q:
            u = self.bfs_q[0]
            if self.bfs_idx[u] < len(self.bfs_neighbors[u]): break
            self.bfs_q.popleft()

        if not self.bfs_q:
            self.current_wave_edges = []
            self._redraw_dynamic_edges()
            return False

        u = self.bfs_q[0]
        v = self.bfs_neighbors[u][self.bfs_idx[u]]
        self.bfs_idx[u] += 1

        # STORE NODES, NOT COORDINATES
        self.bfs_checked_edges.append((u, v))
        self.current_wave_edges = [(u, v)]
        self._redraw_dynamic_edges()  # Trigger redraw

        if v not in self.bfs_visited:
            self.bfs_visited.add(v)
            self.bfs_parent[v] = u
            self.bfs_q.append(v)

        self.visited_sc.set_offsets([self.pos[n] for n in self.bfs_visited])
        self.frontier_sc.set_offsets([self.pos[n] for n in self.bfs_q] if self.bfs_q else self._empty_offsets())

        if self.discovery_targets is not None and v in self.discovery_targets:
            self.discovery_targets_found.add(v)

        self._set_msg(f"Checked: {u} → {v}")
        return True

    # ... (Helpers _build_bfs_parents... remain same) ...
    def _build_bfs_parents_full(self):
        self._init_bfs_discovery(None)
        while self._step_bfs_one_edge(): pass
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.current_wave_edges = [];
        self.bfs_checked_edges = []  # Clear discovery visuals
        self._redraw_dynamic_edges()

    def _build_bfs_parents_until_targets(self, targets):
        self._init_bfs_discovery(targets)
        while True:
            if not self._step_bfs_one_edge(): break
            if self.discovery_targets_found == set(targets): break
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.current_wave_edges = [];
        self.bfs_checked_edges = []
        self._redraw_dynamic_edges()

    def _reconstruct_path(self, dst):
        if dst not in self.bfs_parent: return []
        p = [];
        cur = dst
        while cur is not None: p.append(cur); cur = self.bfs_parent.get(cur)
        p.reverse();
        return p

    # ---------- init delivery ----------
    def _init_unicast_delivery_direct(self):
        self.sim_phase = "DELIVERY"
        try:
            self.path = nx.shortest_path(self.H, self.src, self.dst)
        except nx.NetworkXNoPath:
            self.sim_phase = "DONE"; self.sim_active = False; self._set_msg("No path."); return
        self.path_i = 0
        self.packet_sc.set_offsets([self.pos[self.path[0]]])
        self.delivered_edges = []
        self.current_wave_edges = []
        self._redraw_dynamic_edges()

    def _init_tree_delivery_from_parents(self, all_targets):
        self.sim_phase = "DELIVERY"
        self.tree_children = {}
        for v, p in self.bfs_parent.items():
            if p is None: continue
            self.tree_children.setdefault(p, []).append(v)
        self.received = {self.src};
        self.frontier = [self.src]
        self.delivered_edges = [];
        self.current_wave_edges = []
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.bfs_checked_edges = []
        self._redraw_dynamic_edges()

    def _init_multicast_tree_delivery(self, targets):
        edges = set();
        used_nodes = {self.src}
        for t in targets:
            path = self._reconstruct_path(t)
            if not path: continue
            used_nodes.update(path)
            for i in range(len(path) - 1): edges.add((path[i], path[i + 1]))
        self.tree_children = {}
        for a, b in edges: self.tree_children.setdefault(a, []).append(b)
        self.sim_phase = "DELIVERY"
        self.received = {self.src};
        self.frontier = [self.src]
        self.delivered_edges = [];
        self.current_wave_edges = []
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.bfs_checked_edges = []
        self._redraw_dynamic_edges()

    # ---------- stepping ----------
    def step_once(self):
        if not self.sim_active or self.sim_phase in ("IDLE", "DONE"): return
        if self.sim_phase == "DISCOVERY":
            if not self._step_bfs_one_edge(): self._finish_discovery(); self.fig.canvas.draw_idle(); return
            if self.cast == "UNICAST" and self.dst in self.bfs_visited:
                self._finish_discovery()
            elif self.cast == "MULTICAST" and self.discovery_targets and self.discovery_targets_found == self.discovery_targets:
                self._finish_discovery()
        elif self.sim_phase == "DELIVERY":
            if self.cast == "UNICAST":
                self._step_unicast_one_hop()
            elif self.cast == "BROADCAST":
                self._step_tree_one_wave("Broadcast wave")
            elif self.cast == "MULTICAST":
                self._step_tree_one_wave("Multicast wave")
        self._update_hud();
        self.fig.canvas.draw_idle()

    def _finish_discovery(self):
        if self.cast == "UNICAST":
            self.path = self._reconstruct_path(self.dst)
            if not self.path: self.sim_phase = "DONE"; self.sim_active = False; self._set_msg("No path."); return
            self.sim_phase = "DELIVERY";
            self.path_i = 0
            self.packet_sc.set_offsets([self.pos[self.path[0]]])
            self.delivered_edges = [];
            self.current_wave_edges = []
            self.bfs_checked_edges = []  # Clear discovery
            self._redraw_dynamic_edges()
            self._set_msg("Path found. Step to deliver.")
            self.frontier_sc.set_offsets(self._empty_offsets());
            self.visited_sc.set_offsets(self._empty_offsets())
        elif self.cast == "BROADCAST":
            self._init_tree_delivery_from_parents(True);
            self._set_msg("Broadcast tree ready.")
        elif self.cast == "MULTICAST":
            self._init_multicast_tree_delivery(set(self.dsts));
            self._set_msg("Multicast tree ready.")

    def _step_unicast_one_hop(self):
        if not self.path: self.sim_phase = "DONE"; self.sim_active = False; self._set_msg("No path."); return
        if self.path_i >= len(self.path) - 1:
            self.sim_phase = "DONE";
            self.sim_active = False;
            self._set_msg("Done.");
            self.current_wave_edges = [];
            self._redraw_dynamic_edges();
            return

        a = self.path[self.path_i];
        b = self.path[self.path_i + 1]
        self.path_i += 1

        # STORE NODES
        self.delivered_edges.append((a, b))
        self.current_wave_edges = [(a, b)]
        self._redraw_dynamic_edges()

        self.packet_sc.set_offsets([self.pos[b]])
        self._set_msg(f"Hop: {a} → {b}")

    def _step_tree_one_wave(self, label):
        if not self.frontier:
            self.sim_phase = "DONE";
            self.sim_active = False;
            self._set_msg("Done.");
            self.current_wave_edges = [];
            self._redraw_dynamic_edges();
            return

        next_frontier = [];
        wave_edges = []
        for u in self.frontier:
            for v in self.tree_children.get(u, []):
                if v in self.received: continue
                self.received.add(v)
                next_frontier.append(v)
                wave_edges.append((u, v))

        self.delivered_edges.extend(wave_edges)
        self.current_wave_edges = wave_edges
        self._redraw_dynamic_edges()

        self.frontier = next_frontier
        self.frontier_sc.set_offsets([self.pos[n] for n in self.frontier] if self.frontier else self._empty_offsets())

        if wave_edges:
            self._set_msg(f"{label}: +{len(next_frontier)} nodes")
        else:
            self._set_msg("Done."); self.sim_phase = "DONE"; self.sim_active = False

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

    m = compute_metrics_ua(G, cfg, scope="all")
    print_metrics_ua(m, scope="all")

    viewer = CastViewer(G, pos, cfg)
    plt.show()


