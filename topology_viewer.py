import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import numpy as np
import numpy as np
from collections import deque
import zlib
import random


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
def draw_topology_base(G: nx.MultiGraph, pos: Dict[str, Tuple[float, float]], cfg: TopologyConfig) -> None:
    fig, ax = plt.subplots(figsize=cfg.figsize)
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
               edgecolors="black", linewidths=0.8, zorder=10, label="Комутатори")

    ax.scatter([pos[n][0] for n in compute_nodes], [pos[n][1] for n in compute_nodes],
               s=cfg.node_size_compute, c="tab:blue",
               edgecolors="black", linewidths=0.5, zorder=10, label="Обчислювальні вузли")

    fig.tight_layout()
    ax.legend(scatterpoints=1, frameon=False, loc="upper left")
    return fig, ax


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


class UnicastViewer:
    def __init__(self, G: nx.MultiGraph, pos: Dict[str, Tuple[float, float]], cfg: TopologyConfig):
        self.show_discovery = True

        # simulation machine
        self.sim_active = False
        self.sim_phase = "IDLE"  # IDLE, DISCOVERY, DELIVERY, DONE
        self.H = None  # active simple graph

        # BFS state (for discovery)
        self.bfs_q = deque()
        self.bfs_parent = {}
        self.bfs_visited = set()
        self.bfs_neighbors = {}  # node -> sorted neighbor list
        self.bfs_idx = {}  # node -> next neighbor index
        self.bfs_checked_segments = []  # list of ((x1,y1),(x2,y2))

        # delivery state
        self.path = []
        self.path_i = 0  # current index in path (packet at path[path_i])

        self.G = G
        self.pos = pos
        self.cfg = cfg

        # state
        self.mode = "VIEW"  # VIEW, SELECT_SOURCE, SELECT_DEST
        self.cast = "UNICAST"
        self.src: Optional[str] = None
        self.dst: Optional[str] = None

        # animation state
        self.frames: List[dict] = []
        self.frame_idx = 0
        self.running = False
        self.paused = False
        self.timer = None

        # draw base
        self.fig, self.ax = draw_topology_base(G, pos, cfg)

        # overlays
        self._init_overlays()

        # UI text
        self.hud = self.ax.text(
            0.02, 0.02, self._hud_text(),
            transform=self.ax.transAxes,
            fontsize=10, va="bottom", ha="left"
        )

        # events
        self.cid_click = self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.cid_key = self.fig.canvas.mpl_connect("key_press_event", self._on_key)


    def _empty_offsets(self):
        return np.empty((0, 2))

    # ---------- overlays ----------
    def _init_overlays(self):
        # source/dest markers
        self.src_sc = self.ax.scatter([], [], s=220, facecolors="none", edgecolors="black",
                                      linewidths=2.2, zorder=30)
        self.dst_sc = self.ax.scatter([], [], s=220, facecolors="none", edgecolors="black",
                                      linewidths=2.2, zorder=30)

        # discovery overlay
        self.frontier_sc = self.ax.scatter([], [], s=180, facecolors="none", edgecolors="black",
                                           linewidths=1.8, alpha=0.9, zorder=25)
        self.visited_sc = self.ax.scatter([], [], s=120, facecolors="none", edgecolors="black",
                                          linewidths=1.0, alpha=0.25, zorder=20)

        # path overlay
        self.path_bg = LineCollection([], linewidths=3.0, alpha=0.25, zorder=22)
        self.path_fg = LineCollection([], linewidths=3.5, alpha=0.85, zorder=23)
        self.ax.add_collection(self.path_bg)
        self.ax.add_collection(self.path_fg)

        # packet marker
        self.packet_sc = self.ax.scatter([], [], s=90, zorder=35)

        # message
        self.msg = self.ax.text(
            0.5, 0.98, "",
            transform=self.ax.transAxes,
            fontsize=11, va="top", ha="center"
        )

        self.checked_edges_lc = LineCollection([], colors="red", linewidths=1.5, alpha=0.35, zorder=24)
        self.ax.add_collection(self.checked_edges_lc)

        self.current_edge_lc = LineCollection([], colors="red", linewidths=3.2, alpha=0.9, zorder=26)
        self.ax.add_collection(self.current_edge_lc)


    def _reset_overlays(self, keep_src_dst: bool = True):
        # stop sim
        self.sim_active = False
        self.sim_phase = "IDLE"
        self.H = None

        # reset BFS state
        self.bfs_q.clear()
        self.bfs_parent.clear()
        self.bfs_visited.clear()
        self.bfs_neighbors.clear()
        self.bfs_idx.clear()
        self.bfs_checked_segments = []

        # reset delivery
        self.path = []
        self.path_i = 0

        self.frames = []
        self.frame_idx = 0
        self.running = False
        self.paused = False
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

        self.frontier_sc.set_offsets(self._empty_offsets())
        self.visited_sc.set_offsets(self._empty_offsets())
        self.path_bg.set_segments([])
        self.path_fg.set_segments([])
        self.packet_sc.set_offsets(self._empty_offsets())
        self.msg.set_text("")

        # NEW: clear red discovery edges
        self.checked_edges_lc.set_segments([])
        self.current_edge_lc.set_segments([])

        if not keep_src_dst:
            self.src = None
            self.dst = None
            self.src_sc.set_offsets(self._empty_offsets())
            self.dst_sc.set_offsets(self._empty_offsets())


        self._update_src_dst_markers()
        self._update_hud()
        self.fig.canvas.draw_idle()


    # ---------- graph for routing ----------
    def _active_graph(self) -> nx.Graph:
        # For Stage 2 unicast we assume everything is enabled.
        # (We’ll extend this later to respect disabled nodes/edges.)
        H = nx.Graph()
        H.add_nodes_from(self.G.nodes())
        for u, v, d in self.G.edges(data=True):
            H.add_edge(u, v)  # Multi edges collapse to one
        return H

    # ---------- selection helpers ----------
    def _nearest_node(self, x: float, y: float) -> Optional[str]:
        # distance threshold depends on layout scale
        # ring radius ~ 12, so 0.5 is a nice click radius
        thr2 = 0.55 ** 2
        best = None
        best_d2 = 1e18
        for n, (nx_, ny_) in self.pos.items():
            d2 = (nx_ - x) ** 2 + (ny_ - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = n
        return best if best is not None and best_d2 <= thr2 else None

    def _update_src_dst_markers(self):
        if self.src is not None:
            self.src_sc.set_offsets([self.pos[self.src]])
        else:
            self.src_sc.set_offsets(self._empty_offsets())

        if self.dst is not None:
            self.dst_sc.set_offsets([self.pos[self.dst]])
        else:
            self.dst_sc.set_offsets(self._empty_offsets())

    # ---------- HUD ----------
    def _hud_text(self) -> str:
        def short(n: Optional[str]) -> str:
            return n if n is not None else "—"

        return (
            f"Mode: {self.mode} | Cast: {self.cast} | Discovery: {'ON' if self.show_discovery else 'OFF'}\n"
            f"src: {short(self.src)}\n"
            f"dst: {short(self.dst)}\n"
            "Keys: s(src) d(dst) u(unicast) w(toggle discovery) Enter(run) Space(pause) .(step) r(reset)"
        )

    def _update_hud(self):
        self.hud.set_text(self._hud_text())

    def _set_msg(self, text: str):
        self.msg.set_text(text)

    # ---------- events ----------
    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return

        n = self._nearest_node(event.xdata, event.ydata)
        if n is None:
            return

        if self.mode == "SELECT_SOURCE":
            self.src = n
            self._set_msg(f"Source set: {n}")
            self.mode = "VIEW"
            self._update_src_dst_markers()
            self._update_hud()
            self.fig.canvas.draw_idle()

        elif self.mode == "SELECT_DEST":
            self.dst = n
            self._set_msg(f"Destination set: {n}")
            self.mode = "VIEW"
            self._update_src_dst_markers()
            self._update_hud()
            self.fig.canvas.draw_idle()

    def _on_key(self, event):
        k = (event.key or "").lower()

        if k == "w":
            self.show_discovery = not self.show_discovery
            self._set_msg(f"Discovery wave: {'ON' if self.show_discovery else 'OFF'}")
            self._update_hud()
            self.fig.canvas.draw_idle()
            return

        if k == "s":
            self.mode = "SELECT_SOURCE"
            self._set_msg("Click a node to set SOURCE")
            self._update_hud()
            self.fig.canvas.draw_idle()
            return

        if k == "d":
            self.mode = "SELECT_DEST"
            self._set_msg("Click a node to set DESTINATION")
            self._update_hud()
            self.fig.canvas.draw_idle()
            return

        if k == "u":
            self.cast = "UNICAST"
            self._set_msg("Unicast selected. Set src/dst and press Enter.")
            self._update_hud()
            self.fig.canvas.draw_idle()
            return

        if k == "r":
            self._reset_overlays(keep_src_dst=True)
            return

        if k in (" ", "space", "n"):
            self.step_once()
            return

        if k in ("enter", "return"):
            self.prepare_unicast()  # does NOT step
            return

    def prepare_unicast(self):
        # clear dynamic overlays but keep src/dst markers
        self._reset_overlays(keep_src_dst=True)

        if self.src is None or self.dst is None:
            self._set_msg("Set src and dst first (s + click, d + click).")
            self.fig.canvas.draw_idle()
            return

        self.H = self._active_graph()

        if self.src not in self.H or self.dst not in self.H:
            self._set_msg("src/dst not in active graph.")
            self.fig.canvas.draw_idle()
            return

        self.sim_active = True

        if self.show_discovery:
            # initialize BFS state, but DO NOT check any edge yet
            self.sim_phase = "DISCOVERY"
            self.bfs_q = deque([self.src])
            self.bfs_parent = {self.src: None}
            self.bfs_visited = {self.src}
            self.bfs_checked_segments = []

            # deterministic neighbor order
            self.bfs_neighbors = {}
            for u in self.H.nodes():
                nbrs = sorted(list(self.H.neighbors(u)))  # stable base
                # per-node deterministic seed based on cfg.seed and node name
                seed_u = (int(self.cfg.seed) * 1000003) ^ zlib.adler32(str(u).encode("utf-8"))
                rnd_u = random.Random(seed_u)
                rnd_u.shuffle(nbrs)
                self.bfs_neighbors[u] = nbrs
            self.bfs_idx = {u: 0 for u in self.H.nodes()}

            # show initial visited/frontier (optional)
            self.visited_sc.set_offsets([self.pos[self.src]])
            self.frontier_sc.set_offsets([self.pos[self.src]])
            self._set_msg("Prepared DISCOVERY. Press Space to check next edge.")
        else:
            # skip discovery: compute path now, but DO NOT traverse any hop
            try:
                self.path = nx.shortest_path(self.H, self.src, self.dst)
            except nx.NetworkXNoPath:
                self.sim_phase = "DONE"
                self.sim_active = False
                self._set_msg("No path.")
                self.fig.canvas.draw_idle()
                return

            self.sim_phase = "DELIVERY"
            self.path_i = 0

            # path background for context (optional)
            segs = [(self.pos[self.path[i]], self.pos[self.path[i+1]]) for i in range(len(self.path)-1)]
            self.path_bg.set_segments(segs)
            self.path_fg.set_segments([])

            # packet starts at src but we don't move yet
            self.packet_sc.set_offsets([self.pos[self.path[0]]])

            self._set_msg("Prepared DELIVERY. Press Space to traverse next hop.")

        self._update_src_dst_markers()
        self._update_hud()
        self.fig.canvas.draw_idle()


    def step_once(self):
        if not self.sim_active or self.sim_phase in ("IDLE", "DONE"):
            return

        if self.sim_phase == "DISCOVERY":
            self._step_discovery_one_edge()
        elif self.sim_phase == "DELIVERY":
            self._step_delivery_one_hop()

        self.fig.canvas.draw_idle()


    def _step_discovery_one_edge(self):
        # find next node in queue that still has unchecked neighbors
        while self.bfs_q:
            u = self.bfs_q[0]
            nbrs = self.bfs_neighbors[u]
            i = self.bfs_idx[u]
            if i < len(nbrs):
                break
            self.bfs_q.popleft()

        if not self.bfs_q:
            # BFS exhausted
            self.sim_phase = "DONE"
            self.sim_active = False
            self.current_edge_lc.set_segments([])
            self._set_msg("Discovery finished: destination unreachable (no path).")
            return

        u = self.bfs_q[0]
        nbrs = self.bfs_neighbors[u]
        i = self.bfs_idx[u]
        v = nbrs[i]
        self.bfs_idx[u] = i + 1

        # "check" edge (u,v)
        seg = (self.pos[u], self.pos[v])
        self.bfs_checked_segments.append(seg)
        self.checked_edges_lc.set_segments(self.bfs_checked_segments)
        self.current_edge_lc.set_segments([seg])

        # BFS relax / discover
        if v not in self.bfs_visited:
            self.bfs_visited.add(v)
            self.bfs_parent[v] = u
            self.bfs_q.append(v)

        # update overlays: visited and frontier (= queue)
        self.visited_sc.set_offsets([self.pos[n] for n in self.bfs_visited] if self.bfs_visited else self._empty_offsets())
        self.frontier_sc.set_offsets([self.pos[n] for n in self.bfs_q] if self.bfs_q else self._empty_offsets())

        # if reached destination, finalize path but don't move yet
        if v == self.dst:
            self.path = self._reconstruct_path(self.dst)
            self.sim_phase = "DELIVERY"
            self.path_i = 0

            segs = [(self.pos[self.path[i]], self.pos[self.path[i+1]]) for i in range(len(self.path)-1)]
            self.path_bg.set_segments(segs)
            self.path_fg.set_segments([])
            self.packet_sc.set_offsets([self.pos[self.path[0]]])

            self.current_edge_lc.set_segments([])  # stop highlighting checks
            self._set_msg("Path found. Press Space to traverse next hop (DELIVERY).")
        else:
            self._set_msg(f"Checked edge: {u} → {v}")

    def _reconstruct_path(self, dst: str) -> List[str]:
        path = []
        cur = dst
        while cur is not None:
            path.append(cur)
            cur = self.bfs_parent.get(cur)
        path.reverse()
        return path

    def _step_delivery_one_hop(self):
        if not self.path or len(self.path) == 1:
            self.sim_phase = "DONE"
            self.sim_active = False
            self._set_msg("Trivial path (src==dst). Done.")
            return

        if self.path_i >= len(self.path) - 1:
            self.sim_phase = "DONE"
            self.sim_active = False
            self._set_msg("Delivered. Done.")
            return

        a = self.path[self.path_i]
        b = self.path[self.path_i + 1]

        # move packet to next node
        self.path_i += 1
        self.packet_sc.set_offsets([self.pos[b]])

        # extend highlighted delivered segments
        segs_done = [(self.pos[self.path[i]], self.pos[self.path[i+1]]) for i in range(self.path_i)]
        self.path_fg.set_segments(segs_done)

        self._set_msg(f"Traversed hop: {a} → {b}")

        self.frontier_sc.set_offsets(self._empty_offsets())
        self.visited_sc.set_offsets(self._empty_offsets())

    # ---------- unicast: frames ----------
    def start_unicast(self):
        self._reset_overlays(keep_src_dst=True)

        if self.src is None or self.dst is None:
            self._set_msg("Set src and dst first (press s/d, then click nodes).")
            self.fig.canvas.draw_idle()
            return

        H = self._active_graph()

        if self.src not in H or self.dst not in H:
            self._set_msg("src/dst not in active graph.")
            self.fig.canvas.draw_idle()
            return

        try:
            # shortest path for delivery
            path = nx.shortest_path(H, self.src, self.dst)
        except nx.NetworkXNoPath:
            self._set_msg("No path (graph disconnected or failures).")
            self.fig.canvas.draw_idle()
            return

        # build discovery wave (BFS frontiers) until dst reached
        discovery = self._bfs_discovery_frames(H, self.src, self.dst) if self.show_discovery else []
        delivery = self._delivery_frames(path)
        self.frames = discovery + delivery
        self.frame_idx = 0
        self.running = True
        self.paused = False

        # pre-draw full path background (for clarity)
        segs = [(self.pos[path[i]], self.pos[path[i+1]]) for i in range(len(path)-1)]
        self.path_bg.set_segments(segs)
        self.path_fg.set_segments([])

        # start timer
        self.timer = self.fig.canvas.new_timer(interval=260)
        self.timer.add_callback(self._advance_frame)
        self.timer.start()

        self._set_msg("Running unicast…")
        self._update_src_dst_markers()
        self._update_hud()
        self.fig.canvas.draw_idle()

    def _bfs_discovery_frames(self, H: nx.Graph, src: str, dst: str) -> List[dict]:
        visited: Set[str] = set([src])
        frontier: List[str] = [src]
        frames: List[dict] = []

        while frontier:
            frames.append({
                "phase": "DISCOVERY",
                "frontier": list(frontier),
                "visited": list(visited),
            })
            if dst in frontier:
                break

            nxt = []
            for u in frontier:
                for v in H.neighbors(u):
                    if v not in visited:
                        visited.add(v)
                        nxt.append(v)
            frontier = nxt

        return frames

    def _delivery_frames(self, path: List[str]) -> List[dict]:
        frames: List[dict] = []
        # packet starts at path[0], then moves each hop
        for i in range(len(path)):
            frames.append({
                "phase": "DELIVERY",
                "path": path,
                "hop": i,  # index in path where packet is
            })
        return frames

    def _advance_frame(self):
        if not self.running:
            return
        if self.paused:
            return

        if self.frame_idx >= len(self.frames):
            self.running = False
            self._set_msg("Done.")
            self.fig.canvas.draw_idle()
            if self.timer is not None:
                self.timer.stop()
                self.timer = None
            return

        fr = self.frames[self.frame_idx]
        self._render_frame(fr)
        self.frame_idx += 1
        self.fig.canvas.draw_idle()

    def _render_frame(self, fr: dict):
        if fr["phase"] == "DISCOVERY":
            frontier_pts = [self.pos[n] for n in fr["frontier"]]
            visited_pts = [self.pos[n] for n in fr["visited"]]

            self.frontier_sc.set_offsets(frontier_pts if frontier_pts else [])
            self.visited_sc.set_offsets(visited_pts if visited_pts else [])

            # during discovery, no packet movement yet
            self.packet_sc.set_offsets(self._empty_offsets())
            self.path_fg.set_segments([])

        elif fr["phase"] == "DELIVERY":
            path = fr["path"]
            hop = fr["hop"]

            # hide discovery overlays
            self.frontier_sc.set_offsets(self._empty_offsets())
            self.visited_sc.set_offsets(self._empty_offsets())

            # progress path highlight up to current hop
            segs_done = []
            for i in range(min(hop, len(path)-1)):
                segs_done.append((self.pos[path[i]], self.pos[path[i+1]]))
            self.path_fg.set_segments(segs_done)

            # packet position
            self.packet_sc.set_offsets([self.pos[path[hop]]])

            if hop == len(path) - 1:
                self._set_msg("Delivered.")



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

    viewer = UnicastViewer(G, pos, cfg)
    plt.show()


