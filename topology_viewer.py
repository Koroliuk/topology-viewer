import math
import zlib
import random
from collections import Counter
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyArrowPatch


@dataclass
class TopologyConfig:
    groups: int = 6
    subgroups_per_group: int = 4
    compute_nodes_per_subgroup: int = 8
    switches_per_subgroup: int = 4
    ports_per_switch: int = 26
    min_links_per_group_pair: int = 2
    seed: int = 42
    metric_scope: str = "switches"
    sample_pairs: int = 4000
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


def generate_dragonfly(cfg: TopologyConfig) -> nx.MultiGraph:
    random.Random(cfg.seed)
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
            subgroup_switches: List[str] = []
            for s in range(cfg.switches_per_subgroup):
                sid = f"g{gid}_sg{sg}_s{s}"
                G.add_node(sid, kind="switch", group=gid, subgroup=sg, switch_in_subgroup=s)
                subgroup_switches.append(sid)

            for n in range(cfg.compute_nodes_per_subgroup):
                nid = f"g{gid}_sg{sg}_n{n}"
                G.add_node(nid, kind="compute", group=gid, subgroup=sg, node_in_subgroup=n)
                for sid in subgroup_switches:
                    G.add_edge(sid, nid, kind="inj")

            for i in range(len(subgroup_switches)):
                for j in range(i + 1, len(subgroup_switches)):
                    G.add_edge(subgroup_switches[i], subgroup_switches[j], kind="subgroup_intra")

            s1, s2, s3, s4 = subgroup_switches
            add_k_edges(s1, s2, 2, kind="local_extra")
            add_k_edges(s3, s4, 2, kind="local_extra")

        sw = ordered_switches(gid)
        for i in range(len(sw)):
            for j in range(i + 1, len(sw)):
                a, b = sw[i], sw[j]
                if G.nodes[a]["subgroup"] != G.nodes[b]["subgroup"]:
                    G.add_edge(a, b, kind="group_intra")

    # ---- global links ----
    group_pairs = [(a, b) for a in range(cfg.groups) for b in range(a + 1, cfg.groups)]
    if not group_pairs:
        return G

    rr_idx = {gid: 0 for gid in range(cfg.groups)}

    def remaining_ports(sw: str) -> int:
        return cfg.ports_per_switch - G.degree(sw)

    def pick_switch_with_free_port(gid: int) -> str | None:
        sw = ordered_switches(gid)
        if not sw: return None
        start = rr_idx[gid] % len(sw)
        for t in range(len(sw)):
            s = sw[(start + t) % len(sw)]
            if remaining_ports(s) > 0:
                rr_idx[gid] = (start + t + 1) % len(sw)
                return s
        return None

    for (ga, gb) in group_pairs:
        for _ in range(cfg.min_links_per_group_pair):
            sa = pick_switch_with_free_port(ga)
            sb = pick_switch_with_free_port(gb)
            if sa is None or sb is None:
                raise TopologyConfigError("Not enough ports for min global connectivity")
            G.add_edge(sa, sb, kind="global")

    progress = True
    while progress:
        progress = False
        for (ga, gb) in group_pairs:
            sa = pick_switch_with_free_port(ga)
            sb = pick_switch_with_free_port(gb)
            if sa is None or sb is None: continue
            G.add_edge(sa, sb, kind="global")
            progress = True

    return G


# ----------------------------
# Ring layout
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

    compute_nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == "compute"]
    for n in compute_nodes:
        gid = G.nodes[n]["group"]
        sg = G.nodes[n]["subgroup"]
        subgroup_switches = [s for s in group_to_switches[gid] if G.nodes[s]["subgroup"] == sg]
        if not subgroup_switches: continue
        cx = sum(pos[s][0] for s in subgroup_switches) / len(subgroup_switches)
        cy = sum(pos[s][1] for s in subgroup_switches) / len(subgroup_switches)
        norm = math.hypot(cx, cy) or 1.0
        ux, uy = (cx / norm, cy / norm)
        nz = G.nodes[n].get("node_in_subgroup", 0)
        lateral = (nz - (cfg.compute_nodes_per_subgroup - 1) / 2) * 0.15
        px, py = (-uy, ux)
        pos[n] = (cx + ux * cfg.node_radius_offset + px * lateral, cy + uy * cfg.node_radius_offset + py * lateral)
    return pos


# ----------------------------
# Draw (Updated for Registry)
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
        nx_, ny_ = (-dy, dx)
        oux, ouy = out_vec.get(u, (0.0, 0.0))
        ovx, ovy = out_vec.get(v, (0.0, 0.0))
        outx, outy = ((oux + ovx) / 2.0, (ouy + ovy) / 2.0)
        inx, iny = (-outx, -outy)
        return mag if (nx_ * inx + ny_ * iny) > 0 else -mag

    global_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "global"]
    group_intra = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "group_intra"]
    local_extra = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "local_extra"]
    inj_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("kind") == "inj"]

    if global_edges:
        pair_counts = Counter()
        for u, v in global_edges:
            a, b = (u, v) if u < v else (v, u)
            pair_counts[(a, b)] += 1
        for (u, v), cnt in pair_counts.items():
            for i in range(cnt):
                mag = 0.22 + 0.07 * min(i, 6)
                rad = rad_away_from_compute(u, v, mag)
                edge_registry[(u, v)] = rad
                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v], connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-", lw=1.6, color="black", alpha=0.65, zorder=0.2
                ))

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
            mag = 0.22 + 0.04 * min(dist, 12)
            rad = rad_away_from_compute(u, v, mag)
            edge_registry[(u, v)] = -rad
            ax.add_patch(FancyArrowPatch(
                posA=pos[u], posB=pos[v], connectionstyle=f"arc3,rad={-rad}",
                arrowstyle="-", lw=0.7, color="black", alpha=0.65, zorder=0.3
            ))

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
            edge_registry[(u, v)] = inward
            for rad in ([inward, outward] if cnt >= 2 else [inward]):
                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v], connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-", lw=2.4, color="black", alpha=0.65, zorder=1.0
                ))

    for gid in range(cfg.groups):
        for sg in range(cfg.subgroups_per_group):
            sw = [n for n, d in G.nodes(data=True) if
                  d.get("kind") == "switch" and d.get("group") == gid and d.get("subgroup") == sg]
            sw.sort(key=lambda n: G.nodes[n].get("switch_in_subgroup", 0))
            if len(sw) != 4: continue
            s1, s2, s3, s4 = sw[0], sw[1], sw[2], sw[3]
            edge_registry[(s2, s3)] = 0.0
            ax.plot([pos[s2][0], pos[s3][0]], [pos[s2][1], pos[s3][1]], color="black", alpha=0.55, lw=1.2, zorder=1.2)
            for (u, v, mag) in [(s1, s2, 0.18), (s3, s4, 0.18), (s1, s3, 0.22), (s2, s4, 0.22), (s1, s4, 0.32)]:
                rad = rad_away_from_compute(u, v, mag)
                edge_registry[(u, v)] = -rad
                ax.add_patch(FancyArrowPatch(
                    posA=pos[u], posB=pos[v], connectionstyle=f"arc3,rad={-rad}",
                    arrowstyle="-", lw=1.2, color="black", alpha=0.50, zorder=1.2
                ))

    if inj_edges:
        nx.draw_networkx_edges(nx.Graph(inj_edges), pos, alpha=0.45, width=1.6, edge_color="black")

    ax.scatter([pos[n][0] for n in switches], [pos[n][1] for n in switches], s=cfg.node_size_switch, c="tab:green",
               edgecolors="black", linewidths=0.8, zorder=10, label="Switches")
    ax.scatter([pos[n][0] for n in compute_nodes], [pos[n][1] for n in compute_nodes], s=cfg.node_size_compute,
               c="tab:blue", edgecolors="black", linewidths=0.5, zorder=10, label="Compute")
    fig.tight_layout()
    ax.legend(scatterpoints=1, frameon=False, loc="upper left")
    return fig, ax, edge_registry


# ----------------------------
# Cast Viewer (Complete)
# ----------------------------
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

        # NEW: Track dead nodes
        self.dead_nodes: Set[str] = set()

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
        self.bfs_checked_edges = []
        self.discovery_targets = None
        self.discovery_targets_found = set()

        # Delivery state
        self.path = []
        self.path_i = 0
        self.tree_children = {}
        self.received = set()
        self.frontier = []
        self.delivered_edges = []
        self.current_wave_edges = []

        # --- Base Draw with Registry ---
        self.fig, self.ax, self.edge_registry = draw_topology_base(G, pos, cfg)
        self.fig = self.ax.figure

        # --- Dynamic Patches Storage ---
        self.active_patches: List[FancyArrowPatch] = []

        self._init_overlays()

        self.hud = self.ax.text(0.02, 0.02, self._hud_text(), transform=self.ax.transAxes, fontsize=10, va="bottom")
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _active_graph(self) -> nx.Graph:
        """Builds graph H excluding dead nodes."""
        H = nx.Graph()
        alive_nodes = [n for n in self.G.nodes() if n not in self.dead_nodes]
        H.add_nodes_from(alive_nodes)
        for u, v, _d in self.G.edges(data=True):
            if u not in self.dead_nodes and v not in self.dead_nodes:
                H.add_edge(u, v)
        return H

    def _empty_offsets(self):
        return np.empty((0, 2))

    def _nearest_node(self, x: float, y: float) -> Optional[str]:
        thr2 = 0.75 ** 2
        best, best_d2 = None, 1e18
        for n, (nx_, ny_) in self.pos.items():
            d2 = (nx_ - x) ** 2 + (ny_ - y) ** 2
            if d2 < best_d2:
                best, best_d2 = n, d2
        return best if best is not None and best_d2 <= thr2 else None

    def _set_msg(self, txt: str):
        self.msg.set_text(txt)

    def _hud_text(self) -> str:
        def short(x): return x if x is not None else "—"

        return (
            f"Cast: {self.cast} | Phase: {self.sim_phase} | Discovery: {'ON' if self.show_discovery else 'OFF'}\n"
            f"Mode: {self.mode} | Dead Nodes: {len(self.dead_nodes)}\n"
            f"src: {short(self.src)} | dst: {short(self.dst)} | m_dsts: {len(self.dsts)}\n"
            "Keys: f(failures) s(src) d(dst) Space(step) r(reset) Enter(run) c(clear)"
        )

    def _update_hud(self):
        self.hud.set_text(self._hud_text())

    def _init_overlays(self):
        self.dead_sc = self.ax.scatter([], [], s=180, marker="x", c="gray", linewidths=3.0, zorder=50)
        self.src_sc = self.ax.scatter([], [], s=240, facecolors="none", edgecolors="black", linewidths=2.2, zorder=40)
        self.dst_sc = self.ax.scatter([], [], s=240, facecolors="none", edgecolors="black", linewidths=2.2, zorder=40)
        self.dsts_sc = self.ax.scatter([], [], s=190, facecolors="none", edgecolors="black", linewidths=1.6, zorder=39)
        self.frontier_sc = self.ax.scatter([], [], s=170, facecolors="none", edgecolors="red", linewidths=2.0,
                                           zorder=35)
        self.visited_sc = self.ax.scatter([], [], s=120, facecolors="none", edgecolors="red", linewidths=1.0,
                                          alpha=0.25, zorder=34)
        self.packet_sc = self.ax.scatter([], [], s=95, c="red", zorder=45)
        self.msg = self.ax.text(0.5, 0.98, "", transform=self.ax.transAxes, fontsize=11, va="top", ha="center")

        self.show_dsts_panel = True
        self.dsts_panel = self.ax.text(0.98, 0.02, "", transform=self.ax.transAxes, fontsize=9, va="bottom", ha="right",
                                       bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="black", alpha=0.75),
                                       zorder=60)
        self.tooltip = self.ax.text(0, 0, "", fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="black", alpha=0.85), zorder=70,
                                    visible=False)

    def _update_dsts_panel(self):
        if not getattr(self, "show_dsts_panel", True):
            self.dsts_panel.set_visible(False)
            return
        if self.cast != "MULTICAST":
            self.dsts_panel.set_visible(False)
            return
        self.dsts_panel.set_visible(True)
        items = sorted(self.dsts)
        if not items:
            self.dsts_panel.set_text("Multicast group:\n(empty)")
            return
        N = 10
        head = items[:N]
        more = len(items) - len(head)
        text = "Multicast group:\n" + "\n".join(head)
        if more > 0:
            text += f"\n… +{more} more"
        self.dsts_panel.set_text(text)

    def _update_src_dst_markers(self):
        self.src_sc.set_offsets([self.pos[self.src]] if self.src else self._empty_offsets())
        if self.cast == "MULTICAST":
            self.dst_sc.set_offsets(self._empty_offsets())
        else:
            self.dst_sc.set_offsets([self.pos[self.dst]] if self.dst else self._empty_offsets())
        self.dsts_sc.set_offsets([self.pos[n] for n in sorted(self.dsts)] if self.dsts else self._empty_offsets())

        if self.dead_nodes:
            self.dead_sc.set_offsets([self.pos[n] for n in self.dead_nodes])
        else:
            self.dead_sc.set_offsets(self._empty_offsets())

        self._update_dsts_panel()

    def _show_tooltip(self, node_id, x, y):
        d = self.G.nodes[node_id]
        self.tooltip.set_text(f"{node_id}\n{d.get('kind', '?')}")
        self.tooltip.set_position((x, y))
        self.tooltip.set_visible(True)

    def _hide_tooltip(self):
        self.tooltip.set_visible(False)

    def _draw_curved_edge(self, u, v, color, lw, alpha, zorder):
        rad = 0.0
        if (u, v) in self.edge_registry:
            rad = self.edge_registry[(u, v)]
        elif (v, u) in self.edge_registry:
            rad = -self.edge_registry[(v, u)]
        patch = FancyArrowPatch(
            posA=self.pos[u], posB=self.pos[v],
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-", color=color, lw=lw, alpha=alpha, zorder=zorder
        )
        self.ax.add_patch(patch)
        self.active_patches.append(patch)

    def _redraw_dynamic_edges(self):
        for p in self.active_patches:
            p.remove()
        self.active_patches.clear()
        for u, v in self.bfs_checked_edges:
            self._draw_curved_edge(u, v, "red", 1.6, 0.25, 30)
        for u, v in self.delivered_edges:
            self._draw_curved_edge(u, v, "red", 3.0, 0.55, 32)
        for u, v in self.current_wave_edges:
            self._draw_curved_edge(u, v, "red", 4.2, 0.95, 33)

    def _on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None: return
        n = self._nearest_node(event.xdata, event.ydata)
        if n is None: return

        if event.button == 3:
            self._show_tooltip(n, event.xdata, event.ydata)
            self.fig.canvas.draw_idle()
            return
        else:
            self._hide_tooltip()

        if self.mode == "SELECT_FAILURES":
            if n in self.dead_nodes:
                self.dead_nodes.remove(n)
                self._set_msg(f"Node restored: {n}")
            else:
                self.dead_nodes.add(n)
                self._set_msg(f"Node KILLED: {n}")
                if self.src == n: self.src = None
                if self.dst == n: self.dst = None
                if n in self.dsts: self.dsts.remove(n)
            self._reset_overlays(keep_selection=True)
            return

        if self.mode == "SELECT_SOURCE":
            if n in self.dead_nodes:
                self._set_msg(f"Dead node {n}!")
                return
            self.src = None if self.src == n else n
            self._set_msg(f"Source: {self.src}")
            self.mode = "VIEW"
            self._update_src_dst_markers();
            self._update_hud();
            self.fig.canvas.draw_idle()
            return

        if self.mode == "SELECT_DEST_ONE":
            if n in self.dead_nodes:
                self._set_msg(f"Dead node {n}!")
                return
            self.dst = None if self.dst == n else n
            self._set_msg(f"Dest: {self.dst}")
            self.mode = "VIEW"
            self._update_src_dst_markers();
            self._update_hud();
            self.fig.canvas.draw_idle()
            return

        if self.mode == "SELECT_DEST_MANY":
            if n in self.dead_nodes:
                self._set_msg(f"Dead node {n}!")
                return
            if n in self.dsts:
                self.dsts.remove(n)
            else:
                self.dsts.add(n)
            self._update_src_dst_markers();
            self._update_hud();
            self.fig.canvas.draw_idle()
            return

    def _on_key(self, event):
        k = (event.key or "").lower()
        if k == "f":
            self.mode = "SELECT_FAILURES"
            self._set_msg("FAILURE MODE: Click nodes to kill/restore.")
            self._update_hud();
            self.fig.canvas.draw_idle()
            return
        if k == "u":
            self.cast = "UNICAST";
            self.mode = "VIEW";
            self._set_msg("Unicast")
            self._update_hud();
            self.fig.canvas.draw_idle()
        elif k == "b":
            self.cast = "BROADCAST";
            self.mode = "VIEW";
            self._set_msg("Broadcast")
            self._update_hud();
            self.fig.canvas.draw_idle()
        elif k == "m":
            self.cast = "MULTICAST";
            self.mode = "VIEW";
            self._set_msg("Multicast")
            self._update_hud();
            self.fig.canvas.draw_idle()
        elif k == "s":
            self.mode = "SELECT_SOURCE";
            self._set_msg("Click Source")
            self._update_hud();
            self.fig.canvas.draw_idle()
        elif k == "d":
            self.mode = "SELECT_DEST_MANY" if self.cast == "MULTICAST" else "SELECT_DEST_ONE"
            self._set_msg("Click Dest")
            self._update_hud();
            self.fig.canvas.draw_idle()
        elif k == "c":
            self.dsts.clear();
            self._update_src_dst_markers();
            self.fig.canvas.draw_idle()
        elif k == "w":
            self.show_discovery = not self.show_discovery
            self._update_hud();
            self.fig.canvas.draw_idle()
        elif k == "r":
            self._reset_overlays(True)
        elif k in ("enter", "return"):
            self.prepare()
        elif k in (" ", "space", "n"):
            self.step_once()
        elif k == "l":
            self.show_dsts_panel = not getattr(self, "show_dsts_panel", True)
            self._update_src_dst_markers();
            self.fig.canvas.draw_idle()
        elif k == "z":
            self.src = None;
            self._update_src_dst_markers();
            self.fig.canvas.draw_idle()
        elif k == "x":
            self.dst = None;
            self._update_src_dst_markers();
            self.fig.canvas.draw_idle()

    def prepare(self):
        self._reset_overlays(keep_selection=True)
        if self.src is None:
            self._set_msg("Set src first.")
            self.fig.canvas.draw_idle();
            return

        self.H = self._active_graph()
        if self.src not in self.H:
            self._set_msg("Source node is DEAD.");
            return
        if self.cast == "UNICAST" and self.dst and self.dst not in self.H:
            self._set_msg("Destination node is DEAD.");
            return

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
            alive_dsts = {d for d in self.dsts if d in self.H}
            if len(alive_dsts) < len(self.dsts):
                self._set_msg(f"Warning: {len(self.dsts) - len(alive_dsts)} dead targets.")
            if not alive_dsts: self._set_msg("All targets are dead."); return
            if self.show_discovery:
                self._init_bfs_discovery(targets=alive_dsts)
            else:
                self._build_bfs_parents_until_targets(alive_dsts); self._init_multicast_tree_delivery(alive_dsts)

        self.sim_active = True
        self._update_src_dst_markers();
        self._update_hud();
        self.fig.canvas.draw_idle()

    def _init_bfs_discovery(self, targets):
        self.sim_phase = "DISCOVERY";
        self.discovery_targets = targets;
        self.discovery_targets_found = set()
        self.bfs_q = deque([self.src]);
        self.bfs_parent = {self.src: None};
        self.bfs_visited = {self.src}
        self.bfs_checked_edges = [];
        self.current_wave_edges = [];
        self.bfs_neighbors = {};
        self.bfs_idx = {}
        for u in self.H.nodes():
            nbrs = sorted(list(self.H.neighbors(u)))
            seed_u = (int(self.cfg.seed) * 1000003) ^ zlib.adler32(str(u).encode("utf-8"))
            rnd_u = random.Random(seed_u);
            rnd_u.shuffle(nbrs)
            self.bfs_neighbors[u] = nbrs;
            self.bfs_idx[u] = 0
        self.visited_sc.set_offsets([self.pos[self.src]]);
        self.frontier_sc.set_offsets([self.pos[self.src]])
        self._redraw_dynamic_edges()

    def _step_bfs_one_edge(self, update_visuals=True):
        while self.bfs_q:
            u = self.bfs_q[0]
            if self.bfs_idx[u] < len(self.bfs_neighbors[u]): break
            self.bfs_q.popleft()

        if not self.bfs_q:
            if update_visuals:
                self.current_wave_edges = []
                self._redraw_dynamic_edges()
            return False

        u = self.bfs_q[0]
        v = self.bfs_neighbors[u][self.bfs_idx[u]]
        self.bfs_idx[u] += 1

        self.bfs_checked_edges.append((u, v))

        if update_visuals:
            self.current_wave_edges = [(u, v)]
            self._redraw_dynamic_edges()

        if v not in self.bfs_visited:
            self.bfs_visited.add(v)
            self.bfs_parent[v] = u
            self.bfs_q.append(v)

        if update_visuals:
            self.visited_sc.set_offsets([self.pos[n] for n in self.bfs_visited])
            self.frontier_sc.set_offsets([self.pos[n] for n in self.bfs_q] if self.bfs_q else self._empty_offsets())

        if self.discovery_targets is not None and v in self.discovery_targets:
            self.discovery_targets_found.add(v)

        if update_visuals:
            self._set_msg(f"Checked: {u} → {v}")

        return True

    def _build_bfs_parents_full(self):
        self._init_bfs_discovery(None)
        while self._step_bfs_one_edge(update_visuals=False): pass
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.current_wave_edges = [];
        self.bfs_checked_edges = [];
        self._redraw_dynamic_edges()

    def _build_bfs_parents_until_targets(self, targets):
        self._init_bfs_discovery(targets)
        while True:
            if not self._step_bfs_one_edge(update_visuals=False):
                break
            if self.discovery_targets_found == set(targets):
                break

        self.frontier_sc.set_offsets(self._empty_offsets())
        self.visited_sc.set_offsets(self._empty_offsets())
        self.current_wave_edges = []
        self.bfs_checked_edges = []
        self._redraw_dynamic_edges()

    def _reconstruct_path(self, dst):
        if dst not in self.bfs_parent: return []
        p = [];
        cur = dst
        while cur is not None: p.append(cur); cur = self.bfs_parent.get(cur)
        p.reverse();
        return p

    def _init_unicast_delivery_direct(self):
        self.sim_phase = "DELIVERY"
        try:
            self.path = nx.shortest_path(self.H, self.src, self.dst)
        except nx.NetworkXNoPath:
            self.sim_phase = "DONE"; self.sim_active = False; self._set_msg("No path."); return
        self.path_i = 0;
        self.packet_sc.set_offsets([self.pos[self.path[0]]]);
        self.delivered_edges = []
        self.current_wave_edges = [];
        self._redraw_dynamic_edges()

    def _init_tree_delivery_from_parents(self, all_targets):
        self.sim_phase = "DELIVERY";
        self.tree_children = {}
        for v, p in self.bfs_parent.items():
            if p is None: continue
            self.tree_children.setdefault(p, []).append(v)
        self.received = {self.src};
        self.frontier = [self.src];
        self.delivered_edges = [];
        self.current_wave_edges = []
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.bfs_checked_edges = [];
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
        self.sim_phase = "DELIVERY";
        self.received = {self.src};
        self.frontier = [self.src]
        self.delivered_edges = [];
        self.current_wave_edges = []
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.bfs_checked_edges = [];
        self._redraw_dynamic_edges()

    def step_once(self):
        if not self.sim_active or self.sim_phase in ("IDLE", "DONE"): return

        if self.sim_phase == "DISCOVERY":
            if not self._step_bfs_one_edge(update_visuals=True):
                self._finish_discovery()
                self.fig.canvas.draw_idle()
                return

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

        self._update_hud()
        self.fig.canvas.draw_idle()

    def _finish_discovery(self):
        if self.cast == "UNICAST":
            self.path = self._reconstruct_path(self.dst)
            if not self.path: self.sim_phase = "DONE"; self.sim_active = False; self._set_msg(
                "No path (unreachable)."); return
            self.sim_phase = "DELIVERY";
            self.path_i = 0
            self.packet_sc.set_offsets([self.pos[self.path[0]]])
            self.delivered_edges = [];
            self.current_wave_edges = [];
            self.bfs_checked_edges = []
            self._redraw_dynamic_edges();
            self._set_msg("Path found. Step to deliver.")
            self.frontier_sc.set_offsets(self._empty_offsets());
            self.visited_sc.set_offsets(self._empty_offsets())
        elif self.cast == "BROADCAST":
            self._init_tree_delivery_from_parents(True);
            self._set_msg("Broadcast tree ready.")
        elif self.cast == "MULTICAST":
            alive_dsts = {d for d in self.dsts if d not in self.dead_nodes}
            self._init_multicast_tree_delivery(alive_dsts);
            self._set_msg("Multicast tree ready.")

    def _step_unicast_one_hop(self):
        if not self.path: self.sim_phase = "DONE"; self.sim_active = False; self._set_msg("No path."); return
        if self.path_i >= len(self.path) - 1:
            self.sim_phase = "DONE";
            self.sim_active = False;
            self._set_msg("Done.")
            self.current_wave_edges = [];
            self._redraw_dynamic_edges();
            return
        a = self.path[self.path_i];
        b = self.path[self.path_i + 1];
        self.path_i += 1
        self.delivered_edges.append((a, b));
        self.current_wave_edges = [(a, b)];
        self._redraw_dynamic_edges()
        self.packet_sc.set_offsets([self.pos[b]]);
        self._set_msg(f"Hop: {a} → {b}")

    def _step_tree_one_wave(self, label):
        if not self.frontier:
            self.sim_phase = "DONE";
            self.sim_active = False;
            self._set_msg("Done.")
            self.current_wave_edges = [];
            self._redraw_dynamic_edges();
            return
        next_frontier = [];
        wave_edges = []
        for u in self.frontier:
            for v in self.tree_children.get(u, []):
                if v in self.received: continue
                self.received.add(v);
                next_frontier.append(v);
                wave_edges.append((u, v))
        self.delivered_edges.extend(wave_edges);
        self.current_wave_edges = wave_edges;
        self._redraw_dynamic_edges()
        self.frontier = next_frontier
        self.frontier_sc.set_offsets([self.pos[n] for n in self.frontier] if self.frontier else self._empty_offsets())
        if wave_edges:
            self._set_msg(f"{label}: +{len(next_frontier)} nodes")
        else:
            self._set_msg("Done."); self.sim_phase = "DONE"; self.sim_active = False

    def _reset_overlays(self, keep_selection: bool = True):
        self.sim_active = False;
        self.sim_phase = "IDLE";
        self.H = None
        self.bfs_q.clear();
        self.bfs_parent.clear();
        self.bfs_visited.clear()
        self.bfs_neighbors.clear();
        self.bfs_idx.clear()
        self.bfs_checked_edges = [];
        self.discovery_targets = None;
        self.discovery_targets_found = set()
        self.path = [];
        self.path_i = 0;
        self.tree_children = {};
        self.received = set();
        self.frontier = []
        self.delivered_edges = [];
        self.current_wave_edges = []
        self.frontier_sc.set_offsets(self._empty_offsets());
        self.visited_sc.set_offsets(self._empty_offsets())
        self.packet_sc.set_offsets(self._empty_offsets());
        self._redraw_dynamic_edges();
        self._set_msg("")
        if not keep_selection: self.src = None; self.dst = None; self.dsts.clear(); self.dead_nodes.clear()
        self._update_src_dst_markers();
        self._update_hud()
        self.fig.canvas.draw_idle()


def compute_metrics_custom(G_multi: nx.MultiGraph) -> dict:
    G_simple = nx.Graph(G_multi)

    N = G_multi.number_of_nodes()
    if N == 0: return {}

    degrees = [d for n, d in G_multi.degree()]
    S = max(degrees) if degrees else 0

    # Check connectivity first
    if nx.is_connected(G_simple):
        D = nx.diameter(G_simple)
        D_s = nx.average_shortest_path_length(G_simple)
    else:
        # If graph is not connected, use the largest component
        largest_cc = max(nx.connected_components(G_simple), key=len)
        subG = G_simple.subgraph(largest_cc)
        D = nx.diameter(subG)
        D_s = nx.average_shortest_path_length(subG)
        print(f"Warning: Graph is not connected. Metrics calculated on largest component ({len(largest_cc)} nodes).")

    T = (2 * D_s) / S if S > 0 else 0.0
    C = D * N * S

    return {
        "N": N,
        "S": S,
        "D": D,
        "D_s": D_s,
        "T": T,
        "C": C
    }


def print_metrics_custom(m: dict):
    print("\n" + "=" * 40)
    print(" ОБЧИСЛЕННЯ ПАРАМЕТРІВ МЕРЕЖІ")
    print("=" * 40)
    print(f"Кількість вузлів (N):         {m['N']}")
    print(f"Ступінь топології (S):        {m['S']}")
    print(f"Діаметр топології (D):        {m['D']}")
    print(f"Середній діаметр (D_s):       {m['D_s']:.4f}")
    print(f"Топологічний трафік (T):      {m['T']:.6f}  (Formula: 2 * D_s / S)")
    print(f"Вартість системи (C):         {m['C']}       (Formula: D * N * S)")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    plt.rcParams['keymap.fullscreen'] = []
    plt.rcParams['keymap.save'] = []
    plt.rcParams['keymap.yscale'] = []

    # cfg = TopologyConfig(
    #     groups=40,
    #     subgroups_per_group=4,
    #     compute_nodes_per_subgroup=8,
    #     ports_per_switch=34,
    #     min_links_per_group_pair=2,
    # )

    cfg = TopologyConfig(
        groups=8,
        subgroups_per_group=4,
        compute_nodes_per_subgroup=8,
        ports_per_switch=26,
        min_links_per_group_pair=2,
    )
    #
    # cfg = TopologyConfig(
    #     groups=6,
    #     subgroups_per_group=4,
    #     compute_nodes_per_subgroup=8,
    #     ports_per_switch=26,
    #     min_links_per_group_pair=2,
    # )
    G = generate_dragonfly(cfg)
    pos = dragonfly_ring_positions(G, cfg)

    metrics = compute_metrics_custom(G)
    print_metrics_custom(metrics)

    viewer = CastViewer(G, pos, cfg)
    plt.show()