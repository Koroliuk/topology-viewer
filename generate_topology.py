from __future__ import annotations

import argparse
import csv
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

import networkx as nx

Edge = Tuple[int, int]
EdgeSet = FrozenSet[Edge]


@dataclass(frozen=True)
class Config:
    n: int = 20
    deg_min: int = 2
    deg_max: int = 6

    # Genetic algo params
    population: int = 120
    generations: int = 500
    elitism: int = 5
    tournament_k: int = 3

    # Mutation
    child_mutation_steps_min: int = 1
    child_mutation_steps_max: int = 3
    mutation_attempts_per_child: int = 25

    # Failure model
    mu: float = 1.5
    k_max: int = 8

    # Adaptive Simulation (monte-carlo)
    mc_trials_fast: int = 20
    mc_trials_final: int = 300

    # Runtime
    time_limit_sec: int = 15 * 60
    seed: int = 42
    top_n: int = 10
    out_dir: str = "ga_parallel_out"
    save_png: bool = True
    verbose_every: int = 10


@dataclass
class Metrics:
    R: float
    Q: float
    P: float
    n_edges: int
    S: float
    apl: float
    diam: int


# Util functions
def norm_edge(u: int, v: int) -> Edge:
    return (u, v) if u < v else (v, u)


def edgeset_from_graph(g: nx.Graph) -> EdgeSet:
    return frozenset(norm_edge(u, v) for (u, v) in g.edges())


def graph_from_edgeset(n: int, es: EdgeSet) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(range(n))
    g.add_edges_from(es)
    return g


def average_degree(n: int, m_edges: int) -> float:
    return (2.0 * m_edges) / float(n) if n > 0 else 0.0


def poisson_weights_truncated(mu: float, k_max: int) -> List[float]:
    if k_max < 0: return []
    p = [0.0] * (k_max + 1)
    p[0] = math.exp(-mu)
    for k in range(1, k_max + 1):
        p[k] = p[k - 1] * (mu / float(k))
    s = sum(p)
    return [x / s for x in p] if s > 0 else [1.0] + [0.0] * k_max


def is_simple_undirected_valid(g: nx.Graph, cfg: Config) -> bool:
    if g.number_of_nodes() != cfg.n: return False
    if any(u == v for (u, v) in g.edges()): return False
    degs = dict(g.degree())
    if any(d < cfg.deg_min for d in degs.values()): return False
    if any(d > cfg.deg_max for d in degs.values()): return False
    if not nx.is_connected(g): return False
    return True


# Worker Function
def evaluate_worker(args) -> Optional[Metrics]:
    edges, cfg, seed, fast_mode = args
    rng = random.Random(seed)

    g = graph_from_edgeset(cfg.n, edges)

    if not is_simple_undirected_valid(g, cfg):
        return None

    # Determine trials count
    trials = cfg.mc_trials_fast if fast_mode else cfg.mc_trials_final

    m_edges = g.number_of_edges()
    S = average_degree(cfg.n, m_edges)
    apl = float(nx.average_shortest_path_length(g))
    diam = int(nx.diameter(g))

    Q = (2.0 * apl / S) if S > 0 else float("inf")
    P = float(diam) * S

    # Robustness Calculation
    weights = poisson_weights_truncated(cfg.mu, cfg.k_max)
    R = 0.0

    if m_edges > 0:
        g_edges = list(g.edges())
        n_nodes = cfg.n

        for k, pk in enumerate(weights):
            if pk == 0.0: continue
            if k == 0:
                R += pk * 1.0
                continue

            # Monte Carlo for specific k
            acc_lcc = 0.0

            # Loop optimization: avoid function call overhead
            for _ in range(trials):
                # Fail k edges
                k_eff = min(k, m_edges)
                failed = rng.sample(g_edges, k_eff)

                # Build G'
                g2 = g.copy()
                g2.remove_edges_from(failed)

                # Get LCC
                if nx.is_empty(g2):
                    max_sz = 1
                else:
                    visited = set()
                    max_sz = 0
                    for node in range(n_nodes):
                        if node not in visited:
                            c = set(nx.bfs_tree(g2, node))
                            visited.update(c)
                            if len(c) > max_sz:
                                max_sz = len(c)

                acc_lcc += (max_sz / float(n_nodes))

            R += pk * (acc_lcc / trials)

    return Metrics(R=R, Q=Q, P=P, n_edges=m_edges, S=S, apl=apl, diam=diam)


# --- Graph Generation Helpers (Same as before) ---
def make_cycle(n: int) -> nx.Graph:
    return nx.cycle_graph(n)


def random_valid_graph(cfg: Config, rng: random.Random) -> nx.Graph:
    # Simplified generator
    g = make_cycle(cfg.n)
    extra = rng.randint(0, cfg.n)
    nodes = list(g.nodes())
    for _ in range(extra):
        u, v = rng.sample(nodes, 2)
        if not g.has_edge(u, v) and g.degree(u) < cfg.deg_max and g.degree(v) < cfg.deg_max:
            g.add_edge(u, v)
    return g


def seed_population(cfg: Config, rng: random.Random) -> List[nx.Graph]:
    seeds = []
    seeds.append(nx.star_graph(cfg.n))
    seeds.append(nx.cycle_graph(cfg.n))
    seeds.append(nx.wheel_graph(cfg.n))
    seeds.append(nx.complete_graph(cfg.n))
    if cfg.n % 2 == 0:
        seeds.append(nx.circular_ladder_graph(cfg.n // 2))
    while len(seeds) < 20:
        seeds.append(random_valid_graph(cfg, rng))
    return seeds


# Mutations
def mutate_graph(parent: nx.Graph, cfg: Config, rng: random.Random) -> nx.Graph:
    g = parent.copy()
    steps = rng.randint(cfg.child_mutation_steps_min, cfg.child_mutation_steps_max)
    nodes = list(g.nodes())

    for _ in range(steps):
        op = rng.random()
        # 40% Swap (Double edge), 30% Rewire, 15% Add, 15% Remove
        if op < 0.4:  # Swap
            if g.number_of_edges() >= 2:
                edges = list(g.edges())
                if len(edges) > 2:
                    (u, v), (x, y) = rng.sample(edges, 2)
                    if len({u, v, x, y}) == 4:
                        # Try swap
                        if rng.random() < 0.5:
                            cand = [(u, x), (v, y)]
                        else:
                            cand = [(u, y), (v, x)]

                        # Check existence
                        if not any(g.has_edge(a, b) for a, b in cand):
                            g.remove_edge(u, v)
                            g.remove_edge(x, y)
                            g.add_edges_from(cand)
                            if not is_simple_undirected_valid(g, cfg):
                                g.remove_edges_from(cand)
                                g.add_edge(u, v)
                                g.add_edge(x, y)

        elif op < 0.7:  # Rewire
            if g.number_of_edges() > 0:
                edges = list(g.edges())
                u, v = rng.choice(edges)
                keep, rem = (u, v) if rng.random() < 0.5 else (v, u)

                # Try to connect keep to random z
                z = rng.choice(nodes)
                if z != keep and z != rem and not g.has_edge(keep, z):
                    if g.degree(z) < cfg.deg_max and g.degree(rem) > cfg.deg_min:
                        g.remove_edge(keep, rem)
                        g.add_edge(keep, z)
                        if not is_simple_undirected_valid(g, cfg):
                            g.remove_edge(keep, z)
                            g.add_edge(keep, rem)

        elif op < 0.85:  # Add
            u, v = rng.sample(nodes, 2)
            if not g.has_edge(u, v):
                if g.degree(u) < cfg.deg_max and g.degree(v) < cfg.deg_max:
                    g.add_edge(u, v)

        else:  # Remove
            if g.number_of_edges() > cfg.n:
                edges = list(g.edges())
                u, v = rng.choice(edges)
                if g.degree(u) > cfg.deg_min and g.degree(v) > cfg.deg_min:
                    g.remove_edge(u, v)
                    if not is_simple_undirected_valid(g, cfg):
                        g.add_edge(u, v)

    return g


@dataclass(frozen=True)
class Individual:
    edges: EdgeSet


def fitness_key(m: Metrics) -> Tuple[float, float, float]:
    # Max R -> min -R; Min Q; Min P
    return (-m.R, m.Q, m.P)


# --- Output (Modified) ---
def save_results(top, cfg, sub_folder: str):
    """
    Saves results to cfg.out_dir / sub_folder.
    Useful for saving 'initial' vs 'final' populations.
    """
    path = os.path.join(cfg.out_dir, sub_folder)
    os.makedirs(path, exist_ok=True)

    # CSV
    with open(f"{path}/summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Rank", "R", "Q", "D*S", "Edges", "Diam", "APL"])
        for i, (ind, m) in enumerate(top, 1):
            w.writerow([i, m.R, m.Q, m.P, m.n_edges, m.diam, m.apl])

            # Edgelist
            g = graph_from_edgeset(cfg.n, ind.edges)
            nx.write_edgelist(g, f"{path}/top_{i}.edgelist", data=False)

            # Plot
            if cfg.save_png and i <= 5:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(5, 5))
                    pos = nx.spring_layout(g, seed=cfg.seed)
                    nx.draw(g, pos, node_size=300, node_color='lightblue', with_labels=True)
                    plt.title(f"Rank {i}\nR={m.R:.3f} Q={m.Q:.3f}")
                    plt.savefig(f"{path}/top_{i}.png")
                    plt.close()
                except ImportError:
                    pass
    print(f"Results saved to {path}/")


def evolve_parallel(cfg: Config):
    # Detect CPU count
    workers = os.cpu_count() or 4
    print(f"=== Starting GA on {workers} cores ===")

    rng = random.Random(cfg.seed)

    # 1. Initialize
    raw_graphs = seed_population(cfg, rng)
    pop_inds = [Individual(edges=edgeset_from_graph(g)) for g in raw_graphs]

    while len(pop_inds) < cfg.population:
        g = random_valid_graph(cfg, rng)
        pop_inds.append(Individual(edges=edgeset_from_graph(g)))

    # Cache to avoid re-evaluating elites
    # Key: EdgeSet, Value: Metrics
    cache: Dict[EdgeSet, Metrics] = {}

    def parallel_evaluate(inds: List[Individual], fast: bool) -> List[Optional[Metrics]]:
        # Identify what needs calc
        todo_args = []
        indices_to_calc = []
        results = [None] * len(inds)

        for i, ind in enumerate(inds):
            if ind.edges in cache:
                results[i] = cache[ind.edges]
            else:
                indices_to_calc.append(i)
                # Unique seed for each worker task
                w_seed = rng.randint(0, 10 ** 9)
                todo_args.append((ind.edges, cfg, w_seed, fast))

        if not todo_args:
            return results

        # Execute
        with ProcessPoolExecutor(max_workers=workers) as executor:
            computed = list(executor.map(evaluate_worker, todo_args))

        for i, res in zip(indices_to_calc, computed):
            if res is not None:
                cache[inds[i].edges] = res  # Update cache
                results[i] = res

        return results

    # Initial Eval
    metrics = parallel_evaluate(pop_inds, fast=True)
    pop = []
    for ind, m in zip(pop_inds, metrics):
        if m is not None:
            pop.append((ind, m))

    pop.sort(key=lambda x: fitness_key(x[1]))

    # --- SAVE INITIAL POPULATION ---
    print("\n>>> Saving Initial Population (Gen 0) for comparison...")
    save_results(pop, cfg, sub_folder="initial_population")
    print(">>> Starting Evolution...\n")

    start_time = time.time()

    for gen in range(1, cfg.generations + 1):
        if time.time() - start_time > cfg.time_limit_sec:
            print("[STOP] Time limit.")
            break

        # Elitism
        next_pop = pop[:cfg.elitism]

        # Create candidates
        candidates = []
        needed = cfg.population - len(next_pop)

        while len(candidates) < needed:
            # Tournament
            t_inds = random.sample(pop, cfg.tournament_k)
            t_inds.sort(key=lambda x: fitness_key(x[1]))
            parent = t_inds[0][0]

            p_graph = graph_from_edgeset(cfg.n, parent.edges)

            # Mutate (Retry logic inside loop to ensure we get candidates)
            for _ in range(10):
                child_g = mutate_graph(p_graph, cfg, rng)
                if is_simple_undirected_valid(child_g, cfg):
                    candidates.append(Individual(edges=edgeset_from_graph(child_g)))
                    break
            else:
                candidates.append(parent)

        candidates = candidates[:needed]

        cand_metrics = parallel_evaluate(candidates, fast=True)

        for c, m in zip(candidates, cand_metrics):
            if m is not None:
                next_pop.append((c, m))
            else:
                # If invalid logically, grab random elite
                next_pop.append(pop[0])

        next_pop.sort(key=lambda x: fitness_key(x[1]))
        pop = next_pop

        if gen % cfg.verbose_every == 0 or gen == 1:
            best = pop[0][1]
            elapsed = time.time() - start_time
            print(
                f"[Gen {gen:4d}] R={best.R:.4f} Q={best.Q:.4f} D*S={best.P:.4f} Edges={best.n_edges} Time: {elapsed:.1f}s")

    # --- Final High-Precision Evaluation ---
    print("\n>>> Re-evaluating Top-20 with High Precision (300 trials)...")
    top_inds = [x[0] for x in pop[:20]]

    # Clear cache for these to force re-calc
    for ind in top_inds:
        if ind.edges in cache:
            del cache[ind.edges]

    final_metrics = parallel_evaluate(top_inds, fast=False)

    final_pop = []
    for ind, m in zip(top_inds, final_metrics):
        if m is not None:
            final_pop.append((ind, m))

    final_pop.sort(key=lambda x: fitness_key(x[1]))
    return final_pop


if __name__ == "__main__":
    # Windows/MacOS multiprocessing support requires this protection
    args = argparse.ArgumentParser()
    args.add_argument("--mu", type=float, default=1.5)
    parsed = args.parse_args()

    cfg = Config(mu=parsed.mu)
    print("Config:", cfg)

    top = evolve_parallel(cfg)

    print("\n=== FINAL TOP 5 ===")
    for i, (ind, m) in enumerate(top[:5], 1):
        print(f"#{i} R={m.R:.5f} Q={m.Q:.5f} D*S={m.P:.5f} (Edges={m.n_edges})")

    # Save final
    save_results(top[:cfg.top_n], cfg, sub_folder="final_population")