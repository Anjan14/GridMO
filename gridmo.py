"""
gridmo: multi-objective trajectories on a grid.

A small, self-contained framework for the grid problem:

* an environment (grid, start, goal, landmarks such as grass and a pool,
  obstacles, moving people) with a fixed number of steps T;
* objectives built from a per-step closeness score, summed over the trajectory:

      o = sum_{t=0..T} w(t) * k(d_t)      (optionally divided by sum_t w(t))

  where d_t is the distance at step t from the agent to the target,
  k is the closeness kernel (default 2^-d) and w is the time weight
  (default t, as in the original formula; 1 and gamma^t are alternatives);
* exact Pareto fronts (brute-force enumeration for small cases, an exact
  dynamic program over (cell, time) for larger ones);
* simulated agents (uniform random, myopic greedy with preferences,
  weighted-sum optimal, Chebyshev optimal);
* evaluation of any set of trajectories against the exact front
  (validity, on-front rate, coverage, hypervolume) -- the template a
  generative model will later be scored with;
* random environments and a JSON-lines dataset for conditional generation;
* matplotlib figures.

Coordinates are (row, col), 1-indexed,
row 1 at the top. Objective values are reported in the user's orientation
(larger is better for "max" objectives, smaller is better for "min").
Internally every objective is turned into "larger is better" (min
objectives are negated) so a single dominance test works throughout.
"""
from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

Cell = Tuple[int, int]
ROUND = 9  # decimals used to decide that two objective vectors are equal

# Environment
@dataclass
class Mover:
    """A moving object (e.g. a person) with a fixed, known schedule.

    path : cells visited at t = 0, 1, 2, ...
    mode : what happens after the path ends
        "loop"     - start the path again
        "pingpong" - walk it back and forth
        "hold"     - stay on the last cell
    """
    name: str
    path: List[Cell]
    mode: str = "loop"

    def position(self, t: int) -> Cell:
        n = len(self.path)
        if n == 1:
            return tuple(self.path[0])
        if self.mode == "loop":
            return tuple(self.path[t % n])
        if self.mode == "pingpong":
            period = 2 * (n - 1)
            k = t % period
            return tuple(self.path[k if k < n else period - k])
        if self.mode == "hold":
            return tuple(self.path[min(t, n - 1)])
        raise ValueError(f"unknown mover mode {self.mode!r}")


@dataclass
class GridEnv:
    """A fixed grid environment.

    rows, cols        grid size
    start, goal       (row, col), 1-indexed, row 1 at the top
    T                 number of steps; every trajectory has T + 1 positions
    landmarks         name -> list of cells (e.g. {"grass": [(2, 2)]})
    impassable        landmark names the agent cannot step on (e.g. a pool)
    obstacles         cells nobody can enter
    movers            moving objects with known schedules
    allow_stay        the agent may wait in place for a step
    diagonal          8-connected moves instead of 4-connected
    must_end_at_goal  trajectories must be at the goal at step T
    absorbing_goal    once at the goal, the agent stays there
    """
    rows: int
    cols: int
    start: Cell
    goal: Cell
    T: int
    landmarks: Dict[str, List[Cell]] = field(default_factory=dict)
    impassable: List[str] = field(default_factory=list)
    obstacles: List[Cell] = field(default_factory=list)
    movers: List[Mover] = field(default_factory=list)
    allow_stay: bool = True
    diagonal: bool = False
    must_end_at_goal: bool = True
    absorbing_goal: bool = False

    # indexing
    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def idx(self, cell: Cell) -> int:
        r, c = cell
        if not (1 <= r <= self.rows and 1 <= c <= self.cols):
            raise ValueError(f"cell {cell} is outside the {self.rows}x{self.cols} grid")
        return (r - 1) * self.cols + (c - 1)

    def cell(self, i: int) -> Cell:
        return (i // self.cols + 1, i % self.cols + 1)

    # movement
    def blocked(self) -> set:
        b = {tuple(c) for c in self.obstacles}
        for name in self.impassable:
            b |= {tuple(c) for c in self.landmarks.get(name, [])}
        return b

    def passable_mask(self) -> np.ndarray:
        m = np.ones(self.n_cells, bool)
        for c in self.blocked():
            m[self.idx(c)] = False
        return m

    def step_offsets(self) -> List[Tuple[int, int]]:
        offs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if self.diagonal:
            offs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        if self.allow_stay:
            offs = [(0, 0)] + offs
        return offs

    def successors(self) -> List[List[int]]:
        """succ[i] = cells reachable from cell i in one step (incl. staying)."""
        ok = self.passable_mask()
        g = self.idx(self.goal)
        succ = []
        for i in range(self.n_cells):
            if not ok[i]:
                succ.append([])
                continue
            if self.absorbing_goal and i == g:
                succ.append([g])
                continue
            r, c = self.cell(i)
            out = []
            for dr, dc in self.step_offsets():
                rr, cc = r + dr, c + dc
                if 1 <= rr <= self.rows and 1 <= cc <= self.cols:
                    j = self.idx((rr, cc))
                    if ok[j]:
                        out.append(j)
            succ.append(out)
        return succ

    def validate(self) -> None:
        ok = self.passable_mask()
        for name, c in (("start", self.start), ("goal", self.goal)):
            if not ok[self.idx(c)]:
                raise ValueError(f"{name} {c} is on a blocked cell")
        for name, cells in self.landmarks.items():
            for c in cells:
                self.idx(c)
        if self.T < 0:
            raise ValueError("T must be >= 0")

    # serialization
    def to_dict(self) -> dict:
        d = asdict(self)
        d["start"] = list(self.start)
        d["goal"] = list(self.goal)
        d["landmarks"] = {k: [list(c) for c in v] for k, v in self.landmarks.items()}
        d["obstacles"] = [list(c) for c in self.obstacles]
        d["movers"] = [{"name": m.name, "path": [list(c) for c in m.path], "mode": m.mode}
                       for m in self.movers]
        return d

    @staticmethod
    def from_dict(d: dict) -> "GridEnv":
        d = dict(d)
        d["start"] = tuple(d["start"])
        d["goal"] = tuple(d["goal"])
        d["landmarks"] = {k: [tuple(c) for c in v] for k, v in d.get("landmarks", {}).items()}
        d["obstacles"] = [tuple(c) for c in d.get("obstacles", [])]
        d["movers"] = [Mover(m["name"], [tuple(c) for c in m["path"]], m.get("mode", "loop"))
                       for m in d.get("movers", [])]
        return GridEnv(**d)


# objectives
KERNELS: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    # closeness k(d); `p` is the kernel parameter
    "exp2": lambda d, p: np.power(2.0, -d),                       # 2^-d  (default)
    "exp": lambda d, p: np.power(p, -d),                          # p^-d
    "inverse": lambda d, p: 1.0 / (1.0 + d),                      # 1/(1+d)
    "linear": lambda d, p: np.clip(1.0 - d / p, 0.0, None),       # max(0, 1 - d/p)
    "within": lambda d, p: (d <= p).astype(float),                # 1 if d <= p
}

TIME_WEIGHTS: Dict[str, Callable[[np.ndarray, float], np.ndarray]] = {
    "linear": lambda t, g: t.astype(float),          # w(t) = t   (default)
    "uniform": lambda t, g: np.ones_like(t, float),  # w(t) = 1
    "discount": lambda t, g: np.power(g, t),         # w(t) = gamma^t
}


@dataclass
class Objective:
    """One objective of a trajectory.

    kind
        "closeness" - sum over steps of w(t) * k(distance to target)
        "time"      - sum over steps of w(t) * [agent not at the goal]
                      (with uniform weights and an absorbing goal this is
                      the arrival time)
    target      landmark name or mover name (closeness only)
    sense       "max" (more is better) or "min" (less is better)
    kernel      key of KERNELS; kernel_param is its parameter
    time_weight key of TIME_WEIGHTS; gamma is used by "discount"
    normalize   divide by sum_t w(t) so the value lies in [0, 1]
    metric      "manhattan", "euclidean", "chebyshev" or "walk"
                ("walk" = shortest walking distance around blocked cells)
    """
    name: str
    target: Optional[str] = None
    sense: str = "max"
    kind: str = "closeness"
    kernel: str = "exp2"
    kernel_param: float = 2.0
    time_weight: str = "linear"
    gamma: float = 0.9
    normalize: bool = True
    metric: str = "manhattan"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Objective":
        return Objective(**d)

    def describe(self) -> str:
        w = {"linear": "t", "uniform": "1", "discount": f"{self.gamma}^t"}[self.time_weight]
        if self.kind == "time":
            core = f"sum_t {w} * [not at goal]"
        else:
            k = {"exp2": "2^-d", "exp": f"{self.kernel_param}^-d", "inverse": "1/(1+d)",
                 "linear": f"max(0,1-d/{self.kernel_param})",
                 "within": f"[d<={self.kernel_param}]"}[self.kernel]
            core = f"sum_t {w} * {k},  d = {self.metric} distance to {self.target}"
        norm = f" / sum_t {w}" if self.normalize else ""
        return f"{self.name} ({self.sense}): {core}{norm}"


def time_weights(obj: Objective, T: int) -> np.ndarray:
    return TIME_WEIGHTS[obj.time_weight](np.arange(T + 1), obj.gamma)


def _walk_distances(env: GridEnv, sources: Sequence[int]) -> np.ndarray:
    """BFS distance from a set of source cells, moving 4/8-connected around
    obstacles (sources themselves may be impassable, e.g. the pool)."""
    blocked = env.passable_mask() == False  # noqa: E712
    dist = np.full(env.n_cells, np.inf)
    q = deque()
    for s in sources:
        dist[s] = 0
        q.append(s)
    offs = [o for o in env.step_offsets() if o != (0, 0)]
    while q:
        i = q.popleft()
        r, c = env.cell(i)
        for dr, dc in offs:
            rr, cc = r + dr, c + dc
            if 1 <= rr <= env.rows and 1 <= cc <= env.cols:
                j = env.idx((rr, cc))
                if not blocked[j] and dist[j] == np.inf:
                    dist[j] = dist[i] + 1
                    q.append(j)
    return dist


def distance_to(env: GridEnv, targets: Sequence[Cell], metric: str) -> np.ndarray:
    """Distance from every cell to the nearest target cell (length n_cells)."""
    if metric == "walk":
        return _walk_distances(env, [env.idx(t) for t in targets])
    rr, cc = np.divmod(np.arange(env.n_cells), env.cols)
    rr, cc = rr + 1, cc + 1
    best = np.full(env.n_cells, np.inf)
    for (tr, tc) in targets:
        dr, dc = np.abs(rr - tr), np.abs(cc - tc)
        if metric == "manhattan":
            d = dr + dc
        elif metric == "euclidean":
            d = np.sqrt(dr ** 2 + dc ** 2)
        elif metric == "chebyshev":
            d = np.maximum(dr, dc)
        else:
            raise ValueError(f"unknown metric {metric!r}")
        best = np.minimum(best, d)
    return best


def per_step_scores(env: GridEnv, obj: Objective, weighted: bool = True) -> np.ndarray:
    """Array (T+1, n_cells): the contribution of being on each cell at each
    step, in the user's orientation (before the sense flip). With
    weighted=False the time weight and normalisation are left out (this is
    the raw closeness k(d_t), used by the greedy agent)."""
    T = env.T
    out = np.zeros((T + 1, env.n_cells))
    movers = {m.name: m for m in env.movers}
    if obj.kind == "time":
        base = np.ones(env.n_cells)
        base[env.idx(env.goal)] = 0.0
        out[:] = base
    elif obj.kind == "closeness":
        kern = KERNELS[obj.kernel]
        if obj.target in env.landmarks:
            d = distance_to(env, env.landmarks[obj.target], obj.metric)
            out[:] = kern(d, obj.kernel_param)
        elif obj.target in movers:
            m = movers[obj.target]
            cache = {}
            for t in range(T + 1):
                p = m.position(t)
                if p not in cache:
                    cache[p] = kern(distance_to(env, [p], obj.metric), obj.kernel_param)
                out[t] = cache[p]
        else:
            raise ValueError(f"objective {obj.name!r}: unknown target {obj.target!r}")
    else:
        raise ValueError(f"unknown objective kind {obj.kind!r}")
    if weighted:
        w = time_weights(obj, T)
        out = out * w[:, None]
        if obj.normalize:
            s = w.sum()
            out = out / (s if s > 0 else 1.0)
    return out


def score_table(env: GridEnv, objectives: Sequence[Objective]) -> np.ndarray:
    """Array (T+1, n_cells, m) in internal orientation (larger is better):
    the value of a trajectory is the sum of S[t, cell_t] over t."""
    cols = []
    for o in objectives:
        s = per_step_scores(env, o, weighted=True)
        cols.append(s if o.sense == "max" else -s)
    return np.stack(cols, axis=-1)


def signs(objectives: Sequence[Objective]) -> np.ndarray:
    return np.array([1.0 if o.sense == "max" else -1.0 for o in objectives])


def trajectory_values(env: GridEnv, objectives: Sequence[Objective], trajs: np.ndarray,
                      S: Optional[np.ndarray] = None, internal: bool = False) -> np.ndarray:
    """Objective values (N, m) of trajectories given as cell-index arrays (N, T+1)."""
    trajs = np.atleast_2d(np.asarray(trajs, dtype=int))
    if S is None:
        S = score_table(env, objectives)
    V = S[np.arange(env.T + 1)[None, :], trajs, :].sum(axis=1)
    return V if internal else V * signs(objectives)


# Counting, uniform sampling, enumeration
def completion_counts(env: GridEnv) -> List[List[int]]:
    """N[t][i] = number of ways to finish a valid trajectory from cell i at
    step t (exact Python integers)."""
    env.validate()
    succ = env.successors()
    ok = env.passable_mask()
    T, n = env.T, env.n_cells
    N = [[0] * n for _ in range(T + 1)]
    for i in range(n):
        if ok[i] and (not env.must_end_at_goal or i == env.idx(env.goal)):
            N[T][i] = 1
    for t in range(T - 1, -1, -1):
        nxt = N[t + 1]
        N[t] = [sum(nxt[j] for j in succ[i]) for i in range(n)]
    return N


def count_trajectories(env: GridEnv) -> int:
    """Number of valid trajectories (revisits allowed)."""
    return completion_counts(env)[0][env.idx(env.start)]


def sample_uniform(env: GridEnv, k: int, rng: np.random.Generator,
                   N: Optional[List[List[int]]] = None) -> np.ndarray:
    """k trajectories drawn uniformly at random from ALL valid trajectories
    (exact: each next cell is chosen in proportion to its completion count)."""
    if N is None:
        N = completion_counts(env)
    succ = env.successors()
    s = env.idx(env.start)
    if N[0][s] == 0:
        raise ValueError("no valid trajectory exists for this environment and T")
    out = np.empty((k, env.T + 1), dtype=int)
    for a in range(k):
        i = s
        out[a, 0] = i
        for t in range(env.T):
            nxt = succ[i]
            cnt = [N[t + 1][j] for j in nxt]
            tot = sum(cnt)
            # exact integer draw (counts may exceed float precision)
            u = int(rng.integers(0, 2 ** 62)) * tot // (2 ** 62)
            acc = 0
            for j, cj in zip(nxt, cnt):
                acc += cj
                if u < acc:
                    i = j
                    break
            out[a, t + 1] = i
    return out


def enumerate_trajectories(env: GridEnv, allow_revisit: bool = True,
                           limit: int = 3_000_000) -> np.ndarray:
    """Every valid trajectory as an array (N, T+1) of cell indices.

    allow_revisit=False keeps only simple paths (no cell twice, so no
    waiting either). Raises if more than `limit` trajectories exist."""

    env.validate()
    succ = env.successors()
    T = env.T
    s, g = env.idx(env.start), env.idx(env.goal)
    if allow_revisit:
        N = completion_counts(env)
        total = N[0][s]
        if total > limit:
            raise ValueError(f"{total:,} trajectories exceed limit={limit:,}; use solve_front()")
        out = np.empty((total, T + 1), dtype=int)
        path = [s]
        row = 0

        def rec(i, t):
            nonlocal row
            if t == T:
                out[row, :] = path
                row += 1
                return
            for j in succ[i]:
                if N[t + 1][j]:
                    path.append(j)
                    rec(j, t + 1)
                    path.pop()

        if total:
            rec(s, 0)
        return out
    # simple paths of exactly T steps
    dist_goal = _walk_distances(env, [g])
    res, path, seen = [], [s], {s}

    def rec2(i, t):
        if len(res) > limit:
            raise ValueError(f"more than limit={limit:,} simple paths")
        if t == T:
            if (not env.must_end_at_goal) or i == g:
                res.append(list(path))
            return
        for j in succ[i]:
            if j in seen:
                continue
            if env.must_end_at_goal and dist_goal[j] > T - t - 1:
                continue
            seen.add(j)
            path.append(j)
            rec2(j, t + 1)
            path.pop()
            seen.remove(j)

    rec2(s, 0)
    return np.array(res, dtype=int).reshape(-1, T + 1)


# Pareto utilities (internal orientation: larger is better in every column)
def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a >= b) and np.any(a > b))


def nondominated_indices(V: np.ndarray) -> np.ndarray:
    """Indices of the rows of V not dominated by any other row (ties kept)."""
    V = np.asarray(V, float)
    if len(V) == 0:
        return np.zeros(0, int)
    R = np.round(V, ROUND)
    # Descending lexicographic order: anything that dominates a point comes
    # before it, so each point only needs checking against points kept so far.
    order = np.lexsort(R.T[::-1])[::-1]
    keep = []
    if R.shape[1] == 2:
        best_y, x_at_best = -np.inf, None
        for i in order:
            x, y = R[i]
            if y > best_y:
                best_y, x_at_best = y, x
                keep.append(i)
            elif y == best_y and x == x_at_best:
                keep.append(i)  # exact tie with a kept point
        return np.sort(np.array(keep, dtype=int))
    F = np.empty_like(R)
    k = 0
    for i in order:
        v = R[i]
        if k:
            G = F[:k]
            if np.any(np.all(G >= v, axis=1) & np.any(G > v, axis=1)):
                continue
        keep.append(i)
        F[k] = v
        k += 1
    return np.sort(np.array(keep, dtype=int))


def unique_rows(V: np.ndarray):
    R = np.round(np.asarray(V, float), ROUND)
    U, first, inv, cnt = np.unique(R, axis=0, return_index=True, return_inverse=True,
                                   return_counts=True)
    return U, first, inv.ravel(), cnt


@dataclass
class Front:
    """An exact Pareto front.

    values (K, m) objective values in the user's orientation
    internal (K, m) larger-is-better version used for dominance
    counts number of trajectories that reach each front vector exactly
    reps (K, T+1) one representative trajectory per front vector
    """
    values: np.ndarray
    internal: np.ndarray
    counts: List[int]
    reps: np.ndarray
    method: str = ""

    def __len__(self):
        return len(self.values)

    @property
    def n_trajectories(self) -> int:
        return int(sum(self.counts))


def front_by_enumeration(env: GridEnv, objectives: Sequence[Objective],
                         trajs: Optional[np.ndarray] = None) -> Front:
    """Exact front by listing every trajectory (small cases; used to check
    solve_front)."""
    if trajs is None:
        trajs = enumerate_trajectories(env)
    Vi = trajectory_values(env, objectives, trajs, internal=True)
    U, first, inv, cnt = unique_rows(Vi)
    nd = nondominated_indices(U)
    sg = signs(objectives)
    return Front(values=U[nd] * sg, internal=U[nd], counts=[int(cnt[k]) for k in nd],
                 reps=trajs[first[nd]], method="enumeration")


def _merge_and_filter(V, counts, parents):
    """Merge equal vectors (summing counts) then drop dominated ones."""
    R = np.round(V, ROUND)
    U, first, inv = np.unique(R, axis=0, return_index=True, return_inverse=True)
    inv = inv.ravel()
    c = [0] * len(U)
    for a, k in enumerate(inv):
        c[k] += counts[a]
    Vm = V[first]
    pm = [parents[f] for f in first]
    nd = nondominated_indices(Vm)
    return Vm[nd], [c[k] for k in nd], [pm[k] for k in nd]


def solve_front(env: GridEnv, objectives: Sequence[Objective]) -> Front:
    """Exact Pareto front by dynamic programming over (cell, time).

    Every objective adds up a per-step score that depends only on the cell
    and the step, so two partial trajectories that reach the same cell at
    the same step get the same future. A partial trajectory dominated there
    can never become nondominated, so it is dropped (Bellman's principle for
    Pareto sets). Revisits must be allowed (the default)."""
    env.validate()
    S = score_table(env, objectives)
    T, n, m = env.T, env.n_cells, len(objectives)
    succ = env.successors()
    pred = [[] for _ in range(n)]
    for i in range(n):
        for j in succ[i]:
            pred[j].append(i)
    N = completion_counts(env)
    s = env.idx(env.start)
    if N[0][s] == 0:
        raise ValueError("no valid trajectory exists for this environment and T")
    # labels[t][cell] = (V (L,m), counts [L], parents [(prev_cell, prev_label)])
    labels: List[Dict[int, tuple]] = [dict() for _ in range(T + 1)]
    labels[0][s] = (S[0, s][None, :].copy(), [1], [None])
    for t in range(1, T + 1):
        prev = labels[t - 1]
        for j in range(n):
            if N[t][j] == 0:
                continue
            Vs, cs, ps = [], [], []
            for i in pred[j]:
                if i in prev:
                    V, c, _ = prev[i]
                    Vs.append(V + S[t, j])
                    cs.extend(c)
                    ps.extend((i, k) for k in range(len(c)))
            if not Vs:
                continue
            labels[t][j] = _merge_and_filter(np.vstack(Vs), cs, ps)
    # collect end labels
    if env.must_end_at_goal:
        ends = [env.idx(env.goal)] if env.idx(env.goal) in labels[T] else []
    else:
        ends = list(labels[T].keys())
    Vs, cs, ps = [], [], []
    for j in ends:
        V, c, _ = labels[T][j]
        Vs.append(V)
        cs.extend(c)
        ps.extend((j, k) for k in range(len(c)))
    V, c, p = _merge_and_filter(np.vstack(Vs), cs, ps)
    # backtrack one representative per front vector
    reps = np.empty((len(V), T + 1), dtype=int)
    for a, (j, k) in enumerate(p):
        cell, lab = j, k
        for t in range(T, -1, -1):
            reps[a, t] = cell
            par = labels[t][cell][2][lab]
            if par is None:
                break
            cell, lab = par
    order = np.lexsort(np.round(V, ROUND).T[::-1])
    V = V[order]
    return Front(values=V * signs(objectives), internal=V, counts=[c[k] for k in order],
                 reps=reps[order], method="dp")


def on_front(front: Front, Vi: np.ndarray) -> np.ndarray:
    """Boolean per row of Vi (internal orientation): equal to a front vector."""
    F = np.round(front.internal, ROUND)
    R = np.round(np.atleast_2d(Vi), ROUND)
    return np.array([bool(np.any(np.all(F == r, axis=1))) for r in R])


def is_nondominated(front: Front, Vi: np.ndarray) -> np.ndarray:
    """Boolean per row: not dominated by any front vector (i.e. P+)."""
    F = np.round(front.internal, ROUND)
    R = np.round(np.atleast_2d(Vi), ROUND)
    return np.array([not bool(np.any(np.all(F >= r, axis=1) & np.any(F > r, axis=1)))
                     for r in R])


# agents
def weighted_sum_choice(front: Front, w: Sequence[float]) -> int:
    """Index of the front vector an agent maximising sum_j w_j * o_j picks
    (min objectives enter as -o_j). This is also its best trajectory overall:
    a weighted-sum optimum is always on the front."""
    return int(np.argmax(front.internal @ np.asarray(w, float)))


def chebyshev_choice(front: Front, w: Sequence[float], ideal: Optional[np.ndarray] = None) -> int:
    """Index of the front vector minimising max_j w_j * (ideal_j - o_j)."""
    if ideal is None:
        ideal = front.internal.max(axis=0) + 1e-9
    gap = (ideal - front.internal) * np.asarray(w, float)
    return int(np.argmin(gap.max(axis=1)))


def weight_grid(m: int, steps: int) -> np.ndarray:
    """All weight vectors on the simplex with the given resolution."""
    if m == 1:
        return np.ones((1, 1))
    out = []

    def rec(prefix, left, k):
        if k == 1:
            out.append(prefix + [left])
            return
        for a in range(left + 1):
            rec(prefix + [a], left - a, k - 1)

    rec([], steps, m)
    return np.array(out, float) / steps


def sweep(front: Front, rule: str = "weighted_sum", steps: int = 200) -> Dict[int, List[np.ndarray]]:
    """Which front vectors are ever chosen as the weights vary.
    Returns {front index: [weights that choose it]}."""
    found: Dict[int, List[np.ndarray]] = {}
    for w in weight_grid(front.internal.shape[1], steps):
        k = weighted_sum_choice(front, w) if rule == "weighted_sum" else chebyshev_choice(front, w)
        found.setdefault(k, []).append(w)
    return found


def simulate_greedy(env: GridEnv, objectives: Sequence[Objective], n_agents: int,
                    rng: np.random.Generator, temperature: float = 0.1,
                    weights: Optional[np.ndarray] = None):
    """Myopic agents: at each step an agent looks only at the next cell's
    raw closeness values k(d), weighted by its own preference vector, and
    picks a move by softmax(utility / temperature) among the moves that can
    still reach the goal in time. Preferences are Dirichlet(1,...,1) unless
    given. Returns (trajectories (n, T+1), weights (n, m))."""
    m = len(objectives)
    if weights is None:
        weights = rng.dirichlet(np.ones(m), size=n_agents)
    weights = np.atleast_2d(weights)
    if len(weights) == 1 and n_agents > 1:
        weights = np.repeat(weights, n_agents, axis=0)
    K = np.stack([per_step_scores(env, o, weighted=False) * (1 if o.sense == "max" else -1)
                  for o in objectives], axis=-1)  # (T+1, n, m)
    N = completion_counts(env)
    succ = env.successors()
    s = env.idx(env.start)
    out = np.empty((n_agents, env.T + 1), dtype=int)
    for a in range(n_agents):
        i = s
        out[a, 0] = i
        for t in range(env.T):
            nxt = [j for j in succ[i] if N[t + 1][j] > 0]
            u = np.array([K[t + 1, j] @ weights[a] for j in nxt])
            if temperature <= 0:
                best = np.flatnonzero(u == u.max())
                i = nxt[int(rng.choice(best))]
            else:
                z = (u - u.max()) / temperature
                p = np.exp(z)
                p /= p.sum()
                i = nxt[int(rng.choice(len(nxt), p=p))]
            out[a, t + 1] = i
    return out, weights


#evaluation
def hypervolume(P: np.ndarray, ref: np.ndarray) -> float:
    """Hypervolume of points P (larger is better in every column) above the
    reference point `ref`. Exact recursive slicing; fine for small fronts."""
    P = np.asarray(P, float)
    ref = np.asarray(ref, float)
    if len(P) == 0:
        return 0.0
    P = P[np.all(P > ref, axis=1)]
    if len(P) == 0:
        return 0.0
    P = P[nondominated_indices(P)]
    if P.shape[1] == 1:
        return float(P[:, 0].max() - ref[0])
    order = np.argsort(-P[:, -1])
    P = P[order]
    total = 0.0
    for k in range(len(P)):
        lower = P[k + 1, -1] if k + 1 < len(P) else ref[-1]
        h = P[k, -1] - lower
        if h > 0:
            total += h * hypervolume(P[: k + 1, :-1], ref[:-1])
    return float(total)


def default_reference(objectives: Sequence[Objective]) -> Optional[np.ndarray]:
    """Worst possible internal value per objective when all are normalised
    (closeness and time lie in [0, 1]): 0 for max objectives, -1 for min."""
    if not all(o.normalize for o in objectives):
        return None
    return np.array([0.0 if o.sense == "max" else -1.0 for o in objectives])


def check_validity(env: GridEnv, trajs: np.ndarray) -> np.ndarray:
    """Boolean per trajectory: right length, starts at start, every step is a
    legal move, ends at the goal when required."""
    trajs = np.atleast_2d(np.asarray(trajs))
    succ = [set(x) for x in env.successors()]
    s, g = env.idx(env.start), env.idx(env.goal)
    ok = []
    for tr in trajs:
        good = len(tr) == env.T + 1 and int(tr[0]) == s
        if good:
            for a, b in zip(tr[:-1], tr[1:]):
                if not (0 <= int(a) < env.n_cells) or int(b) not in succ[int(a)]:
                    good = False
                    break
        if good and env.must_end_at_goal:
            good = int(tr[-1]) == g
        ok.append(good)
    return np.array(ok, bool)


def nadir_reference(front: Front, margin: float = 0.1) -> np.ndarray:
    """Reference point just below the front's nadir (internal orientation):
    nadir - margin * (ideal - nadir). Points worse than the front's worst
    value in any objective then add (almost) nothing, which keeps the
    hypervolume ratio sensitive to how close a set gets to the front."""
    ideal, nadir = front.internal.max(0), front.internal.min(0)
    span = np.where(ideal > nadir, ideal - nadir, 1.0)
    return nadir - margin * span


def assess(env: GridEnv, objectives: Sequence[Objective], trajs: np.ndarray, front: Front,
           ref="nadir") -> dict:
    """Score a set of trajectories (from any agent or generator) against the
    exact front.

    valid_rate      share of trajectories that are legal
    on_front_rate   share of valid trajectories that land exactly on the front
    nondominated_rate  share of valid trajectories no front vector dominates
                       (equals on_front_rate for an exact front)
    coverage        share of front vectors hit by at least one trajectory
    hv_ratio        hypervolume of what was produced / hypervolume of the front

    ref: "nadir" (default, see nadir_reference), "worst" (theoretical worst
    value, needs normalised objectives) or an explicit internal-orientation
    point. With "worst", almost any set scores a high hv_ratio because the
    box under the front dominates the volume.
    """
    trajs = np.atleast_2d(np.asarray(trajs, dtype=int))
    valid = check_validity(env, trajs)
    out = {"n": int(len(trajs)), "valid_rate": float(valid.mean()) if len(trajs) else 0.0}
    tv = trajs[valid]
    if len(tv) == 0:
        out.update(on_front_rate=0.0, nondominated_rate=0.0, coverage=0.0, hv_ratio=0.0)
        return out
    Vi = trajectory_values(env, objectives, tv, internal=True)
    of = on_front(front, Vi)
    nd = is_nondominated(front, Vi)
    F = np.round(front.internal, ROUND)
    hit = {tuple(r) for r in np.round(Vi[of], ROUND)}
    cov = sum(tuple(r) in hit for r in F) / len(F)
    if isinstance(ref, str):
        if ref == "nadir":
            ref = nadir_reference(front)
        elif ref == "worst":
            ref = default_reference(objectives)
            if ref is None:
                raise ValueError("ref='worst' needs normalised objectives")
        else:
            raise ValueError(f"unknown ref {ref!r}")
    hv_f = hypervolume(front.internal, ref)
    hv_g = hypervolume(Vi, ref)
    out.update(on_front_rate=float(of.mean()), nondominated_rate=float(nd.mean()),
               coverage=float(cov), hv=hv_g, hv_front=hv_f,
               hv_ratio=float(hv_g / hv_f) if hv_f > 0 else float("nan"))
    return out


# trajectory helpers
MOVE_CHARS = {(0, 0): ".", (-1, 0): "U", (1, 0): "D", (0, 1): "R", (0, -1): "L",
              (-1, 1): "9", (-1, -1): "7", (1, 1): "3", (1, -1): "1"}
CHAR_MOVES = {v: k for k, v in MOVE_CHARS.items()}


def to_cells(env: GridEnv, traj: Sequence[int]) -> List[Cell]:
    return [env.cell(int(i)) for i in traj]


def to_moves(env: GridEnv, traj: Sequence[int]) -> str:
    """Move string: U/D/L/R, '.' = wait (diagonals use numpad digits)."""
    cells = to_cells(env, traj)
    return "".join(MOVE_CHARS[(b[0] - a[0], b[1] - a[1])] for a, b in zip(cells, cells[1:]))


def from_moves(env: GridEnv, moves: str) -> np.ndarray:
    r, c = env.start
    out = [env.idx((r, c))]
    for ch in moves:
        dr, dc = CHAR_MOVES[ch]
        r, c = r + dr, c + dc
        out.append(env.idx((r, c)) if (1 <= r <= env.rows and 1 <= c <= env.cols) else -1)
    return np.array(out, dtype=int)


# ready-made settings
def example_env(T: int = 10, **settings) -> GridEnv:
    """The 4x4 example grid: start (4,1), goal (1,4), grass (2,2), pool (3,3)."""
    return GridEnv(rows=4, cols=4, start=(4, 1), goal=(1, 4), T=T,
                   landmarks={"grass": [(2, 2)], "pool": [(3, 3)]}, **settings)


def example_objectives(time_weight: str = "linear", normalize: bool = True,
                       kernel: str = "exp2", people: Sequence[str] = (),
                       **settings) -> List[Objective]:
    """Closeness to the grass and closeness to the pool (more is better),
    plus closeness to each named person (less is better)."""
    objectives = [Objective("grass", "grass", "max", time_weight=time_weight,
                            normalize=normalize, kernel=kernel, **settings),
                  Objective("pool", "pool", "max", time_weight=time_weight,
                            normalize=normalize, kernel=kernel, **settings)]
    for person in people:
        objectives.append(Objective(f"near_{person}", person, "min", time_weight=time_weight,
                                    normalize=normalize, kernel=kernel, **settings))
    return objectives


# random env and datasets (for conditional generation later0
def random_walk_mover(env: GridEnv, rng: np.random.Generator, name: str = "person",
                      start: Optional[Cell] = None, length: Optional[int] = None) -> Mover:
    """A person doing a random walk on passable cells (may pause)."""
    ok = env.passable_mask()
    if start is None:
        free = [env.cell(i) for i in range(env.n_cells) if ok[i]
                and env.cell(i) not in (tuple(env.start), tuple(env.goal))]
        start = free[int(rng.integers(len(free)))]
    length = env.T + 1 if length is None else length
    path = [tuple(start)]
    offs = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(length - 1):
        r, c = path[-1]
        opts = [(r + dr, c + dc) for dr, dc in offs
                if 1 <= r + dr <= env.rows and 1 <= c + dc <= env.cols
                and ok[env.idx((r + dr, c + dc))]]
        path.append(opts[int(rng.integers(len(opts)))])
    return Mover(name, path, "hold")


def random_env(rng: np.random.Generator, rows: int = 6, cols: int = 6, slack: int = 4,
               landmark_names: Sequence[str] = ("grass", "pool"), n_obstacles: int = 3,
               n_people: int = 0, max_tries: int = 200) -> GridEnv:
    """A random layout: start and goal far apart, one cell per landmark,
    a few obstacles, optional random-walking people. T = shortest walking
    distance from start to goal + slack."""
    for _ in range(max_tries):
        cells = [(r, c) for r in range(1, rows + 1) for c in range(1, cols + 1)]
        perm = rng.permutation(len(cells))
        start = cells[perm[0]]
        far = [cells[k] for k in perm[1:]
               if abs(cells[k][0] - start[0]) + abs(cells[k][1] - start[1]) >= (rows + cols) // 2]
        if not far:
            continue
        goal = far[0]
        rest = [cells[k] for k in perm[1:] if cells[k] != goal]
        lms = {name: [rest[a]] for a, name in enumerate(landmark_names)}
        obst = rest[len(landmark_names): len(landmark_names) + n_obstacles]
        env = GridEnv(rows, cols, start, goal, T=0, landmarks=lms, obstacles=obst)
        d = _walk_distances(env, [env.idx(goal)])[env.idx(start)]
        if not np.isfinite(d):
            continue
        env.T = int(d) + slack
        for p in range(n_people):
            env.movers.append(random_walk_mover(env, rng, name=f"person{p + 1}"))
        if count_trajectories(env) > 0:
            return env
    raise RuntimeError("could not build a feasible random environment")


def env_to_tensor(env: GridEnv) -> Dict[str, np.ndarray]:
    """Array encoding for a conditional generator.

    static  (C, rows, cols): blocked, start, goal, one channel per landmark
    movers  (T+1, M, rows, cols): position of each mover at each step
    """
    C = 3 + len(env.landmarks)
    st = np.zeros((C, env.rows, env.cols), np.float32)
    for (r, c) in env.blocked():
        st[0, r - 1, c - 1] = 1
    st[1, env.start[0] - 1, env.start[1] - 1] = 1
    st[2, env.goal[0] - 1, env.goal[1] - 1] = 1
    for k, (name, cells) in enumerate(sorted(env.landmarks.items())):
        for (r, c) in cells:
            st[3 + k, r - 1, c - 1] = 1
    mv = np.zeros((env.T + 1, len(env.movers), env.rows, env.cols), np.float32)
    for k, m in enumerate(env.movers):
        for t in range(env.T + 1):
            r, c = m.position(t)
            mv[t, k, r - 1, c - 1] = 1
    return {"static": st, "movers": mv}


def make_record(env: GridEnv, objectives: Sequence[Objective], rng: np.random.Generator,
                n_random: int = 50, front: Optional[Front] = None) -> dict:
    """One dataset record: environment, objectives, the exact front (one
    representative per front vector plus how many trajectories tie it) and
    a uniform random sample labelled P+ (nondominated) / P- (dominated)."""
    if front is None:
        front = solve_front(env, objectives)
    rec = {"env": env.to_dict(), "objectives": [o.to_dict() for o in objectives],
           "n_trajectories": str(count_trajectories(env)),
           "front": [{"cells": [list(c) for c in to_cells(env, tr)], "moves": to_moves(env, tr),
                      "values": [float(x) for x in v], "n_tied": str(int(c))}
                     for tr, v, c in zip(front.reps, front.values, front.counts)]}
    if n_random:
        R = sample_uniform(env, n_random, rng)
        Vi = trajectory_values(env, objectives, R, internal=True)
        nd = is_nondominated(front, Vi)
        rec["random"] = [{"moves": to_moves(env, tr),
                          "values": [float(x) for x in v * signs(objectives)],
                          "label": "P+" if lab else "P-"}
                         for tr, v, lab in zip(R, Vi, nd)]
    return rec


def make_dataset(n_envs: int, rng: np.random.Generator,
                 objectives_for: Callable[[GridEnv], List[Objective]],
                 env_kwargs: Optional[dict] = None, n_random: int = 50) -> List[dict]:
    env_kwargs = env_kwargs or {}
    out = []
    for _ in range(n_envs):
        env = random_env(rng, **env_kwargs)
        out.append(make_record(env, objectives_for(env), rng, n_random=n_random))
    return out


def save_jsonl(records: Sequence[dict], path: str) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


#matplotlib figure
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots: validated all-pairs
LANDMARK_FILL = {"grass": "#d6ecd2", "pool": "#cde2fb"}
OTHER_FILL = "#efe6f7"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ax.spines.values():
        s.set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    ax.title.set_color(INK)


def plot_env(env: GridEnv, trajs: Optional[np.ndarray] = None, ax=None, title: Optional[str] = None,
             show_movers: bool = True, label_trajs: Optional[Sequence[str]] = None):
    """Draw the grid (row 1 at the top). Up to three trajectories are drawn
    in distinct colours, slightly offset so overlaps stay visible; waits
    are marked with a ring and the number of steps waited."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    if ax is None:
        _, ax = plt.subplots(figsize=(0.9 * env.cols + 1.2, 0.9 * env.rows + 0.9))
    ax.set_facecolor(SURFACE)
    blocked = env.blocked()

    def xy(cell):
        r, c = cell
        return c - 0.5, env.rows - r + 0.5

    for r in range(1, env.rows + 1):
        for c in range(1, env.cols + 1):
            fill = SURFACE
            text = ""
            for name, cells in env.landmarks.items():
                if (r, c) in [tuple(x) for x in cells]:
                    fill = LANDMARK_FILL.get(name, OTHER_FILL)
                    text = name
            if (r, c) in {tuple(x) for x in env.obstacles}:
                fill, text = AXIS, "wall"
            ax.add_patch(Rectangle((c - 1, env.rows - r), 1, 1, facecolor=fill,
                                   edgecolor=GRID, linewidth=1))
            if text:
                ax.text(c - 0.5, env.rows - r + 0.18, text, ha="center", va="center",
                        fontsize=8, color=INK2)
            ax.text(c - 0.94, env.rows - r + 0.93, f"{r},{c}", ha="left", va="top",
                    fontsize=6, color=MUTED)
    for cell, lab in ((env.start, "S"), (env.goal, "G")):
        x, y = xy(cell)
        ax.text(x, y + 0.3, lab, ha="center", va="center", fontsize=11, weight="bold", color=INK)

    handles = []
    if show_movers:
        for m in env.movers:
            pts = [xy(m.position(t)) for t in range(env.T + 1)]
            xs, ys = zip(*pts)
            h, = ax.plot(xs, ys, ls=(0, (2, 2)), color=MUTED, lw=1.2, zorder=2,
                         label=f"{m.name} (step numbers)")
            handles.append(h)
            seen = {}
            for t, p in enumerate(pts):
                seen.setdefault(p, []).append(t)
            for p, ts in seen.items():  # top-right corner of the cell
                ax.text(p[0] + 0.46, p[1] + 0.44,
                        ",".join(map(str, ts[:4])) + ("…" if len(ts) > 4 else ""),
                        fontsize=6, color=INK2, ha="right", va="top")

    if trajs is not None:
        trajs = np.atleast_2d(trajs)
        if len(trajs) > len(SERIES):
            raise ValueError("draw at most three trajectories per panel; use plot_trajectory_panels")
        offs = np.linspace(-0.14, 0.14, len(trajs)) if len(trajs) > 1 else [0.0]
        lane = 0.06  # each move is drawn slightly to its right, so U-then-D shows two lines
        for k, tr in enumerate(trajs):
            col = SERIES[k]
            cen = np.array([xy(env.cell(int(i))) for i in tr]) + np.array([offs[k], -offs[k] - 0.1])
            # the moves (waits removed), each shifted to its own right-hand side
            segs = [(a, b) for a, b in zip(cen[:-1], cen[1:]) if np.hypot(*(b - a)) > 0]
            if segs:
                nrms = [np.array([(b - a)[1], -(b - a)[0]]) / np.hypot(*(b - a)) * lane
                        for a, b in segs]
                pts = [segs[0][0] + nrms[0]]
                for (a, b), n1, n2 in zip(segs[:-1], nrms[:-1], nrms[1:]):
                    if np.allclose(n1, n2):          # straight on
                        pts.append(b + n1)
                    elif np.allclose(n1, -n2):       # U-turn: short hairpin
                        pts += [b + n1, b + n2]
                    else:                            # turn: meet at the corner
                        pts.append(b + n1 + n2)
                pts.append(segs[-1][1] + nrms[-1])
            else:
                pts = [cen[0]]
            pts = np.array(pts)
            h, = ax.plot(pts[:, 0], pts[:, 1], color=col, lw=2, solid_capstyle="round",
                         solid_joinstyle="miter", zorder=3,
                         label=label_trajs[k] if label_trajs else None)
            handles.append(h)
            ax.scatter(pts[:1, 0], pts[:1, 1], s=40, color=col, zorder=4, edgecolor=SURFACE,
                       linewidth=2)
            if segs:  # arrow on the last move (into the goal)
                (a, b), nrm = segs[-1], nrms[-1]
                d = b - a
                ax.annotate("", xy=b + nrm, xytext=a + nrm + d * 0.45,
                            arrowprops=dict(arrowstyle="-|>", color=col, lw=2, mutation_scale=12,
                                            shrinkA=0, shrinkB=0), zorder=4)
            runs = {}
            for t in range(1, len(tr)):
                if tr[t] == tr[t - 1]:
                    runs[int(tr[t])] = runs.get(int(tr[t]), 0) + 1
            for i, w in runs.items():
                x, y = xy(env.cell(i))
                c0 = (x + offs[k], y - offs[k] - 0.1)
                ax.add_patch(Circle(c0, 0.16, facecolor=SURFACE, edgecolor=col, lw=2, zorder=5))
                ax.text(*c0, str(w), ha="center", va="center", fontsize=7, color=INK, zorder=6)
    if label_trajs or (handles and env.movers and show_movers):
        labelled = [h for h in handles if not h.get_label().startswith("_")]
        if labelled:
            ax.legend(handles=labelled, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                      ncol=min(len(labelled), 4), frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_xlim(0, env.cols)
    ax.set_ylim(0, env.rows)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, color=INK, loc="left")
    return ax


def plot_trajectory_panels(env: GridEnv, trajs: np.ndarray, titles: Optional[Sequence[str]] = None,
                           ncols: int = 4, panel: float = 2.6):
    """Small multiples: one trajectory per panel."""
    import matplotlib.pyplot as plt
    trajs = np.atleast_2d(trajs)
    n = len(trajs)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, min(ncols, n), figsize=(panel * min(ncols, n), panel * nrows + 0.3))
    axes = np.atleast_1d(axes).ravel()
    for k, ax in enumerate(axes):
        if k < n:
            plot_env(env, trajs[k][None, :], ax=ax, title=titles[k] if titles else None)
        else:
            ax.axis("off")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig


def plot_front(front: Front, objectives: Sequence[Objective], all_values: Optional[np.ndarray] = None,
               pair: Tuple[int, int] = (0, 1), picks: Optional[Dict[str, Sequence[int]]] = None,
               ax=None, title: Optional[str] = None, annotate_counts: bool = True):
    """Objective space for two objectives: every trajectory (grey), the exact
    front (blue, joined in order) and optional highlighted picks
    (e.g. {'weighted-sum agents': [indices into the front]})."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.4))
    _style(ax)
    a, b = pair
    if all_values is not None and len(all_values):
        U = np.unique(np.round(all_values[:, [a, b]], ROUND), axis=0)
        ax.scatter(U[:, 0], U[:, 1], s=10, color=AXIS, label="all trajectories", zorder=1,
                   linewidths=0)
    F = front.values[:, [a, b]]
    if len(objectives) == 2:
        order = np.argsort(F[:, 0])
        ax.plot(F[order, 0], F[order, 1], color=SERIES[0], lw=1, alpha=0.5, zorder=2)
    ax.scatter(F[:, 0], F[:, 1], s=46, color=SERIES[0], edgecolor=SURFACE, linewidth=1.5,
               label="exact front" if len(objectives) == 2 else "front (projected)", zorder=3)
    if annotate_counts and len(objectives) == 2 and len(F) <= 20:
        for (x, y), c in zip(F, front.counts):
            ax.annotate(f"{c}", (x, y), xytext=(5, 4), textcoords="offset points",
                        fontsize=7, color=INK2)
    if picks:
        markers = ["o", "s"]
        for k, (lab, idx) in enumerate(picks.items()):
            idx = list(idx)
            ax.scatter(F[idx, 0], F[idx, 1], s=150, facecolor="none",
                       edgecolor=SERIES[1 + k % 2], linewidth=2, marker=markers[k % 2],
                       label=lab, zorder=4)
    ax.set_xlabel(f"{objectives[a].name} ({'higher' if objectives[a].sense == 'max' else 'lower'} is better)")
    ax.set_ylabel(f"{objectives[b].name} ({'higher' if objectives[b].sense == 'max' else 'lower'} is better)")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="best")
    if title:
        ax.set_title(title, fontsize=10, loc="left")
    return ax


def plot_step_contributions(env: GridEnv, obj_variants: Dict[str, Objective], traj: Sequence[int],
                            panel: Tuple[float, float] = (3.4, 2.4)):
    """For one trajectory: the share of the objective earned at each step,
    one panel per objective variant (e.g. time weight 1 vs t vs 0.9^t)."""
    import matplotlib.pyplot as plt
    k = len(obj_variants)
    fig, axes = plt.subplots(1, k, figsize=(panel[0] * k, panel[1]), sharey=True)
    axes = np.atleast_1d(axes)
    t = np.arange(env.T + 1)
    for ax, (lab, o) in zip(axes, obj_variants.items()):
        _style(ax)
        s = per_step_scores(env, o, weighted=True)[t, np.asarray(traj, int)]
        total = s.sum()
        share = s / total if total > 0 else s
        ax.bar(t, share, width=0.7, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        ax.set_title(f"{lab}\nvalue = {total:.3f}", fontsize=9, loc="left")
        ax.set_xlabel("step t")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xticks(t)
    axes[0].set_ylabel("share of the value earned")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    return fig
