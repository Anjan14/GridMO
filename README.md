# GridMO v0: multi-objective trajectories on a grid
### work in progress work

A working starting point for the grid trajectory problem: an agent walks a fixed number of steps from a start to a goal, and each objective adds up how close it is to something (grass, pool, a person) at every step. The framework finds the exact nondominated trajectories, simulates agents, scores any set of trajectories against the exact front, and writes a dataset for the later "environment given → produce plans" model.

## Files

| file | what it is |
|---|---|
| `GridMO_v0.ipynb` | the walkthrough notebook (already run, with all outputs) |
| `gridmo.py` | the framework (one file, numpy + matplotlib only) |
| `test_gridmo.py` | 51 checks, fixed before the code was written |
| `data/grid_dataset_v0.jsonl` | 20 random 6×6 environments with exact fronts (made by section 9) |

## Run it

**Colab.** Unzip, upload the `GridMO` folder to `My Drive/`, open `GridMO_v0.ipynb` in Colab and choose *Run all*. The first cell mounts Drive and imports `gridmo.py` from `My Drive/GridMO`. Everything runs in about 40 seconds on a normal CPU; no GPU needed.

**Locally.** `pip install numpy pandas matplotlib pytest`, then open the notebook from inside the folder. Checks: `pytest -q test_gridmo.py`.

## The formula, and how to change it

As first written: Σ_{t=0}^{n} 2^(−d) × t. The code reads it as

```
o = Σ_t w(t) · k(d_t)  /  Σ_t w(t)      d_t = distance from the agent at step t to the target
k(d) = 2^-d        w(t) = t             (normalised to [0, 1])
```

Each part is a setting of `gridmo.Objective`:

```python
import gridmo

env = gridmo.example_env(T=10)          # 4x4, start (4,1), goal (1,4), grass (2,2), pool (3,3)

objectives = gridmo.example_objectives()                                    # 2^-d, weight t, normalised
objectives = gridmo.example_objectives(time_weight="uniform")               # every step counts the same
objectives = gridmo.example_objectives(time_weight="discount", gamma=0.9)   # earlier steps count more
objectives = gridmo.example_objectives(kernel="inverse")                    # 1/(1+d) instead of 2^-d
objectives = gridmo.example_objectives(normalize=False)                     # raw sums

front = gridmo.solve_front(env, objectives)                                 # exact Pareto front
```

Other settings: `metric` (`manhattan`, `euclidean`, `chebyshev`, `walk` around walls), `Objective(kind="time")` for an arrival-time objective, `GridEnv(impassable=["pool"], obstacles=[...], movers=[gridmo.Mover(...)], allow_stay=..., diagonal=..., absorbing_goal=...)`.

## What is in the notebook

1. The 4×4 example environment.
2. The formula: four example trajectories under w = t, 1 and 0.9^t, and where each version puts its weight.
3. Trajectory counts (exact, by dynamic programming). Check: 184 simple paths, matching OEIS A007764.
4. The exact front for T = 10 (29 vectors, 240 trajectories), checked against brute force over all 23,508 trajectories.
5. How the time weight changes the front (11 vectors with w = 1, 29 with w = t, 34 with 0.9^t at T = 10).
6. Simulated agents: linear-preference agents (e.g. 0.6 grass / 0.4 pool) only ever pick the two extreme trajectories; Chebyshev agents reach every front point; greedy agents and a uniform-random null, all scored with `gridmo.assess`.
7. A person walking around the centre: a third objective (avoid people), and two trajectories through the same cells that differ only in timing.
8. A random 8×8 layout with walls and a person: the exact solver still works there (5.5 million trajectories, 229 front vectors), and how its run time grows with grid size.
9. A dataset of random environments for conditional generation, and the tensor encoding a model would take as input.
10. The checks.
11. Open questions.

## How the exact solver works

Every objective adds up a score that depends only on the cell and the step. Two partial trajectories that reach the same cell at the same step therefore have the same possible futures, so one that is dominated there can never become nondominated. `solve_front` keeps only the nondominated partial scores at each (cell, step) and steps forward. This is multi-objective dynamic programming on the time-expanded grid, the same idea as exact multi-objective shortest-path search. It is exact: the tests compare it with brute force on many settings.

## Limits of v0

* The exact solver needs revisits allowed; simple paths (no revisits) are handled by enumeration only.
* People follow fixed, known schedules. What the agent can see ("the human can see") is not modelled yet.
* With three objectives the front grows to thousands of vectors on 10×10 grids and the solver takes tens of seconds; larger grids will need sampling or a learned generator.
* The 4×4 grid is too small to compare methods: random draws already hit the front. Use 8×8 or larger for comparisons.
* The explanations in the notebook are drafts; rewrite them in your own words before sharing.

## Questions to explore

1. Is the formula Σ_t t·2^(−d_t) (later steps count more) or Σ_t 2^(−d_t)? Normalised?
2. Are 0.6 and 0.4 objective values or an agent's preference weights?
3. What fixed number of steps T? Must the agent end at the goal, and may it wait or backtrack?
4. Is "count time step" a separate objective, or the t in the formula?
5. Can the agent step onto the pool? Manhattan or Euclidean distance?
6. How do people move, does the agent know where they will be, and what does "the human can see" mean?
7. What varies between environments for conditional generation, and should the model output one trajectory per preference or the whole front?
