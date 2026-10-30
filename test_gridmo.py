"""Checks for gridmo. Run with:  pytest -q test_gridmo.py

The expected numbers were fixed BEFORE this framework existed, from an
independent brute-force script on the 4x4 example grid (start (4,1),
goal (1,4), grass (2,2), pool (3,3), Manhattan distance, closeness 2^-d).
"""
import numpy as np
import pytest

import gridmo


# counting
@pytest.mark.parametrize("T,expected", [(6, 20), (8, 896), (10, 23_508), (12, 508_464)])
def test_trajectory_counts(T, expected):
    env = gridmo.example_env(T)
    assert gridmo.count_trajectories(env) == expected
    if T <= 10:
        assert len(gridmo.enumerate_trajectories(env)) == expected


def test_simple_paths_match_oeis():
    # 184 self-avoiding corner-to-corner paths on a 4x4 grid (OEIS A007764),
    # 20 of them shortest; simple paths have even lengths 6..14 here.
    total = 0
    for T in range(0, 16):
        env = gridmo.example_env(T, allow_stay=False)
        total += len(gridmo.enumerate_trajectories(env, allow_revisit=False))
    assert total == 184
    shortest = gridmo.enumerate_trajectories(gridmo.example_env(6, allow_stay=False),
                                             allow_revisit=False)
    assert len(shortest) == 20


def test_shortest_paths_without_waiting():
    env = gridmo.example_env(6, allow_stay=False)
    assert gridmo.count_trajectories(env) == 20


# recorded front sizes
RECORDED = {  # (T, time weight) -> (front vectors, trajectories on the front)
    (6, "uniform"): (2, 8), (8, "uniform"): (7, 80), (10, "uniform"): (11, 464),
    (12, "uniform"): (15, 1360),
    (6, "linear"): (2, 8), (8, "linear"): (14, 72), (10, "linear"): (29, 240),
    (12, "linear"): (44, 440),
}


@pytest.mark.parametrize("key", list(RECORDED))
def test_recorded_front_sizes(key):
    T, time_weight = key
    env = gridmo.example_env(T)
    for normalize in (False, True):  # dividing by a constant cannot change the front
        objectives = gridmo.example_objectives(time_weight=time_weight, normalize=normalize)
        front = gridmo.solve_front(env, objectives)
        assert (len(front), front.n_trajectories) == RECORDED[key]


def test_recorded_fronts_with_a_person():
    ring = [(2, 3), (3, 3), (3, 2), (2, 2)]
    for T, expected in [(8, (37, 92)), (10, (98, 393))]:
        env = gridmo.example_env(T, movers=[gridmo.Mover("person", ring, "loop")])
        objectives = gridmo.example_objectives(time_weight="uniform", normalize=False,
                                               people=["person"])
        front = gridmo.solve_front(env, objectives)
        assert (len(front), front.n_trajectories) == expected


# exact solver = brute force
def same_front(a, b):
    first = {tuple(v): c for v, c in zip(np.round(a.internal, 9), a.counts)}
    second = {tuple(v): c for v, c in zip(np.round(b.internal, 9), b.counts)}
    return first == second


@pytest.mark.parametrize("T", [6, 8, 10])
@pytest.mark.parametrize("time_weight", ["uniform", "linear", "discount"])
@pytest.mark.parametrize("kernel", ["exp2", "inverse", "linear"])
def test_solver_matches_brute_force(T, time_weight, kernel):
    env = gridmo.example_env(T)
    objectives = gridmo.example_objectives(time_weight=time_weight, kernel=kernel,
                                           kernel_param=3.0)
    assert same_front(gridmo.solve_front(env, objectives),
                      gridmo.front_by_enumeration(env, objectives))


def test_solver_matches_brute_force_with_people_and_walls():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(12):
        env = gridmo.random_env(rng, rows=4, cols=5, slack=3, n_obstacles=2, n_people=1)
        if gridmo.count_trajectories(env) > 300_000:
            continue
        objectives = gridmo.example_objectives(time_weight="linear", people=["person1"])
        solved = gridmo.solve_front(env, objectives)
        assert same_front(solved, gridmo.front_by_enumeration(env, objectives))
        # each representative must be valid and score exactly its front value
        assert gridmo.check_validity(env, solved.reps).all()
        values = gridmo.trajectory_values(env, objectives, solved.reps, internal=True)
        assert np.allclose(values, solved.internal)
        checked += 1
    assert checked >= 8


def test_absorbing_goal_and_time_objective():
    env = gridmo.example_env(10, absorbing_goal=True)
    objectives = gridmo.example_objectives(time_weight="uniform") + [
        gridmo.Objective("time", kind="time", sense="min", time_weight="uniform")]
    assert same_front(gridmo.solve_front(env, objectives),
                      gridmo.front_by_enumeration(env, objectives))


# agents
def test_weighted_sum_only_finds_the_extremes():
    env = gridmo.example_env(10)
    for time_weight in ("uniform", "linear"):
        front = gridmo.solve_front(env, gridmo.example_objectives(time_weight=time_weight))
        found = gridmo.sweep(front, "weighted_sum", steps=1000)
        assert len(found) == 2
        assert set(found) == {int(np.argmax(front.internal[:, 0])),
                              int(np.argmax(front.internal[:, 1]))}


def test_chebyshev_reaches_every_front_point_in_2d():
    env = gridmo.example_env(10)
    front = gridmo.solve_front(env, gridmo.example_objectives(time_weight="uniform"))
    assert len(gridmo.sweep(front, "chebyshev", steps=2000)) == len(front)


def test_greedy_agents_produce_valid_trajectories():
    env = gridmo.example_env(10)
    objectives = gridmo.example_objectives()
    trajectories, weights = gridmo.simulate_greedy(env, objectives, 50, np.random.default_rng(1))
    assert gridmo.check_validity(env, trajectories).all() and weights.shape == (50, 2)


# sampling
def test_uniform_sampler_is_uniform():
    env = gridmo.example_env(6)
    samples = gridmo.sample_uniform(env, 60_000, np.random.default_rng(2))
    _, counts = np.unique(samples, axis=0, return_counts=True)
    assert len(counts) == 20
    expected = 60_000 / 20
    assert np.max(np.abs(counts - expected)) < 5 * np.sqrt(expected)  # within 5 sd


# evaluation
def test_hypervolume_known_values():
    assert gridmo.hypervolume(np.array([[1.0, 1.0]]), np.zeros(2)) == pytest.approx(1.0)
    assert gridmo.hypervolume(np.array([[2.0, 1.0], [1.0, 2.0]]), np.zeros(2)) == \
        pytest.approx(3.0)
    assert gridmo.hypervolume(np.array([[1.0, 1.0, 1.0], [2.0, 0.5, 0.5]]), np.zeros(3)) == \
        pytest.approx(1.0 + 0.25)


def test_assess_scores_the_front_perfectly_and_flags_invalid_moves():
    env = gridmo.example_env(8)
    objectives = gridmo.example_objectives()
    front = gridmo.solve_front(env, objectives)
    result = gridmo.assess(env, objectives, front.reps, front)
    assert result["valid_rate"] == 1 and result["on_front_rate"] == 1 and result["coverage"] == 1
    assert result["hv_ratio"] == pytest.approx(1.0)
    teleporting = front.reps.copy()
    teleporting[:, 1] = env.idx((1, 1))
    assert gridmo.assess(env, objectives, teleporting, front)["valid_rate"] == 0


def test_moves_round_trip_and_json():
    env = gridmo.example_env(8, movers=[gridmo.Mover("p", [(2, 3), (3, 2)], "pingpong")])
    trajectory = gridmo.enumerate_trajectories(env)[123]
    assert np.array_equal(gridmo.from_moves(env, gridmo.to_moves(env, trajectory)), trajectory)
    assert gridmo.GridEnv.from_dict(env.to_dict()).to_dict() == env.to_dict()
    objectives = gridmo.example_objectives(people=["p"])
    record = gridmo.make_record(env, objectives, np.random.default_rng(3), n_random=10)
    assert len(record["front"]) == len(gridmo.solve_front(env, objectives))
    assert {r["label"] for r in record["random"]} <= {"P+", "P-"}
