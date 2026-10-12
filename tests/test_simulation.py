"""
Automated checks for the AGV simulation.  Run from the project folder with:

    python -m pytest -v
"""

import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import simpy

from analysis import aggregate, fleet_sizing, mean_std_ci, min_fleet_meeting
from simulation import (EXPECTED_EVENTS, POLICIES, STATIONS, AGV, Factory, Params,
                        TransportJob, check_invariants, overlap, run_replication,
                        travel_time_min)

P = Params()


# ---------------------------------------------------------------- equations
def test_travel_time_units():
    # 100 m at 0.8 m/s = 125 s = 2.0833 min
    assert travel_time_min(100, 0.8) == pytest.approx(125 / 60)
    assert travel_time_min(300, 0.8) == pytest.approx(375 / 60)
    assert travel_time_min(0, 0.8) == 0.0


def test_travel_time_rejects_bad_inputs():
    with pytest.raises(ValueError):
        travel_time_min(-5, 0.8)
    with pytest.raises(ValueError):
        travel_time_min(10, 0.0)


def test_overlap_with_window():
    assert overlap(50, 70, 60, 540) == 10        # starts in warm-up
    assert overlap(530, 550, 60, 540) == 10      # runs past the end
    assert overlap(10, 20, 60, 540) == 0         # entirely in warm-up
    assert overlap(100, 110, 60, 540) == 10      # entirely inside


# ---------------------------------------------------------------- whole runs
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("n_agvs", [1, 3])
def test_small_run_passes_all_invariants(policy, n_agvs):
    small = replace(P, sim_time_min=150, warmup_min=30)
    row, f = run_replication(small, n_agvs, policy, 0)   # raises if any invariant fails
    assert row["completed_in_window"] > 0
    assert check_invariants(f, row) == []


@pytest.mark.parametrize("policy", POLICIES)
def test_full_run_route_order_and_capacity(policy):
    row, f = run_replication(P, 2, policy, 3)
    # every completed component went RM -> M1 -> M2 -> FG in that order
    for c in f.components.values():
        if c.completed is not None:
            assert [(e, s) for _, e, s in c.history] == EXPECTED_EVENTS
    # machines never overlap
    for log in f.machine_log.values():
        ordered = sorted(log)
        for (s1, e1, _), (s2, _, _) in zip(ordered, ordered[1:]):
            assert s2 >= e1 - 1e-9
    # each AGV handles one job at a time
    for agv in f.agvs:
        jobs = sorted((j for j in f.jobs if j.agv_id == agv.agv_id), key=lambda j: j.assigned)
        for a, b in zip(jobs, jobs[1:]):
            assert b.assigned >= a.t_done - 1e-9
    assert 0.0 <= row["agv_utilisation"] <= 1.0


def test_completion_requires_delivery_to_fg():
    row, f = run_replication(P, 1, "FIFO", 0)
    lo, hi = P.warmup_min, P.sim_time_min
    delivered = sum(1 for c in f.components.values()
                    if c.completed is not None and lo <= c.completed < hi)
    finished_m2_in_window = sum(1 for c in f.components.values()
                                for t, e, s in c.history
                                if e == "finish" and s == "M2" and lo <= t < hi)
    assert row["completed_in_window"] == delivered
    # with 1 AGV many parts finish M2 but wait for transport - they must NOT count
    assert finished_m2_in_window > delivered


def test_reproducible_with_same_seed():
    a, _ = run_replication(P, 2, "FIFO", 5)
    b, _ = run_replication(P, 2, "FIFO", 5)
    for k in a:
        if isinstance(a[k], float) and math.isnan(a[k]):
            assert math.isnan(b[k])
        else:
            assert a[k] == b[k]


def test_common_random_numbers_across_cases():
    arrivals = {run_replication(P, n, pol, 7)[0]["components_arrived"]
                for pol in POLICIES for n in (1, 5)}
    assert len(arrivals) == 1


def test_checker_detects_a_planted_violation():
    _, f = run_replication(P, 2, "FIFO", 0)
    s, e, cid = f.machine_log["M2"][0]
    f.machine_log["M2"].append((s + 1.0, e + 1.0, cid))   # fake overlapping operation
    assert any("M2 processed two components" in p for p in check_invariants(f))


# ---------------------------------------------------------------- dispatch rules
def _factory_with(policy, agvs, jobs):
    env = simpy.Environment()
    f = Factory(env, P, n_agvs=len(agvs), policy=policy, rng=np.random.default_rng(0))
    f.agvs = agvs
    f.idle_agvs = list(agvs)
    f.waiting_jobs = jobs
    return f


def _job(job_id, pickup, created):
    return TransportJob(job_id=job_id, component_id=job_id, leg=1, pickup=pickup,
                        dest="M1", created=created, done=None)


def test_fifo_takes_oldest_job_and_longest_idle_agv():
    agvs = [AGV(0, STATIONS["FG"], idle_since=5.0), AGV(1, STATIONS["M2"], idle_since=2.0)]
    jobs = [_job(0, "M1", created=3.0), _job(1, "RM", created=1.0)]
    agv, job = _factory_with("FIFO", agvs, jobs)._select_fifo()
    assert job.job_id == 1 and agv.agv_id == 1


def test_nearest_picks_shortest_empty_distance():
    agvs = [AGV(0, STATIONS["RM"], 0.0), AGV(1, STATIONS["M2"], 0.0)]
    jobs = [_job(0, "RM", created=9.0), _job(1, "FG", created=1.0)]
    agv, job = _factory_with("NEAREST", agvs, jobs)._select_nearest()
    assert (agv.agv_id, job.job_id) == (0, 0)          # distance 0 beats distance 100


def test_nearest_tie_breaks_by_creation_time_then_agv_id():
    # Both AGVs at M1; both jobs 100 m away -> older job first, then lower AGV id.
    agvs = [AGV(1, STATIONS["M1"], 0.0), AGV(0, STATIONS["M1"], 0.0)]
    jobs = [_job(0, "M2", created=4.0), _job(1, "RM", created=2.0)]
    agv, job = _factory_with("NEAREST", agvs, jobs)._select_nearest()
    assert job.job_id == 1 and agv.agv_id == 0


# ---------------------------------------------------------------- statistics
def test_confidence_interval_formula():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = mean_std_ci(x)
    half = 2.776445 * np.std(x, ddof=1) / math.sqrt(5)    # t_{0.975,4}
    assert s["mean"] == pytest.approx(3.0)
    assert s["ci_high"] - s["mean"] == pytest.approx(half, rel=1e-5)


def test_aggregate_uses_data_not_constants():
    raw = pd.DataFrame({"policy": ["FIFO"] * 3, "n_agvs": [1] * 3})
    from analysis import METRICS
    for m in METRICS:
        raw[m] = [1.0, 2.0, 6.0]
    agg = aggregate(raw)
    assert agg.loc[0, "throughput_per_h_mean"] == pytest.approx(3.0)


def _agg(means):
    return pd.DataFrame({"policy": ["FIFO"] * len(means),
                         "n_agvs": list(range(1, len(means) + 1)),
                         "throughput_per_h_mean": means})


def test_fleet_sizing_normal_case():
    s = fleet_sizing(_agg([1.0, 4.0, 6.5, 6.6, 6.6])).iloc[0]
    assert s["status"] == "reached" and s["min_fleet_size"] == 3
    assert "plateau is visible" in s["note"]


def test_fleet_sizing_flags_missing_plateau():
    s = fleet_sizing(_agg([1.0, 2.0, 3.0, 4.0, 5.0])).iloc[0]
    assert s["min_fleet_size"] == 5
    assert "insufficient to establish a plateau" in s["note"]


def test_fleet_sizing_handles_no_throughput():
    s = fleet_sizing(_agg([0.0, 0.0, 0.0])).iloc[0]
    assert s["status"] == "not determined"


def test_threshold_not_reached_returns_none():
    assert min_fleet_meeting({1: 2.0, 2: 3.0}, target=6.0) is None
    s = fleet_sizing(_agg([1.0, 2.0, 3.0]), bottleneck_capacity=6.67).iloc[0]
    assert s["min_fleet_for_95pct_of_bottleneck"] is None or \
        pd.isna(s["min_fleet_for_95pct_of_bottleneck"])
    assert "No tested fleet reached" in s["note"]


def test_paired_difference_verdicts():
    from analysis import paired_differences
    rows = []
    for r in range(5):
        rows.append({"policy": "FIFO", "n_agvs": 1, "replication": r,
                     "throughput_per_h": 5.0 + r, "mean_transport_wait_min": 2.0,
                     "empty_travel_km": 10.0, "agv_utilisation": 0.5})
        rows.append({"policy": "NEAREST", "n_agvs": 1, "replication": r,
                     "throughput_per_h": 6.0 + r, "mean_transport_wait_min": 2.0,
                     "empty_travel_km": 9.0 + (0.1 if r % 2 else -0.1), "agv_utilisation": 0.5})
    out = paired_differences(pd.DataFrame(rows)).set_index("metric")
    assert out.loc["throughput_per_h", "mean_diff_nearest_minus_fifo"] == pytest.approx(1.0)
    assert out.loc["throughput_per_h", "verdict_95pct"] == "NEAREST higher"
    assert out.loc["empty_travel_km", "verdict_95pct"] == "NEAREST lower"
    assert out.loc["agv_utilisation", "verdict_95pct"] == "no significant difference"
