"""
main.py - Factory material-handling and AGV fleet-sizing simulation
====================================================================

Run everything with ONE command (from this folder):

    python main.py

What it does, in order:
  1. Prints simple hand-calculated capacity checks.
  2. Runs built-in validation checks (small simulations, reproducibility, ...).
  3. Runs the experiment: 2 dispatch policies x 5 fleet sizes x 20 replications.
  4. Writes per-replication results, aggregated statistics (mean, std, 95% CI)
     and the fleet-sizing decision to CSV files in results/.
  5. Re-reads those CSV files and draws the figures from them (PNG).
  6. Writes a plain-text summary and a validation report.

Optional:  python main.py --replications 5     (faster trial run)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd

from analysis import aggregate, fleet_sizing, make_figures, paired_differences
from simulation import (POLICIES, Params, SimulationError, analytic_capacity,
                        run_replication, travel_time_min)


# ---------------------------------------------------------------------------
# Validation checks that run before the main experiment
# ---------------------------------------------------------------------------

def pre_experiment_checks(params: Params) -> list[tuple[str, bool, str]]:
    """Each entry is (check name, passed?, detail)."""
    results = []

    # 1. Travel-time equation: dimensionally consistent and non-negative.
    t100 = travel_time_min(100.0, params.agv_speed_mps)
    ok = abs(t100 - 100.0 / 48.0) < 1e-12 and travel_time_min(0.0, 0.8) == 0.0
    try:
        travel_time_min(-1.0, 0.8)
        ok = False  # a negative distance must be rejected
    except ValueError:
        pass
    results.append(("Travel time: 100 m at 0.8 m/s = 2.083 min; 0 m -> 0 min; "
                    "negative distance rejected", ok, f"t(100 m) = {t100:.4f} min"))

    # 2. Both policies complete a small simulation (invariants checked inside).
    small = replace(params, sim_time_min=120.0, warmup_min=20.0)
    for policy in POLICIES:
        try:
            row, f = run_replication(small, n_agvs=2, policy=policy, rep_index=0)
            done = sum(1 for c in f.components.values() if c.completed is not None)
            results.append((f"Small run (120 min, 2 AGVs) completes with policy {policy}",
                            done > 0, f"{done} components completed, all invariants passed"))
        except SimulationError as e:
            results.append((f"Small run with policy {policy}", False, str(e)))

    # 3. Reproducibility: same seed -> identical results.
    a, _ = run_replication(params, 3, "NEAREST", 0)
    b, _ = run_replication(params, 3, "NEAREST", 0)
    same = all((a[k] == b[k]) or (isinstance(a[k], float) and math.isnan(a[k])
                                  and math.isnan(b[k])) for k in a)
    results.append(("Same seed gives identical results", same, f"seed {a['seed']}"))

    # 4. Different seeds give different arrival streams.
    c, _ = run_replication(params, 3, "NEAREST", 1)
    results.append(("Different seeds give different arrival streams",
                    a["components_arrived"] != c["components_arrived"]
                    or a["throughput_per_h"] != c["throughput_per_h"],
                    f"arrivals {a['components_arrived']} vs {c['components_arrived']}"))
    return results


def post_experiment_checks(raw: pd.DataFrame, params: Params) -> list[tuple[str, bool, str]]:
    results = []
    expected_rows = len(POLICIES) * len(params.fleet_sizes) * params.n_replications
    results.append(("Every replication finished with all invariants satisfied "
                    "(route order, one load per AGV, machine capacity 1, "
                    "non-negative travel times)", len(raw) == expected_rows,
                    f"{len(raw)} of {expected_rows} replications"))

    u = raw["agv_utilisation"]
    results.append(("AGV utilisation within [0, 100] %", bool(((u >= 0) & (u <= 1 + 1e-9)).all()),
                    f"min {u.min():.3f}, max {u.max():.3f}"))

    max_th = (math.floor(params.measure_min / params.m2_process_min) + 1) / (params.measure_min / 60)
    th = raw["throughput_per_h"]
    results.append(("Throughput never exceeds what Machine 2 can physically finish in the window",
                    bool((th <= max_th + 1e-9).all()),
                    f"max observed {th.max():.3f} /h, physical limit {max_th:.3f} /h"))

    # Common random numbers: with the same replication index, every policy and
    # fleet size must see exactly the same number of arrivals.
    crn = raw.groupby("replication")["components_arrived"].nunique()
    results.append(("Common random numbers: same arrivals for every policy/fleet size "
                    "within a replication", bool((crn == 1).all()),
                    f"{len(crn)} replications checked"))

    nonneg = all((raw[c] >= 0).all() for c in ["empty_travel_km", "loaded_travel_km",
                                                 "mean_transport_wait_min"])
    results.append(("Distances and waiting times are non-negative", nonneg, ""))
    return results


def write_report(path: Path, sections: dict) -> bool:
    all_ok = True
    lines = ["VALIDATION REPORT", "=================", ""]
    for title, checks in sections.items():
        lines.append(title)
        for name, ok, detail in checks:
            all_ok &= ok
            lines.append(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
        lines.append("")
    lines.append("OVERALL: " + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return all_ok


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(params: Params) -> pd.DataFrame:
    rows = []
    for policy in POLICIES:
        for n in params.fleet_sizes:
            for r in range(params.n_replications):
                row, _ = run_replication(params, n, policy, r)
                rows.append(row)
            print(f"  finished {policy:<8} with {n} AGV(s): {params.n_replications} replications")
    return pd.DataFrame(rows)


def summary_text(agg: pd.DataFrame, sizing: pd.DataFrame, cap: dict, params: Params,
                 paired: pd.DataFrame) -> str:
    L = []
    L.append("RESULTS SUMMARY (generated by main.py - do not edit by hand)")
    L.append("=" * 60)
    L.append(f"Replications per case: {params.n_replications}; seeds "
             f"{params.base_seed}..{params.base_seed + params.n_replications - 1}; "
             f"measurement window {params.warmup_min:g}-{params.sim_time_min:g} min")
    L.append(f"Arrival rate {cap['arrival_rate_per_h']:.2f}/h, M1 capacity "
             f"{cap['m1_capacity_per_h']:.2f}/h, M2 capacity {cap['m2_capacity_per_h']:.2f}/h "
             f"-> bottleneck {cap['bottleneck']} (load ratio {cap['m2_load_ratio']:.2f})")
    L.append("")
    hdr = (f"{'policy':<8} {'N':>2} | {'throughput/h':>20} | {'wait min':>24} | "
           f"{'empty km':>22} | {'AGV util %':>22} | {'unassigned':>10}")
    L.append(hdr)
    L.append("-" * len(hdr))
    for _, r in agg.sort_values(["policy", "n_agvs"]).iterrows():
        def f(m, scale=1.0, d=2):
            return (f"{r[m + '_mean'] * scale:.{d}f} "
                    f"[{r[m + '_ci_low'] * scale:.{d}f},{r[m + '_ci_high'] * scale:.{d}f}]")
        L.append(f"{r['policy']:<8} {int(r['n_agvs']):>2} | {f('throughput_per_h'):>20} | "
                 f"{f('mean_transport_wait_min'):>24} | {f('empty_travel_km'):>22} | "
                 f"{f('agv_utilisation', 100, 1):>22} | {r['jobs_unassigned_at_end_mean']:>10.1f}")
    L.append("(values are mean [95% CI])")
    L.append("")
    L.append("FLEET SIZING (minimum fleet with mean throughput >= 95% of best tested mean)")
    for _, s in sizing.iterrows():
        if s["status"] == "reached":
            L.append(f"  {s['policy']}: {int(s['min_fleet_size'])} AGV(s) "
                     f"(best mean {s['max_mean_throughput_per_h']:.3f}/h at "
                     f"{int(s['fleet_size_at_max'])} AGVs; threshold {s['threshold_per_h']:.3f}/h)")
        else:
            L.append(f"  {s['policy']}: {s['status']}")
        L.append(f"     {s['note']}")
    L.append("")
    L.append("PAIRED COMPARISON: NEAREST minus FIFO (same seeds), mean difference [95% CI]")
    for _, r in paired.iterrows():
        L.append(f"  N={int(r['n_agvs'])} {r['metric']:<26} {r['mean_diff_nearest_minus_fifo']:+9.3f} "
                 f"[{r['diff_ci_low']:+.3f}, {r['diff_ci_high']:+.3f}]  -> {r['verdict_95pct']}")
    return "\n".join(L) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="AGV fleet-sizing simulation")
    parser.add_argument("--replications", type=int, default=20,
                        help="replications per (policy, fleet size); default 20")
    parser.add_argument("--results-dir", default="results", help="output folder")
    args = parser.parse_args()

    params = Params(n_replications=args.replications)
    out = Path(args.results_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Hand calculations
    cap = analytic_capacity(params)
    pd.DataFrame([cap]).to_csv(out / "analytic_checks.csv", index=False)
    print("Hand-calculation checks:", {k: (round(v, 3) if isinstance(v, float) else v)
                                       for k, v in cap.items()})

    # 2. Pre-experiment validation
    print("\nRunning validation checks ...")
    pre = pre_experiment_checks(params)
    if not all(ok for _, ok, _ in pre):
        write_report(out / "validation_report.txt", {"Pre-experiment checks": pre})
        print("Validation failed - experiment not run.")
        return 1

    # 3. Experiment
    print("\nRunning experiment ...")
    raw = run_experiment(params)
    raw_path = out / "raw_replications.csv"
    raw.to_csv(raw_path, index=False)

    # 4. Statistics - computed from the CSV that was just written
    raw_from_disk = pd.read_csv(raw_path)
    agg = aggregate(raw_from_disk)
    agg_path = out / "aggregated_results.csv"
    agg.to_csv(agg_path, index=False)

    agg_from_disk = pd.read_csv(agg_path)
    sizing = fleet_sizing(agg_from_disk, fraction=0.95, plateau_tol=0.02,
                          bottleneck_capacity=cap["bottleneck_capacity_per_h"])
    sizing.to_csv(out / "fleet_sizing.csv", index=False)

    paired = paired_differences(raw_from_disk)
    paired.to_csv(out / "paired_comparison.csv", index=False)

    # 5. Figures - drawn from the CSV files, never from hard-coded numbers
    files = make_figures(agg_from_disk, sizing, params, out)

    # 6. Reports
    post = post_experiment_checks(raw_from_disk, params)
    ok = write_report(out / "validation_report.txt",
                      {"Pre-experiment checks": pre, "Post-experiment checks": post})
    text = summary_text(agg_from_disk, sizing, cap, params, paired)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    print("\n" + text)

    print("Files written:")
    for p in sorted(out.iterdir()):
        print("  ", p)
    print(f"\nDone in {time.time() - t0:.1f} s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
