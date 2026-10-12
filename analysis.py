"""
analysis.py
===========
Statistics and figures. Everything here works on the CSV files that the
experiment wrote, so every plotted number can be traced back to program output.

* ``aggregate``      : mean, sample standard deviation and 95 % t-confidence
                       interval of every metric for each (policy, fleet size).
* ``fleet_sizing``   : the minimum fleet reaching 95 % of the best mean
                       throughput among the tested fleet sizes.
* ``make_figures``   : the PNG plots.
"""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# Metrics that are aggregated (mean, std, CI) for every case.
METRICS = [
    "throughput_per_h",
    "completed_in_window",
    "mean_transport_wait_min",
    "mean_transport_wait_lower_bound_min",
    "wait_leg1_RM_M1_min",
    "wait_leg2_M1_M2_min",
    "wait_leg3_M2_FG_min",
    "jobs_created_in_window",
    "jobs_unassigned_at_end",
    "empty_travel_km",
    "loaded_travel_km",
    "empty_to_loaded_ratio",
    "agv_utilisation",
    "m1_utilisation",
    "m2_utilisation",
    "components_arrived",
    "wip_at_end",
    "m1_queue_at_end",
    "m2_queue_at_end",
]

# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def mean_std_ci(values, confidence: float = 0.95) -> dict:
    """Mean, sample std (n-1) and a t-based confidence interval for the mean.

        half-width = t_{1-alpha/2, n-1} * s / sqrt(n)

    With n = 20 replications, t_{0.975, 19} = 2.093.
    """
    x = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))],
                   dtype=float)
    n = len(x)
    if n == 0:
        return {"mean": math.nan, "std": math.nan, "n": 0,
                "ci_low": math.nan, "ci_high": math.nan, "ci_half": math.nan}
    mean = float(x.mean())
    if n == 1:
        return {"mean": mean, "std": math.nan, "n": 1,
                "ci_low": math.nan, "ci_high": math.nan, "ci_half": math.nan}
    std = float(x.std(ddof=1))
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, df=n - 1)
    half = t_crit * std / math.sqrt(n)
    return {"mean": mean, "std": std, "n": n,
            "ci_low": mean - half, "ci_high": mean + half, "ci_half": half}


def aggregate(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per (policy, n_agvs); columns <metric>_mean/_std/_ci_low/_ci_high."""
    rows = []
    for (policy, n_agvs), grp in raw.groupby(["policy", "n_agvs"], sort=True):
        row = {"policy": policy, "n_agvs": int(n_agvs), "replications": len(grp)}
        for m in METRICS:
            s = mean_std_ci(grp[m].tolist())
            row[f"{m}_mean"] = s["mean"]
            row[f"{m}_std"] = s["std"]
            row[f"{m}_ci_low"] = s["ci_low"]
            row[f"{m}_ci_high"] = s["ci_high"]
        rows.append(row)
    return pd.DataFrame(rows)


COMPARE_METRICS = ["throughput_per_h", "mean_transport_wait_min",
                   "empty_travel_km", "agv_utilisation"]


def paired_differences(raw: pd.DataFrame, metrics=COMPARE_METRICS) -> pd.DataFrame:
    """NEAREST minus FIFO, paired by replication (same seed = same arrivals).

    Because both policies see identical arrivals in replication r, we can
    analyse the 20 differences d_r = X_r(NEAREST) - X_r(FIFO) directly.
    Pairing removes the arrival-to-arrival noise that both policies share,
    so the confidence interval of the difference is usually much narrower.
    If the CI excludes 0 the difference is statistically significant at 95 %.
    """
    rows = []
    for n_agvs, grp in raw.groupby("n_agvs", sort=True):
        fifo = grp[grp["policy"] == "FIFO"].set_index("replication")
        near = grp[grp["policy"] == "NEAREST"].set_index("replication")
        common = fifo.index.intersection(near.index)
        for m in metrics:
            d = (near.loc[common, m] - fifo.loc[common, m]).tolist()
            s = mean_std_ci(d)
            if s["n"] < 2 or math.isnan(s["ci_low"]):
                verdict = "not enough data"
            elif s["ci_low"] > 0:
                verdict = "NEAREST higher"
            elif s["ci_high"] < 0:
                verdict = "NEAREST lower"
            else:
                verdict = "no significant difference"
            rows.append({"n_agvs": int(n_agvs), "metric": m,
                         "fifo_mean": float(fifo.loc[common, m].mean()),
                         "nearest_mean": float(near.loc[common, m].mean()),
                         "mean_diff_nearest_minus_fifo": s["mean"],
                         "diff_ci_low": s["ci_low"], "diff_ci_high": s["ci_high"],
                         "pairs": s["n"], "verdict_95pct": verdict})
    return pd.DataFrame(rows)


def min_fleet_meeting(means_by_n: dict, target: float):
    """Smallest fleet size whose mean is >= target, or None if none is."""
    for n in sorted(means_by_n):
        m = means_by_n[n]
        if m is not None and not math.isnan(m) and m >= target:
            return n
    return None


def fleet_sizing(agg: pd.DataFrame, fraction: float = 0.95, plateau_tol: float = 0.02,
                 bottleneck_capacity: float | None = None) -> pd.DataFrame:
    """Apply the fleet-sizing rule separately for each policy.

    Rule:  N* = min { N : mean_TH(N) >= fraction * max_N mean_TH(N) }

    This is relative to the TESTED fleet sizes only. If the largest tested
    fleet still increases throughput by more than ``plateau_tol`` (relative)
    over the next-smaller fleet, a plateau has not been demonstrated.
    """
    out = []
    for policy, sub in agg.groupby("policy", sort=True):
        sub = sub.sort_values("n_agvs")
        means = dict(zip(sub["n_agvs"].astype(int), sub["throughput_per_h_mean"]))
        valid = {n: m for n, m in means.items() if not math.isnan(m)}
        row = {"policy": policy, "tested_fleet_sizes": " ".join(str(n) for n in sorted(means))}

        if not valid or max(valid.values()) <= 0:
            row.update(status="not determined", min_fleet_size=None,
                       note="No positive throughput was observed, so no threshold can be set.")
            out.append(row)
            continue

        best_n = max(valid, key=lambda n: (valid[n], -n))  # ties -> smaller fleet
        best = valid[best_n]
        target = fraction * best
        n_star = min_fleet_meeting(valid, target)

        row.update(
            max_mean_throughput_per_h=best,
            fleet_size_at_max=best_n,
            threshold_fraction=fraction,
            threshold_per_h=target,
            min_fleet_size=n_star,
            throughput_at_min_fleet=valid.get(n_star, math.nan) if n_star else math.nan,
            status="reached" if n_star is not None else "not reached",
        )

        notes = []
        sizes = sorted(valid)
        if len(sizes) >= 2:
            last, prev = sizes[-1], sizes[-2]
            gain = (valid[last] - valid[prev]) / valid[prev] if valid[prev] > 0 else math.inf
            row["gain_between_two_largest_fleets"] = gain
            if gain > plateau_tol:
                notes.append(f"Throughput still rises by {gain:.1%} from {prev} to {last} AGVs: "
                             "the tested range may be insufficient to establish a plateau.")
            else:
                notes.append(f"Throughput changes by {gain:+.1%} from {prev} to {last} AGVs: "
                             "a plateau is visible within the tested range.")
        notes.append("Threshold is relative to the best TESTED fleet size, not the true "
                     "maximum capacity of a real factory.")

        # Secondary, absolute check against the analytical bottleneck capacity.
        if bottleneck_capacity is not None:
            abs_target = fraction * bottleneck_capacity
            n_abs = min_fleet_meeting(valid, abs_target)
            row["bottleneck_capacity_per_h"] = bottleneck_capacity
            row["best_as_fraction_of_bottleneck"] = best / bottleneck_capacity
            row["min_fleet_for_95pct_of_bottleneck"] = n_abs
            if n_abs is None:
                notes.append("No tested fleet reached 95% of the analytical bottleneck capacity.")
        row["note"] = " ".join(notes)
        out.append(row)
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

# Colours: two validated categorical slots (blue, orange) + recessive greys.
STYLE = {
    "FIFO": {"color": "#2a78d6", "marker": "o", "linestyle": "-",
             "label": "FIFO dispatch"},
    "NEAREST": {"color": "#eb6834", "marker": "s", "linestyle": "--",
                "label": "Nearest-pickup dispatch"},
}
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"


def _setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK_2,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
    })


def _errorbars(sub, metric, positive_only=False):
    mean = sub[f"{metric}_mean"].to_numpy()
    lo = sub[f"{metric}_ci_low"].to_numpy()
    hi = sub[f"{metric}_ci_high"].to_numpy()
    if positive_only:  # needed on a log axis: never draw below a tiny positive value
        lo = np.maximum(lo, mean * 0.05)
    return mean, np.vstack([mean - lo, hi - mean])


def _line_plot(ax, agg, metric, positive_only=False, offset=0.06):
    """Mean +/- 95% CI for both policies, slightly offset in x so bars don't overlap."""
    for i, policy in enumerate(["FIFO", "NEAREST"]):
        sub = agg[agg["policy"] == policy].sort_values("n_agvs")
        if sub.empty:
            continue
        st = STYLE[policy]
        x = sub["n_agvs"].to_numpy() + (i - 0.5) * 2 * offset
        mean, err = _errorbars(sub, metric, positive_only)
        ax.errorbar(x, mean, yerr=err, color=st["color"], marker=st["marker"],
                    linestyle=st["linestyle"], capsize=4, elinewidth=1.4,
                    label=st["label"] + " (mean, 95% CI)",
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_xticks(sorted(agg["n_agvs"].unique()))
    ax.set_xlabel("AGV fleet size (vehicles)")


def make_figures(agg: pd.DataFrame, sizing: pd.DataFrame, params, out_dir) -> list[str]:
    """Create the PNG figures from the aggregated results. Returns file paths."""
    _setup_matplotlib()
    files = []
    caption = (f"{params.n_replications} replications per point; error bars = 95% t-CI of the mean; "
               f"measurement window {params.warmup_min:.0f}-{params.sim_time_min:.0f} min")
    cap = 60.0 / params.m2_process_min

    # ---- Figure 1: throughput ------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _line_plot(ax, agg, "throughput_per_h")
    ax.axhline(cap, color=INK_2, linestyle=":", linewidth=1.3)
    ax.text(0.02, cap + 0.08, f"Machine 2 capacity = 60/{params.m2_process_min:g} = {cap:.2f} /h",
            transform=ax.get_yaxis_transform(), ha="left", va="bottom", color=INK_2, fontsize=9)
    for _, r in sizing.iterrows():
        if r.get("status") == "reached":
            st = STYLE[r["policy"]]
            ax.axhline(r["threshold_per_h"], color=st["color"], linewidth=0.9, alpha=0.5,
                       linestyle=st["linestyle"])
    ax.set_ylabel("Throughput (components per hour)")
    ax.set_ylim(0, cap * 1.18)
    ax.set_title("Production throughput vs AGV fleet size")
    ax.legend(loc="lower right")
    fig.text(0.01, 0.01, caption + "; thin coloured lines = 95%-of-best thresholds",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    files.append(_save(fig, out_dir, "fig1_throughput_vs_fleet.png"))

    # ---- Figure 2: transport waiting time (log scale) --------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _line_plot(ax, agg, "mean_transport_wait_min", positive_only=True)
    for policy in ["FIFO", "NEAREST"]:
        sub = agg[agg["policy"] == policy].sort_values("n_agvs")
        st = STYLE[policy]
        ax.plot(sub["n_agvs"], sub["mean_transport_wait_lower_bound_min_mean"],
                color=st["color"], linestyle=":", linewidth=1.3, marker="x", markersize=6,
                label=st["label"].split(" ")[0] + " incl. still-waiting jobs (lower bound)")
    ax.set_yscale("log")
    ax.set_ylabel("Mean transport waiting time (min, log scale)")
    ax.set_title("Transport waiting time (request → AGV assigned)")
    ax.legend(loc="upper right")
    fig.text(0.01, 0.01, caption + "; jobs created in window", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    files.append(_save(fig, out_dir, "fig2_waiting_time_vs_fleet.png"))

    # ---- Figure 3: empty travel ----------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    _line_plot(ax, agg, "empty_travel_km")
    ax.set_ylabel("Empty travel distance in window (km)")
    ax.set_ylim(bottom=0)
    ax.set_title("Empty (unloaded) AGV travel vs fleet size")
    ax.legend(loc="lower right")
    fig.text(0.01, 0.01, caption + "; total for whole fleet", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    files.append(_save(fig, out_dir, "fig3_empty_travel_vs_fleet.png"))

    # ---- Figure 4: policy comparison (small multiples) -----------------
    panels = [
        ("throughput_per_h", "Throughput (components/h)", False, 1.0),
        ("mean_transport_wait_min", "Transport wait (min, log)", True, 1.0),
        ("empty_travel_km", "Empty travel (km)", False, 1.0),
        ("agv_utilisation", "AGV utilisation (%)", False, 100.0),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    sizes = sorted(agg["n_agvs"].unique())
    width = 0.38
    for ax, (metric, label, log, scale) in zip(axes.flat, panels):
        for i, policy in enumerate(["FIFO", "NEAREST"]):
            sub = agg[agg["policy"] == policy].set_index("n_agvs").reindex(sizes)
            mean = sub[f"{metric}_mean"].to_numpy() * scale
            lo = sub[f"{metric}_ci_low"].to_numpy() * scale
            hi = sub[f"{metric}_ci_high"].to_numpy() * scale
            if log:
                lo = np.maximum(lo, mean * 0.05)
            x = np.arange(len(sizes)) + (i - 0.5) * (width + 0.02)
            st = STYLE[policy]
            ax.bar(x, mean, width=width, color=st["color"], label=st["label"],
                   edgecolor=SURFACE, linewidth=1.0,
                   hatch=None if policy == "FIFO" else "//")
            ax.errorbar(x, mean, yerr=np.vstack([mean - lo, hi - mean]), fmt="none",
                        ecolor=INK, elinewidth=1.0, capsize=3)
        ax.set_xticks(np.arange(len(sizes)))
        ax.set_xticklabels([str(s) for s in sizes])
        ax.set_xlabel("AGV fleet size")
        ax.set_title(label, fontsize=11)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        if log:
            ax.set_yscale("log")
        if metric == "agv_utilisation":
            ax.set_ylim(0, 110)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.suptitle("FIFO vs nearest-pickup dispatch across fleet sizes", fontsize=13,
                 fontweight="bold", color=INK, y=0.995)
    fig.text(0.01, 0.01, caption + "; hatched bars = nearest-pickup", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    files.append(_save(fig, out_dir, "fig4_policy_comparison.png"))

    return files


def _save(fig, out_dir, name) -> str:
    path = f"{out_dir}/{name}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
