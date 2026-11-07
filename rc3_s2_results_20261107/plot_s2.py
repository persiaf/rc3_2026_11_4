"""Plot only the CSV output of analyze_s2.py; no live log or tensor access."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from paper_plot_style import plt, COLORS, save

SEEDS = (11622026, 11622027, 11622028)


def read_rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def number(row, key):
    value = row.get(key, "")
    return float(value) if value not in ("", "None", None) else float("nan")


def smooth(values, window=16):
    # Explicit trailing window, unweighted over the displayed update means.
    return [sum(values[max(0, i-window+1):i+1]) / len(values[max(0, i-window+1):i+1]) for i in range(len(values))]


def selected(rows, seed, phase="RL"):
    return sorted([r for r in rows if int(r["seed"]) == seed and r["phase"] == phase], key=lambda r: int(r["update"]))


def absent(ax, phase):
    ax.text(.5, .5, f"{phase}: no completed updates", transform=ax.transAxes, ha="center", va="center", fontsize=9)


def curve(ax, rows, key, label, color=COLORS[0], linestyle="-", show_raw=False):
    x, y = [int(r["cursor_after"]) for r in rows], [number(r, key) for r in rows]
    if show_raw:
        ax.plot(x, y, color=color, alpha=.18, linewidth=.7)
    ax.plot(x, smooth(y), color=color, linewidth=1.3, linestyle=linestyle, label=label)


def axes_grid(nrows=1, ncols=3, height=3.0):
    return plt.subplots(nrows, ncols, figsize=(10.2, height * nrows), squeeze=False, constrained_layout=True)


def label_axis(ax, seed, phase="RL"):
    ax.set_xlabel(f"Completed {phase} update (seed {seed})")
    ax.grid(alpha=.12, linewidth=.5)


def fig01(updates, groups, folder):
    fig, axes = axes_grid()
    for ax, seed in zip(axes[0], SEEDS):
        rows = selected(updates, seed)
        if rows:
            curve(ax, rows, "return__mean", "16-update trailing mean", show_raw=True)
        else:
            absent(ax, "RL")
        label_axis(ax, seed)
        ax.set_ylabel("Complete return, $-J_H$ (m equivalent)")
        if rows:
            ax.legend(frameon=False)
    return save(fig, folder, "fig01_rl_return")


def fig02(updates, groups, folder):
    fig, axes = axes_grid()
    for ax, seed in zip(axes[0], SEEDS):
        rows = selected(updates, seed)
        if rows:
            curve(ax, rows, "loss", "REINFORCE objective (trailing16)", show_raw=True)
            ax.axhline(0, color=".5", linewidth=.5)
            ax.legend(frameon=False)
        else:
            absent(ax, "RL")
        label_axis(ax, seed)
        ax.set_ylabel("Signed mixed REINFORCE objective")
    return save(fig, folder, "fig02_rl_objective")


def fig03(updates, groups, folder):
    fig, axes = axes_grid(3, height=2.3)
    for j, seed in enumerate(SEEDS):
        rows = selected(updates, seed)
        for i, metric in enumerate(("J", "L", "K")):
            ax = axes[i][j]
            if rows:
                curve(ax, rows, f"raw_{metric}_mean", "Before H", COLORS[1], "--")
                curve(ax, rows, f"H_{metric}_mean", "After H", COLORS[0])
                if i == 0:
                    ax.legend(frameon=False)
            else:
                absent(ax, "RL")
            label_axis(ax, seed)
            ax.set_ylabel({"J": "Objective $J$ (m equivalent)", "L": "Route length $L$ (m)", "K": "Service stops $K$"}[metric])
    return save(fig, folder, "fig03_raw_and_H_costs")


def fig04(updates, groups, folder):
    fig, axes = axes_grid(2, height=2.5)
    for j, seed in enumerate(SEEDS):
        for i, phase in enumerate(("RL", "continued_SL")):
            ax, rows = axes[i][j], selected(updates, seed, phase)
            if rows:
                x = [int(r["cursor_after"]) for r in rows]
                for key, label, color, ls in (("gradient_norm_before_clip", "Before clipping", COLORS[1], "-"),
                                               ("gradient_norm_after_clip", "After clipping", COLORS[0], "--")):
                    ax.plot(x, [max(number(r, key), 1e-14) for r in rows], label=label, color=color, linewidth=.8, linestyle=ls)
                ax.set_yscale("log")
                ax.axhline(1., color=".5", linewidth=.6)
                if i == 0:
                    ax.legend(frameon=False)
            else:
                absent(ax, phase)
            label_axis(ax, seed, phase)
            ax.set_ylabel("Global gradient norm (log scale)")
    return save(fig, folder, "fig04_gradient_clipping")


def fig05(updates, groups, folder):
    fig, axes = axes_grid(3, height=2.35)
    for j, seed in enumerate(SEEDS):
        rows = selected(updates, seed)
        for ax in axes[:, j]:
            label_axis(ax, seed)
        if not rows:
            for ax in axes[:, j]:
                absent(ax, "RL")
            continue
        x = [int(r["cursor_after"]) for r in rows]
        ax = axes[0][j]
        ax.fill_between(x, [number(r, "advantage_p05") for r in rows], [number(r, "advantage_p95") for r in rows],
                        color=COLORS[0], alpha=.2, linewidth=0, label="Within-update p05-p95")
        ax.plot(x, [number(r, "advantage_median") for r in rows], color=COLORS[0], linewidth=.6, label="Median")
        ax.axhline(0, color=".5", linewidth=.6)
        ax.set_ylabel("Detached leave-one-out advantage")
        ax.legend(frameon=False)
        ax = axes[1][j]
        ax.plot(x, [number(r, "advantage_exact_zero_n") / number(r, "advantage_n") for r in rows], color=COLORS[1], linewidth=.8)
        ax.set_ylim(-.03, 1.03)
        ax.set_ylabel("Exact-zero advantage proportion")
        ax = axes[2][j]
        curve(ax, rows, "decision_entropy_mean", "Decision entropy (trailing16)", COLORS[2], show_raw=True)
        ax.set_ylabel("Mean sampled-decision entropy (nats)")
        ax.legend(frameon=False)
    return save(fig, folder, "fig05_advantage_and_entropy")


def fig06(updates, groups, folder):
    fig, axes = axes_grid()
    for ax, seed in zip(axes[0], SEEDS):
        rows = selected(updates, seed, "continued_SL")
        if rows:
            curve(ax, rows, "loss", "Route-mean CE (trailing16)", COLORS[2], show_raw=True)
            ax.legend(frameon=False)
        else:
            absent(ax, "continued_SL")
        label_axis(ax, seed, "continued_SL")
        ax.set_ylabel("Masked teacher cross-entropy (nats)")
    return save(fig, folder, "fig06_continued_SL_CE")


def fig07(updates, groups, folder):
    fig, axes = axes_grid(2, height=2.6)
    for j, seed in enumerate(SEEDS):
        rl, sl = selected(updates, seed), selected(updates, seed, "continued_SL")
        ax = axes[0][j]
        for phase, rows, offset, color in (("RL", rl, 0, COLORS[0]), ("continued SL", sl, 512, COLORS[2])):
            if rows:
                ax.plot([int(r["cursor_after"]) + offset for r in rows], [number(r, "stage_elapsed_s") / 60 for r in rows],
                        label=phase, color=color, linewidth=1.0)
        if not rl and not sl:
            absent(ax, "S2")
        else:
            ax.legend(frameon=False)
        ax.axhline(120, color=".5", linestyle=":", linewidth=.7, label="120-min cap")
        ax.set_xlabel(f"Phase update sequence (seed {seed})")
        ax.set_ylabel("Cumulative shared S2 wall time (min)")
        ax = axes[1][j]
        if rl:
            for key, label, color, ls in (("H_s_mean", "H", COLORS[0], "-"), ("export_s_mean", "Export", COLORS[1], "--"),
                                         ("validation_s_mean", "Validation", COLORS[2], ":")):
                curve(ax, rl, key, label, color, ls)
            ax.legend(frameon=False)
        else:
            absent(ax, "RL")
        label_axis(ax, seed)
        ax.set_ylabel("Mean charged trajectory component (s)")
    return save(fig, folder, "fig07_wall_and_terminal_components")


def fig08(updates, groups, folder):
    fig, axes = axes_grid()
    windows = ("full10x10", "local4x3", "local6x6", "local8x8")
    for ax, seed in zip(axes[0], SEEDS):
        for metric, offset, label, color, marker in (("raw_J", -.08, "Before H", COLORS[1], "o"),
                                                   ("H_J", .08, "After H", COLORS[0], "s")):
            members = [next((r for r in groups if r["group_dimensions"] == "seed|window_group" and
                int(r["seed"]) == seed and r["window_group"] == window and r["metric"] == metric), None) for window in windows]
            x = [i + offset for i, r in enumerate(members) if r]
            y = [number(r, "mean") for r in members if r]
            # Asymmetric distances around the mean can be negative when a mean
            # lies outside the IQR; draw IQR segments separately from mean dots.
            ax.scatter(x, y, marker=marker, color=color, label=label, s=22)
            for xi, r in zip(x, [r for r in members if r]):
                ax.vlines(xi, number(r, "p25"), number(r, "p75"), color=color, linewidth=1.3)
        ax.set_xticks(range(4), ("Full10x10", "Local4x3", "Local6x6", "Local8x8"), rotation=25, ha="right")
        ax.set_xlabel(f"Observed training window (seed {seed})")
        ax.set_ylabel("Objective $J$ (m equivalent)")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=.12, linewidth=.5)
    return save(fig, folder, "fig08_observed_cost_by_window")


def fig09(updates, groups, folder):
    fig, axes = axes_grid()
    for ax, seed in zip(axes[0], SEEDS):
        rows = selected(updates, seed)
        if rows:
            curve(ax, rows, "H_gain_mean", "Frozen H gain (trailing16)", COLORS[1], show_raw=True)
            ax.axhline(0, color=".5", linewidth=.6)
            ax.legend(frameon=False)
        else:
            absent(ax, "RL")
        label_axis(ax, seed)
        ax.set_ylabel("$J_{raw}-J_H$ (m equivalent)")
    return save(fig, folder, "fig09_frozen_H_gain")


FUNCTIONS = {i: globals()[f"fig{i:02}"] for i in range(1, 10)}


def run_figure(index):
    parser = argparse.ArgumentParser(description="Plot saved S2 analysis CSVs only")
    parser.add_argument("--analysis", type=Path, default=Path(__file__).parent / "analysis")
    args = parser.parse_args()
    analysis = args.analysis.resolve()
    report = Path(__file__).resolve().parent
    if not analysis.is_relative_to(report):
        raise ValueError("Figure inputs/outputs must remain inside the new report directory")
    files = FUNCTIONS[index](read_rows(analysis / "updates.csv"), read_rows(analysis / "group_summary.csv"), analysis / "figures")
    for file in files:
        print(file)
