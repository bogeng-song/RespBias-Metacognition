"""
figure2.py
Generates the example-participant figure (manuscript Figure 3):
  "Empirical relationships between response bias and confidence"

Note on numbering: in the current manuscript, Figure 1 is the generative SDT
simulation (see scripts_analysis/simulation.py), Figure 2 is the task schematic
(not code-generated), and this example-participant figure is Figure 3.

Panel A: Theoretical illustration — optimal confidence vs. response bias
         (4-choice and 8-choice), derived from average empirical response biases
         across all participants.
Panel B: Example participant — bias~confidence, bias~accuracy, and
         bias~confidence-residual (after controlling for accuracy),
         for both the 4-choice (top row) and 8-choice (bottom row).

Usage:
    python notebooks/figure2.py \
        --data4 data/human_data/Experiment1_4_choice.csv \
        --data8 data/human_data/Experiment1_8_choice.csv \
        --example_sub 12 \
        --output figure3_example_participant.pdf
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import linregress
import matplotlib as mpl

# ──────────────────────────────────────────────────────────────────────────────
# Style constants (matches manuscript / Nature Human Behaviour style)
# ──────────────────────────────────────────────────────────────────────────────
DIGIT_COLORS_4 = {5: "#9467bd", 6: "#8c564b", 7: "#e377c2", 8: "#7f7f7f"}
DIGIT_COLORS_8 = {
    1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728",
    5: "#9467bd", 6: "#8c564b", 7: "#e377c2", 8: "#7f7f7f"
}
PANEL_A_COLORS = {"4-choice": "#4E79A7", "8-choice": "#F28E2B"}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
})


# ──────────────────────────────────────────────────────────────────────────────
# Helper: compute per-subject per-digit statistics
# ──────────────────────────────────────────────────────────────────────────────
def compute_digit_stats(df: pd.DataFrame, digits: list) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
    Sub_id, Digit, HR, FAR, Accuracy, Confidence, Conf_residual
    where Conf_residual = confidence with accuracy linearly regressed out (raw scale).
    """
    records = []
    for sub_id, sub_df in df.groupby("Sub_id"):
        hrs, fars, accs, confs = [], [], [], []
        for d in digits:
            stim     = sub_df[sub_df["Stimulus"] == d]
            non_stim = sub_df[sub_df["Stimulus"] != d]

            hr  = stim["Correct"].mean()             if len(stim) > 0     else np.nan
            far = (non_stim["Response"] == d).mean() if len(non_stim) > 0 else np.nan
            acc = hr
            conf = stim["Confidence"].mean()         if len(stim) > 0     else np.nan

            hrs.append(hr); fars.append(far); accs.append(acc); confs.append(conf)

        # Confidence residual: regress out accuracy (simple OLS, raw scale)
        accs_a  = np.array(accs, dtype=float)
        confs_a = np.array(confs, dtype=float)
        valid   = ~(np.isnan(accs_a) | np.isnan(confs_a))
        conf_residual = np.full_like(confs_a, np.nan)
        if valid.sum() >= 2 and np.std(accs_a[valid]) > 0:
            slope, intercept, *_ = linregress(accs_a[valid], confs_a[valid])
            conf_residual[valid] = confs_a[valid] - (slope * accs_a[valid] + intercept)

        for i, d in enumerate(digits):
            records.append({
                "Sub_id": sub_id, "Digit": d,
                "HR": hrs[i], "FAR": fars[i],
                "Accuracy": accs[i], "Confidence": confs[i],
                "Conf_residual": conf_residual[i],
            })
    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# Panel A helpers
# ──────────────────────────────────────────────────────────────────────────────
def optimal_confidence(hr: np.ndarray, far: np.ndarray, n_alt: int) -> np.ndarray:
    """
    P(correct | choice = digit) using Bayes' rule with a uniform stimulus prior.
    = HR * (1/N) / [HR * (1/N) + FAR * ((N-1)/N)]
    """
    prior  = 1.0 / n_alt
    p_corr = hr * prior / (hr * prior + far * (1 - prior))
    return p_corr


def mean_hr_far(df: pd.DataFrame, digits: list) -> tuple:
    """Average HR and FAR across all subjects for each digit."""
    hr_all, far_all = [], []
    for d in digits:
        hrs, fars = [], []
        for _, sub_df in df.groupby("Sub_id"):
            stim     = sub_df[sub_df["Stimulus"] == d]
            non_stim = sub_df[sub_df["Stimulus"] != d]
            if len(stim) > 0:
                hrs.append(stim["Correct"].mean())
            if len(non_stim) > 0:
                fars.append((non_stim["Response"] == d).mean())
        hr_all.append(np.nanmean(hrs))
        far_all.append(np.nanmean(fars))
    return np.array(hr_all), np.array(far_all)


# ──────────────────────────────────────────────────────────────────────────────
# Scatter helper with regression line + β annotation
# ──────────────────────────────────────────────────────────────────────────────
def scatter_with_regression(ax, x, y, colors_dict, digits, xlabel, ylabel,
                             beta_pos="lower_right", point_size=90, fontsize=11):
    """
    Scatter plot with labelled points + dashed OLS regression line + β annotation.
    beta_pos: 'lower_right' or 'upper_left'
    """
    slope, intercept, r, p, _ = linregress(x, y)

    xx = np.linspace(np.min(x), np.max(x), 200)
    ax.plot(xx, slope * xx + intercept, "--", color="black", lw=1.5, zorder=1)

    # Point jitter offset for labels
    dx = 0.012 * (np.max(x) - np.min(x)) if np.ptp(x) > 1e-9 else 0.001
    dy = 0.012 * (np.max(y) - np.min(y)) if np.ptp(y) > 1e-9 else 0.001

    for xi, yi, d in zip(x, y, digits):
        c = colors_dict.get(int(d), "gray")
        ax.scatter(xi, yi, s=point_size, color=c, edgecolor="black",
                   linewidth=0.7, zorder=3, alpha=0.92)
        ax.text(xi + dx, yi + dy, str(int(d)),
                fontsize=fontsize - 2, color=c, zorder=4, fontweight="bold")

    # β annotation
    beta_str = f"β = {slope:.2f}"
    if beta_pos == "upper_left":
        tx, ty, ha, va = 0.05, 0.92, "left", "top"
    elif beta_pos == "lower_right":
        tx, ty, ha, va = 0.95, 0.08, "right", "bottom"
    elif beta_pos == "lower_left":
        tx, ty, ha, va = 0.05, 0.08, "left", "bottom"
    else:
        tx, ty, ha, va = 0.05, 0.92, "left", "top"

    ax.text(tx, ty, beta_str, transform=ax.transAxes,
            fontsize=fontsize, ha=ha, va=va, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"))

    ax.set_xlabel(xlabel, fontsize=fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=fontsize, fontweight="bold")
    ax.tick_params(labelsize=fontsize - 1)
    return slope, p


# ──────────────────────────────────────────────────────────────────────────────
# Main figure function
# ──────────────────────────────────────────────────────────────────────────────
def make_figure2(df4: pd.DataFrame, df8: pd.DataFrame,
                 example_sub: int = 12,
                 output_path: str = "figure2.pdf",
                 fontsize: int = 11):

    digits4 = list(range(5, 9))
    digits8 = list(range(1, 9))

    # ── Compute aggregate stats ────────────────────────────────────────────────
    hr4, far4 = mean_hr_far(df4, digits4)
    hr8, far8 = mean_hr_far(df8, digits8)
    opt4 = optimal_confidence(hr4, far4, n_alt=4)
    opt8 = optimal_confidence(hr8, far8, n_alt=8)

    # ── Compute example participant stats ──────────────────────────────────────
    agg4 = compute_digit_stats(df4, digits4)
    agg8 = compute_digit_stats(df8, digits8)

    ex4 = agg4[agg4["Sub_id"] == example_sub].sort_values("Digit")
    ex8 = agg8[agg8["Sub_id"] == example_sub].sort_values("Digit")

    if len(ex4) == 0 or len(ex8) == 0:
        raise ValueError(f"Subject {example_sub} not found in one or both datasets.")

    # ── Layout: 2 rows ─────────────────────────────────────────────────────────
    # Row 0 = Panel A (2 scatter plots)
    # Row 1 = Panel B (2 rows × 3 cols = 6 scatter plots, shown as one 2×3 sub-grid)
    fig = plt.figure(figsize=(14, 10))
    gs_outer = gridspec.GridSpec(2, 1, figure=fig,
                                 height_ratios=[1, 1.8],
                                 hspace=0.42)

    # ── PANEL A ────────────────────────────────────────────────────────────────
    gs_a = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_outer[0],
                                            wspace=0.40)
    ax_a4 = fig.add_subplot(gs_a[0])
    ax_a8 = fig.add_subplot(gs_a[1])

    # 4-choice
    slope_a4, _ = scatter_with_regression(
        ax_a4, far4, opt4,
        colors_dict=DIGIT_COLORS_4, digits=digits4,
        xlabel="Response Bias (FAR)",
        ylabel="Optimal Confidence",
        beta_pos="lower_left", fontsize=fontsize)
    ax_a4.set_title("4-Choice", fontsize=fontsize + 1, fontweight="bold", pad=6)

    # 8-choice
    slope_a8, _ = scatter_with_regression(
        ax_a8, far8, opt8,
        colors_dict=DIGIT_COLORS_8, digits=digits8,
        xlabel="Response Bias (FAR)",
        ylabel="Optimal Confidence",
        beta_pos="lower_left", fontsize=fontsize)
    ax_a8.set_title("8-Choice", fontsize=fontsize + 1, fontweight="bold", pad=6)

    # Panel A label
    fig.text(0.01, 0.97, "A", fontsize=16, fontweight="bold", va="top")

    # ── PANEL B ────────────────────────────────────────────────────────────────
    # 2 rows (4-choice top, 8-choice bottom) × 3 cols
    gs_b = gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=gs_outer[1],
                                            wspace=0.40, hspace=0.50)

    col_titles = [
        "Bias–Confidence\n(without acc. control)",
        "Bias–Accuracy",
        "Bias–Confidence\n(acc. controlled)",
    ]
    row_labels = ["4-Choice", "8-Choice"]

    for row_i, (ex, digits, cmap) in enumerate([
        (ex4, digits4, DIGIT_COLORS_4),
        (ex8, digits8, DIGIT_COLORS_8),
    ]):
        far_vals  = ex["FAR"].values
        acc_vals  = ex["Accuracy"].values
        conf_vals = ex["Confidence"].values
        resid_vals = ex["Conf_residual"].values
        digs       = ex["Digit"].values

        # Column 0: Confidence vs. FAR (no accuracy control)
        ax = fig.add_subplot(gs_b[row_i, 0])
        scatter_with_regression(ax, far_vals, conf_vals, cmap, digs,
                                 "Response Bias (FAR)", "Confidence",
                                 beta_pos="upper_left", fontsize=fontsize)
        if row_i == 0:
            ax.set_title(col_titles[0], fontsize=fontsize, pad=6)
        ax.set_ylabel(f"{row_labels[row_i]}\nConfidence",
                      fontsize=fontsize, fontweight="bold")

        # Column 1: Accuracy vs. FAR
        ax = fig.add_subplot(gs_b[row_i, 1])
        scatter_with_regression(ax, far_vals, acc_vals, cmap, digs,
                                 "Response Bias (FAR)", "Accuracy",
                                 beta_pos="upper_left", fontsize=fontsize)
        if row_i == 0:
            ax.set_title(col_titles[1], fontsize=fontsize, pad=6)
        ax.set_ylabel("Accuracy", fontsize=fontsize, fontweight="bold")

        # Column 2: Confidence residual vs. FAR (after accuracy controlled)
        ax = fig.add_subplot(gs_b[row_i, 2])
        scatter_with_regression(ax, far_vals, resid_vals, cmap, digs,
                                 "Response Bias (FAR)",
                                 "Confidence\n(acc. controlled)",
                                 beta_pos="lower_left", fontsize=fontsize)
        if row_i == 0:
            ax.set_title(col_titles[2], fontsize=fontsize, pad=6)
        ax.set_ylabel("Confidence residual", fontsize=fontsize, fontweight="bold")
        ax.axhline(0, color="gray", lw=0.8, ls="--", zorder=0)

    # Panel B label
    fig.text(0.01, 0.67, "B", fontsize=16, fontweight="bold", va="top")

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate Figure 2")
    p.add_argument("--data4",       default="data/human_data/Experiment1_4_choice.csv")
    p.add_argument("--data8",       default="data/human_data/Experiment1_8_choice.csv")
    p.add_argument("--example_sub", type=int, default=12,
                   help="Sub_id for the example participant in Panel B")
    p.add_argument("--output",      default="figure2.pdf")
    p.add_argument("--fontsize",    type=int, default=11)
    args = p.parse_args()

    df4 = pd.read_csv(args.data4)
    df8 = pd.read_csv(args.data8)
    make_figure2(df4, df8,
                 example_sub=args.example_sub,
                 output_path=args.output,
                 fontsize=args.fontsize)


if __name__ == "__main__":
    main()
