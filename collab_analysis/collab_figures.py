# =============================================================================
# collab_figures.py
# =============================================================================
"""
collab_figures.py
-----------------
Five-panel behavioural summary for the collaborator dataset.
 
Panels
------
A  Preference index (vertical zones, X axis) over time
B  Group centroid X position over time
C  Mean VX over time
D  Mean average neighbour distance over time
E  Mean nearest neighbour distance over time
 
All panels show mean ± SEM shading.
One figure per condition; call in a loop for multiple conditions.
Palette is a simple qualitative scheme ordered by concentration then genotype,
with no assumptions about starvation state.
"""
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
 
# ---------------------------------------------------------------------------
# Palette builder (collaborator-specific — no starvation grouping)
# ---------------------------------------------------------------------------
 
def build_collab_palette(
    df: pd.DataFrame,
    condition_col: str    = "Condition",
    concentration_col: str = "Concentration",
    genotype_col: str     = "Genotype",
) -> tuple:
    """
    Build a qualitative colour palette and ordered condition list.
 
    Ordering: Concentration (ascending) → Genotype (alphabetical).
    Colours: evenly spaced from seaborn's 'tab10' palette.
 
    Returns
    -------
    condition_colors : dict  {condition: rgb_tuple}
    ordered_conditions : list[str]
    """
    ordered = (
        df[[condition_col, concentration_col, genotype_col]]
        .drop_duplicates()
        .sort_values([concentration_col, genotype_col])
        [condition_col]
        .tolist()
    )
 
    palette = sns.color_palette("tab10", len(ordered))
    colors  = {cond: col for cond, col in zip(ordered, palette)}
 
    return colors, ordered
 
 
# ---------------------------------------------------------------------------
# Helper: line + SEM shading for one metric
# ---------------------------------------------------------------------------
 
def _line_sem(ax, sub, x_col, y_col, sem_col, color, label):
    """Draw a mean line with SEM shading on ax."""
    sub = sub.sort_values(x_col)
    ax.plot(sub[x_col], sub[y_col], color=color, linewidth=1.8, label=label)
    ax.fill_between(
        sub[x_col],
        sub[y_col] - sub[sem_col],
        sub[y_col] + sub[sem_col],
        color=color, alpha=0.2,
    )
 
 
# ---------------------------------------------------------------------------
# Main figure function
# ---------------------------------------------------------------------------
 
def collab_summary(
    df: pd.DataFrame,
    bin_size: int       = 100,
    zone_width: float   = None,
    arena_width: float  = None,
    display_labels: dict = None,
):
    """
    Five-panel behavioural summary for the collaborator dataset.
    All conditions in df are plotted together on each panel.
 
    Parameters
    ----------
    df : pd.DataFrame
        Output of collab_io.preprocess_collab().
    bin_size : int
        Frame bin width. Default 100.
    zone_width : float or None
        Width of left/right zones for PI. If None, equal thirds of X range.
    arena_width : float or None
        Total arena width. Used only when zone_width is provided.
    display_labels : dict or None
        Optional {condition: short_label} for legend entries.
 
    Returns
    -------
    pi_df, centroid_df, vx_df, neighbour_df
        The four underlying metric DataFrames.
    """
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
 
    colors, order = build_collab_palette(df)
 
    def _label(cond):
        return display_labels[cond] if display_labels and cond in display_labels else cond
 
    # --- Compute all metrics ---
    import collab_analysis.collab_metrics as collab_metrics
 
    pi_df        = collab_metrics.preference_index_vertical(
        df, bin_size=bin_size, zone_width=zone_width, arena_width=arena_width,
    )
    centroid_df  = collab_metrics.centroid_position(df, bin_size=bin_size)
    vx_df        = collab_metrics.mean_vx(df, bin_size=bin_size)
    neighbour_df = collab_metrics.neighbour_distances(df, bin_size=bin_size)
 
    # Aggregate PI to mean ± SEM across individuals within each
    # (Condition, Trial, FrameBin), then across trials within each
    # (Condition, FrameBin). This two-step average is correct when
    # there are multiple trials: first collapse within trial, then
    # across trials so each trial contributes equally regardless of
    # how many individuals it has.
    pi_trial = (
        pi_df.groupby(["Condition", "Trial", "FrameBin"])["PreferenceIndex"]
        .mean()
        .reset_index()
    )
    pi_agg = (
        pi_trial.groupby(["Condition", "FrameBin"])["PreferenceIndex"]
        .agg(Mean="mean", SEM="sem")
        .reset_index()
    )
 
    # --- Build figure ---
    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    panel_labels = ["A", "B", "C", "D", "E"]
    for ax, lbl in zip(axes, panel_labels):
        ax.text(
            -0.12, 1.05, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top",
        )
 
    for cond in order:
        col = colors[cond]
        lbl = _label(cond)
 
        # --- Panel A: Preference index ---
        sub = pi_agg[pi_agg["Condition"] == cond]
        _line_sem(axes[0], sub, "FrameBin", "Mean", "SEM", col, lbl)
 
        # --- Panel B: Centroid X ---
        sub = centroid_df[centroid_df["Condition"] == cond]
        # Average across trials if multiple
        sub_agg = (
            sub.groupby("FrameBin")
            .agg(Mean=("MeanX", "mean"), SEM=("MeanX", "sem"))
            .reset_index()
        )
        _line_sem(axes[1], sub_agg, "FrameBin", "Mean", "SEM", col, lbl)
 
        # --- Panel C: Mean VX ---
        sub = vx_df[vx_df["Condition"] == cond]
        sub_agg = (
            sub.groupby("FrameBin")
            .agg(Mean=("MeanVX", "mean"), SEM=("SEM_VX", "mean"))
            .reset_index()
        )
        _line_sem(axes[2], sub_agg, "FrameBin", "Mean", "SEM", col, lbl)
 
        # --- Panel D: Mean average neighbour distance ---
        sub = neighbour_df[neighbour_df["Condition"] == cond]
        if not sub.empty:
            sub_agg = (
                sub.groupby("FrameBin")
                .agg(
                    Mean=("MeanAvgNeighbour",     "mean"),
                    SEM=("SEM_AvgNeighbour",      "mean"),
                )
                .reset_index()
            )
            _line_sem(axes[3], sub_agg, "FrameBin", "Mean", "SEM", col, lbl)
 
        # --- Panel E: Mean nearest neighbour distance ---
        if not sub.empty:
            sub_agg = (
                sub.groupby("FrameBin")
                .agg(
                    Mean=("MeanNearestNeighbour", "mean"),
                    SEM=("SEM_NearestNeighbour",  "mean"),
                )
                .reset_index()
            )
            _line_sem(axes[4], sub_agg, "FrameBin", "Mean", "SEM", col, lbl)
 
    # --- Axis labels and formatting ---
    axes[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylim(-1, 1)
    axes[0].set_title("Preference Index\n(Left − Right)")
    axes[0].set_xlabel("Frame Bin")
    axes[0].set_ylabel("PI")
 
    axes[1].set_title("Centroid X Position")
    axes[1].set_xlabel("Frame Bin")
    axes[1].set_ylabel("Mean X")
 
    axes[2].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[2].set_title("Mean V\u2093 (X velocity)")
    axes[2].set_xlabel("Frame Bin")
    axes[2].set_ylabel("Mean VX")
 
    axes[3].set_title("Avg Neighbour Distance")
    axes[3].set_xlabel("Frame Bin")
    axes[3].set_ylabel("Distance")
 
    axes[4].set_title("Nearest Neighbour Distance")
    axes[4].set_xlabel("Frame Bin")
    axes[4].set_ylabel("Distance")
 
    # Single shared legend on the rightmost panel
    axes[4].legend(
        title="Condition",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=9,
        frameon=True,
    )
 
    plt.tight_layout()
    plt.show()
 
    return pi_df, centroid_df, vx_df, neighbour_df
 