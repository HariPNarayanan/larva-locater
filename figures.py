"""
figures.py
----------
All plotting functions for the larva behaviour pipeline.

Each function follows the same pattern:
  1. Call config.build_palette(df) once for colours + ordering
  2. Call the relevant metrics.* function(s) for data
  3. Plot and return the underlying DataFrame(s) for downstream use

No computation that belongs in metrics.py lives here.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit

import config
import metrics

# ---------------------------------------------------------------------------
# Shared style helper
# ---------------------------------------------------------------------------

def _apply_style():
    sns.set_style("white")
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # --- Vector rendering settings ---
        "pdf.fonttype": 42,        # embeds fonts as TrueType rather than bitmaps
        "ps.fonttype": 42,         # same for PostScript
        "svg.fonttype": "none",    # keeps text as text in SVG
        "path.simplify": False,    # don't simplify paths — keeps all data points
        "agg.path.chunksize": 0,   # no chunking — renders full paths at once
    })


# ---------------------------------------------------------------------------
# 1. Behavior summary (main 90% function)
# ---------------------------------------------------------------------------

def behavior_summary_overtime(
    df: pd.DataFrame,
    bin_size: int   = 100,
    target_x: float = config.TARGET_X,
    target_y: float = config.TARGET_Y,
    radius: float   = config.SUCCESS_RADIUS,
    zone_bounds: tuple = (10.0, 20.0),
    display_labels: dict = None,
):
    """
    Three-panel behavioural summary:
        Panel A — Cumulative success rate over time (± SEM)
        Panel B — Preference index over time (± SEM)
        Panel C — Post-success dwell time (box + strip, successful only)
 
    Parameters
    ----------
    display_labels : dict or None
        Optional {condition_name: short_label} mapping for axis labels.
 
    Returns
    -------
    cumul_df, pref_df, dwell_df
    """
    _apply_style()
    colors, order = config.build_palette(df)
 
    cumul_df = metrics.cumulative_success(
        df, bin_size=bin_size, target_x=target_x, target_y=target_y, radius=radius
    )
    pref_df  = metrics.preference_index(
        df, bin_size=bin_size, zone_bounds=zone_bounds
    )
    dwell_df = metrics.dwell_time(
        df, target_x=target_x, target_y=target_y, radius=radius
    )
 
    # Apply display labels
    def _label(cond):
        return display_labels[cond] if display_labels and cond in display_labels else cond
 
    label_order = [_label(c) for c in order]
 
    for frame in (cumul_df, pref_df, dwell_df):
        frame["PlotLabel"] = frame[config.COL_CONDITION].map(_label)
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
 
    # --- Panel A: Cumulative success rate ---
    ax = axes[0]
    for cond in order:
        sub = cumul_df[cumul_df[config.COL_CONDITION] == cond].sort_values("FrameBin")
        col = colors[cond]
        lbl = _label(cond)
        ax.plot(sub["FrameBin"], sub["CumulSuccess"], color=col, label=lbl, linewidth=1.8)
        ax.fill_between(
            sub["FrameBin"],
            sub["CumulSuccess"] - sub["SEM"],
            sub["CumulSuccess"] + sub["SEM"],
            color=col, alpha=0.2,
        )
    ax.set_ylim(0, 1)
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_title("Cumulative Success Rate")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Proportion successful (± SEM)")
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
 
    # --- Panel B: Preference index ---
    # dodge=True causes a ZeroDivisionError in seaborn when there is only
    # one hue level (n_hue_levels - 1 == 0). Guard against this explicitly.
    _dodge = len(label_order) > 1
 
    sns.pointplot(
        data=pref_df,
        x="FrameBin",
        y="PreferenceIndex",
        hue="PlotLabel",
        hue_order=label_order,
        palette=colors,
        errorbar="se",
        dodge=_dodge,
        ax=axes[1],
    )
    axes[1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[1].set_title("Preference Index Over Time")
    axes[1].set_ylabel("Preference Index (Z1 − Z3)")
    axes[1].set_xlabel("Frame Bin")
    axes[1].legend(title="Condition", bbox_to_anchor=(1.05, 1))
 
    # --- Panel C: Dwell time ---
    ax = axes[2]
    if not dwell_df.empty:
        sns.boxplot(
            data=dwell_df, x="PlotLabel", y="DwellTime",
            order=label_order, palette=colors,
            showfliers=False, ax=ax,
        )
        sns.stripplot(
            data=dwell_df, x="PlotLabel", y="DwellTime",
            order=label_order, palette=colors,
            jitter=True, size=5, alpha=0.7,
            edgecolor="black", linewidth=0.5, ax=ax,
        )
    ax.set_title("Post-Success Dwell Time")
    ax.set_ylabel("Frames inside after first entry")
    ax.set_xlabel("Condition")
    ax.tick_params(axis="x", rotation=45)
 
    plt.tight_layout()
    plt.show()
 
    return cumul_df, pref_df, dwell_df


# ---------------------------------------------------------------------------
# 2. Trajectory heatmaps (one condition at a time)
# ---------------------------------------------------------------------------

def trajectory_heatmaps(
    df: pd.DataFrame,
    condition: str,
    frame_bin_size: int = 100,
    grid_size: int      = 1,
    title_size: int     = 18,
    label_size: int     = 14,
    tick_size: int      = 12,
    cbar_size: int      = 14,
    save_path: str = None,   # e.g. "figures/fig1.pdf"
):
    """
    Spatial density heatmap split into equal frame bins for a single condition.
    Pass df pre-filtered or supply condition= to filter internally.
    """
    df = df[df[config.COL_CONDITION] == condition].copy()

    min_frame  = df[config.COL_FRAME].min()
    max_frame  = df[config.COL_FRAME].max()
    frame_bins = np.arange(min_frame, max_frame + frame_bin_size, frame_bin_size)
    num_bins   = len(frame_bins) - 1

    fig, axes = plt.subplots(1, num_bins, figsize=(5 * num_bins, 7), sharey=True)
    if num_bins == 1:
        axes = [axes]

    cax = None
    for i in range(num_bins):
        bin_start, bin_end = frame_bins[i], frame_bins[i + 1]
        df_bin = df[
            (df[config.COL_FRAME] >= bin_start) & (df[config.COL_FRAME] < bin_end)
        ]

        x_min, x_max = df_bin[config.COL_X].min(), df_bin[config.COL_X].max()
        y_min, y_max = df_bin[config.COL_Y].min(), df_bin[config.COL_Y].max()

        hist, xedges, yedges = np.histogram2d(
            df_bin[config.COL_X], df_bin[config.COL_Y],
            bins=[
                np.arange(x_min, x_max + grid_size, grid_size),
                np.arange(y_min, y_max + grid_size, grid_size),
            ],
        )
        cax = axes[i].pcolormesh(
            xedges, yedges, hist.T,
            cmap="viridis", shading="auto", norm=LogNorm(vmin=1),
        )
        axes[i].set_title(f"Frames {bin_start}–{bin_end}", fontsize=title_size)
        axes[i].set_xlabel("X (cm)", fontsize=label_size)
        if i == 0:
            axes[i].set_ylabel("Y (cm)", fontsize=label_size)
        axes[i].tick_params(labelsize=tick_size)
        axes[i].grid(True)
        axes[i].set_aspect("equal")
        axes[i].set_xlim([0, config.ARENA_WIDTH])
        axes[i].set_ylim([0, config.ARENA_HEIGHT])

    cbar = fig.colorbar(cax, ax=axes, orientation="vertical", fraction=0.02, pad=0.04)
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label("Density (log scale)", fontsize=cbar_size)

    fig.suptitle(f"Trajectory Heatmap — {condition}", fontsize=title_size + 4)
    plt.subplots_adjust(right=0.85, top=0.85)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# 3. Zone proportions (Early / Mid / Late × Concentration)
# ---------------------------------------------------------------------------

def zone_means(
    df: pd.DataFrame,
    filter_column: str  = config.COL_CONCENTRATION,
    filter_values: list = None,
    titles: list        = ("Early", "Mid", "Late"),
    zone_bounds: tuple  = None,
):
    """
    Grid of mean zone proportions (± SEM) split by time period (rows)
    and a filter column such as Concentration (columns).

    Rows:    Early / Mid / Late thirds of the frame range
    Columns: unique values of filter_column
    """
    _apply_style()
    colors, _ = config.build_palette(df)

    if filter_values is None:
        filter_values = sorted(df[filter_column].dropna().unique())

    min_f = df[config.COL_FRAME].min()
    max_f = df[config.COL_FRAME].max()
    third = (max_f - min_f) // 3

    time_slices = [
        df[df[config.COL_FRAME] <= min_f + third],
        df[(df[config.COL_FRAME] > min_f + third) & (df[config.COL_FRAME] <= min_f + 2 * third)],
        df[df[config.COL_FRAME] > min_f + 2 * third],
    ]

    num_rows = len(time_slices)
    num_cols = len(filter_values)

    fig, axes = plt.subplots(
        num_rows, num_cols,
        figsize=(4 * num_cols, 4 * num_rows),
        sharey=True, sharex=True,
    )
    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    zone_labels = ["Zone 1 (Bottom)", "Zone 2 (Middle)", "Zone 3 (Top)"]

    for row_idx, (slice_df, title) in enumerate(zip(time_slices, titles)):
        for col_idx, value in enumerate(filter_values):
            ax = axes[row_idx][col_idx]
            filtered = slice_df[slice_df[filter_column] == value]

            if filtered.empty:
                ax.set_visible(False)
                continue

            # Use metrics.zone_proportions for computation
            zp = metrics.zone_proportions(filtered, zone_bounds=zone_bounds)

            for cond in zp[config.COL_CONDITION].unique():
                sub = zp[zp[config.COL_CONDITION] == cond]
                means = sub[["Zone1", "Zone2", "Zone3"]].mean().values
                sems  = (
                    sub[["Zone1", "Zone2", "Zone3"]].std(ddof=1).values
                    / np.sqrt(len(sub))
                )
                ax.errorbar(
                    zone_labels, means, yerr=sems,
                    marker="o", markersize=6,
                    linestyle="-", linewidth=3,
                    color=colors.get(cond, "gray"),
                    capsize=5, label=cond,
                )

            ax.set_ylim(0, 1)
            if row_idx == 0:
                ax.set_title(f"{filter_column}: {value}")
                ax.legend(title="Condition", loc="upper right", fontsize=9, frameon=True)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\nMean Proportion")
            if row_idx == num_rows - 1:
                ax.set_xlabel("Zone")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 4. Radial Sholl heatmap (one condition at a time)
# ---------------------------------------------------------------------------

def sholl_heatmap(
    df: pd.DataFrame,
    target_x: float  = config.TARGET_X,
    target_y: float  = config.TARGET_Y,
    bin_size: int    = 100,
    max_radius: int  = 10,
    spatial_bin: int = 2,
):
    """
    Polar heatmap showing the per-bin-normalised proportion of individuals
    at each Sholl ring over time. Pass a single-condition DataFrame.
    """
    condition_name = (
        df[config.COL_CONDITION].iloc[0]
        if config.COL_CONDITION in df.columns and not df.empty
        else "Unknown"
    )

    df = df.copy()
    df["_dist"]  = np.sqrt(
        (df[config.COL_X] - target_x) ** 2 + (df[config.COL_Y] - target_y) ** 2
    )
    df["_ring"]  = (df["_dist"] / spatial_bin).astype(int)
    df           = df[df["_ring"] <= max_radius]
    df["_bin"]   = (df[config.COL_FRAME] // bin_size) * bin_size

    grouped = (
        df.groupby(["_bin", "_ring"])[config.COL_INDIVIDUAL]
        .nunique()
        .reset_index(name="Count")
    )
    totals = grouped.groupby("_bin")["Count"].sum().reset_index(name="Total")
    grouped = grouped.merge(totals, on="_bin")
    grouped["Proportion"] = grouped["Count"] / grouped["Total"]

    frame_bins = sorted(grouped["_bin"].unique())
    num_bins   = len(frame_bins)
    theta      = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    width      = 2 * np.pi / num_bins

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=grouped["Proportion"].max())

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 4)
    ax.set_theta_direction(-1)

    for i, fb in enumerate(frame_bins):
        for r in range(max_radius + 1):
            val = grouped.loc[
                (grouped["_bin"] == fb) & (grouped["_ring"] == r), "Proportion"
            ]
            proportion = val.values[0] if not val.empty else 0
            ax.bar(
                x=theta[i], height=spatial_bin, width=width,
                bottom=r * spatial_bin,
                color=cmap(norm(proportion)), edgecolor="none",
            )

    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_yticks([])
    ax.set_yticklabels([])
    ax.set_xticks(theta)
    ax.set_xticklabels([str(fb) for fb in frame_bins], fontsize=9, rotation=90)
    ax.set_title(
        f"Radial Sholl (per-bin normalised) — {condition_name}",
        va="bottom", fontsize=14,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", fraction=0.046, pad=0.1)
    cbar.set_label("Proportion per time bin", fontsize=12)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 5. Speed (binned boxplot)
# ---------------------------------------------------------------------------

def speed_binned_boxplot(
    df: pd.DataFrame,
    bin_size: int    = 30,
    min_speed: float = config.SPEED_MIN,
    max_speed: float = config.SPEED_MAX,
    display_labels: dict = None,
):
    """
    Boxplot of per-individual mean speed within each frame bin.
    """
    _apply_style()
    colors, order = config.build_palette(df)

    binned = metrics.speed_binned(
        df, bin_size=bin_size, min_speed=min_speed, max_speed=max_speed
    )

    def _label(c):
        return display_labels[c] if display_labels and c in display_labels else c

    binned["PlotLabel"] = binned[config.COL_CONDITION].map(_label)
    label_order = [_label(c) for c in order]

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=binned,
        x="FrameBin",
        y=config.COL_SPEED,
        hue="PlotLabel",
        hue_order=label_order,
        palette=colors,
        showfliers=False,
    )
    plt.title(f"Speed per individual (mean, {bin_size}-frame bins)")
    plt.xlabel("Frame Bin")
    plt.ylabel("Speed (cm/s)")
    plt.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return binned


# ---------------------------------------------------------------------------
# 6. Stopping frequency
# ---------------------------------------------------------------------------

def stopping_frequency(
    df: pd.DataFrame,
    stop_threshold: float = 0.2,
    display_labels: dict  = None,
):
    """
    Boxplot of the fraction of time each individual is 'stopped'
    (Speed < stop_threshold).

    Pass the unfiltered DataFrame so slow frames are present.
    """
    _apply_style()
    colors, order = config.build_palette(df)

    stop_df = metrics.stopping_frequency(df, stop_threshold=stop_threshold)

    def _label(c):
        return display_labels[c] if display_labels and c in display_labels else c

    stop_df["PlotLabel"] = stop_df[config.COL_CONDITION].map(_label)
    label_order = [_label(c) for c in order if c in stop_df[config.COL_CONDITION].values]

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        data=stop_df,
        x="PlotLabel",
        y="StopFrequency",
        order=label_order,
        palette=colors,
        showfliers=False,
    )
    plt.title(f"Stopping frequency (Speed < {stop_threshold} cm/s)")
    plt.ylabel("Fraction of time stopped")
    plt.xlabel("Condition")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return stop_df


# ---------------------------------------------------------------------------
# 7. Central fraction
# ---------------------------------------------------------------------------

def central_fraction(
    df: pd.DataFrame,
    center_x: float = config.ARENA_WIDTH  / 2,
    center_y: float = config.ARENA_HEIGHT / 2,
    radius: float   = 5.0,
    bin_size: int   = 30,
    display_labels: dict = None,
):
    """
    Fraction of individuals within a central radius over time (± SEM).
    Group-size invariant: computed per frame before binning.
    """
    _apply_style()
    colors, order = config.build_palette(df)

    central_df = metrics.central_fraction(
        df, center_x=center_x, center_y=center_y,
        radius=radius, bin_size=bin_size,
    )

    def _label(c):
        return display_labels[c] if display_labels and c in display_labels else c

    central_df["PlotLabel"] = central_df[config.COL_CONDITION].map(_label)
    label_order = [_label(c) for c in order if c in central_df[config.COL_CONDITION].values]

    plt.figure(figsize=(8, 6))
    sns.pointplot(
        data=central_df,
        x="FrameBin",
        y="CentralFraction",
        hue="PlotLabel",
        hue_order=label_order,
        palette=colors,
        errorbar="se",
        dodge=True,
    )
    plt.axhline(0.5, linestyle="--", color="gray", linewidth=1)
    plt.ylim(0, 1)
    plt.title(f"Central occupancy (radius = {radius} cm, bin = {bin_size} frames)")
    plt.ylabel("Fraction in centre (± SEM)")
    plt.xlabel("Frame (binned)")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    return central_df


# ---------------------------------------------------------------------------
# 8. Paired Fed vs 5h preference index (final window)
# ---------------------------------------------------------------------------

def preference_index_paired(
    df: pd.DataFrame,
    frame_range: tuple = None,
    last_n_frames: int = None,
    zone_bounds: tuple = (10.0, 20.0),
    pair_labels: dict  = None,
):
    """
    Box + strip plot comparing Fed vs 5h PI within a specific frame window,
    using per-trial averages as the statistical unit.

    Supply either frame_range=(start, end) or last_n_frames=int.
    pair_labels: optional {(concentration, genotype, collective): label} dict.
    """
    _apply_style()

    pi_trial = metrics.preference_index_final_window(
        df,
        frame_range=frame_range,
        last_n_frames=last_n_frames,
        zone_bounds=zone_bounds,
    )

    palette = {
        "Fed": sns.color_palette("Oranges", 3)[-1],
        "5h":  sns.color_palette("Blues",   3)[-1],
    }

    # Build x-axis order from PairKey
    unique_pairs = (
        pi_trial[[config.COL_CONCENTRATION, config.COL_GENOTYPE, config.COL_COLLECTIVE]]
        .drop_duplicates()
        .sort_values([config.COL_CONCENTRATION, config.COL_GENOTYPE, config.COL_COLLECTIVE])
    )
    ordered_pairs = [
        (r[config.COL_CONCENTRATION], r[config.COL_GENOTYPE], r[config.COL_COLLECTIVE])
        for _, r in unique_pairs.iterrows()
    ]

    if pair_labels:
        pi_trial["PairLabel"] = pi_trial["PairKey"].map(pair_labels)
        x_order = [pair_labels[p] for p in ordered_pairs if p in pair_labels]
    else:
        pi_trial["PairLabel"] = pi_trial["PairKey"].astype(str)
        x_order = [str(p) for p in ordered_pairs]

    pi_trial[config.COL_STARVATION] = pd.Categorical(
        pi_trial[config.COL_STARVATION].astype(str).str.strip(),
        categories=["Fed", "5h"], ordered=True,
    )

    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=pi_trial,
        x="PairLabel", y="PreferenceIndex",
        hue=config.COL_STARVATION,
        order=x_order, palette=palette,
        showfliers=False, hue_order=["Fed", "5h"],
    )
    sns.stripplot(
        data=pi_trial,
        x="PairLabel", y="PreferenceIndex",
        hue=config.COL_STARVATION,
        order=x_order, palette=palette,
        dodge=True, jitter=True, alpha=0.7,
        edgecolor="black", linewidth=0.5,
        hue_order=["Fed", "5h"],
    )
    plt.axhline(0, linestyle="--", color="black", linewidth=1)

    if frame_range:
        plt.ylabel(f"Preference Index (frames {frame_range[0]}–{frame_range[1]}, trial mean)")
    else:
        plt.ylabel(f"Preference Index (last {last_n_frames} frames, trial mean)")

    plt.xlabel("")
    plt.xticks(rotation=45)
    plt.legend(title="Starvation", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()

    return pi_trial


# ---------------------------------------------------------------------------
# 9. Logistic probability curves (10% function)
# ---------------------------------------------------------------------------

def _logistic_4param(r, A, L, k, r0):
    """Decreasing 4-parameter logistic: A + (L-A) / (1 + exp(k*(r-r0)))"""
    return A + (L - A) / (1.0 + np.exp(k * (r - r0)))


def _fit_logistic(r, p, bounds=None, p0=None):
    if bounds is None:
        bounds = ([0.0, 0.0, 0.0, np.min(r)], [1.0, 1.0, 20.0, np.max(r)])
    if p0 is None:
        p0 = [np.min(p), np.max(p), 0.5, np.median(r)]
    popt, pcov = curve_fit(
        _logistic_4param, r, p, p0=p0, bounds=bounds, maxfev=10000
    )
    return popt, pcov


def _prob_given_radius(df, target_x, target_y, radius, n_steps=50):
    """
    Compute P(success | cross r) for each condition.
    Returns a DataFrame with columns [Condition, Radius, P_success_given_r].
    Internal helper — not part of the public API.
    """
    df = df.copy()
    df["_dist"] = np.sqrt(
        (df[config.COL_X] - target_x) ** 2 + (df[config.COL_Y] - target_y) ** 2
    )

    max_d        = df["_dist"].max()
    radius_steps = np.linspace(max_d, radius, n_steps)[::-1]
    conditions   = sorted(df[config.COL_CONDITION].dropna().unique())

    records = []
    for cond in conditions:
        df_c = df[df[config.COL_CONDITION] == cond]
        grp  = df_c.groupby(config.COL_INDIVIDUAL)

        first_success = grp.apply(
            lambda d: d[d["_dist"] <= radius][config.COL_FRAME].min()
        ).replace({np.inf: np.nan})

        for r in radius_steps:
            first_r  = grp.apply(
                lambda d: d[d["_dist"] <= r][config.COL_FRAME].min()
            ).replace({np.inf: np.nan})
            crossed  = first_r.dropna().index
            if len(crossed) == 0:
                records.append({"Condition": cond, "Radius": r, "P_success_given_r": np.nan})
                continue
            succeeded = [
                ind for ind in crossed
                if not np.isnan(first_success.get(ind, np.nan))
                and first_success[ind] >= first_r[ind]
            ]
            records.append({
                "Condition": cond,
                "Radius": r,
                "P_success_given_r": len(succeeded) / len(crossed),
            })

    return pd.DataFrame(records)


def probability_logistic(
    df: pd.DataFrame,
    target_x: float = config.TARGET_X,
    target_y: float = config.TARGET_Y,
    radius: float   = config.SUCCESS_RADIUS,
    n_boot: int     = 500,
    ci: int         = 95,
    random_state: int = 1,
    save_csv: bool  = False,
    csv_path: str   = "logistic_params.csv",
):
    """
    Two-panel figure:
        Top    — empirical P(success | cross r) curves per condition
        Bottom — 4-parameter logistic fits with bootstrap CI bands

    Returns
    -------
    dict with keys: prob_df, params_df, comparison_table, bootstrap_samples
    """
    _apply_style()
    colors, order = config.build_palette(df)
    rng = np.random.default_rng(random_state)

    prob_df = _prob_given_radius(df, target_x, target_y, radius)
    conditions = [c for c in order if c in prob_df["Condition"].values]

    # --- Fit logistic + bootstrap per condition ---
    fit_summary      = []
    bootstrap_results = {}
    alpha   = 100 - ci
    lower_q = alpha / 2
    upper_q = 100 - alpha / 2

    for cond in conditions:
        sub = prob_df[prob_df["Condition"] == cond].sort_values("Radius")
        r   = sub["Radius"].values
        p   = sub["P_success_given_r"].values

        if len(r) < 4 or np.all(np.isnan(p)):
            fit_summary.append(
                {"Condition": cond, "A": np.nan, "L": np.nan,
                 "k": np.nan, "r0": np.nan, "R2": np.nan}
            )
            bootstrap_results[cond] = pd.DataFrame(columns=["A", "L", "k", "r0"])
            continue

        lb = [0.0, 0.0, 0.0, np.nanmin(r)]
        ub = [1.0, 1.0, 50.0, np.nanmax(r)]
        p0 = [np.nanmin(p), np.nanmax(p), 0.5, np.nanmedian(r)]

        try:
            popt, pcov = _fit_logistic(r, p, bounds=(lb, ub), p0=p0)
            A_f, L_f, k_f, r0_f = popt
            if A_f > L_f:
                A_f, L_f, k_f = L_f, A_f, -k_f
            p_pred = _logistic_4param(r, A_f, L_f, k_f, r0_f)
            ss_res = np.sum((p - p_pred) ** 2)
            ss_tot = np.sum((p - np.mean(p)) ** 2)
            R2     = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
        except Exception:
            A_f = L_f = k_f = r0_f = R2 = np.nan

        fit_summary.append(
            {"Condition": cond, "A": A_f, "L": L_f, "k": k_f, "r0": r0_f, "R2": R2}
        )

        boot_params = []
        n_pts = len(r)
        for _ in range(n_boot):
            idx = rng.integers(0, n_pts, n_pts)
            try:
                popt_b, _ = _fit_logistic(r[idx], p[idx], bounds=(lb, ub), p0=p0)
                Ab, Lb, kb, r0b = popt_b
                if Ab > Lb:
                    Ab, Lb, kb = Lb, Ab, -kb
                boot_params.append([Ab, Lb, kb, r0b])
            except Exception:
                continue
        bootstrap_results[cond] = pd.DataFrame(boot_params, columns=["A", "L", "k", "r0"])

    params_df = pd.DataFrame(fit_summary)

    # Add CI columns
    for param in ["A", "L", "k", "r0"]:
        lowers, uppers = [], []
        for cond in params_df["Condition"]:
            bd = bootstrap_results.get(cond, pd.DataFrame())
            if bd.empty:
                lowers.append(np.nan)
                uppers.append(np.nan)
            else:
                lowers.append(np.nanpercentile(bd[param], lower_q))
                uppers.append(np.nanpercentile(bd[param], upper_q))
        params_df[f"{param}_ci_lower"] = lowers
        params_df[f"{param}_ci_upper"] = uppers

    # --- Plot ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for cond in conditions:
        sub = prob_df[prob_df["Condition"] == cond]
        ax1.plot(sub["Radius"], sub["P_success_given_r"], "o-",
                 label=cond, color=colors.get(cond))
    ax1.set_ylabel("P(success | cross r)")
    ax1.set_title("Empirical probability curves")
    ax1.set_ylim(-0.02, 1.02)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="Condition")

    r_dense = np.linspace(prob_df["Radius"].min(), prob_df["Radius"].max(), 400)
    for cond in conditions:
        row = params_df[params_df["Condition"] == cond].iloc[0]
        A_f, L_f, k_f, r0_f = row[["A", "L", "k", "r0"]]
        if np.isnan(A_f):
            continue
        ax2.plot(r_dense, _logistic_4param(r_dense, A_f, L_f, k_f, r0_f),
                 color=colors.get(cond), label=cond)
        bd = bootstrap_results.get(cond, pd.DataFrame())
        if not bd.empty:
            n_take   = min(len(bd), 200)
            s_idx    = np.linspace(0, len(bd) - 1, n_take).astype(int)
            y_samp   = np.stack([
                _logistic_4param(r_dense, *bd.iloc[i][["A", "L", "k", "r0"]].values)
                for i in s_idx
            ])
            ax2.fill_between(
                r_dense,
                np.nanpercentile(y_samp, lower_q, axis=0),
                np.nanpercentile(y_samp, upper_q, axis=0),
                color=colors.get(cond), alpha=0.15,
            )

    ax2.set_xlabel("Radius r (cm)")
    ax2.set_ylabel("Fitted P(success | cross r)")
    ax2.set_title(f"Logistic fits with {ci}% bootstrap CI")
    ax2.set_ylim(-0.02, 1.02)
    ax2.grid(True, alpha=0.3)
    ax2.legend(title="Condition")

    plt.tight_layout()
    plt.show()

    # Comparison table
    def _fmt(est, lo, hi):
        if any(np.isnan(v) for v in [est, lo, hi]):
            return "NA"
        return f"{est:.3f} ± {(hi - lo) / 2:.3f}"

    comparison_table = pd.DataFrame([
        {
            "Condition": row["Condition"],
            "A":  _fmt(row["A"],  row["A_ci_lower"],  row["A_ci_upper"]),
            "L":  _fmt(row["L"],  row["L_ci_lower"],  row["L_ci_upper"]),
            "k":  _fmt(row["k"],  row["k_ci_lower"],  row["k_ci_upper"]),
            "r0": _fmt(row["r0"], row["r0_ci_lower"], row["r0_ci_upper"]),
            "R2": f"{row['R2']:.3f}" if not np.isnan(row["R2"]) else "NA",
        }
        for _, row in params_df.iterrows()
    ]).sort_values("Condition")

    if save_csv:
        params_df.to_csv(csv_path, index=False)
        comparison_table.to_csv(csv_path.replace(".csv", "_comparison.csv"), index=False)

    return {
        "prob_df":          prob_df,
        "params_df":        params_df,
        "comparison_table": comparison_table,
        "bootstrap_samples": bootstrap_results,
    }

def behavior_summary_directional(
    df: pd.DataFrame,
    condition: str,
    targets: dict      = None,
    radius: float      = config.SUCCESS_RADIUS,
    bin_size: int      = 100,
    zone_width: float  = 10.0,
):
    """
    Three-panel behavioural summary for one condition, showing the true
    odour target alongside three null edge positions:
 
        Panel A — Cumulative success rate over time (± SEM)
        Panel B — Preference index over time (± SEM)
        Panel C — Post-first-entry dwell time (box + strip)
 
    The true target is plotted as a solid line / filled box throughout.
    The three null edges are plotted as dashed lines / outlined boxes
    so the comparison is immediately readable.
 
    Parameters
    ----------
    condition : str
        Must match a value in df['Condition'].
    targets : dict or None
        {label: (x, y)} mapping. Defaults to metrics.DIRECTIONAL_TARGETS.
    radius : float
        Radius in cm considered 'at target'. Default config.SUCCESS_RADIUS.
    bin_size : int
        Frame bin width in frames. Default 100.
    zone_width : float
        Width of near/far strips for PI calculation. Default 10 cm
        (= one third of the arena, matching the standard PI definition).
 
    Returns
    -------
    cumul_df, pref_df, dwell_df
        The three underlying DataFrames, each with a 'Target' column
        distinguishing the four positions.
    """
    _apply_style()
 
    if targets is None:
        targets = metrics.DIRECTIONAL_TARGETS
 
    # --- Compute all three metrics ---
    cumul_df = metrics.cumulative_success_directional(
        df, condition=condition, targets=targets,
        radius=radius, bin_size=bin_size,
    )
    pref_df  = metrics.preference_index_directional(
        df, condition=condition, targets=targets,
        bin_size=bin_size, zone_width=zone_width,
    )
    dwell_df = metrics.dwell_time_directional(
        df, condition=condition, targets=targets, radius=radius,
    )
 
    # --- Consistent style map across all three panels ---
    # True target: solid / filled / thicker / high contrast
    # Null edges:  dashed / outlined / thinner / muted
    null_labels  = [l for l in targets if l != "Target (odour)"]
    null_colors  = ["#7fb3c8", "#a0b8a0", "#c4a882"]
    target_order = ["Target (odour)"] + null_labels
 
    style_map = {
        "Target (odour)": {
            "color":     "#1a6e8e",
            "linestyle": "-",
            "linewidth": 2.5,
            "zorder":    3,
        },
    }
    for label, color in zip(null_labels, null_colors):
        style_map[label] = {
            "color":     color,
            "linestyle": "--",
            "linewidth": 1.5,
            "zorder":    2,
        }
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(condition, fontsize=13, y=1.01)
 
    # --- Panel A: Cumulative success rate ---
    ax = axes[0]
    for label in target_order:
        sub = cumul_df[cumul_df["Target"] == label].sort_values("FrameBin")
        s   = style_map[label]
        ax.plot(
            sub["FrameBin"], sub["CumulSuccess"],
            label=label, color=s["color"],
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            zorder=s["zorder"],
        )
        ax.fill_between(
            sub["FrameBin"],
            sub["CumulSuccess"] - sub["SEM"],
            sub["CumulSuccess"] + sub["SEM"],
            color=s["color"], alpha=0.15, zorder=s["zorder"] - 1,
        )
    ax.set_ylim(0, 1)
    ax.axhline(1.0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_title("Cumulative Success Rate")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Proportion reached (± SEM)")
    ax.legend(title="Target", frameon=True, fontsize=9)
 
    # --- Panel B: Preference index ---
    ax = axes[1]
    # Aggregate to mean ± SEM across individuals per (target, frame bin)
    pref_agg = (
        pref_df.groupby(["Target", "FrameBin"])["PreferenceIndex"]
        .agg(Mean="mean", SEM=lambda x: x.sem())
        .reset_index()
    )
    for label in target_order:
        sub = pref_agg[pref_agg["Target"] == label].sort_values("FrameBin")
        s   = style_map[label]
        ax.plot(
            sub["FrameBin"], sub["Mean"],
            label=label, color=s["color"],
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            zorder=s["zorder"],
        )
        ax.fill_between(
            sub["FrameBin"],
            sub["Mean"] - sub["SEM"],
            sub["Mean"] + sub["SEM"],
            color=s["color"], alpha=0.15, zorder=s["zorder"] - 1,
        )
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_title("Preference Index Over Time")
    ax.set_xlabel("Frame Bin")
    ax.set_ylabel("Preference Index (near − far)")
    ax.legend(title="Target", frameon=True, fontsize=9)
 
    # --- Panel C: Dwell time ---
    ax = axes[2]
    # Box per target in consistent order, style matched by edge colour
    box_palette = {label: style_map[label]["color"] for label in target_order}
 
    # Separate boxplot properties for true target vs nulls
    # seaborn doesn't support per-box linestyle natively, so we draw
    # the true target as a filled box and the nulls with lighter alpha
    sns.boxplot(
        data=dwell_df,
        x="Target", y="DwellTime",
        order=target_order,
        palette=box_palette,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=dwell_df,
        x="Target", y="DwellTime",
        order=target_order,
        palette=box_palette,
        jitter=True, size=4, alpha=0.6,
        edgecolor="black", linewidth=0.4,
        ax=ax,
    )
 
    # Visually de-emphasise null boxes by reducing their patch alpha
    for i, patch in enumerate(ax.patches):
        if i > 0:   # index 0 = true target box
            patch.set_alpha(0.45)
 
    ax.axhline(0, linestyle=":", color="gray", linewidth=0.8)
    ax.set_title("Post-Entry Dwell Time")
    ax.set_xlabel("Target")
    ax.set_ylabel("Frames inside after first entry")
    ax.tick_params(axis="x", rotation=20)
 
    plt.tight_layout()
    plt.show()
 
    return cumul_df, pref_df, dwell_df

# ---------------------------------------------------------------------------
# 12. Behaviour summary — current occupancy variant
# ---------------------------------------------------------------------------
 
def behavior_summary_current_occupancy(
    df: pd.DataFrame,
    bin_size: int = 100,
    target_x: float = config.TARGET_X,
    target_y: float = config.TARGET_Y,
    radius: float = config.SUCCESS_RADIUS,
    zone_bounds: tuple = (10.0, 20.0),
    display_labels: dict = None,
    palette_override: dict = None,
    condition_order=None,
    save_path: str = None,   # e.g. "figures/fig1.pdf"
):
    """
    Three-panel behavioural summary:

        Panel A — Preference index over time (± SEM across trials, pointplot
                  at every bin_size-th frame).
        Panel B — Current occupancy: proportion of individuals inside the
                  success radius, plotted as a pointplot at every bin_size-th
                  frame (± SEM across trials), enabling direct statistical
                  comparison across conditions at each sampled timepoint.
        Panel C — Post-first-entry dwell time (box + strip).

    All conditions in df are plotted together, coloured by the standard
    palette from config.build_palette().

    Parameters
    ----------
    bin_size : int
        Frame bin width used for both preference index and occupancy sampling.
    display_labels : dict or None
        Optional {condition_name: short_label} for axis tick labels.

    Returns
    -------
    occupancy_df, pref_df, dwell_df
    """
    _apply_style()
    colors, order = config.build_palette(df)
    
    if condition_order is not None:
        observed = list(df[config.COL_CONDITION].unique())

        order = (
            [c for c in condition_order if c in observed] +
            [c for c in observed if c not in condition_order]
        )

    if palette_override is not None:
        colors = {
            cond: palette_override.get(cond, colors.get(cond))
            for cond in order
        }

    occupancy_df = metrics.current_occupancy(
        df,
        target_x=target_x, target_y=target_y,
        radius=radius, bin_size=bin_size,
    )
    pref_df  = metrics.preference_index(
        df, bin_size=bin_size, zone_bounds=zone_bounds,
    )
    dwell_df = metrics.dwell_time(
        df, target_x=target_x, target_y=target_y, radius=radius,
    )

    def _label(cond):
        return display_labels[cond] if display_labels and cond in display_labels else cond

    label_order = [_label(c) for c in order]

    for frame in (occupancy_df, pref_df, dwell_df):
        frame["PlotLabel"] = frame[config.COL_CONDITION].map(_label)

    # Sample occupancy at every 100th frame to match preference index cadence
    sampled_occupancy_df = occupancy_df[occupancy_df["FrameBin"] % 100 == 0].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # --- Panel A: Preference index ---
    sns.pointplot(
        data=pref_df,
        x="FrameBin",
        y="PreferenceIndex",
        hue="PlotLabel",
        hue_order=label_order,
        palette=colors,
        errorbar="se",
        dodge=True,
        ax=axes[0],
    )
    axes[0].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[0].set_ylim(-0.6, 1)
    axes[0].set_title("Preference Index Over Time")
    axes[0].set_ylabel("Preference Index (Z1 − Z3)")
    axes[0].set_xlabel("Frame Bin")
    axes[0].legend(title="Condition", bbox_to_anchor=(1.05, 1))

    # --- Panel B: Current occupancy (pointplot, trial-averaged ± SEM) ---
    sns.pointplot(
        data=occupancy_df,
        x="FrameBin",
        y="Occupancy",           # renamed from MeanOccupancy
        hue="PlotLabel",
        hue_order=label_order,
        palette=colors,
        errorbar="se",
        dodge=True,
        ax=axes[1],
    )
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Current Occupancy")
    axes[1].set_xlabel("Frame Bin")
    axes[1].set_ylabel(f"Proportion inside radius {radius} cm (± SEM)")
    axes[1].legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # --- Panel C: Dwell time ---
    ax = axes[2]
    if not dwell_df.empty:
        sns.boxplot(
            data=dwell_df, x="PlotLabel", y="DwellTime",
            order=label_order, palette=colors,
            showfliers=False, ax=ax,
        )
        sns.stripplot(
            data=dwell_df, x="PlotLabel", y="DwellTime",
            order=label_order, palette=colors,
            jitter=True, size=5, alpha=0.7,
            edgecolor="black", linewidth=0.5, ax=ax,
        )
    ax.set_title("Post-Success Dwell Time")
    ax.set_ylabel("Frames inside after first entry")
    ax.set_xlabel("Condition")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

    return occupancy_df, pref_df, dwell_df
 # ---------------------------------------------------------------------------
# 13. Behaviour summary — directional null + current occupancy (per condition)
# ---------------------------------------------------------------------------
 
def behavior_summary_directional_occupancy(
    df: pd.DataFrame,
    condition: str,
    targets: dict      = None,
    radius: float      = config.SUCCESS_RADIUS,
    bin_size: int      = 100,
    zone_width: float  = 10.0,
    save_path: str = None,   # e.g. "figures/fig1.pdf"
):
    """
    Three-panel behavioural summary for one condition, combining the
    directional null comparison with current occupancy:
 
        Panel A — Current occupancy: proportion of individuals inside
                  each target's radius per frame bin (± SEM across frames).
                  True target solid, null edges dashed.
        Panel B — Preference index toward each target over time (± SEM).
                  True target solid, null edges dashed.
        Panel C — Post-first-entry dwell time for each target (box + strip).
 
    Parameters
    ----------
    condition : str
        Must match a value in df['Condition'].
    targets : dict or None
        {label: (x, y)} mapping. Defaults to metrics.DIRECTIONAL_TARGETS.
    radius : float
        Radius in cm considered 'at target'. Default config.SUCCESS_RADIUS.
    bin_size : int
        Frame bin width in frames. Default 100.
    zone_width : float
        Width of near/far strips for PI calculation. Default 10 cm.
 
    Returns
    -------
    occupancy_df, pref_df, dwell_df
    """
    _apply_style()
 
    if targets is None:
        targets = metrics.DIRECTIONAL_TARGETS
 
    # --- Compute all three metrics ---
    occupancy_df = metrics.current_occupancy_directional(
        df, condition=condition, targets=targets,
        radius=radius, bin_size=bin_size,
    )
    pref_df = metrics.preference_index_directional(
        df, condition=condition, targets=targets,
        bin_size=bin_size, zone_width=zone_width,
    )
    dwell_df = metrics.dwell_time_directional(
        df, condition=condition, targets=targets, radius=radius,
    )
 
    # --- Shared style map ---
    null_labels  = [l for l in targets if l != "Target (odour)"]
    null_colors  = ["#7fb3c8", "#a0b8a0", "#c4a882"]
    target_order = ["Target (odour)"] + null_labels
 
    style_map = {
        "Target (odour)": {
            "color":     "#1a6e8e",
            "linestyle": "-",
            "linewidth": 2.5,
            "zorder":    3,
        },
    }
    for label, color in zip(null_labels, null_colors):
        style_map[label] = {
            "color":     color,
            "linestyle": "--",
            "linewidth": 1.5,
            "zorder":    2,
        }
 
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(condition, fontsize=13, y=1.01)
 
    # --- Panel A: Current occupancy ---
    ax = axes[0]
    for label in target_order:
        sub = occupancy_df[occupancy_df["Target"] == label].sort_values("FrameBin")
        s   = style_map[label]
        ax.plot(
            sub["FrameBin"], sub["MeanOccupancy"],
            label=label, color=s["color"],
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            zorder=s["zorder"],
        )
        ax.fill_between(
            sub["FrameBin"],
            sub["MeanOccupancy"] - sub["SEM"],
            sub["MeanOccupancy"] + sub["SEM"],
            color=s["color"], alpha=0.15, zorder=s["zorder"] - 1,
        )
    ax.set_ylim(0, 1)
    ax.set_title("Current Occupancy")
    ax.set_xlabel("Frame")
    ax.set_ylabel(f"Proportion inside radius {radius} cm (± SEM)")
    ax.legend(title="Target", frameon=True, fontsize=9)
 
    # --- Panel B: Preference index ---
    ax = axes[1]
    pref_agg = (
        pref_df.groupby(["Target", "FrameBin"])["PreferenceIndex"]
        .agg(Mean="mean", SEM=lambda x: x.sem())
        .reset_index()
    )
    for label in target_order:
        sub = pref_agg[pref_agg["Target"] == label].sort_values("FrameBin")
        s   = style_map[label]
        ax.plot(
            sub["FrameBin"], sub["Mean"],
            label=label, color=s["color"],
            linestyle=s["linestyle"], linewidth=s["linewidth"],
            zorder=s["zorder"],
        )
        ax.fill_between(
            sub["FrameBin"],
            sub["Mean"] - sub["SEM"],
            sub["Mean"] + sub["SEM"],
            color=s["color"], alpha=0.15, zorder=s["zorder"] - 1,
        )
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.set_title("Preference Index Over Time")
    ax.set_xlabel("Frame Bin")
    ax.set_ylabel("Preference Index (near − far)")
    ax.legend(title="Target", frameon=True, fontsize=9)
 
    # --- Panel C: Dwell time ---
    ax = axes[2]
    box_palette = {label: style_map[label]["color"] for label in target_order}
 
    sns.boxplot(
        data=dwell_df,
        x="Target", y="DwellTime",
        order=target_order,
        palette=box_palette,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=dwell_df,
        x="Target", y="DwellTime",
        order=target_order,
        palette=box_palette,
        jitter=True, size=4, alpha=0.6,
        edgecolor="black", linewidth=0.4,
        ax=ax,
    )
    for i, patch in enumerate(ax.patches):
        if i > 0:
            patch.set_alpha(0.45)
 
    ax.set_title("Post-Entry Dwell Time")
    ax.set_xlabel("Target")
    ax.set_ylabel("Frames inside after first entry")
    ax.tick_params(axis="x", rotation=20)
 
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
 
    return occupancy_df, pref_df, dwell_df
 