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
    zone_bounds: tuple = (12.0, 20.0),
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

def trajectory_heatmaps_with_marginals(
    df,
    condition,
    frame_bin_size=100,
    grid_size=1,
    x_range=(0, 30),
    y_range=(0, 30),
    marginal_ratio=0.22,     # marginal size, relative to ax_main's rendered box
    cbar_ratio=0.12,         # colorbar width, relative to ax_main's rendered box
    marginal_kind="line",
    marginal_color="steelblue",
    panel_gap=0.008,         # figure-fraction gap between a heatmap and its OWN marginals
    bin_spacing=0.35,        # extra breathing room BETWEEN bins (fraction of axis width) —
                             # this is what stops one bin's y-marginal hitting the next bin
    right_margin=0.16,       # figure-fraction reserved on the right for the last y-marginal + colorbar
    cbar_gap=0.03,
    title_size=18,
    label_size=14,
    tick_size=12,
    cbar_size=14,
    condition_col="Condition",
    frame_col="Frame",
    x_col="X",
    y_col="Y",
    save_path=None,
):
    df = df[df[condition_col] == condition].copy()

    min_frame = df[frame_col].min()
    max_frame = df[frame_col].max()
    frame_bins = np.arange(min_frame, max_frame + frame_bin_size, frame_bin_size)
    num_bins = len(frame_bins) - 1

    x_lo, x_hi = x_range
    y_lo, y_hi = y_range
    x_edges = np.arange(x_lo, x_hi + grid_size, grid_size)
    y_edges = np.arange(y_lo, y_hi + grid_size, grid_size)

    fig = plt.figure(figsize=(5.5 * num_bins, 7.5))

    # `wspace` reserves space BETWEEN GridSpec columns as a fraction of the
    # average axis width — exactly the unit we need, since every ax_main
    # will render to the same size (same data range -> same square).
    # `right` reserves a fixed strip on the right of the whole figure for
    # the LAST bin's y-marginal + colorbar, which live outside the grid.
    gs = gridspec.GridSpec(
        nrows=1, ncols=num_bins, figure=fig,
        left=0.06, right=1 - right_margin, top=0.88, bottom=0.14,
        wspace=marginal_ratio + bin_spacing,
    )

    cax = None
    panels = []

    for i in range(num_bins):
        bin_start, bin_end = frame_bins[i], frame_bins[i + 1]
        df_bin = df[(df[frame_col] >= bin_start) & (df[frame_col] < bin_end)]

        ax_main = fig.add_subplot(gs[0, i])
        ax_xmarg = fig.add_axes([0, 0, 0.1, 0.1], sharex=ax_main)
        ax_ymarg = fig.add_axes([0, 0, 0.1, 0.1], sharey=ax_main)

        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylim(y_lo, y_hi)
        ax_main.autoscale(enable=False)
        ax_xmarg.autoscale(enable=False, axis="x")
        ax_ymarg.autoscale(enable=False, axis="y")

        hist, xedges, yedges = np.histogram2d(
            df_bin[x_col], df_bin[y_col], bins=[x_edges, y_edges]
        )
        cax = ax_main.pcolormesh(
            xedges, yedges, hist.T,
            cmap="viridis", shading="auto", norm=LogNorm(vmin=1),
        )
        ax_main.set_title(f"Frames {bin_start}-{bin_end}", fontsize=title_size)
        if i == 0:
            ax_main.set_ylabel("Y-coordinate (cm)", fontsize=label_size)
        ax_main.tick_params(bottom=False, labelbottom=False, labelsize=tick_size)
        ax_main.set_aspect("equal")
        ax_main.grid(True, alpha=0.3)

        x_counts, _ = np.histogram(df_bin[x_col], bins=x_edges)
        x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
        if marginal_kind == "bar":
            ax_xmarg.bar(x_centers, x_counts, width=grid_size, color=marginal_color)
        else:
            ax_xmarg.plot(x_centers, x_counts, color=marginal_color, lw=1.5)
            ax_xmarg.fill_between(x_centers, x_counts, color=marginal_color, alpha=0.3)
        ax_xmarg.set_ylim(bottom=0)
        ax_xmarg.margins(y=0.08)
        ax_xmarg.invert_yaxis()
        ax_xmarg.set_xlabel("X-coordinate (cm)", fontsize=label_size)
        ax_xmarg.tick_params(labelsize=tick_size - 2)
        if i == 0:
            ax_xmarg.set_ylabel("Freq.", fontsize=label_size - 2)
        else:
            ax_xmarg.tick_params(labelleft=False)
        ax_xmarg.spines[["top", "right"]].set_visible(False)

        y_counts, _ = np.histogram(df_bin[y_col], bins=y_edges)
        y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
        if marginal_kind == "bar":
            ax_ymarg.barh(y_centers, y_counts, height=grid_size, color=marginal_color)
        else:
            ax_ymarg.plot(y_counts, y_centers, color=marginal_color, lw=1.5)
            ax_ymarg.fill_betweenx(y_centers, y_counts, color=marginal_color, alpha=0.3)
        ax_ymarg.set_xlim(left=0)
        ax_ymarg.margins(x=0.08)
        ax_ymarg.tick_params(labelsize=tick_size - 2)
        ax_ymarg.spines[["top", "right"]].set_visible(False)

        is_last = (i == num_bins - 1)
        if is_last:
            ax_ymarg.yaxis.tick_right()
            ax_ymarg.yaxis.set_label_position("right")
            ax_ymarg.set_ylabel("Y-coordinate (cm)", fontsize=label_size - 2)
            ax_ymarg.set_xlabel("Freq.", fontsize=label_size - 2)
        else:
            ax_ymarg.tick_params(labelleft=False, labelright=False)

        ax_main.set_xlim(x_lo, x_hi)
        ax_main.set_ylim(y_lo, y_hi)
        ax_xmarg.set_xlim(x_lo, x_hi)
        ax_ymarg.set_ylim(y_lo, y_hi)

        panels.append((ax_main, ax_xmarg, ax_ymarg))

    fig.suptitle(f"Trajectory Heatmaps + Marginals - Condition: {condition}",
                 fontsize=title_size + 4)

    # ---- Render once, then snap every marginal to its own ax_main's true box ----
    fig.canvas.draw()
    for ax_main, ax_xmarg, ax_ymarg in panels:
        pos = ax_main.get_position()
        xmarg_h = marginal_ratio * pos.height
        ymarg_w = marginal_ratio * pos.width
        ax_xmarg.set_position([pos.x0, pos.y0 - panel_gap - xmarg_h, pos.width, xmarg_h])
        ax_ymarg.set_position([pos.x1 + panel_gap, pos.y0, ymarg_w, pos.height])

    # ---- Colorbar: anchored past the last y-marginal's true right edge,
    # inside the `right_margin` strip we reserved above.
    fig.canvas.draw()
    last_main_pos = panels[-1][0].get_position()
    last_ymarg_pos = panels[-1][2].get_position()

    cbar_x0 = last_ymarg_pos.x1 + cbar_gap
    cbar_w = cbar_ratio * last_main_pos.width
    cbar_ax = fig.add_axes([cbar_x0, last_main_pos.y0, cbar_w, last_main_pos.height])
    cbar = fig.colorbar(cax, cax=cbar_ax, orientation="vertical")
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label("Density (log scale)", fontsize=cbar_size)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def safe_skew(vals, min_points=5, var_eps=1e-8, bias=False):
    """
    Skewness that treats near-constant data as zero skew instead of NaN.

    stats.skew() computes m3 / m2**1.5; when the sample has (near) zero
    variance — e.g. a larva has settled at the target and stopped moving
    for that bin — this is a 0/0 division that returns NaN even though
    there's nothing wrong with the data. A constant sample has, by
    definition, no asymmetry to report, so we return 0.0 explicitly
    rather than letting NaN propagate into downstream aggregates.
    """
    import scipy.stats as stats
    vals = np.asarray(vals, dtype=float)
    vals = vals[~np.isnan(vals)]
    n = len(vals)
    if n < min_points:
        return np.nan, n, False  # genuinely insufficient data — this SHOULD stay NaN
    std = vals.std(ddof=0)
    is_constant = std < var_eps
    if is_constant:
        return 0.0, n, True
    return stats.skew(vals, bias=bias), n, False


def compute_bin_skewness_comparison(
    df,
    condition_a,
    condition_b,
    axis_col,
    bin_size=100,
    frame_col='Frame',
    trial_col='Trial',
    condition_col='Condition',
    min_points_per_trial=5,
    var_eps=1e-8,
    n_boot=2000,
    random_state=0
):
    import scipy.stats as stats
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df['Bin'] = (df[frame_col] // bin_size) * bin_size

    def safe_skew(vals, min_points=5, var_eps=1e-8, bias=False):
        """
        Skewness that treats near-constant data as zero skew instead of NaN.

        stats.skew() computes m3 / m2**1.5; when the sample has (near) zero
        variance — e.g. a larva has settled at the target and stopped moving
        for that bin — this is a 0/0 division that returns NaN even though
        there's nothing wrong with the data. A constant sample has, by
        definition, no asymmetry to report, so we return 0.0 explicitly
        rather than letting NaN propagate into downstream aggregates.
        """
        import scipy.stats as stats
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]
        n = len(vals)
        if n < min_points:
            return np.nan, n, False  # genuinely insufficient data — this SHOULD stay NaN
        std = vals.std(ddof=0)
        is_constant = std < var_eps
        if is_constant:
            return 0.0, n, True
        return stats.skew(vals, bias=bias), n, False

    def trial_skews(cond):
        sub = df[df[condition_col] == cond]
        rows = []
        for (b, trial), g in sub.groupby(['Bin', trial_col]):
            vals = g[axis_col].dropna().values
            skew_val, n, is_const = safe_skew(vals, min_points=min_points_per_trial, var_eps=var_eps)
            if np.isnan(skew_val):
                continue  # only dropped for genuinely too-few points, not for zero variance
            rows.append({'Bin': b, trial_col: trial, 'Skew': skew_val,
                         'N': n, 'IsConstant': is_const})
        return pd.DataFrame(rows)

    skew_a = trial_skews(condition_a)
    skew_b = trial_skews(condition_b)

    def bootstrap_ci(vals, ci=95):
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        boots = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
        lo, hi = np.nanpercentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        return float(np.nanmean(vals)), float(lo), float(hi)

    def cliffs_delta(x, y):
        x, y = np.asarray(x), np.asarray(y)
        gt = (x[:, None] > y[None, :]).sum()
        lt = (x[:, None] < y[None, :]).sum()
        return (gt - lt) / (len(x) * len(y))

    results = []
    all_bins = sorted(set(skew_a['Bin']).union(skew_b['Bin']))
    for b in all_bins:
        vals_a = skew_a.loc[skew_a['Bin'] == b, 'Skew'].values
        vals_b = skew_b.loc[skew_b['Bin'] == b, 'Skew'].values
        n_const_a = skew_a.loc[skew_a['Bin'] == b, 'IsConstant'].sum()
        n_const_b = skew_b.loc[skew_b['Bin'] == b, 'IsConstant'].sum()

        mean_a, lo_a, hi_a = bootstrap_ci(vals_a)
        mean_b, lo_b, hi_b = bootstrap_ci(vals_b)

        if len(vals_a) >= 2 and len(vals_b) >= 2:
            u_stat, p_val = stats.mannwhitneyu(vals_a, vals_b, alternative='two-sided')
            delta = cliffs_delta(vals_a, vals_b)
        else:
            u_stat, p_val, delta = np.nan, np.nan, np.nan

        results.append({
            'Bin': b, 'n_trials_a': len(vals_a), 'n_trials_b': len(vals_b),
            'n_constant_a': int(n_const_a), 'n_constant_b': int(n_const_b),
            'skew_a': mean_a, 'skew_a_lo': lo_a, 'skew_a_hi': hi_a,
            'skew_b': mean_b, 'skew_b_lo': lo_b, 'skew_b_hi': hi_b,
            'u_stat': u_stat, 'p_value': p_val, 'cliffs_delta': delta,
        })

    return pd.DataFrame(results)


def plot_marginal_comparison_ridgeline(
    df,
    condition_a,
    condition_b,
    axis_col='X',
    bin_size=100,
    frame_col='Frame',
    trial_col='Trial',
    condition_col='Condition',
    target_value=None,        # e.g. 15 for X — draws a reference line
    color_a='#d95f02',
    color_b='#1b9e77',
    ridge_overlap=0.6,
    n_boot=2000,
    stats_df=None,            # pass a precomputed table to avoid recomputing
    title=None,
):
    """
    Ridgeline comparison of an axis marginal between two conditions, one
    ridge row per frame bin. Both conditions' full KDEs are drawn — nothing
    is reduced to an average — with trial-level skewness, Cliff's delta, and
    a Mann-Whitney significance marker annotated per row.
    """
    if stats_df is None:
        stats_df = compute_bin_skewness_comparison(
            df, condition_a, condition_b, axis_col=axis_col,
            bin_size=bin_size, frame_col=frame_col, trial_col=trial_col,
            condition_col=condition_col, n_boot=n_boot,
        )

    import scipy.stats as stats

    df = df.copy()
    df['Bin'] = (df[frame_col] // bin_size) * bin_size
    bins = sorted(stats_df['Bin'].unique())

    xmin, xmax = df[axis_col].min(), df[axis_col].max()
    grid = np.linspace(xmin, xmax, 400)

    fig, ax = plt.subplots(figsize=(10, 1.1 * len(bins) + 1))

    for row_idx, b in enumerate(bins):
        y0 = row_idx * ridge_overlap
        for cond, color in [(condition_a, color_a), (condition_b, color_b)]:
            vals = df[(df['Bin'] == b) & (df[condition_col] == cond)][axis_col].dropna().values
            if len(vals) < 5:
                continue
            kde = stats.gaussian_kde(vals)
            density = kde(grid)
            density = density / density.max() * ridge_overlap * 0.9
            ax.fill_between(grid, y0, y0 + density, color=color, alpha=0.45, lw=0)
            ax.plot(grid, y0 + density, color=color, lw=1.2)

        row = stats_df[stats_df['Bin'] == b].iloc[0]
        p = row['p_value']
        star = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns') \
            if not np.isnan(p) else 'n/a'
        annot = (f"skew {row['skew_a']:.2f} vs {row['skew_b']:.2f}  "
                 f"(δ={row['cliffs_delta']:.2f}, {star})")
        ax.text(xmax, y0 + ridge_overlap * 0.5, annot, va='center', ha='left', fontsize=9)
        ax.text(xmin, y0 + ridge_overlap * 0.5, f"{int(b)}", va='center', ha='right',
                fontsize=10, fontweight='bold')

    if target_value is not None:
        ax.axvline(target_value, color='gray', linestyle='--', lw=1, zorder=0)

    ax.set_yticks([])
    ax.set_xlabel(f"{axis_col}-coordinate (cm)")
    ax.set_xlim(xmin, xmax + (xmax - xmin) * 0.45)  # room for annotations
    ax.spines[['top', 'right', 'left']].set_visible(False)

    handles = [
        plt.Line2D([0], [0], color=color_a, lw=6, alpha=0.6, label=condition_a),
        plt.Line2D([0], [0], color=color_b, lw=6, alpha=0.6, label=condition_b),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False)
    fig.suptitle(title or f"{axis_col}-marginal, {condition_a} vs {condition_b} (bin = frame)")
    plt.tight_layout()
    plt.show()

    return stats_df


def plot_skew_trajectory(stats_df, axis_col='X', condition_a='A', condition_b='B',
                          color_a='#d95f02', color_b='#1b9e77'):
    """
    Compact companion plot: skewness (± bootstrap CI) vs frame bin for both
    conditions on one axis — good for seeing whether directedness (skew
    toward the target) builds up over time, rather than reading it bin by
    bin off the ridgeline.
    """

    import scipy.stats as stats

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color, lo_col, hi_col, mean_col in [
        (condition_a, color_a, 'skew_a_lo', 'skew_a_hi', 'skew_a'),
        (condition_b, color_b, 'skew_b_lo', 'skew_b_hi', 'skew_b'),
    ]:
        ax.plot(stats_df['Bin'], stats_df[mean_col], color=color, marker='o', label=label)
        ax.fill_between(stats_df['Bin'], stats_df[lo_col], stats_df[hi_col],
                         color=color, alpha=0.2)

    sig = stats_df[stats_df['p_value'] < 0.05]
    if not sig.empty:
        ax.scatter(sig['Bin'], [ax.get_ylim()[1] * 0.95] * len(sig),
                   marker='*', color='black', s=60, label='p < 0.05')

    ax.axhline(0, color='gray', linestyle='--', lw=1)
    ax.set_xlabel('Frame bin')
    ax.set_ylabel(f'{axis_col} skewness (mean over trials ± 95% CI)')
    ax.set_title(f'{axis_col} skewness over time: {condition_a} vs {condition_b}')
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt

def compute_bin_distance_comparison(
    df,
    condition_a,
    condition_b,
    target_x=14,
    target_y=2,
    distance_type='euclidean',   # 'euclidean', 'x' (|X - target_x|), or 'y' (|Y - target_y|)
    stat='median',               # which per-trial summary drives the significance test: 'mean' or 'median'
    bin_size=100,
    frame_col='Frame',
    x_col='X',
    y_col='Y',
    trial_col='Trial',
    condition_col='Condition',
    min_points_per_trial=5,
    n_boot=2000,
    random_state=0
):
    """
    Per frame bin, compares distance-from-target between two conditions,
    using TRIAL as the unit of replication (never pooled raw frames, for
    the same autocorrelation reason as the skew comparison).

    Returns one row per bin with:
        mean_dist_a/b, median_dist_a/b   : trial-level summaries (+ bootstrap CI on `stat`)
        u_stat, p_value                  : Mann-Whitney U on trial-level `stat` values
        cliffs_delta                     : effect size, range [-1, 1]
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    df['Bin'] = (df[frame_col] // bin_size) * bin_size

    if distance_type == 'euclidean':
        df['_dist'] = np.sqrt((df[x_col] - target_x) ** 2 + (df[y_col] - target_y) ** 2)
    elif distance_type == 'x':
        df['_dist'] = np.abs(df[x_col] - target_x)
    elif distance_type == 'y':
        df['_dist'] = np.abs(df[y_col] - target_y)
    else:
        raise ValueError("distance_type must be 'euclidean', 'x', or 'y'")

    def trial_distances(cond):
        sub = df[df[condition_col] == cond]
        rows = []
        for (b, trial), g in sub.groupby(['Bin', trial_col]):
            vals = g['_dist'].dropna().values
            if len(vals) < min_points_per_trial:
                continue
            rows.append({
                'Bin': b, trial_col: trial,
                'MeanDist': np.mean(vals), 'MedianDist': np.median(vals), 'N': len(vals)
            })
        return pd.DataFrame(rows)

    dist_a = trial_distances(condition_a)
    dist_b = trial_distances(condition_b)
    stat_col = 'MeanDist' if stat == 'mean' else 'MedianDist'

    def bootstrap_ci(vals, agg_fn, ci=95):
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        boots = np.array([
            agg_fn(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)
        ])
        lo, hi = np.percentile(boots, [(100 - ci) / 2, 100 - (100 - ci) / 2])
        return float(agg_fn(vals)), float(lo), float(hi)

    def cliffs_delta(x, y):
        x, y = np.asarray(x), np.asarray(y)
        gt = (x[:, None] > y[None, :]).sum()
        lt = (x[:, None] < y[None, :]).sum()
        return (gt - lt) / (len(x) * len(y))

    results = []
    all_bins = sorted(set(dist_a['Bin']).union(dist_b['Bin']))
    for b in all_bins:
        rows_a = dist_a[dist_a['Bin'] == b]
        rows_b = dist_b[dist_b['Bin'] == b]

        mean_a, mean_lo_a, mean_hi_a = bootstrap_ci(rows_a['MeanDist'].values, np.mean)
        mean_b, mean_lo_b, mean_hi_b = bootstrap_ci(rows_b['MeanDist'].values, np.mean)
        med_a, med_lo_a, med_hi_a = bootstrap_ci(rows_a['MedianDist'].values, np.median)
        med_b, med_lo_b, med_hi_b = bootstrap_ci(rows_b['MedianDist'].values, np.median)

        vals_a_stat = rows_a[stat_col].values
        vals_b_stat = rows_b[stat_col].values
        if len(vals_a_stat) >= 2 and len(vals_b_stat) >= 2:
            u_stat, p_val = stats.mannwhitneyu(vals_a_stat, vals_b_stat, alternative='two-sided')
            delta = cliffs_delta(vals_a_stat, vals_b_stat)
        else:
            u_stat, p_val, delta = np.nan, np.nan, np.nan

        results.append({
            'Bin': b, 'n_trials_a': len(rows_a), 'n_trials_b': len(rows_b),
            'mean_dist_a': mean_a, 'mean_dist_a_lo': mean_lo_a, 'mean_dist_a_hi': mean_hi_a,
            'mean_dist_b': mean_b, 'mean_dist_b_lo': mean_lo_b, 'mean_dist_b_hi': mean_hi_b,
            'median_dist_a': med_a, 'median_dist_a_lo': med_lo_a, 'median_dist_a_hi': med_hi_a,
            'median_dist_b': med_b, 'median_dist_b_lo': med_lo_b, 'median_dist_b_hi': med_hi_b,
            'u_stat': u_stat, 'p_value': p_val, 'cliffs_delta': delta,
        })

    trial_level = pd.concat([
        dist_a.assign(**{condition_col: condition_a}),
        dist_b.assign(**{condition_col: condition_b}),
    ], ignore_index=True)

    return pd.DataFrame(results), trial_level


def plot_distance_trajectory(
    stats_df,
    condition_a='A', condition_b='B',
    stat='median',                 # 'mean' or 'median' — which summary line to draw
    color_a='#d95f02', color_b='#1b9e77',
    title=None,
):
    """
    Distance-from-target (± bootstrap CI) vs frame bin for both conditions,
    with Mann-Whitney significance markers per bin. Analogous to
    plot_skew_trajectory but for distance.
    """
    prefix = 'mean_dist' if stat == 'mean' else 'median_dist'

    fig, ax = plt.subplots(figsize=(9, 5))
    for label, color in [(condition_a, color_a), (condition_b, color_b)]:
        suffix = 'a' if label == condition_a else 'b'
        ax.plot(stats_df['Bin'], stats_df[f'{prefix}_{suffix}'],
                color=color, marker='o', label=label)
        ax.fill_between(stats_df['Bin'], stats_df[f'{prefix}_{suffix}_lo'],
                         stats_df[f'{prefix}_{suffix}_hi'], color=color, alpha=0.2)

    sig = stats_df[stats_df['p_value'] < 0.05]
    if not sig.empty:
        y_marker = ax.get_ylim()[1] * 0.97
        ax.scatter(sig['Bin'], [y_marker] * len(sig),
                   marker='*', color='black', s=70, label='p < 0.05', zorder=5)

    ax.set_xlabel('Frame bin')
    ax.set_ylabel(f'{stat.capitalize()} distance from target (cm, ± 95% CI)')
    ax.set_title(title or f'Distance from target over time: {condition_a} vs {condition_b}')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_distance_boxplots_over_time(
    trial_level_df,
    condition_a='A', condition_b='B',
    stat='median',                  # which per-trial column to plot: 'mean' or 'median'
    condition_col='Condition',
    color_a='#d95f02', color_b='#1b9e77',
    title=None,
):
    """
    Per-trial distance-from-target as boxplot + stripplot, grouped by frame
    bin and condition. This is the dimensionality-preserving companion to
    plot_distance_trajectory: rather than collapsing to a single line ± CI,
    every trial's summary value is a visible point, so you can see spread,
    outliers, and bimodality (e.g. some larvae reaching target, others not)
    that a mean/CI band would hide.
    """
    value_col = 'MeanDist' if stat == 'mean' else 'MedianDist'
    palette = {condition_a: color_a, condition_b: color_b}

    fig, ax = plt.subplots(figsize=(max(10, trial_level_df['Bin'].nunique() * 1.2), 6))
    sns.boxplot(
        data=trial_level_df, x='Bin', y=value_col, hue=condition_col,
        palette=palette, showfliers=False, ax=ax
    )
    sns.stripplot(
        data=trial_level_df, x='Bin', y=value_col, hue=condition_col,
        palette=palette, dodge=True, jitter=True, size=4, alpha=0.6,
        edgecolor='black', linewidth=0.3, ax=ax, legend=False
    )

    ax.set_xlabel('Frame bin')
    ax.set_ylabel(f'Per-trial {stat} distance from target (cm)')
    ax.set_title(title or f'Per-trial distance distributions: {condition_a} vs {condition_b}')
    ax.tick_params(axis='x', rotation=45)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_individual_odour_pi(
    df,
    condition,
    bin_size=100,
    zone_width=10.0,
    arena_height=30.0,
    frame_col='Frame',
    individual_col='Individual',
    y_col='Y',
    condition_col='Condition',
    collective_col='Collective',
    collective_filter=None,   # e.g. 'Single' to restrict the pool, 'Group' for the real condition
):
    """
    Per-individual, per-bin Preference Index toward the odour target only.

    near = Y <= zone_width          (close to the odour, bottom edge)
    far  = Y >= arena_height - zone_width   (far strip, top edge)
    PI = (n_near - n_far) / (n_near + n_far)

    Purely temporal: each individual's PI comes from their own frames,
    with no reference to other individuals.
    """
    sub = df[df[condition_col] == condition].copy()
    if collective_filter is not None:
        sub = sub[sub[collective_col] == collective_filter]
    if sub.empty:
        raise ValueError(f"No rows found for condition '{condition}'"
                          f"{' / Collective=' + collective_filter if collective_filter else ''}.")

    sub["FrameBin"] = (sub[frame_col] // bin_size) * bin_size

    records = []
    for ind, ind_df in sub.groupby(individual_col):
        ind_df = ind_df.sort_values(frame_col)
        for bin_id, bin_df in ind_df.groupby("FrameBin"):
            n_near = (bin_df[y_col] <= zone_width).sum()
            n_far = (bin_df[y_col] >= (arena_height - zone_width)).sum()
            denom = n_near + n_far
            pi = np.nan if denom == 0 else (n_near - n_far) / denom
            records.append({
                individual_col: ind, "FrameBin": bin_id, "PreferenceIndex": pi,
            })

    return pd.DataFrame(records)


def bootstrap_pseudogroup_odour_pi(
    ind_pi_df,
    n_virtual_individuals=15,
    n_boot=500,
    individual_col='Individual',
    random_state=0,
):
    """
    Bootstraps over WHICH individuals get averaged together per FrameBin,
    reusing each individual's already-computed temporal PI value -- same
    logic as before, just without the Target grouping.
    """
    rng = np.random.default_rng(random_state)
    unique_inds = ind_pi_df[individual_col].unique()
    if len(unique_inds) == 0:
        raise ValueError("No individuals found in ind_pi_df.")

    records = []
    for b in range(n_boot):
        drawn = rng.choice(unique_inds, size=n_virtual_individuals, replace=True)

        pieces = []
        for virtual_id, real_ind in enumerate(drawn):
            piece = ind_pi_df[ind_pi_df[individual_col] == real_ind].copy()
            piece['VirtualIndividual'] = virtual_id
            pieces.append(piece)
        cohort = pd.concat(pieces, ignore_index=True)

        bin_means = cohort.groupby("FrameBin")["PreferenceIndex"].mean().reset_index()
        bin_means['BootstrapID'] = b
        records.append(bin_means)

    return pd.concat(records, ignore_index=True)


def summarize_pseudogroup_vs_real_odour(pseudogroup_df, real_pi_df, ci=95):
    """
    Collapses bootstrap draws into mean ± CI per FrameBin, next to the
    real Group PI's mean ± SEM across its actual individuals.
    """
    alpha = 100 - ci
    lo_q, hi_q = alpha / 2, 100 - alpha / 2

    pseudo_summary = (
        pseudogroup_df.groupby("FrameBin")["PreferenceIndex"]
        .agg(Mean="mean",
             CI_lo=lambda x: np.nanpercentile(x, lo_q),
             CI_hi=lambda x: np.nanpercentile(x, hi_q))
        .reset_index()
    )
    pseudo_summary['Source'] = 'Pseudo-Group (bootstrapped Singles)'

    real_summary = (
        real_pi_df.groupby("FrameBin")["PreferenceIndex"]
        .agg(Mean="mean", SEM=lambda x: x.sem())
        .reset_index()
    )
    real_summary['CI_lo'] = real_summary['Mean'] - real_summary['SEM']
    real_summary['CI_hi'] = real_summary['Mean'] + real_summary['SEM']
    real_summary['Source'] = 'Real Group'

    cols = ["FrameBin", "Mean", "CI_lo", "CI_hi", "Source"]
    return pd.concat([pseudo_summary[cols], real_summary[cols]], ignore_index=True)


def plot_pseudogroup_vs_real_odour(
    comparison_df,
    condition_label="Fed",
    real_color="#c1440e",
    pseudo_color="#555555",
    save_path=None,
):
    """
    Single-panel comparison of real Group PI vs bootstrapped pseudo-group
    PI toward the odour target only.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    style_map = {
        "Real Group": {"color": real_color, "linestyle": "-", "linewidth": 2.2, "zorder": 3},
        "Pseudo-Group (bootstrapped Singles)": {
            "color": pseudo_color, "linestyle": "--", "linewidth": 1.6, "zorder": 2
        },
    }

    for source, s in style_map.items():
        sub = comparison_df[comparison_df["Source"] == source].sort_values("FrameBin")
        if sub.empty:
            continue
        ax.plot(
            sub["FrameBin"], sub["Mean"],
            label=source, color=s["color"],
            linestyle=s["linestyle"], linewidth=s["linewidth"], zorder=s["zorder"],
        )
        ax.fill_between(
            sub["FrameBin"], sub["CI_lo"], sub["CI_hi"],
            color=s["color"], alpha=0.18, zorder=s["zorder"] - 1,
        )

    ax.axhline(0, linestyle=":", color="black", linewidth=1)
    ax.set_title(f"Odour Preference Index — Real Group vs Pseudo-Group ({condition_label})")
    ax.set_xlabel("Frame Bin")
    ax.set_ylabel("Preference Index (near − far)")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()

def run_odour_pi_comparison(
    df,
    n_boot=500,
    bin_size=100,
    zone_width=10.0,
    arena_height=30.0,
    random_state=0,
    condition_col='Condition',
    collective_col='Collective',
    individual_col='Individual',
    plot=True,
):
    """
    Runs the full Single-vs-Group odour PI comparison on a dataframe that
    contains exactly one Single condition and one Group condition. Infers
    both the condition labels and the real group size from the data --
    nothing hardcoded.
    """
    single_conds = df.loc[df[collective_col] == 'Single', condition_col].unique()
    group_conds  = df.loc[df[collective_col] == 'Group',  condition_col].unique()

    if len(single_conds) != 1 or len(group_conds) != 1:
        raise ValueError(
            f"Expected exactly one Single and one Group condition, got "
            f"Single={list(single_conds)}, Group={list(group_conds)}."
        )
    single_cond, group_cond = single_conds[0], group_conds[0]

    # Match the pseudo-group size to the REAL group's actual individual
    # count, rather than hardcoding 15 -- keeps the comparison fair even
    # if group size varies across datasets.
    n_virtual_individuals = df.loc[
        (df[condition_col] == group_cond) & (df[collective_col] == 'Group'),
        individual_col
    ].nunique()

    ind_pi_single = compute_individual_odour_pi(
        df, condition=single_cond, collective_filter='Single',
        bin_size=bin_size, zone_width=zone_width, arena_height=arena_height,
    )
    real_pi_group = compute_individual_odour_pi(
        df, condition=group_cond, collective_filter='Group',
        bin_size=bin_size, zone_width=zone_width, arena_height=arena_height,
    )
    pseudogroup_df = bootstrap_pseudogroup_odour_pi(
        ind_pi_single, n_virtual_individuals=n_virtual_individuals,
        n_boot=n_boot, random_state=random_state,
    )
    comparison_df = summarize_pseudogroup_vs_real_odour(pseudogroup_df, real_pi_group)

    if plot:
        plot_pseudogroup_vs_real_odour(comparison_df, condition_label=group_cond)

    return comparison_df