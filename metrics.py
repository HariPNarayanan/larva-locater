"""
metrics.py
----------
Pure computation functions. Every function here:
  - Takes a DataFrame and parameters
  - Returns a tidy DataFrame of results
  - Does NOT import matplotlib or produce any plots

This separation means metrics can be reused by figures.py, called
independently for statistical analysis, or tested without a display.
"""

import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Success rate
# ---------------------------------------------------------------------------

def success_rate(
    df: pd.DataFrame,
    target_x: float  = config.TARGET_X,
    target_y: float  = config.TARGET_Y,
    radius: float    = config.SUCCESS_RADIUS,
    frame_col: str   = config.COL_FRAME,
    individual_col: str = config.COL_INDIVIDUAL,
    x_col: str       = config.COL_X,
    y_col: str       = config.COL_Y,
    condition_col: str = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Compute per-individual binary success (ever entered the target radius).

    Returns
    -------
    pd.DataFrame with columns: [Condition, Individual, Success]
        Success is 1 if the individual reached the target, 0 otherwise.

    Notes
    -----
    Unsuccessful individuals are scored 0 (not dropped), so that mean()
    across a condition gives the true success rate including failures.
    This is what seaborn's estimator=np.mean + errorbar='se' needs for SEM.
    """
    records = []
    for (cond, ind), sub in df.groupby([condition_col, individual_col]):
        sub = sub.sort_values(frame_col)
        dist = np.sqrt((sub[x_col] - target_x) ** 2 + (sub[y_col] - target_y) ** 2)
        records.append({
            condition_col:  cond,
            individual_col: ind,
            "Success":      int((dist <= radius).any()),
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Cumulative success rate over time
# ---------------------------------------------------------------------------

def cumulative_success(
    df: pd.DataFrame,
    bin_size: int    = 100,
    target_x: float  = config.TARGET_X,
    target_y: float  = config.TARGET_Y,
    radius: float    = config.SUCCESS_RADIUS,
    frame_col: str   = config.COL_FRAME,
    individual_col: str = config.COL_INDIVIDUAL,
    x_col: str       = config.COL_X,
    y_col: str       = config.COL_Y,
    condition_col: str = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Compute the cumulative proportion of individuals that have reached the
    target by the end of each frame bin.

    Returns
    -------
    pd.DataFrame with columns: [Condition, FrameBin, CumulSuccess, SEM]
    """
    # First-success frame per individual
    dist = np.sqrt((df[x_col] - target_x) ** 2 + (df[y_col] - target_y) ** 2)
    df = df.copy()
    df["_inside"] = dist <= radius

    first_success = (
        df[df["_inside"]]
        .groupby([condition_col, individual_col])[frame_col]
        .min()
    )

    frame_bins = sorted({(f // bin_size) * bin_size for f in df[frame_col].unique()})

    records = []
    for cond in df[condition_col].unique():
        ind_list = df.loc[df[condition_col] == cond, individual_col].unique()
        n_total  = len(ind_list)
        fs_vals  = np.array([
            first_success.get((cond, ind), np.inf) for ind in ind_list
        ])

        for bin_start in frame_bins:
            successes = (fs_vals < bin_start + bin_size).astype(float)
            records.append({
                condition_col: cond,
                "FrameBin":    bin_start,
                "CumulSuccess": successes.mean(),
                "SEM": successes.std(ddof=0) / np.sqrt(n_total) if n_total > 1 else 0.0,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Preference index
# ---------------------------------------------------------------------------

def preference_index(
    df: pd.DataFrame,
    bin_size: int  = 100,
    y_col: str     = config.COL_Y,
    frame_col: str = config.COL_FRAME,
    individual_col: str = config.COL_INDIVIDUAL,
    condition_col: str  = config.COL_CONDITION,
    zone_bounds: tuple  = (10.0, 20.0),
) -> pd.DataFrame:
    """
    Compute the Preference Index per individual per frame bin.

    PI = (n_bottom - n_top) / (n_bottom + n_top)
    where bottom = Y <= zone_bounds[0], top = Y >= zone_bounds[1].

    A positive PI means more time near the odour source (bottom zone).
    NaN is returned for bins where both zones are empty.

    Parameters
    ----------
    zone_bounds : tuple (lower_y, upper_y)
        Y thresholds separating bottom / middle / top zones.
        Default (10, 20) matches the fixed arena thirds for a 30 cm arena.

    Returns
    -------
    pd.DataFrame with columns: [Condition, Individual, FrameBin, PreferenceIndex]
    """
    lower, upper = zone_bounds
    records = []

    for (cond, ind), sub in df.groupby([condition_col, individual_col]):
        sub = sub.sort_values(frame_col).copy()
        sub["FrameBin"] = (sub[frame_col] // bin_size) * bin_size

        for bin_id, bin_df in sub.groupby("FrameBin"):
            y_vals = bin_df[y_col]
            bottom = (y_vals <= lower).sum()
            top    = (y_vals >= upper).sum()
            denom  = bottom + top
            pi     = np.nan if denom == 0 else (bottom - top) / denom

            records.append({
                condition_col:  cond,
                individual_col: ind,
                "FrameBin":     bin_id,
                "PreferenceIndex": pi,
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Dwell time
# ---------------------------------------------------------------------------

def dwell_time(
    df: pd.DataFrame,
    target_x: float  = config.TARGET_X,
    target_y: float  = config.TARGET_Y,
    radius: float    = config.SUCCESS_RADIUS,
    frame_col: str   = config.COL_FRAME,
    individual_col: str = config.COL_INDIVIDUAL,
    x_col: str       = config.COL_X,
    y_col: str       = config.COL_Y,
    condition_col: str = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Compute post-success dwell time per individual.

    Dwell time = total frames inside the target radius after the individual's
    first entry (including re-entries). Individuals that never reach the
    target are excluded from the returned DataFrame.

    Returns
    -------
    pd.DataFrame with columns: [Condition, Individual, DwellTime]
    """
    df = df.copy()
    dist = np.sqrt((df[x_col] - target_x) ** 2 + (df[y_col] - target_y) ** 2)
    df["_inside"] = dist <= radius

    records = []
    for (cond, ind), sub in df.groupby([condition_col, individual_col]):
        sub = sub.sort_values(frame_col)
        inside_frames = sub.loc[sub["_inside"], frame_col].values
        if len(inside_frames) == 0:
            continue
        first_entry = inside_frames[0]
        dwell = sub.loc[sub[frame_col] >= first_entry, "_inside"].sum()
        if dwell > 0:
            records.append({
                condition_col:  cond,
                individual_col: ind,
                "DwellTime":    int(dwell),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Zone proportions (for zone-means plot)
# ---------------------------------------------------------------------------

def zone_proportions(
    df: pd.DataFrame,
    y_col: str     = config.COL_Y,
    trial_col: str = config.COL_TRIAL,
    condition_col: str = config.COL_CONDITION,
    starvation_col: str = config.COL_STARVATION,
    zone_bounds: tuple  = None,
) -> pd.DataFrame:
    """
    Compute per-replicate proportions of time spent in each of three Y zones.

    Zones are defined by equal thirds of the observed Y range unless
    zone_bounds is provided explicitly.

    Parameters
    ----------
    zone_bounds : tuple (lower_y, upper_y) or None
        If None, bounds are computed from data as equal thirds.

    Returns
    -------
    pd.DataFrame with columns:
        [Condition, Starvation, ReplicateID, Zone1, Zone2, Zone3]
    where Zone1/2/3 are proportions summing to 1 per replicate.
    """
    df = df.copy()

    if "ReplicateID" not in df.columns:
        if trial_col in df.columns:
            df["ReplicateID"] = df[condition_col].astype(str) + "_" + df[trial_col].astype(str)
        else:
            df["ReplicateID"] = df[condition_col]

    if zone_bounds is None:
        arena_min = df[y_col].min()
        arena_max = df[y_col].max()
        span = arena_max - arena_min
        lower = arena_min + span / 3
        upper = arena_min + 2 * span / 3
    else:
        lower, upper = zone_bounds

    records = []
    for (cond, rep), sub in df.groupby([condition_col, "ReplicateID"]):
        starvation = sub[starvation_col].iloc[0] if starvation_col in sub.columns else None
        y = sub[y_col]
        p1 = (y < lower).mean()
        p2 = ((y >= lower) & (y < upper)).mean()
        p3 = (y >= upper).mean()
        records.append({
            condition_col:   cond,
            starvation_col:  starvation,
            "ReplicateID":   rep,
            "Zone1":         p1,
            "Zone2":         p2,
            "Zone3":         p3,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Stopping frequency
# ---------------------------------------------------------------------------

def stopping_frequency(
    df: pd.DataFrame,
    stop_threshold: float = 0.2,
    speed_col: str        = config.COL_SPEED,
    individual_col: str   = config.COL_INDIVIDUAL,
    condition_col: str    = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Compute the fraction of frames each individual is 'stopped'
    (Speed < stop_threshold).

    Note: call this on the *unfiltered* dataframe (before drop_invalid_speed)
    so that slow frames are present.

    Returns
    -------
    pd.DataFrame with columns: [Condition, Individual, StopFrequency]
    """
    df = df.copy()
    df["_stopped"] = df[speed_col] < stop_threshold

    result = (
        df.groupby([condition_col, individual_col])["_stopped"]
        .mean()
        .reset_index()
        .rename(columns={"_stopped": "StopFrequency"})
    )
    return result


# ---------------------------------------------------------------------------
# Speed (binned per individual)
# ---------------------------------------------------------------------------

def speed_binned(
    df: pd.DataFrame,
    bin_size: int       = 30,
    min_speed: float    = config.SPEED_MIN,
    max_speed: float    = config.SPEED_MAX,
    speed_col: str      = config.COL_SPEED,
    frame_col: str      = config.COL_FRAME,
    individual_col: str = config.COL_INDIVIDUAL,
    condition_col: str  = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Per-individual mean speed within each frame bin, with speed filtering.

    Returns
    -------
    pd.DataFrame with columns: [Condition, Individual, FrameBin, Speed]
    """
    df = df.copy()
    df = df[(df[speed_col] >= min_speed) & (df[speed_col] <= max_speed)]
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size

    result = (
        df.groupby([condition_col, individual_col, "FrameBin"])[speed_col]
        .mean()
        .reset_index()
    )
    return result


# ---------------------------------------------------------------------------
# Central fraction
# ---------------------------------------------------------------------------

def central_fraction(
    df: pd.DataFrame,
    center_x: float  = config.ARENA_WIDTH  / 2,
    center_y: float  = config.ARENA_HEIGHT / 2,
    radius: float    = 5.0,
    bin_size: int    = 30,
    x_col: str       = config.COL_X,
    y_col: str       = config.COL_Y,
    frame_col: str   = config.COL_FRAME,
    individual_col: str = config.COL_INDIVIDUAL,
    condition_col: str  = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Fraction of individuals within a central radius, averaged per frame bin.

    Group-size invariant: computed as n_inside / n_total per frame before
    binning, so groups of different sizes are comparable.

    Returns
    -------
    pd.DataFrame with columns: [Condition, FrameBin, CentralFraction]
    """
    df = df.copy()
    df["_dist_center"] = np.sqrt(
        (df[x_col] - center_x) ** 2 + (df[y_col] - center_y) ** 2
    )
    df["_inside"] = df["_dist_center"] <= radius

    # Fraction per frame
    frame_records = []
    for (cond, frame), sub in df.groupby([condition_col, frame_col]):
        total = sub[individual_col].nunique()
        if total == 0:
            continue
        inside = sub.loc[sub["_inside"], individual_col].nunique()
        frame_records.append({
            condition_col: cond,
            frame_col:     frame,
            "CentralFraction": inside / total,
        })

    frame_df = pd.DataFrame(frame_records)
    frame_df["FrameBin"] = (frame_df[frame_col] // bin_size) * bin_size

    result = (
        frame_df
        .groupby([condition_col, "FrameBin"])["CentralFraction"]
        .mean()
        .reset_index()
    )
    return result


# ---------------------------------------------------------------------------
# Final-window preference index (for paired Fed vs 5h comparison)
# ---------------------------------------------------------------------------

def preference_index_final_window(
    df: pd.DataFrame,
    frame_range: tuple  = None,
    last_n_frames: int  = None,
    y_col: str          = config.COL_Y,
    frame_col: str      = config.COL_FRAME,
    individual_col: str = config.COL_INDIVIDUAL,
    trial_col: str      = config.COL_TRIAL,
    condition_col: str  = config.COL_CONDITION,
    genotype_col: str   = config.COL_GENOTYPE,
    concentration_col: str = config.COL_CONCENTRATION,
    collective_col: str    = config.COL_COLLECTIVE,
    starvation_col: str    = config.COL_STARVATION,
    zone_bounds: tuple     = (10.0, 20.0),
) -> pd.DataFrame:
    """
    Per-trial mean Preference Index within a specific frame window.
    Used for the paired Fed vs 5h comparison plot.

    Provide exactly one of:
        frame_range=(start, end)  — inclusive
        last_n_frames=int

    Returns
    -------
    pd.DataFrame with columns:
        [Condition, Trial, Starvation, Genotype, Concentration,
         Collective, PreferenceIndex, PairKey]
    """
    if frame_range is None and last_n_frames is None:
        raise ValueError("Provide either frame_range or last_n_frames.")
    if frame_range is not None and last_n_frames is not None:
        raise ValueError("Provide only one of frame_range or last_n_frames.")

    lower, upper = zone_bounds
    records = []

    for (cond, trial, ind), sub in df.groupby([condition_col, trial_col, individual_col]):
        sub = sub.sort_values(frame_col)

        if frame_range is not None:
            start, end = frame_range
            window = sub[(sub[frame_col] >= start) & (sub[frame_col] <= end)]
        else:
            window = sub.tail(last_n_frames)

        if window.empty:
            continue

        y = window[y_col]
        bottom = (y <= lower).sum()
        top    = (y >= upper).sum()
        denom  = bottom + top
        pi     = np.nan if denom == 0 else (bottom - top) / denom

        records.append({
            condition_col:    cond,
            trial_col:        trial,
            individual_col:   ind,
            "PreferenceIndex": pi,
        })

    pi_individual = pd.DataFrame(records)

    # Average to trial level
    pi_trial = (
        pi_individual
        .groupby([condition_col, trial_col])["PreferenceIndex"]
        .mean()
        .reset_index()
    )

    # Merge metadata
    meta_cols = [condition_col, genotype_col, concentration_col,
                 collective_col, starvation_col]
    meta = df[meta_cols].drop_duplicates()
    pi_trial = pi_trial.merge(meta, on=condition_col, how="left")

    # Pair key for grouping Fed vs 5h in the plot
    pi_trial["PairKey"] = list(zip(
        pi_trial[concentration_col],
        pi_trial[genotype_col],
        pi_trial[collective_col],
    ))

    return pi_trial

DIRECTIONAL_TARGETS = {
    "Target (odour)": (14.0,  2.0),   # true odour source — bottom edge
    "Null (top)":     (14.0, 25.0),   # opposite edge
    "Null (left)":    ( 3.0, 15.0),   # left edge
    "Null (right)":   (25.0, 15.0),   # right edge
}
 
 
def cumulative_success_directional(
    df: pd.DataFrame,
    condition: str,
    targets: dict        = None,
    radius: float        = config.SUCCESS_RADIUS,
    bin_size: int        = 100,
    frame_col: str       = config.COL_FRAME,
    individual_col: str  = config.COL_INDIVIDUAL,
    x_col: str           = config.COL_X,
    y_col: str           = config.COL_Y,
    condition_col: str   = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Compute cumulative success rate over time for one condition against
    four arena positions: the true odour source and three null edges.
 
    Each 'target' is treated identically — an individual is counted as
    successful for a given target the first time it enters within `radius`
    of that target's coordinates. This lets you compare true chemotactic
    success against chance-level edge visits within the same dataset.
 
    Parameters
    ----------
    condition : str
        The condition to compute this for. Filter before passing if preferred,
        but this parameter makes the condition-specific intent explicit.
    targets : dict or None
        {label: (x, y)} mapping. Defaults to DIRECTIONAL_TARGETS, which
        defines the odour source and the three null edges.
    radius : float
        Radius considered 'at target' in cm. Same default as SUCCESS_RADIUS.
    bin_size : int
        Frame bin width.
 
    Returns
    -------
    pd.DataFrame with columns:
        [Target, FrameBin, CumulSuccess, SEM]
 
        One row per (target, frame bin). The Target column contains the
        human-readable label from the targets dict, e.g. "Target (odour)".
    """
    if targets is None:
        targets = DIRECTIONAL_TARGETS
 
    df = df[df[condition_col] == condition].copy()
    if df.empty:
        raise ValueError(f"No rows found for condition '{condition}'.")
 
    individuals = df[individual_col].unique()
    n_total     = len(individuals)
    frame_bins  = sorted({(f // bin_size) * bin_size for f in df[frame_col].unique()})
 
    records = []
 
    for label, (tx, ty) in targets.items():
        # First frame each individual enters this target's radius
        dist = np.sqrt((df[x_col] - tx) ** 2 + (df[y_col] - ty) ** 2)
        df[f"_inside_{label}"] = dist <= radius
 
        first_success = (
            df[df[f"_inside_{label}"]]
            .groupby(individual_col)[frame_col]
            .min()
        )
 
        fs_vals = np.array([
            first_success.get(ind, np.inf) for ind in individuals
        ])
 
        for bin_start in frame_bins:
            successes = (fs_vals < bin_start + bin_size).astype(float)
            records.append({
                "Target":       label,
                "FrameBin":     bin_start,
                "CumulSuccess": successes.mean(),
                "SEM": (
                    successes.std(ddof=0) / np.sqrt(n_total)
                    if n_total > 1 else 0.0
                ),
            })
 
    return pd.DataFrame(records)
 


def preference_index_directional(
    df: pd.DataFrame,
    condition: str,
    targets: dict        = None,
    bin_size: int        = 100,
    frame_col: str       = config.COL_FRAME,
    individual_col: str  = config.COL_INDIVIDUAL,
    x_col: str           = config.COL_X,
    y_col: str           = config.COL_Y,
    condition_col: str   = config.COL_CONDITION,
    zone_width: float    = 10.0,
) -> pd.DataFrame:
    """
    Compute a preference index toward each of the four arena positions
    (true target + three null edges) for one condition.
 
    For each target, the arena is divided into a 'near' zone and a 'far'
    zone based on distance to that target's coordinates. The near zone is
    defined as the closest `zone_width` cm along the relevant axis:
 
        Target (odour) — bottom edge  → near = Y <= zone_width
        Null (top)     — top edge     → near = Y >= (arena_height - zone_width)
        Null (left)    — left edge    → near = X <= zone_width
        Null (right)   — right edge   → near = X >= (arena_width - zone_width)
 
    PI = (n_near - n_far) / (n_near + n_far)
    where 'far' is always the opposite strip of the same width.
 
    A positive PI means more time in the near zone than the far zone.
    This is comparable across all four targets because the zone geometry
    is symmetric in each case.
 
    Parameters
    ----------
    targets : dict or None
        {label: (x, y)} mapping. Defaults to DIRECTIONAL_TARGETS.
        The zone axis (X or Y) is inferred from the target coordinates.
    zone_width : float
        Width in cm of the near and far strips. Default 10 cm = one third
        of the 30 cm arena, matching the standard PI zone definition.
 
    Returns
    -------
    pd.DataFrame with columns:
        [Target, Individual, FrameBin, PreferenceIndex]
    """
    if targets is None:
        targets = DIRECTIONAL_TARGETS
 
    df = df[df[condition_col] == condition].copy()
    if df.empty:
        raise ValueError(f"No rows found for condition '{condition}'.")
 
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size
 
    # Define near/far zone functions per target
    # Each returns a boolean Series: True = in near zone, True = in far zone
    def _zone_masks(target_label, tx, ty, subdf):
        if target_label == "Target (odour)":
            # bottom edge: near = low Y, far = high Y
            near = subdf[y_col] <= zone_width
            far  = subdf[y_col] >= (config.ARENA_HEIGHT - zone_width)
        elif target_label == "Null (top)":
            # top edge: near = high Y, far = low Y
            near = subdf[y_col] >= (config.ARENA_HEIGHT - zone_width)
            far  = subdf[y_col] <= zone_width
        elif target_label == "Null (left)":
            # left edge: near = low X, far = high X
            near = subdf[x_col] <= zone_width
            far  = subdf[x_col] >= (config.ARENA_WIDTH - zone_width)
        elif target_label == "Null (right)":
            # right edge: near = high X, far = low X
            near = subdf[x_col] >= (config.ARENA_WIDTH - zone_width)
            far  = subdf[x_col] <= zone_width
        else:
            # Fallback for custom targets: use Euclidean proximity
            # near = closest third of max observed distance
            dist      = np.sqrt((subdf[x_col] - tx)**2 + (subdf[y_col] - ty)**2)
            max_dist  = dist.max()
            near = dist <= max_dist / 3
            far  = dist >= 2 * max_dist / 3
        return near, far
 
    records = []
 
    for label, (tx, ty) in targets.items():
        for ind, ind_df in df.groupby(individual_col):
            ind_df = ind_df.sort_values(frame_col)
 
            for bin_id, bin_df in ind_df.groupby("FrameBin"):
                near, far = _zone_masks(label, tx, ty, bin_df)
                n_near = near.sum()
                n_far  = far.sum()
                denom  = n_near + n_far
                pi     = np.nan if denom == 0 else (n_near - n_far) / denom
 
                records.append({
                    "Target":          label,
                    individual_col:    ind,
                    "FrameBin":        bin_id,
                    "PreferenceIndex": pi,
                })
 
    return pd.DataFrame(records)
 
 
# ---------------------------------------------------------------------------
# Directional dwell time
# ---------------------------------------------------------------------------
 
def dwell_time_directional(
    df: pd.DataFrame,
    condition: str,
    targets: dict        = None,
    radius: float        = config.SUCCESS_RADIUS,
    frame_col: str       = config.COL_FRAME,
    individual_col: str  = config.COL_INDIVIDUAL,
    x_col: str           = config.COL_X,
    y_col: str           = config.COL_Y,
    condition_col: str   = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Compute post-first-entry dwell time for each of the four arena positions
    for one condition.
 
    For each target, dwell time = total frames within `radius` of that
    target after the individual's first entry. Individuals that never
    reach a given target contribute 0 to that target's distribution
    (rather than being dropped), so the distributions are always over the
    same n and are directly comparable.
 
    Returns
    -------
    pd.DataFrame with columns: [Target, Individual, DwellTime]
    """
    if targets is None:
        targets = DIRECTIONAL_TARGETS
 
    df = df[df[condition_col] == condition].copy()
    if df.empty:
        raise ValueError(f"No rows found for condition '{condition}'.")
 
    individuals = df[individual_col].unique()
    records     = []
 
    for label, (tx, ty) in targets.items():
        dist            = np.sqrt((df[x_col] - tx)**2 + (df[y_col] - ty)**2)
        df["_inside"]   = dist <= radius
 
        for ind in individuals:
            sub = df[df[individual_col] == ind].sort_values(frame_col)
            inside_frames = sub.loc[sub["_inside"], frame_col].values
 
            if len(inside_frames) == 0:
                dwell = 0
            else:
                first_entry = inside_frames[0]
                dwell = int(sub.loc[sub[frame_col] >= first_entry, "_inside"].sum())
 
            records.append({
                "Target":        label,
                individual_col:  ind,
                "DwellTime":     dwell,
            })
 
    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Current occupancy — individuals inside success radius per frame bin
# ---------------------------------------------------------------------------
 
def current_occupancy(
    df: pd.DataFrame,
    target_x: float      = config.TARGET_X,
    target_y: float      = config.TARGET_Y,
    radius: float        = config.SUCCESS_RADIUS,
    bin_size: int        = 100,
    frame_col: str       = config.COL_FRAME,
    individual_col: str  = config.COL_INDIVIDUAL,
    x_col: str           = config.COL_X,
    y_col: str           = config.COL_Y,
    condition_col: str   = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Compute per-trial occupancy proportion at each frame bin, returning
    trial-level rows so that downstream plotting (e.g. sns.pointplot with
    errorbar="se") computes SEM across trials rather than across frames.

    For each trial (individual) and each frame bin, the occupancy proportion
    is the mean of the binary inside/outside indicator across all frames
    within that bin — i.e. the fraction of frames in the bin during which
    that individual was inside the radius.

    Unlike the previous version, this does NOT pre-aggregate across trials.
    The caller is responsible for any further aggregation (e.g. pointplot
    will handle the cross-trial mean and SEM automatically).

    Returns
    -------
    pd.DataFrame with columns:
        [condition_col, individual_col, FrameBin, Occupancy]

        Occupancy is the proportion of frames within the bin during which
        the individual was inside the radius. One row per individual per
        FrameBin per condition.
    """
    df = df.copy()
    dist          = np.sqrt((df[x_col] - target_x) ** 2 + (df[y_col] - target_y) ** 2)
    df["_inside"] = dist <= radius
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size

    # Mean of binary indicator per trial per bin = fraction of bin spent inside
    trial_binned = (
        df.groupby([condition_col, individual_col, "FrameBin"])["_inside"]
        .mean()
        .reset_index()
        .rename(columns={"_inside": "Occupancy"})
    )

    return trial_binned
# ---------------------------------------------------------------------------
# Directional current occupancy
# ---------------------------------------------------------------------------
 
def current_occupancy_directional(
    df: pd.DataFrame,
    condition: str,
    targets: dict        = None,
    radius: float        = config.SUCCESS_RADIUS,
    bin_size: int        = 100,
    frame_col: str       = config.COL_FRAME,
    individual_col: str  = config.COL_INDIVIDUAL,
    x_col: str           = config.COL_X,
    y_col: str           = config.COL_Y,
    condition_col: str   = config.COL_CONDITION,
) -> pd.DataFrame:
    """
    Proportion of individuals currently inside each target's radius,
    averaged per frame bin, for one condition.
 
    Mirrors current_occupancy() but computed simultaneously for all four
    arena positions (true target + three null edges), so the four curves
    are directly comparable on the same axes.
 
    Parameters
    ----------
    condition : str
        Must match a value in df['Condition'].
    targets : dict or None
        {label: (x, y)} mapping. Defaults to DIRECTIONAL_TARGETS.
    radius : float
        Radius in cm considered 'at target'. Default config.SUCCESS_RADIUS.
    bin_size : int
        Frame bin width in frames. Default 100.
 
    Returns
    -------
    pd.DataFrame with columns:
        [Target, FrameBin, MeanOccupancy, SEM]
 
        MeanOccupancy is the mean proportion of individuals inside the
        radius across all frames within each bin.
        SEM is across frames within the bin.
    """
    if targets is None:
        targets = DIRECTIONAL_TARGETS
 
    df = df[df[condition_col] == condition].copy()
    if df.empty:
        raise ValueError(f"No rows found for condition '{condition}'.")
 
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size
    records = []
 
    for label, (tx, ty) in targets.items():
        dist          = np.sqrt((df[x_col] - tx) ** 2 + (df[y_col] - ty) ** 2)
        df["_inside"] = dist <= radius
 
        # Proportion inside at each individual frame
        frame_props = (
            df.groupby(frame_col)["_inside"]
            .mean()
            .reset_index()
            .rename(columns={"_inside": "Proportion"})
        )
        frame_props["FrameBin"] = (frame_props[frame_col] // bin_size) * bin_size
 
        binned = (
            frame_props.groupby("FrameBin")["Proportion"]
            .agg(
                MeanOccupancy="mean",
                SEM=lambda x: x.sem(),
            )
            .reset_index()
        )
        binned["Target"] = label
        records.append(binned)
 
    return pd.concat(records, ignore_index=True)
 