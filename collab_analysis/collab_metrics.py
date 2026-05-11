# =============================================================================
# collab_metrics.py
# =============================================================================
"""
collab_metrics.py
-----------------
Pure computation functions for the collaborator dataset.
 
Metrics
-------
1. preference_index_vertical  — PI along X axis (left / centre / right zones)
2. centroid_position          — mean X and Y per frame bin
3. mean_vx                    — mean velocity in X direction per frame bin
4. neighbour_distances        — mean and nearest neighbour distance per frame bin
 
All functions return tidy DataFrames and produce no plots.
They are grouped by (Condition, Trial, FrameBin) throughout so the pipeline
is immediately robust to multiple trials per condition.
"""
 
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix as _dist_matrix
 
 
def preference_index_vertical(
    df: pd.DataFrame,
    bin_size: int       = 100,
    zone_width: float   = None,
    x_col: str          = "X",
    frame_col: str      = "Frame",
    individual_col: str = "Individual",
    trial_col: str      = "Trial",
    condition_col: str  = "Condition",
    arena_width: float  = None,
) -> pd.DataFrame:
    """
    Preference index along the X axis, dividing the arena into three
    vertical zones: Left / Centre / Right.
 
    PI = (n_left - n_right) / (n_left + n_right)
 
    Positive PI = more time on the left side.
    NaN if both left and right zones are empty in a bin.
 
    Zone boundaries default to equal thirds of the observed X range,
    or can be set explicitly via zone_width and arena_width.
 
    Parameters
    ----------
    zone_width : float or None
        Width of left and right zones in data units. If None, the observed
        X range is divided into equal thirds.
    arena_width : float or None
        Total arena width. Used only when zone_width is provided.
 
    Returns
    -------
    pd.DataFrame with columns:
        [Condition, Trial, Individual, FrameBin, PreferenceIndex]
    """
    df = df.copy()
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size
 
    if zone_width is None:
        x_min = df[x_col].min()
        x_max = df[x_col].max()
        span  = x_max - x_min
        left_bound  = x_min + span / 3
        right_bound = x_min + 2 * span / 3
    else:
        x_min       = df[x_col].min()
        left_bound  = x_min + zone_width
        right_bound = (arena_width or df[x_col].max()) - zone_width
 
    records = []
    for (cond, trial, ind), sub in df.groupby(
        [condition_col, trial_col, individual_col]
    ):
        for bin_id, bin_df in sub.groupby("FrameBin"):
            x      = bin_df[x_col]
            n_left  = (x < left_bound).sum()
            n_right = (x > right_bound).sum()
            denom   = n_left + n_right
            pi      = np.nan if denom == 0 else (n_left - n_right) / denom
 
            records.append({
                condition_col:  cond,
                trial_col:      trial,
                individual_col: ind,
                "FrameBin":     bin_id,
                "PreferenceIndex": pi,
            })
 
    return pd.DataFrame(records)
 
 
def centroid_position(
    df: pd.DataFrame,
    bin_size: int       = 100,
    x_col: str          = "X",
    y_col: str          = "Y",
    frame_col: str      = "Frame",
    individual_col: str = "Individual",
    trial_col: str      = "Trial",
    condition_col: str  = "Condition",
) -> pd.DataFrame:
    """
    Mean X and Y position of all individuals per frame bin.
 
    This is the group centroid — a single (X, Y) point per bin summarising
    where the group is on average. SEM is across frames within each bin.
 
    Returns
    -------
    pd.DataFrame with columns:
        [Condition, Trial, FrameBin, MeanX, MeanY, SEM_X, SEM_Y]
    """
    df = df.copy()
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size
 
    # Mean X and Y per frame first, then average within bin
    frame_means = (
        df.groupby([condition_col, trial_col, frame_col, "FrameBin"])[[x_col, y_col]]
        .mean()
        .reset_index()
    )
 
    binned = (
        frame_means.groupby([condition_col, trial_col, "FrameBin"])
        .agg(
            MeanX=(x_col, "mean"),
            MeanY=(y_col, "mean"),
            SEM_X=(x_col, "sem"),
            SEM_Y=(y_col, "sem"),
        )
        .reset_index()
    )
 
    return binned
 
 
def mean_vx(
    df: pd.DataFrame,
    bin_size: int       = 100,
    vx_col: str         = "VX",
    frame_col: str      = "Frame",
    individual_col: str = "Individual",
    trial_col: str      = "Trial",
    condition_col: str  = "Condition",
) -> pd.DataFrame:
    """
    Mean velocity in the X direction per frame bin.
 
    Positive VX = movement toward the right side of the arena.
    Negative VX = movement toward the left.
    SEM is across individuals within each (Condition, Trial, FrameBin).
 
    Returns
    -------
    pd.DataFrame with columns:
        [Condition, Trial, FrameBin, MeanVX, SEM_VX]
    """
    df = df.copy()
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size
 
    # Per-individual mean VX within each bin first
    ind_means = (
        df.groupby([condition_col, trial_col, individual_col, "FrameBin"])[vx_col]
        .mean()
        .reset_index()
        .rename(columns={vx_col: "IndMeanVX"})
    )
 
    # Then mean ± SEM across individuals
    binned = (
        ind_means.groupby([condition_col, trial_col, "FrameBin"])
        .agg(
            MeanVX=("IndMeanVX", "mean"),
            SEM_VX=("IndMeanVX", "sem"),
        )
        .reset_index()
    )
 
    return binned
 
 
def neighbour_distances(
    df: pd.DataFrame,
    bin_size: int       = 100,
    x_col: str          = "X",
    y_col: str          = "Y",
    frame_col: str      = "Frame",
    individual_col: str = "Individual",
    trial_col: str      = "Trial",
    condition_col: str  = "Condition",
    min_group_size: int = 2,
) -> pd.DataFrame:
    """
    Mean and nearest neighbour distance per frame bin.
 
    For each frame, the full pairwise distance matrix across all individuals
    is computed. Each individual's average and nearest neighbour distance is
    extracted, then these are averaged within frame bins.
 
    Frames with fewer than min_group_size individuals are skipped.
 
    Returns
    -------
    pd.DataFrame with columns:
        [Condition, Trial, FrameBin,
         MeanAvgNeighbour, SEM_AvgNeighbour,
         MeanNearestNeighbour, SEM_NearestNeighbour]
    """
    df = df.copy()
    df["FrameBin"] = (df[frame_col] // bin_size) * bin_size
 
    frame_records = []
 
    for (cond, trial, frame, bin_id), group in df.groupby(
        [condition_col, trial_col, frame_col, "FrameBin"]
    ):
        coords = group[[x_col, y_col]].to_numpy()
        if len(coords) < min_group_size:
            continue
 
        dists = _dist_matrix(coords, coords)
        np.fill_diagonal(dists, np.nan)
 
        avg_neighbour     = np.nanmean(dists, axis=1)
        nearest_neighbour = np.nanmin(dists,  axis=1)
 
        for avg, nearest in zip(avg_neighbour, nearest_neighbour):
            frame_records.append({
                condition_col: cond,
                trial_col:     trial,
                "FrameBin":    bin_id,
                "AvgNeighbour":     avg,
                "NearestNeighbour": nearest,
            })
 
    if not frame_records:
        return pd.DataFrame(columns=[
            condition_col, trial_col, "FrameBin",
            "MeanAvgNeighbour", "SEM_AvgNeighbour",
            "MeanNearestNeighbour", "SEM_NearestNeighbour",
        ])
 
    frame_df = pd.DataFrame(frame_records)
 
    binned = (
        frame_df.groupby([condition_col, trial_col, "FrameBin"])
        .agg(
            MeanAvgNeighbour=("AvgNeighbour",     "mean"),
            SEM_AvgNeighbour=("AvgNeighbour",     "sem"),
            MeanNearestNeighbour=("NearestNeighbour", "mean"),
            SEM_NearestNeighbour=("NearestNeighbour", "sem"),
        )
        .reset_index()
    )
 
    return binned