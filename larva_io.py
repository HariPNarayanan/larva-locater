"""
io.py
-----
Data loading, cleaning, and preprocessing.

Replaces cleanup.py. Key differences:
  - Target coordinates are parameters, not hardcoded (defaults from config.py).
  - categorize_values() is dropped — it added a 'Preference Index' column
    from X position that nothing downstream used; zone-based PI is computed
    in metrics.py instead.
  - preprocess() is a single convenience function for the notebook,
    replacing the 6-line setup block.
  - compute_polarization_with_context() and create_combined_real_simulated_df()
    are kept here as data-enrichment steps (they mutate the dataframe rather
    than produce a plot or a pure metric).
"""

import os
import numpy as np
import pandas as pd

import config

# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load(
    pathinfo: str,
    y_col:     str = config.RAW_Y,
    x_col:     str = config.RAW_X,
    speed_col: str = config.RAW_SPEED,
    vy_col:    str = config.RAW_VY,
    frame_col: str = config.RAW_FRAME,
) -> pd.DataFrame:
    """
    Walk a directory tree, read every CSV, and return a single concatenated
    DataFrame with standardised column names.

    Folder naming convention expected:
        <Odour> <Genotype> <Starvation> <Collective> <Concentration>
    e.g.  "ACV Canton-S Fed Single 10-3"

    Parameters
    ----------
    pathinfo : str
        Root directory to search recursively.
    y_col, x_col, speed_col, vy_col, frame_col : str
        Column names as they appear in the raw CSV files.
        Defaults match config.RAW_* constants.

    Returns
    -------
    pd.DataFrame with columns:
        Odour, Y, X, Speed, VY, Frame, Trial, Condition,
        Genotype, Starvation, Collective, Concentration, Individual
    """
    pd.options.mode.use_inf_as_na = True

    df_files = []

    for root, dirs, files in os.walk(pathinfo):
        for f in files:
            if not f.endswith(".csv"):
                continue

            file_path = os.path.join(root, f)
            try:
                a = pd.read_csv(file_path)

                condition_folder = os.path.basename(os.path.dirname(root))
                parts = condition_folder.split()

                if len(parts) == 5:
                    odour, genotype, starvation, collective, concentration = parts
                else:
                    odour = genotype = starvation = collective = concentration = None

                data = {
                    config.COL_ODOUR:         odour,
                    config.COL_Y:             a[y_col],
                    config.COL_X:             a[x_col],
                    config.COL_SPEED:         a[speed_col],
                    "VY":                     a[vy_col],
                    config.COL_FRAME:         a[frame_col],
                    config.COL_TRIAL:         os.path.basename(root),
                    config.COL_CONDITION:     condition_folder,
                    config.COL_GENOTYPE:      genotype,
                    config.COL_STARVATION:    starvation,
                    config.COL_COLLECTIVE:    collective,
                    config.COL_CONCENTRATION: concentration,
                    config.COL_INDIVIDUAL:    f,
                }
                df_files.append(pd.DataFrame(data))

            except KeyError as e:
                print(f"Missing column in {file_path}: {e}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    if not df_files:
        return pd.DataFrame()

    return pd.concat(df_files, ignore_index=True)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def filter_frames(df: pd.DataFrame, max_frame: int = config.MAX_FRAME) -> pd.DataFrame:
    """Drop rows beyond max_frame."""
    return df[df[config.COL_FRAME] < max_frame].copy()


def drop_invalid_speed(
    df: pd.DataFrame,
    min_speed: float = config.SPEED_MIN,
    max_speed: float = config.SPEED_MAX,
) -> pd.DataFrame:
    """
    Remove rows where Speed is outside (min_speed, max_speed).
    Also drops any remaining NaN rows.
    """
    df = df.dropna()
    df = df[df[config.COL_SPEED].between(min_speed, max_speed, inclusive="neither")]
    return df.copy()


def interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Linearly interpolate missing X and Y values within each
    (Trial, Condition) group.
    """
    required = {config.COL_X, config.COL_Y, config.COL_TRIAL, config.COL_CONDITION}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    df = df.copy()
    df[config.COL_TRIAL] = df[config.COL_TRIAL].astype(str)

    for col in (config.COL_X, config.COL_Y):
        df[col] = df.groupby(
            [config.COL_TRIAL, config.COL_CONDITION]
        )[col].transform(
            lambda g: g.interpolate(method="linear", limit_direction="both")
        )

    return df


def add_distance(
    df: pd.DataFrame,
    target_x: float = config.TARGET_X,
    target_y: float = config.TARGET_Y,
) -> pd.DataFrame:
    """
    Add a 'Distance' column: Euclidean distance from each (X, Y) point
    to (target_x, target_y). Rows with NaN coordinates get NaN distance.
    """
    df = df.copy()
    df[config.COL_DISTANCE] = np.where(
        df[[config.COL_X, config.COL_Y]].isnull().any(axis=1),
        np.nan,
        np.hypot(df[config.COL_X] - target_x, df[config.COL_Y] - target_y),
    )
    return df


# ---------------------------------------------------------------------------
# One-call convenience function for the notebook
# ---------------------------------------------------------------------------

def preprocess(
    pathinfo: str,
    max_frame: int   = config.MAX_FRAME,
    target_x: float  = config.TARGET_X,
    target_y: float  = config.TARGET_Y,
    interpolate_xy: bool = True,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline in one call:
        load → filter frames → interpolate → add distance

    Speed filtering is intentionally left out here; it is applied
    on-demand in individual figure functions so that the base dataframe
    retains all rows (e.g. stopping-frequency analysis needs the slow rows).

    Parameters
    ----------
    pathinfo : str
        Root data directory.
    max_frame : int
        Frames >= max_frame are dropped.
    target_x, target_y : float
        Odour source coordinates for distance calculation.
    interpolate_xy : bool
        Whether to interpolate missing X/Y values (default True).

    Returns
    -------
    pd.DataFrame ready for metrics.py and figures.py functions.

    Example
    -------
    >>> import io as larva_io
    >>> df = larva_io.preprocess(filepath)
    """
    df = load(pathinfo)
    df = filter_frames(df, max_frame=max_frame)

    if interpolate_xy:
        df = interpolate(df)

    df = add_distance(df, target_x=target_x, target_y=target_y)

    return df


# ---------------------------------------------------------------------------
# Data-enrichment helpers (kept from cleanup.py)
# ---------------------------------------------------------------------------

def add_polarization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute polarization toward the -Y direction (normalised motion vectors)
    and merge it back as a 'polarization' column.

    Polarization = mean dot product of normalised velocity vectors with [0, -1]
    across all individuals, per (Trial, Condition, Frame).
    """
    df = df.sort_values(
        [config.COL_TRIAL, config.COL_CONDITION, config.COL_INDIVIDUAL, config.COL_FRAME]
    ).copy()

    grp = [config.COL_TRIAL, config.COL_CONDITION, config.COL_INDIVIDUAL]
    df["dx"] = df.groupby(grp)[config.COL_X].diff()
    df["dy"] = df.groupby(grp)[config.COL_Y].diff()

    norm = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2)
    df["dx_norm"] = (df["dx"] / norm).fillna(0)
    df["dy_norm"] = (df["dy"] / norm).fillna(0)
    df["dot_product"] = -df["dy_norm"]

    polarization = (
        df.groupby([config.COL_TRIAL, config.COL_CONDITION, config.COL_FRAME])["dot_product"]
        .mean()
        .reset_index()
        .rename(columns={"dot_product": "polarization"})
    )

    df = df.merge(
        polarization,
        on=[config.COL_TRIAL, config.COL_CONDITION, config.COL_FRAME],
        how="left",
    )
    return df


def create_combined_real_simulated_df(
    df: pd.DataFrame,
    group_size: int = 15,
    n_bootstrap: int = 8,
    seed: int = None,
) -> pd.DataFrame:
    """
    Combine real Group trajectories with bootstrapped pseudo-groups built
    from Single trajectories, adding AvgNeighborDist and NearestNeighborDist.

    Used for collective-behaviour comparisons; not part of the standard
    single-animal pipeline.

    Parameters
    ----------
    df : pd.DataFrame
    group_size : int     Individuals per simulated group (default 15).
    n_bootstrap : int    Simulated groups per condition (default 8).
    seed : int or None   For reproducibility.

    Returns
    -------
    pd.DataFrame with columns: ..., GroupType, Simulated,
                                AvgNeighborDist, NearestNeighborDist
    """
    from scipy.spatial import distance_matrix as _dist_matrix

    if seed is not None:
        np.random.seed(seed)

    df = df.copy()
    df["Simulated"]          = False
    df["GroupType"]          = np.nan
    df["AvgNeighborDist"]    = np.nan
    df["NearestNeighborDist"] = np.nan

    # --- Real group data ---
    real_rows = []
    for (condition, trial, frame), group in df[df[config.COL_COLLECTIVE] == "Group"].groupby(
        [config.COL_CONDITION, config.COL_TRIAL, config.COL_FRAME]
    ):
        coords = group[[config.COL_X, config.COL_Y]].to_numpy()
        if len(coords) < 2:
            continue
        dists = _dist_matrix(coords, coords)
        np.fill_diagonal(dists, np.nan)
        group = group.copy()
        group["AvgNeighborDist"]    = np.nanmean(dists, axis=1)
        group["NearestNeighborDist"] = np.nanmin(dists,  axis=1)
        group["GroupType"]  = "Real"
        group["Simulated"]  = False
        real_rows.append(group)

    df_real = pd.concat(real_rows, ignore_index=True) if real_rows else pd.DataFrame()

    # --- Simulated groups from Singles ---
    simulated_rows = []
    single_df = df[df[config.COL_COLLECTIVE] == "Single"].dropna(
        subset=[config.COL_X, config.COL_Y]
    )

    for condition in single_df[config.COL_CONDITION].unique():
        cond_df  = single_df[single_df[config.COL_CONDITION] == condition]
        n_groups = min(n_bootstrap, len(cond_df) // group_size)

        for sim in range(n_groups):
            sampled = cond_df.sample(n=group_size, replace=False)
            coords  = sampled[[config.COL_X, config.COL_Y]].to_numpy()
            dists   = _dist_matrix(coords, coords)
            np.fill_diagonal(dists, np.nan)
            sampled = sampled.copy()
            sampled["AvgNeighborDist"]    = np.nanmean(dists, axis=1)
            sampled["NearestNeighborDist"] = np.nanmin(dists,  axis=1)
            sampled["Simulated"]  = True
            sampled["GroupType"]  = "Simulated"
            sampled["SimGroupID"] = f"{condition}_s{sim}"
            simulated_rows.append(sampled)

    df_simulated = (
        pd.concat(simulated_rows, ignore_index=True)
        if simulated_rows
        else pd.DataFrame()
    )

    return pd.concat([df_real, df_simulated], ignore_index=True)
