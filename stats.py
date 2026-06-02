"""
stats.py
--------
Statistical testing for the larva behaviour pipeline.

Tests
-----
Preference Index    — Welch's t-test at each frame bin, BH-FDR across bins
Current Occupancy   — Welch's t-test at each frame bin, BH-FDR across bins
Dwell Time          — Mann-Whitney U per condition pair, BH-FDR across pairs

All comparisons are pairwise. Results are printed in a readable table and
returned as a DataFrame for direct use in a results table.

Unit of observation
-------------------
PI tests aggregate to per-trial means before testing, avoiding
pseudoreplication across individuals within the same trial.

CO tests use per-individual per-bin occupancy values directly, since
current_occupancy() now returns one row per individual per bin — each
individual is treated as an independent observation.

Usage
-----
    import stats

    # Run all three tests and print results
    results = stats.test_summary(pref_df, occupancy_df, dwell_df)

    # Or run individually
    pi_results   = stats.test_preference_index(pref_df)
    co_results   = stats.test_current_occupancy(occupancy_df)
    dw_results   = stats.test_dwell_time(dwell_df)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests

import config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    """Convert a p-value to a significance star string."""
    if np.isnan(p):   return "n/a"
    if p < 0.001:     return "***"
    if p < 0.01:      return "**"
    if p < 0.05:      return "*"
    return "ns"


def _bh_correct(pvalues: list) -> np.ndarray:
    """
    Apply Benjamini-Hochberg FDR correction to a list of p-values.
    NaN entries are preserved in position and excluded from correction.
    Returns an array of corrected p-values in the same order.
    """
    pvalues = np.array(pvalues, dtype=float)
    corrected = np.full_like(pvalues, np.nan)

    valid_mask = ~np.isnan(pvalues)
    if valid_mask.sum() == 0:
        return corrected

    _, p_adj, _, _ = multipletests(
        pvalues[valid_mask], alpha=0.05, method="fdr_bh"
    )
    corrected[valid_mask] = p_adj
    return corrected


def _print_header(title: str):
    width = 74
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_table(df: pd.DataFrame):
    """Print a DataFrame as a fixed-width table."""
    print(df.to_string(index=False))
    print()


def _get_conditions(df: pd.DataFrame, condition_col: str) -> list:
    """Return sorted unique conditions."""
    return sorted(df[condition_col].unique())


# ---------------------------------------------------------------------------
# Trial-level aggregation helpers
# ---------------------------------------------------------------------------

def _pi_trial_means(
    pref_df: pd.DataFrame,
    condition_col: str = config.COL_CONDITION,
    individual_col: str = config.COL_INDIVIDUAL,
    trial_col: str = config.COL_TRIAL,
) -> pd.DataFrame:
    """
    Aggregate per-individual PI rows to per-trial means.

    metrics.preference_index() returns one row per (Condition, Individual,
    FrameBin). We first average across individuals within each trial,
    then return (Condition, Trial, FrameBin, PreferenceIndex).

    If the Trial column is absent, Individual is used as the trial proxy
    (appropriate when each individual comes from a different video).
    """
    if trial_col not in pref_df.columns:
        # Fall back: treat each individual as its own trial
        trial_col = individual_col

    return (
        pref_df.groupby([condition_col, trial_col, "FrameBin"])["PreferenceIndex"]
        .mean()
        .reset_index()
    )



# ---------------------------------------------------------------------------
# 1. Preference Index — Welch's t-test per bin, BH-FDR across bins
# ---------------------------------------------------------------------------

def test_preference_index(
    pref_df: pd.DataFrame,
    condition_col: str  = config.COL_CONDITION,
    individual_col: str = config.COL_INDIVIDUAL,
    trial_col: str      = config.COL_TRIAL,
    alpha: float        = 0.05,
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Pairwise Welch's t-test on per-trial mean Preference Index at each
    frame bin. BH-FDR correction applied across bins within each pair.

    Parameters
    ----------
    pref_df : pd.DataFrame
        Output of metrics.preference_index(). Must contain columns:
        [Condition, Individual, FrameBin, PreferenceIndex].
        Trial column used if present; Individual used as fallback.
    alpha : float
        Significance threshold after FDR correction. Default 0.05.
    print_results : bool
        Whether to print the formatted results table.

    Returns
    -------
    pd.DataFrame with columns:
        [Condition_A, Condition_B, FrameBin,
         Mean_A, Mean_B, Mean_Diff,
         t_stat, p_raw, p_corrected, Significant, Stars]
    """
    trial_df   = _pi_trial_means(pref_df, condition_col, individual_col, trial_col)
    conditions = _get_conditions(trial_df, condition_col)
    pairs      = list(combinations(conditions, 2))
    frame_bins = sorted(trial_df["FrameBin"].unique())

    rows = []

    for cond_a, cond_b in pairs:
        raw_pvals = []
        pair_rows = []

        for fb in frame_bins:
            bin_df = trial_df[trial_df["FrameBin"] == fb]
            vals_a = bin_df.loc[bin_df[condition_col] == cond_a, "PreferenceIndex"].dropna().values
            vals_b = bin_df.loc[bin_df[condition_col] == cond_b, "PreferenceIndex"].dropna().values

            if len(vals_a) < 2 or len(vals_b) < 2:
                raw_pvals.append(np.nan)
                pair_rows.append({
                    "Condition_A": cond_a,
                    "Condition_B": cond_b,
                    "FrameBin":    fb,
                    "Mean_A":      np.nanmean(vals_a) if len(vals_a) else np.nan,
                    "Mean_B":      np.nanmean(vals_b) if len(vals_b) else np.nan,
                    "Mean_Diff":   np.nan,
                    "t_stat":      np.nan,
                    "p_raw":       np.nan,
                })
                continue

            t_stat, p_raw = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)
            raw_pvals.append(p_raw)
            pair_rows.append({
                "Condition_A": cond_a,
                "Condition_B": cond_b,
                "FrameBin":    fb,
                "Mean_A":      np.mean(vals_a),
                "Mean_B":      np.mean(vals_b),
                "Mean_Diff":   np.mean(vals_a) - np.mean(vals_b),
                "t_stat":      t_stat,
                "p_raw":       p_raw,
            })

        corrected = _bh_correct(raw_pvals)
        for row, p_adj in zip(pair_rows, corrected):
            row["p_corrected"] = p_adj
            row["Significant"] = (not np.isnan(p_adj)) and (p_adj < alpha)
            row["Stars"]       = _stars(p_adj)

        rows.extend(pair_rows)

    result_df = pd.DataFrame(rows)

    # Round for readability
    for col in ("Mean_A", "Mean_B", "Mean_Diff", "t_stat", "p_raw", "p_corrected"):
        result_df[col] = result_df[col].round(4)

    if print_results:
        _print_header("PREFERENCE INDEX — Welch t-test, BH-FDR corrected (across bins per pair)")
        for cond_a, cond_b in pairs:
            sub = result_df[
                (result_df["Condition_A"] == cond_a) &
                (result_df["Condition_B"] == cond_b)
            ]
            print(f"\n  {cond_a}  vs  {cond_b}")
            print(f"  {'FrameBin':>10}  {'Mean_A':>8}  {'Mean_B':>8}  "
                  f"{'Diff':>8}  {'t':>7}  {'p_raw':>8}  {'p_adj':>8}  {'Sig':>5}")
            print("  " + "-" * 68)
            for _, row in sub.iterrows():
                print(
                    f"  {int(row['FrameBin']):>10}  "
                    f"{row['Mean_A']:>8.4f}  {row['Mean_B']:>8.4f}  "
                    f"{row['Mean_Diff']:>8.4f}  {row['t_stat']:>7.3f}  "
                    f"{row['p_raw']:>8.4f}  {row['p_corrected']:>8.4f}  "
                    f"{row['Stars']:>5}"
                )

    return result_df


# ---------------------------------------------------------------------------
# 2. Current Occupancy — Welch's t-test per bin, BH-FDR across bins
# ---------------------------------------------------------------------------

def test_current_occupancy(
    occupancy_df: pd.DataFrame,
    condition_col: str  = config.COL_CONDITION,
    individual_col: str = config.COL_INDIVIDUAL,
    alpha: float        = 0.05,
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Pairwise Welch's t-test on per-individual current occupancy at each
    frame bin. BH-FDR correction applied across bins within each pair.

    Parameters
    ----------
    occupancy_df : pd.DataFrame
        Output of metrics.current_occupancy(). Must contain columns:
        [Condition, Individual, FrameBin, Occupancy].
        Each row is one individual's occupancy proportion for one bin,
        so each individual contributes one independent observation per bin.
    alpha : float
        Significance threshold after FDR correction. Default 0.05.
    print_results : bool
        Whether to print the formatted results table.

    Returns
    -------
    pd.DataFrame with columns:
        [Condition_A, Condition_B, FrameBin,
         Mean_A, Mean_B, Mean_Diff,
         t_stat, p_raw, p_corrected, Significant, Stars]
    """
    conditions = _get_conditions(occupancy_df, condition_col)
    pairs      = list(combinations(conditions, 2))
    frame_bins = sorted(occupancy_df["FrameBin"].unique())

    rows = []

    for cond_a, cond_b in pairs:
        raw_pvals = []
        pair_rows = []

        for fb in frame_bins:
            bin_df = occupancy_df[occupancy_df["FrameBin"] == fb]
            vals_a = bin_df.loc[bin_df[condition_col] == cond_a, "Occupancy"].dropna().values
            vals_b = bin_df.loc[bin_df[condition_col] == cond_b, "Occupancy"].dropna().values

            if len(vals_a) < 2 or len(vals_b) < 2:
                raw_pvals.append(np.nan)
                pair_rows.append({
                    "Condition_A": cond_a,
                    "Condition_B": cond_b,
                    "FrameBin":    fb,
                    "Mean_A":      np.nanmean(vals_a) if len(vals_a) else np.nan,
                    "Mean_B":      np.nanmean(vals_b) if len(vals_b) else np.nan,
                    "Mean_Diff":   np.nan,
                    "t_stat":      np.nan,
                    "p_raw":       np.nan,
                })
                continue

            t_stat, p_raw = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)
            raw_pvals.append(p_raw)
            pair_rows.append({
                "Condition_A": cond_a,
                "Condition_B": cond_b,
                "FrameBin":    fb,
                "Mean_A":      np.mean(vals_a),
                "Mean_B":      np.mean(vals_b),
                "Mean_Diff":   np.mean(vals_a) - np.mean(vals_b),
                "t_stat":      t_stat,
                "p_raw":       p_raw,
            })

        corrected = _bh_correct(raw_pvals)
        for row, p_adj in zip(pair_rows, corrected):
            row["p_corrected"] = p_adj
            row["Significant"] = (not np.isnan(p_adj)) and (p_adj < alpha)
            row["Stars"]       = _stars(p_adj)

        rows.extend(pair_rows)

    result_df = pd.DataFrame(rows)

    for col in ("Mean_A", "Mean_B", "Mean_Diff", "t_stat", "p_raw", "p_corrected"):
        result_df[col] = result_df[col].round(4)

    if print_results:
        _print_header("CURRENT OCCUPANCY — Welch t-test, BH-FDR corrected (across bins per pair)")
        for cond_a, cond_b in pairs:
            sub = result_df[
                (result_df["Condition_A"] == cond_a) &
                (result_df["Condition_B"] == cond_b)
            ]
            print(f"\n  {cond_a}  vs  {cond_b}")
            print(f"  {'FrameBin':>10}  {'Mean_A':>8}  {'Mean_B':>8}  "
                  f"{'Diff':>8}  {'t':>7}  {'p_raw':>8}  {'p_adj':>8}  {'Sig':>5}")
            print("  " + "-" * 68)
            for _, row in sub.iterrows():
                print(
                    f"  {int(row['FrameBin']):>10}  "
                    f"{row['Mean_A']:>8.4f}  {row['Mean_B']:>8.4f}  "
                    f"{row['Mean_Diff']:>8.4f}  {row['t_stat']:>7.3f}  "
                    f"{row['p_raw']:>8.4f}  {row['p_corrected']:>8.4f}  "
                    f"{row['Stars']:>5}"
                )

    return result_df


# ---------------------------------------------------------------------------
# 3. Dwell Time — Mann-Whitney U per pair, BH-FDR across pairs
# ---------------------------------------------------------------------------

def test_dwell_time(
    dwell_df: pd.DataFrame,
    condition_col: str  = config.COL_CONDITION,
    individual_col: str = config.COL_INDIVIDUAL,
    alpha: float        = 0.05,
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Pairwise Mann-Whitney U test on per-individual dwell times.
    BH-FDR correction applied across all pairs.

    Dwell time is tested once (not per bin) since it is a single value
    per individual summarising total post-entry occupancy.

    Parameters
    ----------
    dwell_df : pd.DataFrame
        Output of metrics.dwell_time(). Must contain columns:
        [Condition, Individual, DwellTime].
        Individuals that never reached the target are excluded (they do
        not appear in dwell_df), which conditions the test on success.
        This is intentional — you are comparing dwell behaviour among
        individuals that did reach the target.

    Returns
    -------
    pd.DataFrame with columns:
        [Condition_A, Condition_B, N_A, N_B,
         Median_A, Median_B, Median_Diff,
         U_stat, p_raw, p_corrected, Significant, Stars]
    """
    conditions = _get_conditions(dwell_df, condition_col)
    pairs      = list(combinations(conditions, 2))

    raw_pvals = []
    pair_rows = []

    for cond_a, cond_b in pairs:
        vals_a = dwell_df.loc[dwell_df[condition_col] == cond_a, "DwellTime"].dropna().values
        vals_b = dwell_df.loc[dwell_df[condition_col] == cond_b, "DwellTime"].dropna().values

        if len(vals_a) < 2 or len(vals_b) < 2:
            raw_pvals.append(np.nan)
            pair_rows.append({
                "Condition_A":  cond_a,
                "Condition_B":  cond_b,
                "N_A":          len(vals_a),
                "N_B":          len(vals_b),
                "Median_A":     np.median(vals_a) if len(vals_a) else np.nan,
                "Median_B":     np.median(vals_b) if len(vals_b) else np.nan,
                "Median_Diff":  np.nan,
                "U_stat":       np.nan,
                "p_raw":        np.nan,
            })
            continue

        u_stat, p_raw = scipy_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
        raw_pvals.append(p_raw)
        pair_rows.append({
            "Condition_A":  cond_a,
            "Condition_B":  cond_b,
            "N_A":          len(vals_a),
            "N_B":          len(vals_b),
            "Median_A":     np.median(vals_a),
            "Median_B":     np.median(vals_b),
            "Median_Diff":  np.median(vals_a) - np.median(vals_b),
            "U_stat":       u_stat,
            "p_raw":        p_raw,
        })

    corrected = _bh_correct(raw_pvals)
    for row, p_adj in zip(pair_rows, corrected):
        row["p_corrected"] = p_adj
        row["Significant"] = (not np.isnan(p_adj)) and (p_adj < alpha)
        row["Stars"]       = _stars(p_adj)

    result_df = pd.DataFrame(pair_rows)

    for col in ("Median_A", "Median_B", "Median_Diff", "U_stat", "p_raw", "p_corrected"):
        if col in result_df.columns:
            result_df[col] = result_df[col].round(4)

    if print_results:
        _print_header("DWELL TIME — Mann-Whitney U, BH-FDR corrected (across pairs)")
        print(f"\n  {'Condition A':<35}  {'Condition B':<35}  "
              f"{'N_A':>4}  {'N_B':>4}  "
              f"{'Median_A':>9}  {'Median_B':>9}  {'Diff':>9}  "
              f"{'U':>8}  {'p_raw':>8}  {'p_adj':>8}  {'Sig':>5}")
        print("  " + "-" * 140)
        for _, row in result_df.iterrows():
            print(
                f"  {row['Condition_A']:<35}  {row['Condition_B']:<35}  "
                f"{int(row['N_A']):>4}  {int(row['N_B']):>4}  "
                f"{row['Median_A']:>9.4f}  {row['Median_B']:>9.4f}  "
                f"{row['Median_Diff']:>9.4f}  "
                f"{row['U_stat']:>8.1f}  {row['p_raw']:>8.4f}  "
                f"{row['p_corrected']:>8.4f}  {row['Stars']:>5}"
            )

    return result_df


# ---------------------------------------------------------------------------
# Convenience wrapper — run all three tests at once
# ---------------------------------------------------------------------------

def test_summary(
    pref_df: pd.DataFrame,
    occupancy_df: pd.DataFrame,
    dwell_df: pd.DataFrame,
    condition_col: str   = config.COL_CONDITION,
    individual_col: str  = config.COL_INDIVIDUAL,
    trial_col: str       = config.COL_TRIAL,
    alpha: float         = 0.05,
) -> dict:
    """
    Run all three tests and return results as a dict of DataFrames.

    Parameters
    ----------
    pref_df, occupancy_df, dwell_df
        Outputs of the corresponding metrics functions.
    alpha : float
        Significance threshold after FDR correction.

    Returns
    -------
    dict with keys: 'preference_index', 'current_occupancy', 'dwell_time'
    Each value is the result DataFrame from the individual test function.

    Example
    -------
    >>> occupancy_df, pref_df, dwell_df = figures.behavior_summary_current_occupancy(df)
    >>> results = stats.test_summary(pref_df, occupancy_df, dwell_df)
    >>> results['preference_index'].to_csv('pi_stats.csv', index=False)
    """
    pi_results = test_preference_index(
        pref_df,
        condition_col=condition_col,
        individual_col=individual_col,
        trial_col=trial_col,
        alpha=alpha,
    )
    co_results = test_current_occupancy(
        occupancy_df,
        condition_col=condition_col,
        individual_col=individual_col,
        alpha=alpha,
    )
    dw_results = test_dwell_time(
        dwell_df,
        condition_col=condition_col,
        individual_col=individual_col,
        alpha=alpha,
    )

    return {
        "preference_index":  pi_results,
        "current_occupancy": co_results,
        "dwell_time":        dw_results,
    }