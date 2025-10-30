import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bootstrap_and_shuffle_pair(
    df: pd.DataFrame,
    value_col: str,
    collective_col: str = "Collective",
    trial_cols: list = ["Trial"],
    value: float = 14,
    width: float = 1.5,
    n_iter: int = 10000,
    random_state: int = 42,
    plot: bool = True
) -> dict:
    """
    Bootstraps Singles and reshuffles Groups for a single genotype+odour+concentration pair.
    Expects df to contain both 'Single' and 'Group' in `collective_col`.
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    PI = df[value_col].apply(
        lambda x: -1 if x > value + width else 
                   0 if value - width <= x <= value + width else 
                   1
    )
    df["_PI_tmp"] = PI

    singles = df[df[collective_col] == "Single"].copy()
    groups = df[df[collective_col] == "Group"].copy()
    
    group_size = groups.groupby(trial_cols).size().max()

    bootstrapped = [
        rng.choice(singles["_PI_tmp"].values, size=group_size, replace=True).mean()
        for _ in range(n_iter)
    ]

    reshuffled = [
        rng.choice(groups["_PI_tmp"].values, size=group_size, replace=False).mean()
        for _ in range(n_iter)
    ]

    real = groups.groupby(trial_cols)["_PI_tmp"].mean().values

    if plot:
        plt.figure(figsize=(8,6))
        bins = np.linspace(-1, 1, 30)
        plt.hist(bootstrapped, bins=bins, alpha=0.5, label="Bootstrapped Singles", density=True)
        plt.hist(reshuffled, bins=bins, alpha=0.5, label="Reshuffled Groups", density=True)
        plt.hist(real, bins=bins, alpha=0.7, label="Real Groups", density=True)
        plt.axvline(np.mean(real), color="k", linestyle="--", label="Mean Real Group PI")
        plt.xlabel("Preference Index (mean per group)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {"group_PI": real, "bootstrapped_single_PIs": np.array(bootstrapped), "reshuffled_group_PIs": np.array(reshuffled)}

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def bootstrap_and_shuffle_framewise(
    df: pd.DataFrame,
    value_col: str,
    frame_col: str = "Frame",
    collective_col: str = "Collective",
    trial_cols: list = ["Trial"],
    value: float = 14,
    width: float = 1.5,
    n_iter: int = 15,
    random_state: int = 42,
    plot: bool = True,
    compute_stats: bool = False,
    smooth_window: int = 5   # rolling mean window for smoothing
) -> dict:
    """
    Bootstraps Singles and reshuffles Groups for each frame.
    Error bars displayed as shaded SEM regions (optionally smoothed).
    """
    rng = np.random.default_rng(random_state)
    df = df.copy()
    
    # Compute framewise PI
    df["_PI_tmp"] = df[value_col].apply(
        lambda x: -1 if x > value + width else 
                   0 if value - width <= x <= value + width else 
                   1
    )
    
    singles = df[df[collective_col] == "Single"]
    groups = df[df[collective_col] == "Group"]

    frames = sorted(df[frame_col].unique())
    
    bootstrapped_matrix = np.zeros((len(frames), n_iter))
    reshuffled_matrix = np.zeros((len(frames), n_iter))
    real_matrix = []
    real_sem = []

    for i, f in enumerate(frames):
        singles_frame = singles[singles[frame_col] == f]
        groups_frame = groups[groups[frame_col] == f]
        
        group_size = groups_frame.groupby(trial_cols).size().max()
        
        # Bootstrapped singles
        bootstrapped_matrix[i] = [
            rng.choice(singles_frame["_PI_tmp"].values, size=group_size, replace=True).mean()
            for _ in range(n_iter)
        ]
        
        # Reshuffled groups
        reshuffled_matrix[i] = [
            rng.choice(groups_frame["_PI_tmp"].values, size=group_size, replace=False).mean()
            for _ in range(n_iter)
        ]
        
        # Real groups: mean and SEM across trials
        group_trial_means = groups_frame.groupby(trial_cols)["_PI_tmp"].mean()
        real_matrix.append(group_trial_means.mean())
        real_sem.append(group_trial_means.std(ddof=1) / np.sqrt(len(group_trial_means)))

    real_matrix = np.array(real_matrix)
    real_sem = np.array(real_sem)

    result = {
        "frames": frames,
        "group_PI": real_matrix,
        "group_PI_SEM": real_sem,
        "bootstrapped_single_PIs": bootstrapped_matrix,
        "reshuffled_group_PIs": reshuffled_matrix
    }

    if compute_stats:
        p_vals_singles = []
        p_vals_reshuffled = []
        for i in range(len(frames)):
            bs = bootstrapped_matrix[i]
            rs = reshuffled_matrix[i]
            real = real_matrix[i]

            p_single = np.mean(np.abs(bs - bs.mean()) >= np.abs(real - bs.mean()))
            p_vals_singles.append(p_single)

            p_rs = np.mean(np.abs(rs - rs.mean()) >= np.abs(real - rs.mean()))
            p_vals_reshuffled.append(p_rs)

        result["p_values_vs_singles"] = np.array(p_vals_singles)
        result["p_values_vs_reshuffled"] = np.array(p_vals_reshuffled)

    if plot:
        plt.figure(figsize=(10, 5))

        # Bootstrapped singles mean ± SEM
        bs_mean = bootstrapped_matrix.mean(axis=1)
        bs_sem = bootstrapped_matrix.std(axis=1) / np.sqrt(n_iter)

        # Reshuffled groups mean ± SEM
        rs_mean = reshuffled_matrix.mean(axis=1)
        rs_sem = reshuffled_matrix.std(axis=1) / np.sqrt(n_iter)

        # Optionally smooth with rolling average
        if smooth_window > 1:
            bs_mean = pd.Series(bs_mean).rolling(smooth_window, center=True).mean()
            bs_sem = pd.Series(bs_sem).rolling(smooth_window, center=True).mean()
            rs_mean = pd.Series(rs_mean).rolling(smooth_window, center=True).mean()
            rs_sem = pd.Series(rs_sem).rolling(smooth_window, center=True).mean()
            real_matrix = pd.Series(real_matrix).rolling(smooth_window, center=True).mean()
            real_sem = pd.Series(real_sem).rolling(smooth_window, center=True).mean()

        # Plot bootstrapped singles
        plt.plot(frames, bs_mean, color="#A9E308C8", label="Bootstrapped Singles")
        plt.fill_between(frames, bs_mean-bs_sem, bs_mean+bs_sem, color="#A9E308C8", alpha=0.2)

        # Plot real groups
        plt.plot(frames, real_matrix, color="#1951DD", label="Real Groups")
        plt.fill_between(frames, real_matrix-real_sem, real_matrix+real_sem, color="#1951DD", alpha=0.2)

        # Add significance markers
        if compute_stats:
            for i, p in enumerate(result["p_values_vs_singles"]):
                if p < 0.05:
                    plt.text(frames[i], real_matrix[i] + real_sem[i] + 0.05, "*", 
                             ha="center", color="red", fontsize=12)

        # Title with condition info
        if "Condition" in df.columns:
            conditions = df["Condition"].unique()
            if len(conditions) == 1:
                condition_str = conditions[0]
            else:
                condition_str = ", ".join(map(str, conditions))
            plt.title(f"Framewise PI: Singles vs Groups (Condition: {condition_str})")
        else:
            plt.title("Framewise PI: Singles vs Groups")
        plt.ylim(-1,1)
        plt.xlabel("Frame")
        plt.ylabel("Preference Index (PI)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def compare_bootstrap_metrics(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    value_col: str,
    frame_col: str = "Frame",
    collective_col: str = "Collective",
    trial_cols: list = ["Trial"],
    value: float = 14,
    width: float = 1.5,
    n_iter: int = 15,
    random_state: int = 42,
    smooth_window: int = 5,
    plot: bool = True,
    compute_stats: bool = True,
    labels: tuple = ("Dataset 1", "Dataset 2")
) -> dict:
    """
    Compare framewise group PI metrics between two datasets that share the same Conditions.
    Uses the bootstrap_and_shuffle_framewise function internally.
    Returns a dictionary with per-frame results and optionally plots comparisons.
    """

    # Run the framewise analysis on both datasets
    res1 = bootstrap_and_shuffle_framewise(
        df1, value_col=value_col, frame_col=frame_col,
        collective_col=collective_col, trial_cols=trial_cols,
        value=value, width=width, n_iter=n_iter,
        random_state=random_state, plot=False
    )
    res2 = bootstrap_and_shuffle_framewise(
        df2, value_col=value_col, frame_col=frame_col,
        collective_col=collective_col, trial_cols=trial_cols,
        value=value, width=width, n_iter=n_iter,
        random_state=random_state, plot=False
    )

    frames = np.intersect1d(res1["frames"], res2["frames"])

    g1 = pd.Series(res1["group_PI"], index=res1["frames"]).reindex(frames).values
    g2 = pd.Series(res2["group_PI"], index=res2["frames"]).reindex(frames).values

    sem1 = pd.Series(res1["group_PI_SEM"], index=res1["frames"]).reindex(frames).values
    sem2 = pd.Series(res2["group_PI_SEM"], index=res2["frames"]).reindex(frames).values

    if smooth_window > 1:
        g1 = pd.Series(g1).rolling(smooth_window, center=True).mean().values
        sem1 = pd.Series(sem1).rolling(smooth_window, center=True).mean().values
        g2 = pd.Series(g2).rolling(smooth_window, center=True).mean().values
        sem2 = pd.Series(sem2).rolling(smooth_window, center=True).mean().values

    results = {
        "frames": frames,
        "dataset1_mean": g1,
        "dataset2_mean": g2,
        "dataset1_sem": sem1,
        "dataset2_sem": sem2
    }

    # Compute statistical comparisons per frame
    if compute_stats:
        p_vals = []
        for i, f in enumerate(frames):
            # Extract bootstrapped distributions for each frame
            bs1 = res1["bootstrapped_single_PIs"][i]
            bs2 = res2["bootstrapped_single_PIs"][i]

            # t-test (can easily replace with Mann–Whitney or permutation)
            _, p = ttest_ind(bs1, bs2, equal_var=False, nan_policy="omit")
            p_vals.append(p)
        p_vals = np.array(p_vals)
        results["p_values"] = p_vals

    # Plot results
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(frames, g1, label=f"{labels[0]} Groups", color="#1951DD")
        plt.fill_between(frames, g1 - sem1, g1 + sem1, color="#1951DD", alpha=0.2)

        plt.plot(frames, g2, label=f"{labels[1]} Groups", color="#E36414")
        plt.fill_between(frames, g2 - sem2, g2 + sem2, color="#E36414", alpha=0.2)

        if compute_stats:
            for i, p in enumerate(p_vals):
                if p < 0.05:
                    plt.text(frames[i], max(g1[i], g2[i]) + 0.05, "*", 
                             ha="center", color="red", fontsize=12)

        if "Condition" in df1.columns:
            conds = df1["Condition"].unique()
            cond_str = ", ".join(map(str, conds))
            plt.title(f"Comparison of Group PI Between Datasets (Condition: {cond_str})")
        else:
            plt.title("Comparison of Group PI Between Datasets")

        plt.xlabel("Frame")
        plt.ylabel("Group PI")
        plt.ylim(-1, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return results
