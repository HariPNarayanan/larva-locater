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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    n_iter: int = 12,
    random_state: int = 42,
    plot: bool = True,
    compute_stats: bool = False
) -> dict:
    """
    Bootstraps Singles and reshuffles Groups for each frame.
    df should contain both 'Single' and 'Group' in collective_col.
    Optionally plots the results and computes framewise p-values.
    Error bars are now SEM for bootstrapped singles, reshuffled groups, and real groups.
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
        # Framewise p-values vs bootstrapped singles and reshuffled groups
        p_vals_singles = []
        p_vals_reshuffled = []
        for i in range(len(frames)):
            bs = bootstrapped_matrix[i]
            rs = reshuffled_matrix[i]
            real = real_matrix[i]

            # Two-sided bootstrap test against singles
            p_single = np.mean(np.abs(bs - bs.mean()) >= np.abs(real - bs.mean()))
            p_vals_singles.append(p_single)

            # Two-sided test against reshuffled groups
            p_rs = np.mean(np.abs(rs - rs.mean()) >= np.abs(real - rs.mean()))
            p_vals_reshuffled.append(p_rs)

        result["p_values_vs_singles"] = np.array(p_vals_singles)
        result["p_values_vs_reshuffled"] = np.array(p_vals_reshuffled)

    if plot:
        plt.figure(figsize=(10, 5))
        
        # Bootstrapped singles mean and SEM
        bs_mean = bootstrapped_matrix.mean(axis=1)
        bs_sem = bootstrapped_matrix.std(axis=1) / np.sqrt(n_iter)
        plt.errorbar(frames, bs_mean, yerr=bs_sem, fmt='-o', color="blue", label="Bootstrapped Singles ± SEM")
        
        # # Reshuffled groups mean and SEM
        # rs_mean = reshuffled_matrix.mean(axis=1)
        # rs_sem = reshuffled_matrix.std(axis=1) / np.sqrt(n_iter)
        # plt.errorbar(frames, rs_mean, yerr=rs_sem, fmt='-o', color="orange", label="Reshuffled Groups ± SEM")
        
        # Real groups mean and SEM
        plt.errorbar(frames, real_matrix, yerr=real_sem, fmt='-o', color="red", label="Real Groups ± SEM")

        # Add significance markers for p < 0.05 vs singles
        if compute_stats:
            for i, p in enumerate(result["p_values_vs_singles"]):
                if p < 0.05:
                    plt.text(frames[i], real_matrix[i] + real_sem[i] + 0.05, "*", ha="center", color="red", fontsize=14)

        # Extract condition name(s) for title
        if "Condition" in df.columns:
            conditions = df["Condition"].unique()
            if len(conditions) == 1:
                condition_str = conditions[0]
            else:
                condition_str = ", ".join(map(str, conditions))
            plt.title(f"Framewise PI: Singles vs Groups (Condition: {condition_str})")
        else:
            plt.title("Framewise PI: Singles vs Groups")

        plt.xlabel("Frame")
        plt.ylabel("Proximity Index (PI)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return result
