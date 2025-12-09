import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For logarithmic normalization

def plot_trajectory_heatmaps(
    dataframe,
    condition,
    frame_bin_size=100,
    grid_size=1,
    title_size=18,
    label_size=14,
    tick_size=12,
    cbar_size=14
):
    """
    Plots heatmaps of the trajectory data based on density in 1x1 cm bins for each frame bin.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing trajectory data.
        condition (str): The condition to filter the dataframe on.
        frame_bin_size (int): Number of frames per bin (default 100).
        grid_size (int): Size of the grid in cm (default 1x1 cm).
        title_size (int): Font size for titles.
        label_size (int): Font size for axis labels.
        tick_size (int): Font size for tick labels.
        cbar_size (int): Font size for colorbar labels.
    """
    # Filter dataframe by condition
    df = dataframe[dataframe['Condition'] == condition]

    # Determine the frame range for each bin
    min_frame = df['Frame'].min()
    max_frame = df['Frame'].max()
    frame_bins = np.arange(min_frame, max_frame + frame_bin_size, frame_bin_size)

    # Create a subplot for each frame bin
    num_bins = len(frame_bins) - 1
    fig, axes = plt.subplots(1, num_bins, figsize=(5 * num_bins, 7), sharey=True)
    if num_bins == 1:
        axes = [axes]

    # Loop through each frame bin
    for i in range(num_bins):
        bin_start, bin_end = frame_bins[i], frame_bins[i + 1]
        df_bin = df[(df['Frame'] >= bin_start) & (df['Frame'] < bin_end)]

        # Define grid for histogram
        x_min, x_max = df_bin['X'].min(), df_bin['X'].max()
        y_min, y_max = df_bin['Y'].min(), df_bin['Y'].max()
        hist, xedges, yedges = np.histogram2d(
            df_bin['X'], df_bin['Y'],
            bins=[np.arange(x_min, x_max + grid_size, grid_size),
                  np.arange(y_min, y_max + grid_size, grid_size)]
        )

        # Plot heatmap
        cax = axes[i].pcolormesh(
            xedges, yedges, hist.T,
            cmap='viridis', shading='auto', norm=LogNorm(vmin=1)
        )

        # Titles and labels
        axes[i].set_title(f"Frames {bin_start}-{bin_end}", fontsize=title_size)
        axes[i].set_xlabel("X-coordinate (cm)", fontsize=label_size)
        if i == 0:
            axes[i].set_ylabel("Y-coordinate (cm)", fontsize=label_size)

        # Ticks
        axes[i].tick_params(axis='both', which='major', labelsize=tick_size)

        # Grid and aspect
        axes[i].grid(True)
        axes[i].set_aspect('equal')
        axes[i].set_xlim([0, 30])
        axes[i].set_ylim([0, 30])

    # Colorbar
    cbar = fig.colorbar(
        cax, ax=axes, orientation='vertical',
        label='Density (log scale)', fraction=0.02, pad=0.04
    )
    cbar.ax.tick_params(labelsize=tick_size)
    cbar.set_label('Density (log scale)', fontsize=cbar_size)

    # Overall title
    fig.suptitle(f"Heatmaps of Trajectories for Condition: {condition}", fontsize=title_size + 4)

    # Layout
    plt.subplots_adjust(right=0.85, top=0.85)
    plt.show()

def plot_prefindex_and_successrate_combined(
    dataframe,
    bin_size=100,
    target_x=14,
    target_y=2,
    radius=2,
    assign_max_if_unreached=True,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    condition_col='Condition',
    concentration_col='Concentration'
):
    """
    Combines Preference Index and Success Rate plots, following the logic of the
    original individual functions, but ensuring that for each concentration,
    'Fed' and '5h' conditions appear adjacent in the plots.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # -------------------------------------------------------------------------
    # 1. Define color palettes for condition types
    # -------------------------------------------------------------------------
    fed_conditions = sorted(dataframe.loc[dataframe[condition_col].str.contains('Fed'), condition_col].unique())
    fiveh_conditions = sorted(dataframe.loc[dataframe[condition_col].str.contains('5h'), condition_col].unique())
    other_conditions = sorted(dataframe.loc[
        ~dataframe[condition_col].str.contains('Fed|5h'), condition_col
    ].unique())

    fed_palette = list(reversed(sns.color_palette("Oranges", n_colors=max(3, len(fed_conditions)))))
    fiveh_palette = list(reversed(sns.color_palette("Blues", n_colors=max(3, len(fiveh_conditions)))))
    other_palette = list(reversed(sns.color_palette("Greens", n_colors=max(3, len(other_conditions)))))


    condition_colors = {}
    for c, color in zip(fed_conditions, fed_palette):
        condition_colors[c] = color
    for c, color in zip(fiveh_conditions, fiveh_palette):
        condition_colors[c] = color
    for c, color in zip(other_conditions, other_palette):
        condition_colors[c] = color

    # -------------------------------------------------------------------------
    # 2. Determine condition order (Fed first, then 5h, grouped by concentration)
    # -------------------------------------------------------------------------
    ordered_conditions = []
    unique_concs = sorted(dataframe[concentration_col].unique())

    for conc in unique_concs:
        conc_subset = dataframe[dataframe[concentration_col] == conc]
        fed = sorted(conc_subset.loc[conc_subset[condition_col].str.contains('Fed'), condition_col].unique())
        fiveh = sorted(conc_subset.loc[conc_subset[condition_col].str.contains('5h'), condition_col].unique())
        others = sorted(conc_subset.loc[
            ~conc_subset[condition_col].str.contains('Fed|5h'), condition_col
        ].unique())

        # Add in logical order: Fed → 5h → any others
        ordered_conditions.extend(fed + fiveh + others)

    # -------------------------------------------------------------------------
    # 3. Compute success rate (same logic as before)
    # -------------------------------------------------------------------------
    success_df = []
    for cond, subdf in dataframe.groupby(condition_col):
        success_per_individual = []
        for ind, df_ind in subdf.groupby(individual_col):
            df_ind = df_ind.sort_values(by=frame_col)
            success = np.sqrt((df_ind[x_col] - target_x)**2 + (df_ind[y_col] - target_y)**2) <= radius
            if assign_max_if_unreached and not success.any():
                success_per_individual.append(0)
            else:
                success_per_individual.append(success.any())
        success_df.append({
            'Condition': cond,
            'Success Rate': np.mean(success_per_individual)
        })
    success_df = pd.DataFrame(success_df)

    # -------------------------------------------------------------------------
    # 4. Compute Preference Index (same logic as before)
    # -------------------------------------------------------------------------
    prefindex_df = []
    for (cond, ind), subdf in dataframe.groupby([condition_col, individual_col]):
        subdf = subdf.sort_values(by=frame_col)
        subdf['Bin'] = (subdf[frame_col] // bin_size) * bin_size
        grouped = subdf.groupby('Bin')[[x_col, y_col]].mean()
        prefindex_df.append(pd.DataFrame({
            'Frame Bin': grouped.index,
            'Preference Index (Z1 - Z3)': -(grouped[y_col] - grouped[x_col]) / (abs(grouped[y_col]) + abs(grouped[x_col]) + 1e-9),
            'Condition': cond
        }))
    prefindex_df = pd.concat(prefindex_df, ignore_index=True)

    # -------------------------------------------------------------------------
    # 5. Plot both panels
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.barplot(
        data=success_df,
        x='Condition',
        y='Success Rate',
        order=ordered_conditions,
        palette=condition_colors,
        ax=axes[0]
    )
    axes[0].set_title("Success Rate per Condition")
    axes[0].set_xlabel("Condition")
    axes[0].set_ylabel("Success Rate")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')

    sns.pointplot(
        data=prefindex_df,
        x='Frame Bin',
        y='Preference Index (Z1 - Z3)',
        hue='Condition',
        hue_order=ordered_conditions,
        palette=condition_colors,
        errorbar='se',          # Use SEM for error bars
        dodge=True,        # Separate points by hue
        markers='o',
        capsize=0.1,
        ax=axes[1]
    )
    axes[1].axhline(0, linestyle='--', color='gray', linewidth=1)
    axes[1].set_title("Preference Index by Frame Bin and Condition")
    axes[1].set_ylabel("Preference Index (Z1 - Z3)")
    axes[1].legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def quantify_trajectory_zone_proportions(dataframe):
    """
    Quantifies the proportion of points in three dynamically defined zones along the y-axis 
    and visualizes them in a boxplot for all conditions in the dataframe.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing trajectory data.
    """
    # Get min/max y-values to dynamically set the zones
    y_min, y_max = dataframe['Y'].min(), dataframe['Y'].max()
    y_range = (y_max - y_min) / 3  # Divide into 3 equal zones
    y_bounds = [y_min + y_range, y_min + 2 * y_range]  # Compute boundaries

    # Initialize storage
    zone_counts = []
 
    # Process each condition and trial
    for condition in dataframe['Condition'].unique():
        df_condition = dataframe[dataframe['Condition'] == condition]
        
        for trial in df_condition['Trial'].unique():
            df_trial = df_condition[df_condition['Trial'] == trial]
            total_points = len(df_trial)  # Total points in trial
            
            # Avoid division by zero (in case a trial has no data)
            if total_points == 0:
                continue
            
            # Count points in each zone
            count_zone_1 = np.sum(df_trial['Y'] < y_bounds[0])  # Bottom zone
            count_zone_2 = np.sum((df_trial['Y'] >= y_bounds[0]) & (df_trial['Y'] < y_bounds[1]))  # Middle zone
            count_zone_3 = np.sum(df_trial['Y'] >= y_bounds[1])  # Top zone
            
            # Convert counts to proportions
            zone_counts.append({'Condition': condition, 'Trial': trial, 'Zone': 'Zone 1', 'Proportion': count_zone_1 / total_points})
            zone_counts.append({'Condition': condition, 'Trial': trial, 'Zone': 'Zone 2', 'Proportion': count_zone_2 / total_points})
            zone_counts.append({'Condition': condition, 'Trial': trial, 'Zone': 'Zone 3', 'Proportion': count_zone_3 / total_points})

    # Convert to DataFrame
    df_zones = pd.DataFrame(zone_counts)

    # Plot boxplot with stripplot overlay
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Zone', y='Proportion', hue='Condition', data=df_zones, palette='pastel', showfliers=False)
    sns.stripplot(x='Zone', y='Proportion', hue='Condition', data=df_zones, 
                  dodge=True, color='black', size=6, alpha=0.7, jitter=True, legend=False)

    # Labels and title
    plt.ylabel("Proportion of Points in Zone")
    plt.title("Zone-wise Proportion of Trajectories")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

def plot_metric_summary(
    df,
    value_col="Distance",
    bin_size=100,
    plot_type="both",  # 'box', 'strip', or 'both'
    condition_colors=None,
    show_summary=True
):
    """
    Visualize a metric (e.g. Distance, Speed, Preference Index) across frame bins for each condition.

    Each data point = mean(value_col) per individual (Trial) per bin.
    Optionally overlays mean ± SEM summary lines.

    Parameters:
        df (pd.DataFrame): DataFrame with columns ['Frame', 'Trial', 'Condition', value_col]
        value_col (str): Column name to analyze (e.g., 'Distance', 'Speed', 'PreferenceIndex')
        bin_size (int): Number of frames per bin
        plot_type (str): 'box', 'strip', or 'both'
        condition_colors (dict/list/None): Optional color mapping for conditions
        show_summary (bool): If True, overlay mean ± SEM per bin
    """

    # --- Copy and preprocess ---
    df = df.copy()

    if value_col not in df.columns:
        raise ValueError(f"'{value_col}' not found in DataFrame columns: {list(df.columns)}")

    # Ensure frame is int and create bins
    df["Frame"] = df["Frame"].astype(int)
    df["FrameBin"] = (df["Frame"] // bin_size) * bin_size
    df["FrameBinLabel"] = df["FrameBin"].astype(str) + "-" + (df["FrameBin"] + bin_size - 1).astype(str)

    # Collapse: mean per trial per bin
    df_agg = (
        df.groupby(["Condition", "Trial", "FrameBinLabel"], as_index=False)
          .agg({value_col: "mean"})
    )

    # Sort bins numerically by start frame
    def parse_start(x):
        try:
            return int(float(x.split("-")[0]))
        except Exception:
            return 0

    df_agg["FrameBinLabel"] = pd.Categorical(
        df_agg["FrameBinLabel"],
        ordered=True,
        categories=sorted(df_agg["FrameBinLabel"].unique(), key=parse_start)
    )

    # --- Colors ---
    unique_conditions = df_agg["Condition"].unique()
    if condition_colors is not None:
        if isinstance(condition_colors, dict):
            palette = {cond: condition_colors[cond] for cond in unique_conditions}
        elif isinstance(condition_colors, list):
            if len(condition_colors) < len(unique_conditions):
                raise ValueError("Not enough colors in list for all conditions.")
            palette = {cond: color for cond, color in zip(unique_conditions, condition_colors)}
        else:
            raise ValueError("condition_colors must be a dict, list, or None")
    else:
        palette = dict(zip(unique_conditions, sns.color_palette("muted", len(unique_conditions))))

    # --- Plot setup ---
    n_conditions = len(unique_conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(12, 4.5 * n_conditions), sharex=True)
    if n_conditions == 1:
        axes = [axes]

    for ax, condition in zip(axes, unique_conditions):
        df_cond = df_agg[df_agg["Condition"] == condition]
        color = palette[condition]

        # --- Boxplot ---
        if plot_type in ("box", "both"):
            sns.boxplot(
                data=df_cond,
                x="FrameBinLabel",
                y=value_col,
                color=color,
                fliersize=0,
                width=0.6,
                ax=ax
            )

        # --- Stripplot ---
        if plot_type in ("strip", "both"):
            sns.stripplot(
                data=df_cond,
                x="FrameBinLabel",
                y=value_col,
                color=color,
                size=5,
                jitter=True,
                alpha=0.6,
                ax=ax
            )

        # --- Summary line (mean ± SEM) ---
        if show_summary:
            df_summary = (
                df_cond.groupby("FrameBinLabel", as_index=False)
                .agg(
                    mean_value=(value_col, "mean"),
                    sem_value=(value_col, lambda x: x.std(ddof=1) / np.sqrt(len(x)))
                )
            )
            xvals = np.arange(len(df_summary))
            ax.plot(
                xvals,
                df_summary["mean_value"],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2,
                label=f"{condition} mean"
            )
            ax.fill_between(
                xvals,
                df_summary["mean_value"] - df_summary["sem_value"],
                df_summary["mean_value"] + df_summary["sem_value"],
                color=color,
                alpha=0.2
            )

        # --- Labels & formatting ---
        pretty_label = value_col.replace("_", " ").title()
        ax.set_title(f"Condition: {condition}", fontsize=14)
        ax.set_xlabel("Frame Bin")
        ax.set_ylabel(f"Mean {pretty_label} per Trial")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xticks(np.arange(len(df_cond["FrameBinLabel"].unique())))
        ax.set_xticklabels(df_cond["FrameBinLabel"].unique())
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def radial_sholl_heatmap_per_bin_normalized(
    df,
    target_x=14,
    target_y=2,
    bin_size=100,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    max_radius=10,
    condition_col='Condition',
    spatial_bin=1
):
    df = df.copy()

    # Get condition name
    condition_name = df[condition_col].iloc[0] if condition_col in df.columns and not df.empty else "Unknown"

    # Compute distances and assign Sholl rings using spatial_bin
    df['Distance'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    df['ShollRing'] = (df['Distance'] / spatial_bin).astype(int)
    df = df[df['ShollRing'] <= max_radius]

    # Assign frame bins
    df['FrameBin'] = (df[frame_col] // bin_size) * bin_size

    # Count unique individuals per ring per bin
    grouped = (
        df.groupby(['FrameBin', 'ShollRing'])[individual_col]
        .nunique()
        .reset_index(name='Count')
    )

    # Total per bin (for normalization)
    total_per_bin = (
        grouped.groupby('FrameBin')['Count']
        .sum()
        .reset_index(name='TotalCount')
    )

    # Merge and normalize per bin
    grouped = grouped.merge(total_per_bin, on='FrameBin')
    grouped['Proportion'] = grouped['Count'] / grouped['TotalCount']

    # Set up polar coordinates
    frame_bins = sorted(grouped['FrameBin'].unique())
    num_bins = len(frame_bins)
    theta = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    width = 2 * np.pi / num_bins

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=grouped['Proportion'].max())

    # Plot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})

    # --- Rotation and direction ---
    ax.set_theta_offset(np.pi / 4)   # start from 1 o’clock 
    ax.set_theta_direction(-1)       # clockwise

    for i, fb in enumerate(frame_bins):
        for r in range(max_radius + 1):
            val = grouped.loc[
                (grouped['FrameBin'] == fb) & (grouped['ShollRing'] == r),
                'Proportion'
            ]
            proportion = val.values[0] if not val.empty else 0

            ax.bar(
                x=theta[i],
                height=spatial_bin,
                width=width,
                bottom=r * spatial_bin,
                color=cmap(norm(proportion)),
                edgecolor='none'
            )

    # Clean up
    ax.grid(False)
    ax.set_frame_on(False)

    # --- Remove ring distance labels (keep ticks invisible) ---
    ax.set_yticks([])
    ax.set_yticklabels([])

    # Keep frame bin labels
    ax.set_xticks(theta)
    ax.set_xticklabels([str(fb) for fb in frame_bins], fontsize=9, rotation=90)

    ax.set_title(f'Radial Sholl (Per-Bin Normalized) – Condition: {condition_name}', va='bottom', fontsize=14)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.1)
    cbar.set_label('Proportion per Time Bin', fontsize=12)

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

def logistic_4param(r, A, L, k, r0):
    """
    Decreasing 4-parameter logistic:
    P(r) = A + (L - A) / (1 + exp(k*(r - r0)))
    - A: lower asymptote (r -> +inf)
    - L: upper asymptote (small r)
    - k: steepness (positive => decreasing)
    - r0: inflection point
    """
    return A + (L - A) / (1.0 + np.exp(k * (r - r0)))

def fit_4param_logistic(r, p, bounds=None, p0=None):
    """
    Fit logistic_4param to (r,p). Returns popt, pcov.
    """
    if bounds is None:
        bounds = ([0.0, 0.0, 0.0, np.min(r)], [1.0, 1.0, 20.0, np.max(r)])
    if p0 is None:
        p0 = [np.min(p), np.max(p), 0.5, np.median(r)]
    popt, pcov = curve_fit(logistic_4param, r, p, p0=p0, bounds=bounds, maxfev=10000)
    return popt, pcov


def plot_probability_and_bootstrap_logistic(
    df,
    target_x=14,
    target_y=2,
    success_radius=2,
    radius_steps=None,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    condition_col='Condition',
    concentration_col='Concentration',
    palette=None,
    n_boot=500,                # number of bootstrap resamples
    ci=95,                     # confidence level (percent)
    save_csv=True,
    csv_path="/mnt/data/logistic_params_bootstrap.csv",
    random_state=1
):
    """
    Fit 4-parameter logistic per condition, bootstrap over aggregated (r,p) pairs,
    plot empirical curves and fitted curves with CI bands, and return parameter table.

    Returns:
        prob_df, params_df, bootstrap_dict
    """

    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import os

    rng = np.random.default_rng(random_state)

    # 1) Compute probability curves
    prob_df, _, _ = compute_probability_derivative_by_condition(
        df=df,
        target_x=target_x,
        target_y=target_y,
        success_radius=success_radius,
        radius_steps=radius_steps,
        frame_col=frame_col,
        individual_col=individual_col,
        x_col=x_col,
        y_col=y_col,
        condition_col=condition_col,
        palette=None,
        plot=False
    )

    conditions = prob_df['Condition'].unique()

    # --------------------------------------------------------------
    # 2) Build palette (unchanged)
    # --------------------------------------------------------------
    if palette is None:
        fed_conditions = sorted(df.loc[df[condition_col].str.contains('Fed', na=False), condition_col].unique())
        fiveh_conditions = sorted(df.loc[df[condition_col].str.contains('5h', na=False), condition_col].unique())
        other_conditions = sorted(df.loc[~df[condition_col].str.contains('Fed|5h', na=False), condition_col].unique())

        fed_palette = list(reversed(sns.color_palette("Blues", n_colors=max(3, len(fed_conditions)))))
        fiveh_palette = list(reversed(sns.color_palette("Oranges", n_colors=max(3, len(fiveh_conditions)))))
        other_palette = list(reversed(sns.color_palette("Greens", n_colors=max(3, len(other_conditions)))))

        condition_colors = {}
        for c, col in zip(fed_conditions, fed_palette):
            condition_colors[c] = col
        for c, col in zip(fiveh_conditions, fiveh_palette):
            condition_colors[c] = col
        for c, col in zip(other_conditions, other_palette):
            condition_colors[c] = col

        palette = condition_colors.copy()

    # --------------------------------------------------------------
    # 3) Fit logistic + bootstrap (unchanged except formatting later)
    # --------------------------------------------------------------
    fit_summary = []
    bootstrap_results = {}

    for cond in conditions:

        sub = prob_df[prob_df['Condition'] == cond].sort_values('Radius')
        r = sub['Radius'].values
        p = sub['P_success_given_r'].values

        if len(r) < 4:
            fit_summary.append({
                "Condition": cond,
                "A": np.nan, "L": np.nan, "k": np.nan, "r0": np.nan,
                "A_se": np.nan, "L_se": np.nan, "k_se": np.nan, "r0_se": np.nan,
                "R2": np.nan
            })
            bootstrap_results[cond] = pd.DataFrame(columns=['A','L','k','r0'])
            continue

        # initial guesses
        A0 = np.clip(np.min(p), 0.0, 1.0)
        L0 = np.clip(np.max(p), 0.0, 1.0)
        k0 = 0.5
        r0_0 = np.median(r)
        lower_bounds = [0.0, 0.0, 0.0, np.min(r)]
        upper_bounds = [1.0, 1.0, 50.0, np.max(r)]

        try:
            popt, pcov = fit_4param_logistic(r, p, bounds=(lower_bounds, upper_bounds), p0=[A0, L0, k0, r0_0])
            perr = np.sqrt(np.diag(pcov))
            A_fit, L_fit, k_fit, r0_fit = popt
        except Exception:
            A_fit, L_fit, k_fit, r0_fit = [np.nan]*4
            perr = [np.nan]*4

        # enforce A <= L
        if (not np.isnan(A_fit)) and (A_fit > L_fit):
            A_fit, L_fit = L_fit, A_fit
            k_fit = -k_fit

        # R2
        if not np.isnan(A_fit):
            p_pred = logistic_4param(r, A_fit, L_fit, k_fit, r0_fit)
            ss_res = np.sum((p - p_pred)**2)
            ss_tot = np.sum((p - np.mean(p))**2)
            R2 = 1.0 - ss_res/ss_tot if ss_tot != 0 else np.nan
        else:
            R2 = np.nan

        fit_summary.append({
            "Condition": cond,
            "A": A_fit, "L": L_fit, "k": k_fit, "r0": r0_fit,
            "A_se": perr[0], "L_se": perr[1], "k_se": perr[2], "r0_se": perr[3],
            "R2": R2
        })

        # ---------------- Bootstrap ----------------
        boot_params = []
        n_pts = len(r)
        for b in range(n_boot):
            idx = rng.integers(0, n_pts, n_pts)
            r_bs = r[idx]
            p_bs = p[idx]

            try:
                popt_b, _ = fit_4param_logistic(r_bs, p_bs,
                                                bounds=(lower_bounds, upper_bounds),
                                                p0=[A0, L0, k0, r0_0])
                A_b, L_b, k_b, r0_b = popt_b
                if A_b > L_b:
                    A_b, L_b = L_b, A_b
                    k_b = -k_b
                boot_params.append([A_b, L_b, k_b, r0_b])
            except Exception:
                continue

        bootstrap_results[cond] = pd.DataFrame(boot_params, columns=['A','L','k','r0'])

    params_df = pd.DataFrame(fit_summary)

    # --------------------------------------------------------------
    # 4) CI computation (unchanged)
    # --------------------------------------------------------------
    alpha = 100 - ci
    lower_q = alpha/2
    upper_q = 100 - alpha/2

    for param in ['A','L','k','r0']:
        lowers, uppers = [], []
        for cond in params_df['Condition']:
            boot_df = bootstrap_results[cond]
            if boot_df.empty:
                lowers.append(np.nan)
                uppers.append(np.nan)
            else:
                lowers.append(np.nanpercentile(boot_df[param], lower_q))
                uppers.append(np.nanpercentile(boot_df[param], upper_q))
        params_df[f"{param}_ci_lower"] = lowers
        params_df[f"{param}_ci_upper"] = uppers

    # --------------------------------------------------------------
    # 5) Plot curves (unchanged)
    # --------------------------------------------------------------
    fig, axes = plt.subplots(2,1,figsize=(12,10),sharex=True)
    ax1, ax2 = axes

    # empirical
    for cond in conditions:
        sub = prob_df[prob_df['Condition']==cond]
        ax1.plot(sub['Radius'], sub['P_success_given_r'], 'o-', label=cond, color=palette.get(cond,None))
    ax1.legend()
    ax1.set_ylim(-0.02,1.02)
    ax1.grid(True)

    # fitted
    r_dense = np.linspace(prob_df['Radius'].min(), prob_df['Radius'].max(), 400)
    for cond in conditions:
        row = params_df[params_df['Condition']==cond].iloc[0]
        A_fit, L_fit, k_fit, r0_fit = row[['A','L','k','r0']]
        if not np.isnan(A_fit):
            ax2.plot(r_dense, logistic_4param(r_dense, A_fit, L_fit, k_fit, r0_fit),
                     color=palette[cond], label=cond)
    ax2.legend()
    ax2.set_ylim(-0.02,1.02)
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------------
    # 6) *** NEW: ± CI FORMAT FOR SUMMARY TABLE ***
    # --------------------------------------------------------------
    comparison_rows = []

    for _, r in params_df.iterrows():
        cond = r['Condition']

        def fmt_pm(est, lo, hi):
            if np.isnan(est) or np.isnan(lo) or np.isnan(hi):
                return "NA"
            err = (hi - lo) / 2
            return f"{est:.3f} ± {err:.3f}"

        comparison_rows.append({
            "Condition": cond,
            "A": fmt_pm(r["A"], r["A_ci_lower"], r["A_ci_upper"]),
            "L": fmt_pm(r["L"], r["L_ci_lower"], r["L_ci_upper"]),
            "k": fmt_pm(r["k"], r["k_ci_lower"], r["k_ci_upper"]),
            "r0": fmt_pm(r["r0"], r["r0_ci_lower"], r["r0_ci_upper"]),
            "R2": f"{r['R2']:.3f}" if not np.isnan(r['R2']) else "NA"
        })

    comparison_df = pd.DataFrame(comparison_rows).sort_values('Condition')

    # --------------------------------------------------------------
    # 7) Save
    # --------------------------------------------------------------
    if save_csv:
        params_df.to_csv(csv_path, index=False)

        comp_path = csv_path.replace(".csv", "_comparison.csv")
        comparison_df.to_csv(comp_path, index=False)

    # Return objects
    return {
        "prob_df": prob_df,
        "params_df": params_df,
        "comparison_table": comparison_df,
        "bootstrap_samples": bootstrap_results,
        "saved_params_csv": csv_path,
        "uploaded_image_path": "/mnt/data/26735b82-7703-41c2-8257-b65300fef926.png"
    }


def plot_post_success_dwell_total_filtered(
    df,
    target_x=14,
    target_y=2,
    success_radius=2,
    frame_col="Frame",
    individual_col="Individual",
    x_col="X",
    y_col="Y",
    condition_col="Condition"
):
    """
    Compute and plot post-success dwell time for each individual.
    Success = first time individual enters the success zone.
    Dwell time = total frames inside after first entry (multiple entries allowed).
    Unsuccessful individuals (dwell = 0) are excluded from plots.
    
    Returns:
        dwell_df: rows = individuals, columns = [Condition, dwell_time]
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # compute distances
    df = df.copy()
    df["dist"] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    df["inside"] = df["dist"] <= success_radius

    dwell_records = []

    for (cond, indiv), sub in df.groupby([condition_col, individual_col]):
        sub = sub.sort_values(frame_col)

        inside_frames = sub[sub["inside"]][frame_col].values
        if len(inside_frames) == 0:
            continue  # skip unsuccessful individuals

        first_entry = inside_frames[0]

        # sum all frames inside AFTER first entry
        after_first = sub[sub[frame_col] >= first_entry]
        dwell = after_first["inside"].sum()  # number of frames inside

        dwell_records.append([cond, indiv, dwell])

    dwell_df = pd.DataFrame(dwell_records, columns=[condition_col, individual_col, "DwellTime_seconds"])

    # Filter out zero dwell times (should already be excluded, but safe)
    dwell_df = dwell_df[dwell_df["DwellTime_seconds"] > 0]

    # Plot histogram + ECDF
    plt.figure(figsize=(12,6))

    # Histogram
    plt.subplot(1,2,1)
    sns.histplot(data=dwell_df, x="DwellTime_seconds", hue=condition_col, kde=False, bins=30)
    plt.xlabel("Post-success dwell time (seconds)")
    plt.ylabel("Count")
    plt.title("Histogram of Dwell Times (successful only)")

    # ECDF
    plt.subplot(1,2,2)
    for cond, sub in dwell_df.groupby(condition_col):
        sorted_vals = np.sort(sub["DwellTime_seconds"])
        yvals = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        plt.plot(sorted_vals, yvals, label=cond)
    plt.xlabel("Post-success dwell time (seconds)")
    plt.ylabel("ECDF")
    plt.title("Empirical CDF of Dwell Times (successful only)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return dwell_df

def run_full_trajectory_analysis(
    df,
    # General shared parameters
    bin_size=100,
    frame_bin_size=100,
    grid_size=1,
    target_x=14,
    target_y=2,
    radius=2,
    spatial_bin=1,
    success_radius=2,
    radius_steps=None,
    n_boot=500,
    ci=95,
    palette=None,
    save_csv=True,
    csv_path="/mnt/data/logistic_params_bootstrap.csv",
    random_state=1
):
    """
    Runs the complete analysis pipeline, producing all figures sequentially.
    Assumes all referenced functions already exist in the environment.
    """

    # ---------------------------------------------------------
    # 1. Trajectory Heatmaps per Condition
    # ---------------------------------------------------------
    print("Generating trajectory heatmaps...")
    for cond in df["Condition"].unique():
        print(f"  - Condition: {cond}")
        plot_trajectory_heatmaps(
            dataframe=df,
            condition=cond,
            frame_bin_size=frame_bin_size,
            grid_size=grid_size
        )

    # ---------------------------------------------------------
    # 2. Preference Index + Success Rate per Condition
    # ---------------------------------------------------------
    print("Generating pref-index and success-rate plots...")
    
    print(f"  - Condition: {cond}")
    plot_prefindex_and_successrate_combined(
        dataframe=df,
        bin_size=bin_size,
        target_x=target_x,
        target_y=target_y,
        radius=radius,
        assign_max_if_unreached=True
    )

    # ---------------------------------------------------------
    # 3. Trajectory Zone Proportions (no condition loop)
    # ---------------------------------------------------------
    print("Quantifying zone proportions...")
    quantify_trajectory_zone_proportions(df)

    # ---------------------------------------------------------
    # 4. Metric Summary (Speed)
    # ---------------------------------------------------------
    print("Generating metric summary plots (Speed)...")
    # plot_metric_summary(
    #     df,
    #     value_col="Speed",
    #     bin_size=bin_size,
    #     plot_type="both",  # box + strip
    #     condition_colors=None,
    #     show_summary=True
    # )

    # ---------------------------------------------------------
    # 5. Radial Sholl Heatmap per Condition
    # ---------------------------------------------------------
    print("Generating radial Sholl heatmaps...")
    for cond in df["Condition"].unique():
        print(f"  - Condition: {cond}")
        radial_sholl_heatmap_per_bin_normalized(
            df=df[df["Condition"] == cond],
            target_x=target_x,
            target_y=target_y,
            bin_size=bin_size,
            max_radius=10,
            spatial_bin=spatial_bin
        )

    # ---------------------------------------------------------
    # 6. Probability + Bootstrap Logistic Regression
    # ---------------------------------------------------------
    print("Generating logistic regression probability curves...")
    # plot_probability_and_bootstrap_logistic(
    #     df=df,
    #     target_x=target_x,
    #     target_y=target_y,
    #     success_radius=success_radius,
    #     radius_steps=radius_steps,
    #     palette=palette,
    #     n_boot=n_boot,
    #     ci=ci,
    #     save_csv=save_csv,
    #     csv_path=csv_path,
    #     random_state=random_state
    # )

      # ---------------------------------------------------------
    # 7. NEW: Post-success dwell analysis
    # ---------------------------------------------------------
    print("Computing post-success dwell time (successful only)...")

    dwell_df = plot_post_success_dwell_total_filtered(
        df=df,
        target_x=target_x,
        target_y=target_y,
        success_radius=success_radius
    )

    print("✔ Full analysis pipeline complete.")
