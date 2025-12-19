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

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_zone_1_over_time(dataframe, y_range=None):
    """
    Plots the mean proportion of points in Zone 1 over time for each condition.

    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing trajectory data.
        y_range (tuple or None): Tuple (ymin, ymax) to set fixed y-axis range. If None, auto-scale is used.
    """
    # Get min/max y-values to define the zones
    y_min, y_max = dataframe['Y'].min(), dataframe['Y'].max()
    y_bound = y_min + (y_max - y_min) / 3  # Zone 1 upper boundary

    frame_data = []

    for condition in dataframe['Condition'].unique():
        df_condition = dataframe[dataframe['Condition'] == condition]

        for trial in df_condition['Trial'].unique():
            df_trial = df_condition[df_condition['Trial'] == trial]

            for frame in df_trial['Frame'].unique():
                df_frame = df_trial[df_trial['Frame'] == frame]
                total_points = len(df_frame)

                if total_points == 0:
                    continue

                count_zone_1 = np.sum(df_frame['Y'] < y_bound)

                frame_data.append({
                    'Condition': condition,
                    'Frame': frame,
                    'Proportion': count_zone_1 / total_points
                })

    df_frames = pd.DataFrame(frame_data)

    # Plot
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        x='Frame', y='Proportion', hue='Condition', data=df_frames,
        estimator='mean', palette='pastel', lw=2
    )
    plt.ylabel("Mean Proportion in Zone 1")
    plt.xlabel("Frame")
    plt.title("Mean Proportion of Points in Zone 1 Over Time")
    plt.legend(title="Condition")

    # Optional y-axis range
    if y_range is not None:
        plt.ylim(y_range)

    plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_zone_means(dataframe):
    """
    Plots the mean proportion of points in each zone and connects them with lines within each condition.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing trajectory data.
        Assumes a column 'Starvation' with values '5h' or 'Fed' exists.
    """
    # Get min/max y-values to define the zones
    y_min, y_max = dataframe['Y'].min(), dataframe['Y'].max()
    y_range = (y_max - y_min) / 3  # Divide into 3 equal zones
    y_bounds = [y_min + y_range, y_min + 2 * y_range]  # Compute boundaries

    # Initialize storage for mean values
    zone_means = []

    # Compute means for each condition
    for condition in dataframe['Condition'].unique():
        df_condition = dataframe[dataframe['Condition'] == condition]
        starvation_status = df_condition['Starvation'].iloc[0]  # Get starvation status

        # Compute mean proportions for each zone
        mean_zone_1 = np.mean(df_condition['Y'] < y_bounds[0])
        mean_zone_2 = np.mean((df_condition['Y'] >= y_bounds[0]) & (df_condition['Y'] < y_bounds[1]))
        mean_zone_3 = np.mean(df_condition['Y'] >= y_bounds[1])

        zone_means.append({'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 1', 'Mean Proportion': mean_zone_1})
        zone_means.append({'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 2', 'Mean Proportion': mean_zone_2})
        zone_means.append({'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 3', 'Mean Proportion': mean_zone_3})

    # Convert to DataFrame
    df_means = pd.DataFrame(zone_means)

    # Create plot
    plt.figure(figsize=(8, 6))

    # Get unique conditions and categorize by starvation status
    conditions = df_means['Condition'].unique()
    starvation_conditions = df_means['Starvation'].unique()

    # Define color palettes
    red_shades = sns.color_palette("Reds", sum(df_means['Starvation'] == '5h'))
    blue_shades = sns.color_palette("Blues", sum(df_means['Starvation'] == 'Fed'))

    # Assign colors to conditions
    color_mapping = {}
    red_idx, blue_idx = 0, 0  # Track color indices for red and blue

    for condition in conditions:
        starvation_status = df_means[df_means['Condition'] == condition]['Starvation'].iloc[0]
        
        if starvation_status == '5h':  # Assign shades of red
            color_mapping[condition] = red_shades[red_idx]
            red_idx += 1
        else:  # Assign shades of blue
            color_mapping[condition] = blue_shades[blue_idx]
            blue_idx += 1

    # Plot means and connect them with lines
    for condition in conditions:
        df_cond = df_means[df_means['Condition'] == condition]
        
        plt.plot(df_cond['Zone'], df_cond['Mean Proportion'], marker='o', markersize=8, 
                 linestyle='-', linewidth=2, color=color_mapping[condition], label=condition)

    # Labels and title
    plt.ylabel("Mean Proportion of Points")
    plt.xlabel("Zone")
    plt.title("Mean Zone Proportions per Condition")
    plt.ylim(0, 1)  # Set y-axis from 0 to 1
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_hued_scatter(df, max_legend_items=10):
    """
    Plots (X, Y) coordinates stored in df['X'] and df['Y'], 
    with colors determined by df['Frame'], and a reduced legend.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing 'X', 'Y', and 'Frame' columns.
        max_legend_items (int): Maximum number of unique 'Frame' values shown in the legend.
    
    Returns:
        None
    """
    df['Frame'] = df['Frame'].astype(str)  # Ensure categorical coloring
    
    plt.figure(figsize=(8, 6))
    scatter = sns.scatterplot(
        data=df, x='X', y='Y', hue='Frame', palette='viridis', edgecolor='black'
    )
    
    # Limit the number of items in the legend
    legend_handles, legend_labels = plt.gca().get_legend_handles_labels()
    if len(legend_labels) > max_legend_items:
        plt.legend(legend_handles[:max_legend_items], legend_labels[:max_legend_items], 
                   title="Frame", fontsize='small', ncol=2, loc='upper left')
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Scatter Plot of (X, Y) Coordinates Colored by Frame")
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_distance_by_condition(
    df,
    frame_column='Frame',
    distance_column='Distance',
    condition_column='Condition',
    bin_size=100,
    num_bins_to_display=4,
    condition_colors=None  # <--- NEW
):
    """
    Plots boxplots of 'Distance' binned by 'Frame' intervals, comparing different conditions.
    Displays only the middle 'num_bins_to_display' bins.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing 'Distance', 'Frame', and 'Condition' columns.
        frame_column (str): The column name representing the frame number.
        distance_column (str): The column name representing the pre-calculated distance.
        condition_column (str): The column name representing the condition for comparison.
        bin_size (int): The number of frames to group together in each bin for boxplots.
        num_bins_to_display (int): The number of middle bins to display (e.g., 3 for the middle 3 bins).
        condition_colors (dict or list or None): Custom colors for conditions.
            - dict: {condition_name: color}
            - list: [color1, color2, ...] in the same order as unique conditions
            - None: defaults to a Viridis palette
    """
    # Ensure the necessary columns exist
    if not {frame_column, distance_column, condition_column}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain '{frame_column}', '{distance_column}', and '{condition_column}' columns.")
    
    # Add a column for the bin (Frame interval)
    df['Bin'] = (df[frame_column] // bin_size) * bin_size
    
    # Find the range of bins
    unique_bins = sorted(df['Bin'].unique())  # Get sorted unique bins
    total_bins = len(unique_bins)
    
    # Find the middle bins
    middle_start_index = (total_bins - num_bins_to_display) // 2
    middle_end_index = middle_start_index + num_bins_to_display
    
    # Filter the data to only include the middle bins
    middle_bins = unique_bins[middle_start_index:middle_end_index]
    df_filtered = df[df['Bin'].isin(middle_bins)]
    
    # Get unique conditions
    unique_conditions = df_filtered[condition_column].unique()
    
    # Handle colors
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
        palette = sns.color_palette("viridis", len(unique_conditions))

    # Plotting the boxplots for Distance within the filtered bins and grouped by Condition
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_filtered,
        x='Bin',
        y=distance_column,
        hue=condition_column,
        palette=palette
    )
    
    plt.title(f'Boxplots of Distance binned by Frame intervals (Middle {num_bins_to_display} bins) and grouped by Condition')
    plt.xlabel('Frame Interval')
    plt.ylabel('Distance')
    plt.xticks(rotation=45)  # Rotate x labels for readability

    # Move the hue legend outside the plot
    plt.legend(title=condition_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_distance_by_condition_average(
    df,
    frame_column='Frame',
    metrics=['Distance', 'AvgNeighborDist'],
    condition_column='Condition',
    trial_column='Trial',
    bin_size=100
):
    """
    Plots boxplots of each metric (e.g., 'Distance', 'AvgNeighborDist') averaged per 'Trial', 
    binned by 'Frame' intervals, comparing different conditions. Displays all bins for both metrics.

    Parameters:
        df (pd.DataFrame): The DataFrame containing metrics, 'Frame', 'Condition', and 'Trial' columns.
        frame_column (str): The column name representing the frame number.
        metrics (list): List of 1 or 2 column names (strings) to plot.
        condition_column (str): The column name representing the condition for comparison.
        trial_column (str): The column representing individual trials.
        bin_size (int): The number of frames to group together in each bin for boxplots.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    if not 1 <= len(metrics) <= 2:
        raise ValueError("You must provide 1 or 2 metric column names.")

    required_columns = {frame_column, condition_column, trial_column, *metrics}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")

    df = df.copy()
    df['Bin'] = (df[frame_column] // bin_size) * bin_size

    plot_data = []
    for metric in metrics:
        df_avg = (
            df.groupby(['Bin', condition_column, trial_column])[metric]
            .mean()
            .reset_index()
        )
        df_avg['Metric'] = metric
        df_avg['Value'] = df_avg[metric]
        plot_data.append(df_avg)

    df_plot = pd.concat(plot_data, ignore_index=True)

    conditions = sorted(df_plot[condition_column].unique())
    num_conditions = len(conditions)
    mid = num_conditions // 2

    red_shades = sns.color_palette("Reds", mid + (num_conditions % 2))
    blue_shades = sns.color_palette("Blues", mid)

    condition_palette = {
        condition: red_shades[i] if i < len(red_shades) else blue_shades[i - len(red_shades)]
        for i, condition in enumerate(conditions)
    }

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 6), sharey=False)

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        metric_df = df_plot[df_plot['Metric'] == metric]

        sns.boxplot(data=metric_df, x='Bin', y='Value', hue=condition_column,
                    palette=condition_palette, ax=ax)
        sns.stripplot(data=metric_df, x='Bin', y='Value', hue=condition_column,
                      dodge=True, color="black", alpha=0.6, jitter=True, legend=False, ax=ax)

        ax.set_title(f'{metric} Averaged per Trial (All Bins)')
        ax.set_xlabel('Frame Interval')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)

        # Put legend inside top right corner of the plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles, labels=labels, title=condition_column,
                  loc='upper right', frameon=True, fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_speed_by_condition(
    df,
    frame_column='Frame',
    speed_column='Speed',
    condition_column='Condition',
    concentration_column='Concentration',   # NEW: for ordering logic
    bin_size=100,
    num_bins_to_display=4
):
    """
    Plots boxplots of 'Speed' binned by 'Frame' intervals, comparing different conditions.
    Uses same colour logic and ordering as the combined PrefIndex/SuccessRate function.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    # -------------------------------------------------------------------------
    # 1. Validate input
    # -------------------------------------------------------------------------
    required = {frame_column, speed_column, condition_column}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")

    if concentration_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{concentration_column}' for colour ordering.")

    # -------------------------------------------------------------------------
    # 2. Build color palette logic (Fed / 5h / Others)
    # -------------------------------------------------------------------------
    fed_conditions = sorted(df.loc[df[condition_column].str.contains('Fed'), condition_column].unique())
    fiveh_conditions = sorted(df.loc[df[condition_column].str.contains('5h'), condition_column].unique())
    other_conditions = sorted(df.loc[~df[condition_column].str.contains('Fed|5h'), condition_column].unique())

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
    # 3. Determine consistent condition order
    # -------------------------------------------------------------------------
    ordered_conditions = []
    unique_concs = sorted(df[concentration_column].unique())

    for conc in unique_concs:
        sub = df[df[concentration_column] == conc]
        fed = sorted(sub.loc[sub[condition_column].str.contains('Fed'), condition_column].unique())
        fiveh = sorted(sub.loc[sub[condition_column].str.contains('5h'), condition_column].unique())
        others = sorted(sub.loc[~sub[condition_column].str.contains('Fed|5h'), condition_column].unique())
        ordered_conditions.extend(fed + fiveh + others)

    # -------------------------------------------------------------------------
    # 4. Bin frames
    # -------------------------------------------------------------------------
    df = df.copy()
    df['Bin'] = (df[frame_column] // bin_size) * bin_size

    unique_bins = sorted(df['Bin'].unique())
    total_bins = len(unique_bins)

    middle_start = (total_bins - num_bins_to_display) // 2
    middle_end = middle_start + num_bins_to_display

    middle_bins = unique_bins[middle_start:middle_end]
    df_filtered = df[df['Bin'].isin(middle_bins)]

    # -------------------------------------------------------------------------
    # 5. Plot boxplots
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=df_filtered,
        x='Bin',
        y=speed_column,
        hue=condition_column,
        hue_order=ordered_conditions,
        palette=condition_colors
    )

    plt.title(f'Boxplots of Speed by Frame Bins (Middle {num_bins_to_display})')
    plt.xlabel('Frame Interval')
    plt.ylabel('Speed')
    plt.xticks(rotation=45)

    plt.legend(title=condition_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_zone_means_subplot(df, filter_column='Concentration', filter_values=None, titles=["Early", "Mid", "Late"], condition_colors=None):
    """
    Creates a subplot panel of mean zone proportions for different time points and a chosen filter column.
    
    Parameters:
        df (DataFrame): The full dataset.
        filter_column (str): The column to filter by for subplot columns (default = 'Concentration').
        filter_values (list or None): List of unique values to use from filter_column. If None, auto-detected.
        titles (list): Titles for each timepoint row (default = ['Early', 'Mid', 'Late']).
    """
    if filter_values is None:
        filter_values = sorted(df[filter_column].dropna().unique())

    min_frame, max_frame = df['Frame'].min(), df['Frame'].max()
    frame_third = (max_frame - min_frame) // 3

    early_df = df[df['Frame'] <= min_frame + frame_third]
    mid_df = df[(df['Frame'] > min_frame + frame_third) & (df['Frame'] <= min_frame + 2 * frame_third)]
    late_df = df[df['Frame'] > min_frame + 2 * frame_third]

    dataframes = [early_df, mid_df, late_df]

    num_rows = len(dataframes)
    num_cols = len(filter_values)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharey=True, sharex=True)

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Default palettes (used only if condition_colors not provided)
    red_shades = sns.color_palette("Reds", 3)
    blue_shades = sns.color_palette("Blues", 3)

    red_shades = [(min(r, 1), min(g, 1), min(b, 1)) for r, g, b in red_shades]
    blue_shades = [(min(r * 0.8, 1), min(g * 0.9, 1), min(b * 1.1, 1)) for r, g, b in blue_shades]

    for row_idx, (df_time, title) in enumerate(zip(dataframes, titles)):
        for col_idx, value in enumerate(filter_values):
            ax = axes[row_idx][col_idx]

            df_filtered = df_time[df_time[filter_column] == value]

            if df_filtered.empty:
                ax.set_visible(False)
                continue

            y_min, y_max = df_filtered['Y'].min(), df_filtered['Y'].max()
            y_range = (y_max - y_min) / 3
            y_bounds = [y_min + y_range, y_min + 2 * y_range]

            zone_means = []

            for condition in df_filtered['Condition'].unique():
                df_condition = df_filtered[df_filtered['Condition'] == condition]
                starvation_status = df_condition['Starvation'].iloc[0]

                mean_zone_1 = np.mean(df_condition['Y'] < y_bounds[0])
                mean_zone_2 = np.mean((df_condition['Y'] >= y_bounds[0]) & (df_condition['Y'] < y_bounds[1]))
                mean_zone_3 = np.mean(df_condition['Y'] >= y_bounds[1])

                zone_means.extend([
                    {'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 1', 'Mean Proportion': mean_zone_1},
                    {'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 2', 'Mean Proportion': mean_zone_2},
                    {'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 3', 'Mean Proportion': mean_zone_3}
                ])

            df_means = pd.DataFrame(zone_means)

            # Handle colors
            if condition_colors is not None:
                if isinstance(condition_colors, dict):
                    color_mapping = {cond: condition_colors[cond] for cond in df_means['Condition'].unique()}
                elif isinstance(condition_colors, list):
                    unique_conditions = df_means['Condition'].unique()
                    color_mapping = {cond: color for cond, color in zip(unique_conditions, condition_colors)}
                else:
                    raise ValueError("condition_colors must be a dict, list, or None")
            else:
                # Fallback: old red/blue logic
                color_mapping = {}
                red_idx, blue_idx = 0, 0
                for condition in df_means['Condition'].unique():
                    starvation_status = df_means[df_means['Condition'] == condition]['Starvation'].iloc[0]
                    if starvation_status == '5h':
                        color_mapping[condition] = blue_shades[blue_idx]
                        blue_idx = (blue_idx + 1) % len(blue_shades)
                    else:
                        color_mapping[condition] = red_shades[red_idx]
                        red_idx = (red_idx + 1) % len(red_shades)

            # Plot
            for condition in df_means['Condition'].unique():
                df_cond = df_means[df_means['Condition'] == condition]
                ax.plot(
                    df_cond['Zone'],
                    df_cond['Mean Proportion'],
                    marker='o', markersize=6,
                    linestyle='-', linewidth=4,
                    color=color_mapping[condition],
                    label=condition
                )

            ax.set_ylim(0, 1)
            if row_idx == 0:
                ax.set_title(f"{filter_column}: {value}")
                ax.legend(title="Condition", loc='upper right', fontsize=10, frameon=True)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\nMean Proportion")
            if row_idx == num_rows - 1:
                ax.set_xlabel("Zone")

    plt.tight_layout()
    plt.show()


# Example Usage:
# dataframes = [primary_df_interp_early, primary_df_interp_mid, primary_df_interp_late]
# concentrations = ['10-3', '10-4', '10-5']
# titles = ["Early", "Mid", "Late"]

# plot_zone_means_subplot(dataframes, concentrations, titles)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_two_distance_by_condition_averages(
    df1,
    df2,
    titles=('Dataset 1', 'Dataset 2'),
    frame_column='Frame',
    distance_column='Distance',
    condition_column='Condition',
    trial_column='Trial',
    bin_size=100
):
    """
    Plots two side-by-side boxplots of 'Distance' averaged per 'Trial', binned by 'Frame' intervals,
    comparing different conditions, one for each input DataFrame.
    
    Parameters:
        df1, df2 (pd.DataFrame): DataFrames with 'Distance', 'Frame', 'Condition', and 'Trial' columns.
        titles (tuple): Titles for the two subplots.
        frame_column (str): Column name for frame number.
        distance_column (str): Column name for pre-calculated distance.
        condition_column (str): Column name for condition comparison.
        trial_column (str): Column name for individual trials.
        bin_size (int): Number of frames to group in each bin.
    """
    def preprocess(df):
        required_columns = {frame_column, distance_column, condition_column, trial_column}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")
        df = df.copy()
        df['Bin'] = (df[frame_column] // bin_size) * bin_size
        df_avg = df.groupby(['Bin', condition_column, trial_column])[distance_column].mean().reset_index()
        return df_avg

    df1_avg = preprocess(df1)
    df2_avg = preprocess(df2)

    # Combine both to determine consistent condition ordering and color palette
    combined_conditions = sorted(set(df1_avg[condition_column].unique()).union(df2_avg[condition_column].unique()))
    num_conditions = len(combined_conditions)
    mid = num_conditions // 2

    red_shades = sns.color_palette("Reds", mid + (num_conditions % 2))
    blue_shades = sns.color_palette("Blues", mid)
    
    condition_palette = {
        condition: red_shades[i] if i < len(red_shades) else blue_shades[i - len(red_shades)]
        for i, condition in enumerate(combined_conditions)
    }

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    for ax, df_avg, title in zip(axes, [df1_avg, df2_avg], titles):
        sns.boxplot(data=df_avg, x='Bin', y=distance_column, hue=condition_column,
                    palette=condition_palette, ax=ax)
        sns.stripplot(data=df_avg, x='Bin', y=distance_column, hue=condition_column,
                      dodge=True, color="black", alpha=0.6, jitter=True, legend=False, ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Frame Interval')
        ax.set_ylabel('Average Distance per Trial to the Odour Source')
        ax.tick_params(axis='x', rotation=45)

    # Legend inside the left plot
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(
        handles, labels,
        title=condition_column,
        loc='upper left',
        bbox_to_anchor=(0, 1),
        frameon=True
    )
    
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sbs

def prepare_aggregated_over_time(df, y_var):
    """
    Aggregates AvgNeighborDist or Distance over time separately for Real (by Trial)
    and Simulated (by SimGroupID), preserving variation across replicates.
    """
    real = df[df['GroupType'] == 'Real'].copy()
    sim = df[df['GroupType'] == 'Simulated'].copy()

    # Add GroupID for consistent naming
    real['GroupID'] = real['Trial']
    sim['GroupID'] = sim['SimGroupID']

    combined = pd.concat([real, sim], ignore_index=True)

    # Now group by replicate (Trial or SimGroupID), Frame, and Condition
    grouped = (
        combined.groupby(['Condition', 'GroupType', 'GroupID', 'Frame'])[y_var]
        .mean()
        .reset_index()
    )

    return grouped

def plot_distances_over_time(df, y_vars=('AvgNeighborDist', 'Distance'), per_condition=False):
    """
    Plots neighbor and center distances over time for Real vs Simulated groups.

    Parameters:
    -----------
    df : pd.DataFrame
        Must contain: ['Frame', 'Condition', 'GroupType', 'AvgNeighborDist', 'Distance']
    y_vars : tuple of str
        Y-axis variables to plot (defaults to neighbor & center distance)
    per_condition : bool
        If True, creates a subplot per condition; else uses 1x2 subplot layout
    """
    import seaborn as sns

    if per_condition:
        for y_var in y_vars:
            if y_var not in df.columns:
                print(f"Missing column: {y_var}")
                continue

            g = sns.relplot(
                data=df,
                x='Frame',
                y=y_var,
                hue='GroupType',
                col='Condition',
                kind='line',
                facet_kws={'sharey': False, 'sharex': True},
                height=4,
                aspect=1.3,
            )
            g.fig.subplots_adjust(top=0.85)
            g.fig.suptitle(f"{y_var} Over Time by Condition")
            plt.show()

    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

        for ax, y_var in zip(axes, y_vars):
            if y_var not in df.columns:
                print(f"Column '{y_var}' not found in DataFrame.")
                continue

            # Aggregate by Frame, Condition, GroupType
            agg_df = prepare_aggregated_over_time(df, y_var)

            sbs.lineplot(
                data=agg_df,
                x='Frame',
                y=y_var,
                hue='GroupType',
                style='Condition',
                errorbar='sd',  # now standard deviation across replicates per frame
                ax=ax
            )

        plt.tight_layout()
        plt.show()

import matplotlib.pyplot as plt
import seaborn as sbs
import numpy as np
import pandas as pd

def plot_binned_summary(df, bin_size=100):
    """
    Plots AvgNeighborDist and Distance as boxplots across binned frames,
    comparing one real and one simulated condition.

    Parameters:
    -----------
    df : pd.DataFrame
        Combined dataframe containing both Real and Simulated data for two conditions.
        Must include columns: ['Frame', 'Distance', 'AvgNeighborDist', 'GroupType', 'Trial', 'SimGroupID', 'Condition']
    
    bin_size : int
        Frame bin size for aggregating data, default = 100.
    """
    df = df.copy()

    # Validate expected structure
    group_types = df['GroupType'].unique()
    if set(group_types) != {'Real', 'Simulated'}:
        raise ValueError(f"Expected GroupType to contain 'Real' and 'Simulated', got {group_types}")

    conditions = df['Condition'].unique()
    if len(conditions) != 2:
        raise ValueError(f"Expected 2 conditions, found {len(conditions)}: {conditions}")

    # Identify real/sim condition labels
    real_condition = df[df['GroupType'] == 'Real']['Condition'].unique()[0]
    sim_condition = df[df['GroupType'] == 'Simulated']['Condition'].unique()[0]

    # Consistent group ID and frame bins
    df['GroupID'] = np.where(df['GroupType'] == 'Real', df['Trial'], df['SimGroupID'])
    df['FrameBin'] = (df['Frame'] // bin_size) * bin_size

    # Metrics to plot
    metrics = ['AvgNeighborDist', 'Distance']
    fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=False)

    for ax, metric in zip(axes, metrics):
        # Mean per (GroupType, GroupID, FrameBin)
        summary_df = (
            df.groupby(['GroupType', 'Condition', 'GroupID', 'FrameBin'])[metric]
            .mean()
            .reset_index()
        )

        sbs.boxplot(
            data=summary_df,
            x='FrameBin',
            y=metric,
            hue='GroupType',
            palette='pastel',
            ax=ax,
            showfliers=False
        )

        sbs.stripplot(
            data=summary_df,
            x='FrameBin',
            y=metric,
            hue='GroupType',
            dodge=True,
            color='black',
            size=3,
            alpha=0.5,
            ax=ax,
            legend=False
        )

        ax.set_title(f"{metric} per {bin_size}-Frame Bin\nReal: {real_condition} | Sim: {sim_condition}")
        ax.set_xlabel("Frame Bin")
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=45)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='GroupType', bbox_to_anchor=(1.02, 0.95), loc='upper left')

    plt.tight_layout()
    plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbs

def boxplot_speed_by_genotype(df, frame_col='Frame', speed_col='Speed', genotype_col='Genotype', bin_size=100):
    """
    Plots a boxplot of Speed per Genotype in 100-frame bins, with Speed clamped between 0.5 and 2.0.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with columns including 'Frame', 'Speed', and 'Genotype'.
    frame_col : str
        Column representing frame/time.
    speed_col : str
        Column representing speed.
    genotype_col : str
        Column representing genotype or grouping.
    bin_size : int
        Frame bin size (default 100).
    """
    df = df.copy()

    # Forward-fill Speed to handle NaNs
    df[speed_col] = df[speed_col].fillna(method='ffill')

    # Clamp Speed values between 0.5 and 2.0
    df[speed_col] = df[speed_col].clip(lower=0.5, upper=2.0)

    # Drop rows with any remaining NaNs
    df.dropna(subset=[speed_col, frame_col, genotype_col], inplace=True)

    # Create frame bins
    df['FrameBin'] = (df[frame_col] // bin_size) * bin_size

    # Plot boxplot
    plt.figure(figsize=(14, 6))
    sbs.boxplot(
        data=df,
        x='FrameBin',
        y=speed_col,
        hue=genotype_col,
        palette='Set2',
        showfliers=False
    )
    plt.xlabel('Frame (binned)')
    plt.ylabel('Speed (clamped to 0.5–2.0)')
    plt.title('Speed Distribution per Genotype Over Time')
    plt.legend(title='Genotype', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sbs
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sbs
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_speed_kde_by_genotype(df, speed_col='Speed', genotype_col='Genotype', clamp_range=(0.5, 2.0)):
    """
    Plots KDEs of Speed values by Genotype, excluding values outside clamp_range.
    Adds vertical lines at the mean speed per genotype.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with speed and genotype columns.
    speed_col : str
        Column name for speed values.
    genotype_col : str
        Column name for genotype categories.
    clamp_range : tuple
        (min_speed, max_speed) to exclude outliers.
    """
    df = df.copy()
    df = df[[speed_col, genotype_col]].dropna()

    # Filter values within clamp range
    df = df[(df[speed_col] >= clamp_range[0]) & (df[speed_col] <= clamp_range[1])]

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("muted", df[genotype_col].nunique())

    for i, (genotype, group) in enumerate(df.groupby(genotype_col)):
        color = palette[i]
        # Plot KDE
        sns.kdeplot(
            data=group,
            x=speed_col,
            fill=False,
            color=color,
            linewidth=2,
            label=genotype
        )
        # Mean line
        mean_speed = group[speed_col].mean()
        plt.axvline(mean_speed, color=color, linestyle='--', alpha=0.7)

    plt.title('Speed Distribution (KDE + Mean Lines)')
    plt.xlabel('Speed')
    plt.ylabel('Density')
    plt.legend(title='Parameter')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_preference_index_over_time(
    dataframe,
    condition_colors=None
):
    """
    Plots the mean Preference Index over time for each condition.
    Preference Index = (Zone 1 - Zone 3) / (Zone 1 + Zone 2 + Zone 3)

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with columns ['Y', 'Frame', 'Trial', 'Condition']
        condition_colors (dict, list, or None): 
            - dict: {"ConditionA": "#FF0000", "ConditionB": "#0000FF"}
            - list: ["#FF0000", "#0000FF", "#00FF00"] assigned in order of unique conditions
            - None: use default red/blue scheme
    """
    # Define Y boundaries for 3 horizontal zones
    y_min, y_max = dataframe['Y'].min(), dataframe['Y'].max()
    y_range = (y_max - y_min) / 3
    zone_bounds = [y_min, y_min + y_range, y_min + 2 * y_range, y_max]

    frame_data = []

    for condition in dataframe['Condition'].unique():
        df_condition = dataframe[dataframe['Condition'] == condition]

        for trial in df_condition['Trial'].unique():
            df_trial = df_condition[df_condition['Trial'] == trial]

            for frame in df_trial['Frame'].unique():
                df_frame = df_trial[df_trial['Frame'] == frame]
                if df_frame.empty:
                    continue

                # Count points in each zone
                z1 = np.sum(df_frame['Y'] < zone_bounds[1])
                z2 = np.sum((df_frame['Y'] >= zone_bounds[1]) & (df_frame['Y'] < zone_bounds[2]))
                z3 = np.sum(df_frame['Y'] >= zone_bounds[2])
                total = z1 + z2 + z3

                if total == 0:
                    continue

                # Compute preference index
                pi = (z1 - z3) / total

                frame_data.append({
                    'Condition': condition,
                    'Trial': trial,
                    'Frame': frame,
                    'PreferenceIndex': pi
                })

    df_frames = pd.DataFrame(frame_data)

    # Handle colors
    unique_conditions = df_frames['Condition'].unique()
    if isinstance(condition_colors, dict):
        palette = condition_colors
    elif isinstance(condition_colors, list):
        palette = dict(zip(unique_conditions, condition_colors))
    else:
        # Default: fallback red/blue scheme
        red_shades = sns.color_palette("Reds", 3).as_hex()
        blue_shades = sns.color_palette("Blues", 3).as_hex()
        palette = {}
        red_idx, blue_idx = 0, 0
        for cond in unique_conditions:
            if "5h" in cond:
                palette[cond] = blue_shades[blue_idx % len(blue_shades)]
                blue_idx += 1
            else:
                palette[cond] = red_shades[red_idx % len(red_shades)]
                red_idx += 1

    # Plot: mean PI over time with confidence intervals per condition
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_frames,
        x='Frame',
        y='PreferenceIndex',
        hue='Condition',
        estimator='mean',
        errorbar='se',
        palette=palette,
        lw=2
    )

    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.ylabel("Preference Index (Z1 - Z3)")
    plt.xlabel("Frame")
    plt.title("Mean Preference Index Over Time")
    plt.ylim(-1, 1)  # Hard limit
    plt.legend(title="Condition")
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_speed_kde_by_starvation(df, speed_col='Speed', starvation_col='Starvation', clamp_range=(0.5, 2.0)):
    """
    Plots KDEs of Speed values by Starvation state, excluding values outside clamp_range.
    Adds vertical lines at the mean speed per starvation group.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with speed and starvation columns.
    speed_col : str
        Column name for speed values.
    starvation_col : str
        Column name for starvation categories.
    clamp_range : tuple
        (min_speed, max_speed) to exclude outliers.
    """
    df = df.copy()
    df = df[[speed_col, starvation_col]].dropna()

    # Filter values within clamp range
    df = df[(df[speed_col] >= clamp_range[0]) & (df[speed_col] <= clamp_range[1])]

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("muted", df[starvation_col].nunique())

    for i, (group_name, group_df) in enumerate(df.groupby(starvation_col)):
        color = palette[i]
        # Plot KDE
        sns.kdeplot(
            data=group_df,
            x=speed_col,
            fill=False,
            color=color,
            linewidth=2,
            label=group_name
        )
        # Mean line
        mean_speed = group_df[speed_col].mean()
        plt.axvline(mean_speed, color=color, linestyle='--', alpha=0.7)

    plt.title('Speed Distribution by Starvation (KDE + Mean Lines)')
    plt.xlabel('Speed')
    plt.ylabel('Density')
    plt.legend(title='Starvation')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_preference_index_boxplots(dataframe, bin_size=100, condition_colors=None):
    """
    Plots boxplots of the Preference Index binned by frame ranges for each condition.

    Preference Index = (Zone 1 - Zone 3) / (Zone 1 + Zone 2 + Zone 3)

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with columns ['Y', 'Frame', 'Trial', 'Condition']
        bin_size (int): Number of frames per bin (default 100)
        condition_colors (dict or list or None): Custom colors for conditions.
            - dict: {condition_name: color}
            - list: [color1, color2, ...] in the same order as unique conditions
            - None: defaults to seaborn 'muted' palette
    """
    # Define Y boundaries for 3 horizontal zones
    y_min, y_max = dataframe['Y'].min(), dataframe['Y'].max()
    y_range = (y_max - y_min) / 3
    zone_bounds = [y_min, y_min + y_range, y_min + 2 * y_range, y_max]

    frame_data = []

    for condition in dataframe['Condition'].unique():
        df_condition = dataframe[dataframe['Condition'] == condition]

        for trial in df_condition['Trial'].unique():
            df_trial = df_condition[df_condition['Trial'] == trial]

            for frame in df_trial['Frame'].unique():
                df_frame = df_trial[df_trial['Frame'] == frame]
                if df_frame.empty:
                    continue

                z1 = np.sum(df_frame['Y'] < zone_bounds[1])
                z2 = np.sum((df_frame['Y'] >= zone_bounds[1]) & (df_frame['Y'] < zone_bounds[2]))
                z3 = np.sum(df_frame['Y'] >= zone_bounds[2])
                total = z1 + z2 + z3

                if total == 0:
                    continue

                pi = (z1 - z3) / total

                frame_int = int(frame)
                bin_start = (frame_int // bin_size) * bin_size
                bin_end = bin_start + bin_size - 1
                bin_label = f"{bin_start}-{bin_end}"

                frame_data.append({
                    'Condition': condition,
                    'Trial': trial,
                    'FrameBin': bin_label,
                    'PreferenceIndex': pi
                })

    df_bins = pd.DataFrame(frame_data)

    # Handle colors
    unique_conditions = df_bins['Condition'].unique()
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
        palette = sns.color_palette("muted", len(unique_conditions))

    # Plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        data=df_bins,
        x='FrameBin',
        y='PreferenceIndex',
        hue='Condition',
        palette=palette
    )

    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.ylabel("Preference Index (Z1 - Z3)")
    plt.xlabel("Frame Bin")
    plt.title("Preference Index by Frame Bin and Condition")
    plt.legend(title="Condition", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_zone_means_subplot_default(df, filter_column='Concentration', filter_values=None, titles=["Early", "Mid", "Late"]):
    """
    Creates a subplot panel of mean zone proportions for different time points and a chosen filter column.

    Parameters:
        df (DataFrame): The full dataset.
        filter_column (str): The column to filter by for subplot columns (default = 'Concentration').
        filter_values (list or None): List of unique values to use from filter_column. If None, auto-detected.
        titles (list): Titles for each timepoint row (default = ['Early', 'Mid', 'Late']).
    """
    if filter_values is None:
        filter_values = sorted(df[filter_column].dropna().unique())

    min_frame, max_frame = df['Frame'].min(), df['Frame'].max()
    frame_third = (max_frame - min_frame) // 3

    early_df = df[df['Frame'] <= min_frame + frame_third]
    mid_df = df[(df['Frame'] > min_frame + frame_third) & (df['Frame'] <= min_frame + 2 * frame_third)]
    late_df = df[df['Frame'] > min_frame + 2 * frame_third]

    dataframes = [early_df, mid_df, late_df]

    num_rows = len(dataframes)
    num_cols = len(filter_values)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharey=True, sharex=True)

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for row_idx, (df_time, title) in enumerate(zip(dataframes, titles)):
        for col_idx, value in enumerate(filter_values):
            ax = axes[row_idx][col_idx]

            df_filtered = df_time[df_time[filter_column] == value]

            if df_filtered.empty:
                ax.set_visible(False)
                continue

            y_min, y_max = df_filtered['Y'].min(), df_filtered['Y'].max()
            y_range = (y_max - y_min) / 3
            y_bounds = [y_min + y_range, y_min + 2 * y_range]

            zone_means = []

            for condition in df_filtered['Condition'].unique():
                df_condition = df_filtered[df_filtered['Condition'] == condition]
                starvation_status = df_condition['Starvation'].iloc[0]

                mean_zone_1 = np.mean(df_condition['Y'] < y_bounds[0])
                mean_zone_2 = np.mean((df_condition['Y'] >= y_bounds[0]) & (df_condition['Y'] < y_bounds[1]))
                mean_zone_3 = np.mean(df_condition['Y'] >= y_bounds[1])

                zone_means.extend([
                    {'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 1', 'Mean Proportion': mean_zone_1},
                    {'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 2', 'Mean Proportion': mean_zone_2},
                    {'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 3', 'Mean Proportion': mean_zone_3}
                ])

            df_means = pd.DataFrame(zone_means)

            for condition in df_means['Condition'].unique():
                df_cond = df_means[df_means['Condition'] == condition]

                ax.plot(df_cond['Zone'], df_cond['Mean Proportion'], marker='o', markersize=6,
                        linestyle='-', linewidth=4, label=condition)  # No color argument here

            ax.set_ylim(0, 1)
            if row_idx == 0:
                ax.set_title(f"{filter_column}: {value}")
                ax.legend(title="Condition", loc='upper right', fontsize=10, frameon=True)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\nMean Proportion")
            if row_idx == num_rows - 1:
                ax.set_xlabel("Zone")

    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import directed_hausdorff

def compute_and_plot_dhd(df, condition_col='Condition', trial_col='Trial', x_col='X', y_col='Y'):
    """
    Compute Directed Hausdorff Distance (DHD) between trials from two unique conditions in a DataFrame.
    Plots DHDs for A→B and B→A directions.

    Args:
        df (pd.DataFrame): Must contain condition_col, trial_col, x_col, y_col.
        condition_col (str): Column name for condition labels (2 unique values).
        trial_col (str): Column name for trial IDs.
        x_col (str): Column name for X coordinate.
        y_col (str): Column name for Y coordinate.
    """

    # Validate input
    conditions = df[condition_col].unique()
    assert len(conditions) == 2, "DataFrame must contain exactly 2 unique conditions."
    cond_a, cond_b = conditions

    # Get trials
    trials_a = df[df[condition_col] == cond_a][trial_col].unique()
    trials_b = df[df[condition_col] == cond_b][trial_col].unique()

    results = []

    # A → B
    for trial_a in trials_a:
        coords_a = df[(df[trial_col] == trial_a) & (df[condition_col] == cond_a)][[x_col, y_col]].values
        for trial_b in trials_b:
            coords_b = df[(df[trial_col] == trial_b) & (df[condition_col] == cond_b)][[x_col, y_col]].values
            dh_ab = directed_hausdorff(coords_a, coords_b)[0]
            results.append({
                'From': cond_a,
                'To': cond_b,
                'Trial_From': trial_a,
                'Trial_To': trial_b,
                'Directed_Hausdorff': dh_ab
            })

    # B → A
    for trial_b in trials_b:
        coords_b = df[(df[trial_col] == trial_b) & (df[condition_col] == cond_b)][[x_col, y_col]].values
        for trial_a in trials_a:
            coords_a = df[(df[trial_col] == trial_a) & (df[condition_col] == cond_a)][[x_col, y_col]].values
            dh_ba = directed_hausdorff(coords_b, coords_a)[0]
            results.append({
                'From': cond_b,
                'To': cond_a,
                'Trial_From': trial_b,
                'Trial_To': trial_a,
                'Directed_Hausdorff': dh_ba
            })

    # Create result DataFrame
    dist_df = pd.DataFrame(results)
    dist_df['Direction'] = dist_df['From'] + ' → ' + dist_df['To']

    # Plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dist_df, x='Direction', y='Directed_Hausdorff', palette='Set2')
    sns.stripplot(data=dist_df, x='Direction', y='Directed_Hausdorff', color='black', alpha=0.5, jitter=True)
    plt.title('Directed Hausdorff Distances Between Trials')
    plt.ylabel('Directed Hausdorff Distance')
    plt.xlabel('Direction')
    plt.tight_layout()
    plt.show()

    return dist_df


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_plot_target_acquisition(
    df,
    target_x=14,
    target_y=2,
    radius=2,
    assign_max_if_unreached=True,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    condition_col='Condition',
    condition_colors=None  # <--- NEW
):
    df = df.copy()
    
    # Compute distance from target
    df['Distance'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)

    # First frame within radius
    within = df[df['Distance'] <= radius]
    first_hits = within.groupby(individual_col)[frame_col].min().reset_index()
    first_hits.columns = [individual_col, 'TargetFrame']

    # Frame bounds for each individual
    frame_bounds = df.groupby(individual_col)[frame_col].agg(['min', 'max']).reset_index()
    frame_bounds.columns = [individual_col, 'StartFrame', 'LastFrame']

    # Merge and compute time to target
    time_to_target = frame_bounds.merge(first_hits, on=individual_col, how='left')

    if assign_max_if_unreached:
        time_to_target['TargetFrame'] = time_to_target['TargetFrame'].fillna(time_to_target['LastFrame'])

    time_to_target['TimeToTarget'] = time_to_target['TargetFrame'] - time_to_target['StartFrame']

    # Merge in condition info
    conditions = df[[individual_col, condition_col]].drop_duplicates()
    time_to_target = time_to_target.merge(conditions, on=individual_col, how='left')

    # Condition order
    condition_order = sorted(time_to_target[condition_col].dropna().unique())

    # Handle colors
    if condition_colors is not None:
        if isinstance(condition_colors, dict):
            palette = {cond: condition_colors[cond] for cond in condition_order}
        elif isinstance(condition_colors, list):
            if len(condition_colors) < len(condition_order):
                raise ValueError("Not enough colors in list for all conditions.")
            palette = {cond: color for cond, color in zip(condition_order, condition_colors)}
        else:
            raise ValueError("condition_colors must be a dict, list, or None")
    else:
        palette = sns.color_palette("muted", len(condition_order))
        palette = {cond: color for cond, color in zip(condition_order, palette)}

    # Plot 1: Time to target
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=time_to_target,
        x=condition_col,
        y='TimeToTarget',
        order=condition_order,
        palette=palette
    )
    sns.stripplot(
        data=time_to_target,
        x=condition_col,
        y='TimeToTarget',
        color='black',
        alpha=0.5,
        order=condition_order
    )
    plt.title('Time to Reach Target by Condition')
    plt.ylabel('Time to Target (frames)')
    plt.xlabel('Condition')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Determine who successfully reached the target before their last frame
    reached = time_to_target[time_to_target['TargetFrame'] < time_to_target['LastFrame']]

    # Plot 2: Success rates per condition
    total_per_condition = time_to_target.groupby(condition_col)[individual_col].nunique()
    success_per_condition = reached.groupby(condition_col)[individual_col].nunique()
    success_rate = (success_per_condition / total_per_condition).fillna(0)

    plot_df = success_rate.reset_index()
    plot_df.columns = [condition_col, 'SuccessRate']
    plot_df = plot_df.sort_values(condition_col)

    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=plot_df,
        x=condition_col,
        y='SuccessRate',
        palette=palette,
        order=condition_order
    )
    plt.ylim(0, 1)
    plt.ylabel('Success Rate')
    plt.xlabel('Condition')
    plt.title('Success Rate per Condition')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Plot 3: Success rate over time
    reached_with_condition = reached[[individual_col, 'TargetFrame', condition_col]].copy()
     
    # Get all unique individuals per condition
    total_inds_per_condition = time_to_target.groupby(condition_col)[individual_col].nunique().to_dict()
    
    # Expand target frames to cumulative count per frame
    frame_range = np.sort(df[frame_col].unique())

    cumulative_success = []
    for condition in condition_order:
        subset = reached_with_condition[reached_with_condition[condition_col] == condition]
        for frame in frame_range:
            count = (subset['TargetFrame'] <= frame).sum()
            rate = count / total_inds_per_condition[condition]
            cumulative_success.append({
                'Frame': frame,
                'Condition': condition,
                'SuccessRate': rate
            })

    cumulative_df = pd.DataFrame(cumulative_success)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=cumulative_df,
        x='Frame',
        y='SuccessRate',
        hue='Condition',
        hue_order=condition_order,
        palette=palette,
        lw=2
    )
    plt.title('Success Rate Over Time')
    plt.ylabel('Cumulative Success Rate')
    plt.xlabel('Frame')
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.tight_layout()
    plt.show()

    return cumulative_df

def plot_target_proximity_by_frame_bins(
    df,
    target_x=14,
    target_y=2,
    radius=2,
    bin_size=100,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    condition_col='Condition'
):
    df = df.copy()

    # Calculate distance from target
    df['Distance'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)

    # Determine if individual is at the target
    df['AtTarget'] = df['Distance'] <= radius

    # Assign frame bins
    df['FrameBin'] = (df[frame_col] // bin_size) * bin_size

    # Get total number of individuals per condition
    total_inds_per_condition = df[[individual_col, condition_col]].drop_duplicates()
    total_counts = total_inds_per_condition.groupby(condition_col)[individual_col].nunique().to_dict()

    # Filter to only those at the target
    at_target_df = df[df['AtTarget']].drop_duplicates(subset=[individual_col, 'FrameBin'])

    # Count individuals at target per frame bin and condition
    bin_counts = (
        at_target_df
        .groupby(['FrameBin', condition_col])[individual_col]
        .nunique()
        .reset_index(name='CountAtTarget')
    )

    # Add total individuals and compute proportion
    bin_counts['TotalIndividuals'] = bin_counts[condition_col].map(total_counts)
    bin_counts['ProportionAtTarget'] = bin_counts['CountAtTarget'] / bin_counts['TotalIndividuals']

    # Ensure full frame bin x condition matrix (fill missing with 0)
    all_bins = df['FrameBin'].unique()
    all_conditions = df[condition_col].unique()
    full_index = pd.MultiIndex.from_product([all_bins, all_conditions], names=['FrameBin', condition_col])
    bin_counts = bin_counts.set_index(['FrameBin', condition_col]).reindex(full_index, fill_value=0).reset_index()

    # Sort for plotting
    bin_counts = bin_counts.sort_values(by=['FrameBin', condition_col])

    # Plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=bin_counts, x='FrameBin', y='ProportionAtTarget', hue=condition_col)
    plt.title('Proportion of Individuals at Target Over Time (Binned)')
    plt.ylabel('Proportion at Target')
    plt.xlabel('Frame Bin')
    plt.ylim(0, 1)
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def sholl_analysis_over_time(
    df,
    target_x=14,
    target_y=2,
    bin_size=100,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    max_radius=10,
    condition_col='Condition'
):
    df = df.copy()
    
    # Get condition name (assumes one condition per call)
    condition_name = df[condition_col].iloc[0] if condition_col in df.columns and not df.empty else "Unknown"

    # Compute distance from target and assign Sholl ring
    df['Distance'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    df['ShollRing'] = df['Distance'].astype(int)  # 0 for 0-1cm, 1 for 1-2cm, etc.
    df = df[df['ShollRing'] <= max_radius]

    # Assign frame bins
    df['FrameBin'] = (df[frame_col] // bin_size) * bin_size

    # Count unique individuals per ring per bin
    grouped = (
        df.groupby(['FrameBin', 'ShollRing'])[individual_col]
        .nunique()
        .reset_index(name='Count')
    )

    # Total individuals per frame bin (for normalization)
    total_per_bin = (
        df.groupby('FrameBin')[individual_col]
        .nunique()
        .reset_index(name='TotalIndividuals')
    )

    # Merge to normalize
    grouped = grouped.merge(total_per_bin, on='FrameBin', how='left')
    grouped['Proportion'] = grouped['Count'] / grouped['TotalIndividuals']

    # Pivot to heatmap format
    heatmap_df = grouped.pivot(index='ShollRing', columns='FrameBin', values='Proportion').fillna(0)

    # Plot heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        heatmap_df, 
        cmap='viridis', 
        cbar_kws={'label': 'Proportion of Individuals'}
    )
    plt.title(f'Sholl Analysis Heatmap (Normalized) – Condition: {condition_name}')
    plt.ylabel('Sholl Ring (Distance from Target in cm)')
    plt.xlabel('Frame Bin')
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

def logistic(x, L, k, x0):
    """Logistic function: L / (1 + exp(-k*(x-x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))

def r_squared(ydata, ypred):
    ss_res = np.sum((ydata - ypred) ** 2)
    ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
    return 1 - (ss_res / ss_tot)

def rmse(ydata, ypred):
    return np.sqrt(np.mean((ydata - ypred) ** 2))

def plot_residuals(xdata, ydata, popt, condition):
    ypred = logistic(xdata, *popt)
    residuals = ydata - ypred
    plt.figure(figsize=(8,4))
    plt.scatter(xdata, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Frame')
    plt.ylabel('Residual (Observed - Predicted)')
    plt.title(f'Residuals Plot for Condition: {condition}')
    plt.show()

def fit_logistic_to_success(
    cumulative_df,
    frame_col='Frame',
    condition_col='Condition',
    success_col='SuccessRate',
    plot_residuals_flag=False,
    print_rmse_flag=False
):
    """
    Fits logistic curves to cumulative success rates per condition.

    Parameters:
    - cumulative_df: DataFrame with columns [Frame, Condition, SuccessRate]
    - frame_col: name of the frame/time column
    - condition_col: name of the condition column
    - success_col: name of the success rate column
    - plot_residuals_flag: bool, whether to plot residuals for each fit
    - print_rmse_flag: bool, whether to print RMSE values for each fit

    Returns:
    - params_df: DataFrame of logistic parameters and R² per condition
    """
    condition_order = sorted(cumulative_df[condition_col].unique())
    params_list = []

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=cumulative_df, x=frame_col, y=success_col, hue=condition_col, palette='tab10', alpha=0.6)

    for condition in condition_order:
        subset = cumulative_df[cumulative_df[condition_col] == condition]
        xdata = subset[frame_col].values
        ydata = subset[success_col].values

        # Initial parameter guesses: L=1, k=0.1, x0=median frame
        p0 = [1, 0.1, np.median(xdata)]

        try:
            popt, _ = curve_fit(logistic, xdata, ydata, p0=p0, bounds=([0, 0, min(xdata)], [1.5, 5, max(xdata)]))
            L, k, x0 = popt

            ypred = logistic(xdata, *popt)
            r2 = r_squared(ydata, ypred)

            if print_rmse_flag:
                error = rmse(ydata, ypred)
                print(f"Condition: {condition}, RMSE: {error:.4f}")

            if plot_residuals_flag:
                plot_residuals(xdata, ydata, popt, condition)

            params_list.append({'Condition': condition, 'L': L, 'k': k, 'x0': x0, 'R2': r2})

            xfit = np.linspace(min(xdata), max(xdata), 200)
            yfit = logistic(xfit, *popt)
            plt.plot(xfit, yfit, label=f'Logistic Fit: {condition} (R²={r2:.2f})')

        except RuntimeError:
            print(f"Fit did not converge for condition: {condition}")
            params_list.append({'Condition': condition, 'L': np.nan, 'k': np.nan, 'x0': np.nan, 'R2': np.nan})

    plt.title('Cumulative Success Rate with Logistic Fits')
    plt.ylabel('Cumulative Success Rate')
    plt.xlabel('Frame')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()

    params_df = pd.DataFrame(params_list)
    return params_df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

def radial_sholl_heatmap(
    df,
    target_x=14,
    target_y=2,
    bin_size=100,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    max_radius=10,
    condition_col='Condition'
):
    df = df.copy()

    # Get condition name
    condition_name = df[condition_col].iloc[0] if condition_col in df.columns and not df.empty else "Unknown"

    # Compute distances and assign Sholl rings
    df['Distance'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    df['ShollRing'] = df['Distance'].astype(int)
    df = df[df['ShollRing'] <= max_radius]

    # Assign frame bins
    df['FrameBin'] = (df[frame_col] // bin_size) * bin_size

    # Count unique individuals per ring per bin
    grouped = (
        df.groupby(['FrameBin', 'ShollRing'])[individual_col]
        .nunique()
        .reset_index(name='Count')
    )

    # Total individuals per time bin (for normalization)
    total_per_bin = (
        df.groupby('FrameBin')[individual_col]
        .nunique()
        .reset_index(name='TotalIndividuals')
    )

    # Normalize
    grouped = grouped.merge(total_per_bin, on='FrameBin', how='left')
    grouped['Proportion'] = grouped['Count'] / grouped['TotalIndividuals']

    # Get sorted frame bins
    frame_bins = sorted(grouped['FrameBin'].unique())
    num_bins = len(frame_bins)

    # Prepare polar coordinates
    theta = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    width = 2 * np.pi / num_bins

    # Set up color map and normalization
    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})

    for i, fb in enumerate(frame_bins):
        for r in range(max_radius + 1):
            val = grouped.loc[
                (grouped['FrameBin'] == fb) & (grouped['ShollRing'] == r),
                'Proportion'
            ]
            proportion = val.values[0] if not val.empty else 0

            # Draw each sector
            ax.bar(
                x=theta[i],
                height=1,
                width=width,
                bottom=r,
                color=cmap(norm(proportion)),
                edgecolor='none'
            )

    # Cleanup: remove radial and circular grid lines
    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_yticks(range(max_radius + 1))
    ax.set_yticklabels([f"{r} cm" for r in range(max_radius + 1)], fontsize=10)
    ax.set_xticks(theta)
    ax.set_xticklabels([str(fb) for fb in frame_bins], fontsize=9, rotation=90)

    ax.set_title(f'Radial Sholl Analysis – Condition: {condition_name}', va='bottom', fontsize=14)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.1)
    cbar.set_label('Proportion of Individuals', fontsize=12)

    plt.tight_layout()
    plt.show()

def radial_sholl_heatmap_global_normalized(
    df,
    target_x=14,
    target_y=2,
    bin_size=100,
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    max_radius=10,
    condition_col='Condition'
):
    df = df.copy()

    # Get condition name
    condition_name = df[condition_col].iloc[0] if condition_col in df.columns and not df.empty else "Unknown"

    # Compute distances and assign Sholl rings
    df['Distance'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    df['ShollRing'] = df['Distance'].astype(int)
    df = df[df['ShollRing'] <= max_radius]

    # Assign frame bins
    df['FrameBin'] = (df[frame_col] // bin_size) * bin_size

    # Count unique individuals per ring per bin
    grouped = (
        df.groupby(['FrameBin', 'ShollRing'])[individual_col]
        .nunique()
        .reset_index(name='Count')
    )

    # Normalize globally: each count / total count over all bins and rings
    total_count = grouped['Count'].sum()
    grouped['Proportion'] = grouped['Count'] / total_count

    # For plotting
    frame_bins = sorted(grouped['FrameBin'].unique())
    num_bins = len(frame_bins)

    theta = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
    width = 2 * np.pi / num_bins

    cmap = plt.cm.viridis
    norm = mpl.colors.Normalize(vmin=0, vmax=grouped['Proportion'].max())

    # Plot
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={'projection': 'polar'})

    for i, fb in enumerate(frame_bins):
        for r in range(max_radius + 1):
            val = grouped.loc[
                (grouped['FrameBin'] == fb) & (grouped['ShollRing'] == r),
                'Proportion'
            ]
            proportion = val.values[0] if not val.empty else 0

            ax.bar(
                x=theta[i],
                height=1,
                width=width,
                bottom=r,
                color=cmap(norm(proportion)),
                edgecolor='none'
            )

    # Clean up
    ax.grid(False)
    ax.set_frame_on(False)
    ax.set_yticks(range(max_radius + 1))
    ax.set_yticklabels([f"{r} cm" for r in range(max_radius + 1)], fontsize=10)
    ax.set_xticks(theta)
    ax.set_xticklabels([str(fb) for fb in frame_bins], fontsize=9, rotation=90)
    ax.set_title(f'Radial Sholl (Global Normalized) – Condition: {condition_name}', va='bottom', fontsize=14)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.1)
    cbar.set_label('Proportion of Total Detections', fontsize=12)

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

def plot_group_means(df, value_col, factor_a, factor_b, min_val=None, max_val=None, trial_averages=True, group_cols=None):
    """
    Quickly plots mean of `value_col` across interaction groups defined by `factor_a` and `factor_b`.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe.
    value_col : str
        Name of the numeric column to plot.
    factor_a : str
        First grouping factor (e.g., 'Genotype').
    factor_b : str
        Second grouping factor (e.g., 'Starvation').
    min_val : float, optional
        Lower bound filter for value_col.
    max_val : float, optional
        Upper bound filter for value_col.
    trial_averages : bool, default=True
        Whether to average over group_cols.
    group_cols : list of str, optional
        Columns to group by for averaging if trial_averages is True.

    Returns:
    --------
    None (plots directly)
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[value_col, factor_a, factor_b])

    # Apply bounds
    if min_val is not None:
        df_clean = df_clean[df_clean[value_col] >= min_val]
    if max_val is not None:
        df_clean = df_clean[df_clean[value_col] <= max_val]

    # Optional averaging per group
    if trial_averages:
        if group_cols is None:
            group_cols = [factor_a, factor_b]
        else:
            group_cols = list(set(group_cols + [factor_a, factor_b]))
        df_clean = df_clean.groupby(group_cols)[value_col].mean().reset_index()

    # Interaction label
    df_clean["Group"] = df_clean[factor_a].astype(str) + "_" + df_clean[factor_b].astype(str)

    # Compute group means
    group_means = df_clean.groupby("Group")[value_col].mean().sort_index()

    # Plot
    plt.figure(figsize=(8, 5))
    group_means.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.ylabel(f"Mean {value_col}")
    plt.title(f"Mean {value_col} by {factor_a} x {factor_b}")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def plot_zone_means_subplot_sem(df, filter_column='Concentration', filter_values=None,
                               titles=["Early", "Mid", "Late"], condition_colors=None):
    """
    Creates a subplot panel of mean zone proportions for different time points and a chosen filter column,
    with SEM (standard error of the mean) added as error bars.

    Parameters:
        df (DataFrame): The full dataset.
        filter_column (str): Column to filter by for subplot columns (default = 'Concentration').
        filter_values (list or None): List of unique values to use from filter_column. If None, auto-detected.
        titles (list): Titles for each timepoint row (default = ['Early', 'Mid', 'Late']).
        condition_colors (dict or list or None): Optional color mapping.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create unique replicate column (Condition + Trial)
    if 'Trial' in df.columns:
        df = df.copy()
        df['ReplicateID'] = df['Condition'].astype(str) + "_" + df['Trial'].astype(str)
    else:
        df['ReplicateID'] = df['Condition']  # fallback

    if filter_values is None:
        filter_values = sorted(df[filter_column].dropna().unique())

    min_frame, max_frame = df['Frame'].min(), df['Frame'].max()
    frame_third = (max_frame - min_frame) // 3

    early_df = df[df['Frame'] <= min_frame + frame_third]
    mid_df = df[(df['Frame'] > min_frame + frame_third) & (df['Frame'] <= min_frame + 2 * frame_third)]
    late_df = df[df['Frame'] > min_frame + 2 * frame_third]

    dataframes = [early_df, mid_df, late_df]

    num_rows = len(dataframes)
    num_cols = len(filter_values)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows), sharey=True, sharex=True)

    if num_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if num_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    # Default palettes (used only if condition_colors not provided)
    red_shades = sns.color_palette("Reds", 3)
    blue_shades = sns.color_palette("Blues", 3)
    red_shades = [(min(r, 1), min(g, 1), min(b, 1)) for r, g, b in red_shades]
    blue_shades = [(min(r * 0.8, 1), min(g * 0.9, 1), min(b * 1.1, 1)) for r, g, b in blue_shades]

    for row_idx, (df_time, title) in enumerate(zip(dataframes, titles)):
        for col_idx, value in enumerate(filter_values):
            ax = axes[row_idx][col_idx]
            df_filtered = df_time[df_time[filter_column] == value]

            if df_filtered.empty:
                ax.set_visible(False)
                continue

            y_min, y_max = df_filtered['Y'].min(), df_filtered['Y'].max()
            y_range = (y_max - y_min) / 3
            y_bounds = [y_min + y_range, y_min + 2 * y_range]

            zone_stats = []

            for condition in df_filtered['Condition'].unique():
                df_condition = df_filtered[df_filtered['Condition'] == condition]
                starvation_status = df_condition['Starvation'].iloc[0]

                grouped = df_condition.groupby('ReplicateID')
                proportions = []
                for rep, df_rep in grouped:
                    p1 = np.mean(df_rep['Y'] < y_bounds[0])
                    p2 = np.mean((df_rep['Y'] >= y_bounds[0]) & (df_rep['Y'] < y_bounds[1]))
                    p3 = np.mean(df_rep['Y'] >= y_bounds[1])
                    proportions.append([p1, p2, p3])
                proportions = np.array(proportions)

                means = proportions.mean(axis=0)
                sems = proportions.std(axis=0, ddof=1) / np.sqrt(proportions.shape[0])

                for zone, mean, sem in zip(['Zone 1', 'Zone 2', 'Zone 3'], means, sems):
                    zone_stats.append({
                        'Condition': condition,
                        'Starvation': starvation_status,
                        'Zone': zone,
                        'Mean': mean,
                        'SEM': sem
                    })

            df_stats = pd.DataFrame(zone_stats)

            # Handle colors
            if condition_colors is not None:
                if isinstance(condition_colors, dict):
                    color_mapping = {cond: condition_colors[cond] for cond in df_stats['Condition'].unique()}
                elif isinstance(condition_colors, list):
                    unique_conditions = df_stats['Condition'].unique()
                    color_mapping = {cond: color for cond, color in zip(unique_conditions, condition_colors)}
                else:
                    raise ValueError("condition_colors must be a dict, list, or None")
            else:
                color_mapping = {}
                red_idx, blue_idx = 0, 0
                for condition in df_stats['Condition'].unique():
                    starvation_status = df_stats[df_stats['Condition'] == condition]['Starvation'].iloc[0]
                    if starvation_status == '5h':
                        color_mapping[condition] = blue_shades[blue_idx]
                        blue_idx = (blue_idx + 1) % len(blue_shades)
                    else:
                        color_mapping[condition] = red_shades[red_idx]
                        red_idx = (red_idx + 1) % len(red_shades)

            # Plot with SEM
            for condition in df_stats['Condition'].unique():
                df_cond = df_stats[df_stats['Condition'] == condition]
                ax.errorbar(
                    df_cond['Zone'],
                    df_cond['Mean'],
                    yerr=df_cond['SEM'],
                    marker='o', markersize=6,
                    linestyle='-', linewidth=3,
                    color=color_mapping[condition],
                    capsize=5,
                    label=condition
                )

            ax.set_ylim(0, 1)
            if row_idx == 0:
                ax.set_title(f"{filter_column}: {value}")
                ax.legend(title="Condition", loc='upper right', fontsize=9, frameon=True)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\nMean Proportion")
            if row_idx == num_rows - 1:
                ax.set_xlabel("Zone")

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_preference_index_comparison(df1, df2, bin_size=100, dataset_labels=("Dataset 1", "Dataset 2"), condition_colors=None):
    """
    Compare Preference Index distributions between two datasets across frame bins for each condition.

    Preference Index = (Zone 1 - Zone 3) / (Zone 1 + Zone 2 + Zone 3)

    Parameters:
        df1, df2 (pd.DataFrame): Input DataFrames with columns ['Y', 'Frame', 'Trial', 'Condition']
        bin_size (int): Number of frames per bin (default 100)
        dataset_labels (tuple): Labels for the two datasets in the legend
        condition_colors (dict or list or None): Optional color control for conditions
    """

    def compute_preference_index(df, dataset_label):
        y_min, y_max = df['Y'].min(), df['Y'].max()
        y_range = (y_max - y_min) / 3
        zone_bounds = [y_min, y_min + y_range, y_min + 2 * y_range, y_max]

        frame_data = []
        for condition in df['Condition'].unique():
            df_condition = df[df['Condition'] == condition]
            for trial in df_condition['Trial'].unique():
                df_trial = df_condition[df_condition['Trial'] == trial]

                for frame in df_trial['Frame'].unique():
                    df_frame = df_trial[df_trial['Frame'] == frame]
                    if df_frame.empty:
                        continue

                    z1 = np.sum(df_frame['Y'] < zone_bounds[1])
                    z2 = np.sum((df_frame['Y'] >= zone_bounds[1]) & (df_frame['Y'] < zone_bounds[2]))
                    z3 = np.sum(df_frame['Y'] >= zone_bounds[2])
                    total = z1 + z2 + z3

                    if total == 0:
                        continue

                    pi = (z1 - z3) / total
                    frame_int = int(frame)
                    bin_start = (frame_int // bin_size) * bin_size
                    bin_end = bin_start + bin_size - 1
                    bin_label = f"{bin_start}-{bin_end}"

                    frame_data.append({
                        'Condition': condition,
                        'Trial': trial,
                        'FrameBin': bin_label,
                        'PreferenceIndex': pi,
                        'Dataset': dataset_label
                    })

        return pd.DataFrame(frame_data)

    # Compute PI data for both datasets
    df1_bins = compute_preference_index(df1, dataset_labels[0])
    df2_bins = compute_preference_index(df2, dataset_labels[1])
    df_combined = pd.concat([df1_bins, df2_bins], ignore_index=True)

    # Color palette
    unique_conditions = df_combined['Condition'].unique()
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

    # Plot setup
    n_conditions = len(unique_conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(12, 5 * n_conditions), sharex=True)

    if n_conditions == 1:
        axes = [axes]  # Ensure iterable if only one condition

    for ax, condition in zip(axes, unique_conditions):
        df_cond = df_combined[df_combined['Condition'] == condition]
        sns.boxplot(
            data=df_cond,
            x='FrameBin',
            y='PreferenceIndex',
            hue='Dataset',
            palette='Set2',
            ax=ax
        )
        ax.axhline(0, color='gray', linestyle='--', lw=1)
        ax.set_title(f"Condition: {condition}", fontsize=14)
        ax.set_xlabel("Frame Bin")
        ax.set_ylabel("Preference Index (Z1 - Z3)")
        ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distance_comparison(df1, df2, bin_size=100, dataset_labels=("Dataset 1", "Dataset 2"), condition_colors=None):
    """
    Compare Distance distributions between two datasets across frame bins for each condition.

    Parameters:
        df1, df2 (pd.DataFrame): Input DataFrames with columns ['Distance', 'Frame', 'Trial', 'Condition']
        bin_size (int): Number of frames per bin (default 100)
        dataset_labels (tuple): Labels for the two datasets in the legend
        condition_colors (dict or list or None): Optional color control for conditions
    """

    def process_dataset(df, dataset_label):
        frame_data = []

        for condition in df['Condition'].unique():
            df_condition = df[df['Condition'] == condition]
            for trial in df_condition['Trial'].unique():
                df_trial = df_condition[df_condition['Trial'] == trial]

                for frame in df_trial['Frame'].unique():
                    df_frame = df_trial[df_trial['Frame'] == frame]
                    if df_frame.empty:
                        continue

                    # Average distance at this frame (or sum if that's more meaningful)
                    dist_val = df_frame['Distance'].mean()

                    frame_int = int(frame)
                    bin_start = (frame_int // bin_size) * bin_size
                    bin_end = bin_start + bin_size - 1
                    bin_label = f"{bin_start}-{bin_end}"

                    frame_data.append({
                        'Condition': condition,
                        'Trial': trial,
                        'FrameBin': bin_label,
                        'Distance': dist_val,
                        'Dataset': dataset_label
                    })

        return pd.DataFrame(frame_data)

    # Process both datasets
    df1_bins = process_dataset(df1, dataset_labels[0])
    df2_bins = process_dataset(df2, dataset_labels[1])
    df_combined = pd.concat([df1_bins, df2_bins], ignore_index=True)

    # Handle colors
    unique_conditions = df_combined['Condition'].unique()
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

    # Plot setup
    n_conditions = len(unique_conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(12, 5 * n_conditions), sharex=True)

    if n_conditions == 1:
        axes = [axes]  # Make iterable

    for ax, condition in zip(axes, unique_conditions):
        df_cond = df_combined[df_combined['Condition'] == condition]
        sns.boxplot(
            data=df_cond,
            x='FrameBin',
            y='Distance',
            hue='Dataset',
            palette='Set2',
            ax=ax
        )
        ax.set_title(f"Condition: {condition}", fontsize=14)
        ax.set_xlabel("Frame Bin")
        ax.set_ylabel("Distance")
        ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distance_comparison_collapsed_with_summary(
    df1, df2,
    bin_size=100,
    dataset_labels=("Dataset 1", "Dataset 2"),
    plot_type="both",  # 'box', 'strip', or 'both'
    condition_colors=None,
    show_summary=True
):
    """
    Compare Distance distributions between two datasets across frame bins for each condition.
    Each data point = mean Distance per individual (Trial) per bin.
    Optionally overlays mean ± SEM summary lines per dataset, color-matched to the data.
    """

    def process_dataset(df, dataset_label):
        df = df.copy()
        # Ensure frames are integers before binning
        df["Frame"] = df["Frame"].astype(int)
        df["FrameBin"] = (df["Frame"] // bin_size) * bin_size
        df["FrameBinLabel"] = df["FrameBin"].astype(str) + "-" + (df["FrameBin"] + bin_size - 1).astype(str)

        # Collapse to one mean Distance per trial per bin
        df_agg = (
            df.groupby(["Condition", "Trial", "FrameBinLabel"], as_index=False)
              .agg({"Distance": "mean"})
        )
        df_agg["Dataset"] = dataset_label
        return df_agg

    # Process both datasets
    df1_bins = process_dataset(df1, dataset_labels[0])
    df2_bins = process_dataset(df2, dataset_labels[1])
    df_combined = pd.concat([df1_bins, df2_bins], ignore_index=True)

    # Fix ordering of bins
    def parse_start(x):
        try:
            return int(float(x.split("-")[0]))
        except Exception:
            return 0

    df_combined["FrameBinLabel"] = pd.Categorical(
        df_combined["FrameBinLabel"],
        ordered=True,
        categories=sorted(df_combined["FrameBinLabel"].unique(), key=parse_start)
    )

    # Color palettes
    unique_conditions = df_combined["Condition"].unique()
    palette = dict(zip(unique_conditions, sns.color_palette("muted", len(unique_conditions)))) \
        if condition_colors is None else condition_colors

    dataset_palette = dict(zip(dataset_labels, sns.color_palette("Set2", len(dataset_labels))))

    # Plot setup
    n_conditions = len(unique_conditions)
    fig, axes = plt.subplots(n_conditions, 1, figsize=(12, 4.5 * n_conditions), sharex=True)
    if n_conditions == 1:
        axes = [axes]

    for ax, condition in zip(axes, unique_conditions):
        df_cond = df_combined[df_combined["Condition"] == condition]

        # --- Boxplot ---
        if plot_type in ("box", "both"):
            sns.boxplot(
                data=df_cond,
                x="FrameBinLabel",
                y="Distance",
                hue="Dataset",
                dodge=True,
                palette=dataset_palette,
                fliersize=0,
                ax=ax,
                width=0.6
            )

        # --- Stripplot ---
        if plot_type in ("strip", "both"):
            sns.stripplot(
                data=df_cond,
                x="FrameBinLabel",
                y="Distance",
                hue="Dataset",
                dodge=True,
                palette=dataset_palette,
                size=5,
                jitter=True,
                alpha=0.6,
                ax=ax
            )

        # Remove duplicate legends if both used
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > len(dataset_labels):
            ax.legend_.remove()

        # --- Summary lines (mean ± SEM) ---
        if show_summary:
            df_summary = (
                df_cond.groupby(["Dataset", "FrameBinLabel"], as_index=False)
                .agg(
                    mean_distance=("Distance", "mean"),
                    sem_distance=("Distance", lambda x: x.std(ddof=1) / np.sqrt(len(x)))
                )
            )

            for dataset in df_summary["Dataset"].unique():
                df_d = df_summary[df_summary["Dataset"] == dataset]
                xvals = np.arange(len(df_d))
                color = dataset_palette[dataset]

                ax.plot(
                    xvals,
                    df_d["mean_distance"],
                    marker="o",
                    linestyle="-",
                    linewidth=2,
                    color=color,
                    label=f"{dataset} mean"
                )
                ax.fill_between(
                    xvals,
                    df_d["mean_distance"] - df_d["sem_distance"],
                    df_d["mean_distance"] + df_d["sem_distance"],
                    color=color,
                    alpha=0.2
                )

        # --- Labels & formatting ---
        ax.set_title(f"Condition: {condition}", fontsize=14)
        ax.set_xlabel("Frame Bin")
        ax.set_ylabel("Mean Distance per Trial")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xticks(np.arange(len(df_cond["FrameBinLabel"].unique())))
        ax.set_xticklabels(df_cond["FrameBinLabel"].unique())
        ax.legend(title="Dataset", bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_distance_summary(
    df,
    bin_size=100,
    plot_type="both",  # 'box', 'strip', or 'both'
    condition_colors=None,
    show_summary=True
):
    """
    Visualize Distance distributions across frame bins for each condition in a single dataset.

    Each data point = mean Distance per individual (Trial) per bin.
    Optionally overlays mean ± SEM summary lines.

    Parameters:
        df (pd.DataFrame): Input DataFrame with ['Distance', 'Frame', 'Trial', 'Condition']
        bin_size (int): Number of frames per bin (default 100)
        plot_type (str): 'box', 'strip', or 'both'
        condition_colors (dict/list/None): Optional color control for conditions
        show_summary (bool): If True, overlay mean ± SEM per bin
    """

    df = df.copy()

    # --- Preprocess ---
    df["Frame"] = df["Frame"].astype(int)
    df["FrameBin"] = (df["Frame"] // bin_size) * bin_size
    df["FrameBinLabel"] = df["FrameBin"].astype(str) + "-" + (df["FrameBin"] + bin_size - 1).astype(str)

    # Collapse to mean Distance per trial per bin
    df_agg = (
        df.groupby(["Condition", "Trial", "FrameBinLabel"], as_index=False)
          .agg({"Distance": "mean"})
    )

    # Sort bins numerically
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
                raise ValueError("Not enough colors for all conditions.")
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
                y="Distance",
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
                y="Distance",
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
                    mean_distance=("Distance", "mean"),
                    sem_distance=("Distance", lambda x: x.std(ddof=1) / np.sqrt(len(x)))
                )
            )
            xvals = np.arange(len(df_summary))
            ax.plot(
                xvals,
                df_summary["mean_distance"],
                color=color,
                marker="o",
                linestyle="-",
                linewidth=2,
                label=f"{condition} mean"
            )
            ax.fill_between(
                xvals,
                df_summary["mean_distance"] - df_summary["sem_distance"],
                df_summary["mean_distance"] + df_summary["sem_distance"],
                color=color,
                alpha=0.2
            )

        # --- Labels & formatting ---
        ax.set_title(f"Condition: {condition}", fontsize=14)
        ax.set_xlabel("Frame Bin")
        ax.set_ylabel("Mean Distance per Trial")
        ax.tick_params(axis="x", rotation=45)
        ax.set_xticks(np.arange(len(df_cond["FrameBinLabel"].unique())))
        ax.set_xticklabels(df_cond["FrameBinLabel"].unique())
        ax.legend(loc="upper left")

    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

