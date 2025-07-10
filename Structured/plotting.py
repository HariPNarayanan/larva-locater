import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm  # For logarithmic normalization

def plot_trajectory_heatmaps(dataframe, condition, frame_bin_size=100, grid_size=1):
    """
    Plots heatmaps of the trajectory data based on density in 1x1 cm bins for each frame bin.
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing trajectory data.
        condition (str): The condition to filter the dataframe on.
        frame_bin_size (int): Number of frames per bin (default 100).
        grid_size (int): Size of the grid in cm (default 1x1 cm).
    """
    # Filter dataframe by condition
    df = dataframe[dataframe['Condition'] == condition]
    # Determine the frame range for each bin
    min_frame = df['Frame'].min()
    max_frame = df['Frame'].max()
    frame_bins = np.arange(min_frame, max_frame + frame_bin_size, frame_bin_size)
    # Create a subplot for each frame bin
    num_bins = len(frame_bins) - 1
    fig, axes = plt.subplots(1, num_bins, figsize=(15, 7), sharey=True)
    if num_bins == 1:  # Handle case where only one frame bin exists
        axes = [axes]
    # Loop through each frame bin
    for i in range(num_bins):
        bin_start, bin_end = frame_bins[i], frame_bins[i + 1]
        # Filter data for the current frame bin
        df_bin = df[(df['Frame'] >= bin_start) & (df['Frame'] < bin_end)]
        # Define grid for 1x1 cm bins (you can adjust based on the scale of your data)
        x_min, x_max = df_bin['X'].min(), df_bin['X'].max()
        y_min, y_max = df_bin['Y'].min(), df_bin['Y'].max()
        # Create a 2D histogram of point density (counts in each 1x1 cm bin)
        hist, xedges, yedges = np.histogram2d(df_bin['X'], df_bin['Y'],
                                               bins=[np.arange(x_min, x_max + grid_size, grid_size),
                                                     np.arange(y_min, y_max + grid_size, grid_size)])
        # Plot heatmap
        cax = axes[i].pcolormesh(xedges, yedges, hist.T, cmap='viridis', shading='auto', norm=LogNorm(vmin=1))
        # Set the title, labels, and grid
        axes[i].set_title(f"Frames {bin_start}-{bin_end}")
        axes[i].set_xlabel("X-coordinate (cm)")
        if i == 0:  # Only label the y-axis on the first plot
            axes[i].set_ylabel("Y-coordinate (cm)")
        axes[i].grid(True)
        # Ensure the aspect ratio is equal to prevent distortion
        axes[i].set_aspect('equal')
        # Set limits for x and y to fix them at 30cm x 30cm
        axes[i].set_xlim([0, 30])  # Set x limit to 30cm
        axes[i].set_ylim([0, 30])  # Set y limit to 30cm
    # Add a colorbar to the figure, and position it to the right
    cbar = fig.colorbar(cax, ax=axes, orientation='vertical', label='Density (log scale)', fraction=0.02, pad=0.04)
    # Add overall title
    fig.suptitle(f"Heatmaps of Trajectories for Condition: {condition}", fontsize=16)
    # Adjust layout to prevent overlap while keeping the colorbar in place
    plt.subplots_adjust(right=0.85, top=0.85)
    # Show the plot
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

def plot_distance_by_condition(df, frame_column='Frame', distance_column='Distance', condition_column='Condition', bin_size=100, num_bins_to_display=4):
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
    
    # Get unique conditions and generate Viridis colors
    unique_conditions = df_filtered[condition_column].nunique()
    viridis_palette = sns.color_palette("viridis", unique_conditions)

    # Plotting the boxplots for Distance within the filtered bins and grouped by Condition
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x='Bin', y=distance_column, hue=condition_column, palette=viridis_palette)
    
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


def plot_speed_by_condition(df, frame_column='Frame', speed_column='Speed', condition_column='Condition', bin_size=100, num_bins_to_display=4):
    """
    Plots boxplots of 'Speed' binned by 'Frame' intervals, comparing different conditions.
    Displays only the middle 'num_bins_to_display' bins.
    
    Parameters:
        df (pd.DataFrame): The DataFrame containing 'Speed', 'Frame', and 'Condition' columns.
        frame_column (str): The column name representing the frame number.
        speed_column (str): The column name representing the pre-calculated speed.
        condition_column (str): The column name representing the condition for comparison.
        bin_size (int): The number of frames to group together in each bin for boxplots.
        num_bins_to_display (int): The number of middle bins to display (e.g., 3 for the middle 3 bins).
    """
    # Ensure the necessary columns exist
    if not {frame_column, speed_column, condition_column}.issubset(df.columns):
        raise ValueError(f"DataFrame must contain '{frame_column}', '{speed_column}', and '{condition_column}' columns.")
    
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
    
    # Plotting the boxplots for Speed within the filtered bins and grouped by Condition
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_filtered, x='Bin', y=speed_column, hue=condition_column)
    
    plt.title(f'Boxplots of Speed binned by Frame intervals (Middle {num_bins_to_display} bins) and grouped by Condition')
    plt.xlabel('Frame Interval')
    plt.ylabel('Speed')
    plt.xticks(rotation=45)  # Rotate x labels for readability

    # Move the hue legend outside the plot
    plt.legend(title=condition_column, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

    import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_zone_means_subplot(df, filter_column='Concentration', filter_values=None, titles=["Early", "Mid", "Late"]):
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

            color_mapping = {}
            red_idx, blue_idx = 0, 0

            # Updated color assignment: flip red and blue use
            for condition in df_means['Condition'].unique():
                starvation_status = df_means[df_means['Condition'] == condition]['Starvation'].iloc[0]

                # Flip here: '5h' now gets blue, others get red
                if starvation_status == '5h':
                    color_mapping[condition] = blue_shades[blue_idx]
                    blue_idx = (blue_idx + 1) % len(blue_shades)
                else:
                    color_mapping[condition] = red_shades[red_idx]
                    red_idx = (red_idx + 1) % len(red_shades)

            for condition in df_means['Condition'].unique():
                df_cond = df_means[df_means['Condition'] == condition]

                ax.plot(df_cond['Zone'], df_cond['Mean Proportion'], marker='o', markersize=6,
                        linestyle='-', linewidth=4, color=color_mapping[condition], label=condition)

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

def plot_preference_index_over_time(dataframe):
    """
    Plots the mean Preference Index over time for each condition.
    Preference Index = (Zone 1 - Zone 3) / (Zone 1 + Zone 2 + Zone 3)

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with columns ['Y', 'Frame', 'Trial', 'Condition']
    """
    # Define Y boundaries for 3 horizontal zones
    y_min, y_max = dataframe['Y'].min(), dataframe['Y'].max()
    y_range = (y_max - y_min) / 3
    zone_bounds = [y_min, y_min + y_range, y_min + 2 * y_range, y_max]  # [Z1_low, Z2_low, Z3_low, y_max]

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

    # Plot: mean PI over time with confidence intervals per condition
    plt.figure(figsize=(12, 6))
    sns.lineplot(
        data=df_frames,
        x='Frame',
        y='PreferenceIndex',
        hue='Condition',
        estimator='mean',
        ci='sd',
        palette='muted',
        lw=2
    )

    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.ylabel("Preference Index (Z1 - Z3)")
    plt.xlabel("Frame")
    plt.title("Mean Preference Index Over Time")
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

def plot_preference_index_boxplots(dataframe, bin_size=100):
    """
    Plots boxplots of the Preference Index binned by frame ranges for each condition.

    Preference Index = (Zone 1 - Zone 3) / (Zone 1 + Zone 2 + Zone 3)

    Parameters:
        dataframe (pd.DataFrame): Input DataFrame with columns ['Y', 'Frame', 'Trial', 'Condition']
        bin_size (int): Number of frames per bin (default 100)
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

    # Ensure frame bins are ordered correctly
    df_bins['FrameBin'] = pd.Categorical(df_bins['FrameBin'], ordered=True,
                                         categories=sorted(df_bins['FrameBin'].unique(),
                                                           key=lambda x: int(x.split('-')[0])))

    plt.figure(figsize=(14, 6))
    sns.boxplot(
        data=df_bins,
        x='FrameBin',
        y='PreferenceIndex',
        hue='Condition',
        palette='muted'
    )

    plt.axhline(0, color='gray', linestyle='--', lw=1)
    plt.ylabel("Preference Index (Z1 - Z3)")
    plt.xlabel("Frame Bins")
    plt.title("Preference Index Distribution in Binned Frames")
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
    condition_col='Condition'
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

    # Alphabetical order of conditions
    condition_order = sorted(time_to_target[condition_col].dropna().unique())

    # Plot 1: Time to target
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=time_to_target, x=condition_col, y='TimeToTarget', order=condition_order)
    sns.stripplot(data=time_to_target, x=condition_col, y='TimeToTarget', color='black', alpha=0.5, order=condition_order)
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
    sns.barplot(data=plot_df, x=condition_col, y='SuccessRate', palette='viridis', order=condition_order)
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
    sns.lineplot(data=cumulative_df, x='Frame', y='SuccessRate', hue='Condition', hue_order=condition_order)
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
    spatial_bin=1  # New parameter
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
    ax.set_yticks([r * spatial_bin for r in range(max_radius + 1)])
    ax.set_yticklabels([f"{r * spatial_bin} units" for r in range(max_radius + 1)], fontsize=10)
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
