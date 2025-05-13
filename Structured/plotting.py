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

def plot_zone_1_over_time(dataframe):
    """
    Plots the mean proportion of points in Zone 1 over time for each condition.
    
    Parameters:
        dataframe (pd.DataFrame): The input dataframe containing trajectory data.
    """
    # Get min/max y-values to define the zones
    y_min, y_max = dataframe['Y'].min(), dataframe['Y'].max()
    y_range = (y_max - y_min) / 3  # Divide into 3 equal zones
    y_bound = y_min + y_range  # Upper boundary of Zone 1

    # Initialize storage for frame-wise zone proportions
    frame_data = []

    # Process each condition, trial, and frame
    for condition in dataframe['Condition'].unique():
        df_condition = dataframe[dataframe['Condition'] == condition]

        for trial in df_condition['Trial'].unique():
            df_trial = df_condition[df_condition['Trial'] == trial]

            for frame in df_trial['Frame'].unique():
                df_frame = df_trial[df_trial['Frame'] == frame]
                total_points = len(df_frame)  # Total points in this frame
                
                if total_points == 0:
                    continue  # Avoid division by zero

                # Count points in Zone 1 (bottom zone)
                count_zone_1 = np.sum(df_frame['Y'] < y_bound)

                # Store proportion
                frame_data.append({
                    'Condition': condition, 
                    'Frame': frame, 
                    'Proportion': count_zone_1 / total_points
                })

    # Convert to DataFrame
    df_frames = pd.DataFrame(frame_data)
    print(df_frames.head())
    print(df_frames.columns)

    # Plot mean proportion over time for each condition (without CI)
    plt.figure(figsize=(12, 6))
    sns.lineplot(x='Frame', y='Proportion', hue='Condition', data=df_frames, 
                 estimator='mean', palette='pastel', lw=2)  # `lw=2` for thicker lines

    # Labels and title
    plt.ylabel("Mean Proportion in Zone 1")
    plt.xlabel("Frame")
    plt.title("Mean Proportion of Points in Zone 1 Over Time")
    plt.legend(title="Condition")
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
    distance_column='Distance',
    condition_column='Condition',
    trial_column='Trial',
    bin_size=100
):
    """
    Plots boxplots of 'Distance' averaged per 'Trial', binned by 'Frame' intervals, comparing different conditions.
    Displays all bins.

    Parameters:
        df (pd.DataFrame): The DataFrame containing 'Distance', 'Frame', 'Condition', and 'Trial' columns.
        frame_column (str): The column name representing the frame number.
        distance_column (str): The column name representing the pre-calculated distance.
        condition_column (str): The column name representing the condition for comparison.
        trial_column (str): The column representing individual trials.
        bin_size (int): The number of frames to group together in each bin for boxplots.
    """
    # Ensure required columns exist
    required_columns = {frame_column, distance_column, condition_column, trial_column}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required_columns}")
    
    # Add bin column
    df['Bin'] = (df[frame_column] // bin_size) * bin_size

    # Group and compute mean
    df_avg = df.groupby(['Bin', condition_column, trial_column])[distance_column].mean().reset_index()
    
    # Identify unique conditions
    conditions = sorted(df_avg[condition_column].unique())
    
    # Assign half to red, half to blue (or any rule you prefer)
    num_conditions = len(conditions)
    mid = num_conditions // 2

    red_shades = sns.color_palette("Reds", mid + (num_conditions % 2))
    blue_shades = sns.color_palette("Blues", mid)

    # Combine into one palette dictionary
    condition_palette = {
        condition: red_shades[i] if i < len(red_shades) else blue_shades[i - len(red_shades)]
        for i, condition in enumerate(conditions)
    }

    # Plot
    plt.figure(figsize=(14, 6))
    sns.boxplot(data=df_avg, x='Bin', y=distance_column, hue=condition_column, palette=condition_palette)
    sns.stripplot(data=df_avg, x='Bin', y=distance_column, hue=condition_column,
                  dodge=True, color="black", alpha=0.6, jitter=True, legend=False)

    plt.title('Boxplots of Averaged Distance per Trial (All Bins) by Condition')
    plt.xlabel('Frame Interval')
    plt.ylabel('Average Distance per Trial to the Odour Source')
    plt.xticks(rotation=45)
    plt.legend(title=condition_column, bbox_to_anchor=(1.05, 1), loc='upper left')
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

def plot_zone_means_subplot(dataframes, concentrations, titles):
    """
    Creates a 3x3 subplot panel of mean zone proportions for different time points and concentrations.
    
    Parameters:
        dataframes (list): List of DataFrames [early, mid, late].
        concentrations (list): List of concentrations ['10-3', '10-4', '10-5'].
        titles (list): Titles for each row (time points: early, mid, late).
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharey=True, sharex=True)
    
    # **Improved Bright Color Palettes** 
    red_shades = sns.color_palette("Reds", 3)  
    blue_shades = sns.color_palette("Blues", 3)  

    # **Clamp values between 0 and 1 to avoid invalid colors**
    red_shades = [(min(r, 1), min(g, 1), min(b, 1)) for r, g, b in red_shades]
    blue_shades = [(min(r*0.8, 1), min(g*0.9, 1), min(b*1.1, 1)) for r, g, b in blue_shades]  # Ensure valid colors

    for row_idx, (df, title) in enumerate(zip(dataframes, titles)):
        for col_idx, conc in enumerate(concentrations):
            ax = axes[row_idx, col_idx]
            
            df_filtered = df[df['Concentration'] == conc]

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

                zone_means.append({'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 1', 'Mean Proportion': mean_zone_1})
                zone_means.append({'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 2', 'Mean Proportion': mean_zone_2})
                zone_means.append({'Condition': condition, 'Starvation': starvation_status, 'Zone': 'Zone 3', 'Mean Proportion': mean_zone_3})

            df_means = pd.DataFrame(zone_means)

            color_mapping = {}
            red_idx, blue_idx = 0, 0  

            for condition in df_means['Condition'].unique():
                starvation_status = df_means[df_means['Condition'] == condition]['Starvation'].iloc[0]

                if starvation_status == '5h':  
                    color_mapping[condition] = red_shades[red_idx]
                    red_idx = (red_idx + 1) % len(red_shades)
                else:  
                    color_mapping[condition] = blue_shades[blue_idx]
                    blue_idx = (blue_idx + 1) % len(blue_shades)

            for condition in df_means['Condition'].unique():
                df_cond = df_means[df_means['Condition'] == condition]

                ax.plot(df_cond['Zone'], df_cond['Mean Proportion'], marker='o', markersize=6, 
                        linestyle='-', linewidth=4, color=color_mapping[condition], label=condition)

            ax.set_ylim(0, 1)
            if row_idx == 0:
                ax.set_title(f"Concentration: {conc}")
                ax.legend(title="Condition", loc='upper right', fontsize=10, frameon=True)
            if col_idx == 0:
                ax.set_ylabel(f"{title}\nMean Proportion")
            if row_idx == 2:
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
