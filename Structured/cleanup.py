import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbs
import math as math
import os
from os import listdir
import tkinter
from tkinter import filedialog
import scipy as scp
import scipy.stats as scpst

pd.options.mode.use_inf_as_na = True

import os
import pandas as pd


def data_to_dataframe(pathinfo, y_col='Y (cm)', x_col='X (cm)', speed_col='SPEED#wcentroid (cm/s)', vy_col='VY (cm/s)', frame_col='frame'):
    """
    Reads CSV files from a specified directory tree, extracts specific columns, and compiles them into a single pandas DataFrame.
    Now also includes 'Odour' as the first part of the condition folder name.

    Parameters:
    -----------
    pathinfo : str
        The root directory from which to start searching for CSV files.
    y_col : str, optional
        Column name for Y coordinates, default is 'Y (cm)'.
    x_col : str, optional
        Column name for X coordinates, default is 'X (cm)'.
    speed_col : str, optional
        Column name for speed, default is 'SPEED#wcentroid (cm/s)'.
    vy_col : str, optional
        Column name for Y velocity, default is 'VY (cm/s)'.
    frame_col : str, optional
        Column name for frame, default is 'frame'.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing concatenated data from all relevant CSV files, now with an additional 'Odour' column.
    """
    import os
    import pandas as pd

    df_files = []

    for root, dirs, files in os.walk(pathinfo):
        for f in files:
            if f.endswith(".csv"):
                file_path = os.path.join(root, f)
                try:
                    a = pd.read_csv(file_path)

                    # Extract metadata from the folder name
                    condition_folder = os.path.basename(os.path.dirname(root))
                    condition_parts = condition_folder.split()  # Expected: Odour Genotype Starvation Collective Concentration

                    if len(condition_parts) == 5:
                        odour, genotype, starvation, collective, concentration = condition_parts
                    else:
                        odour = genotype = starvation = collective = concentration = None  # Fallback for unexpected format
                    
                    data = {
                        "Odour": odour,
                        "Y": a[y_col],
                        "X": a[x_col],
                        "Speed": a[speed_col],
                        "VY": a[vy_col],
                        "Frame": a[frame_col],
                        "Trial": os.path.basename(root),
                        "Condition": condition_folder,
                        "Genotype": genotype,
                        "Starvation": starvation,
                        "Collective": collective,
                        "Concentration": concentration,
                        "Individual": f
                    }
                    df_files.append(pd.DataFrame(data))

                except KeyError as e:
                    print(f"Missing column in file {file_path}: {e}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

    return pd.concat(df_files, ignore_index=True) if df_files else pd.DataFrame()


def clean_dataframe(df):
    """
    Cleans the DataFrame by dropping rows with missing values and filtering 
    rows based on the 'Speed' column.

    This function removes rows with any missing values and retains only rows 
    where the 'Speed' value is between 0.5 and 2.0 (exclusive).

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame to be cleaned.

    Returns:
    --------
    pandas.DataFrame
        The cleaned DataFrame with no missing values and 'Speed' values 
        between 0.5 and 2.0.
    
    Example:
    --------
    >>> df = data_to_dataframe("C:\\data\\experiments")
    >>> clean_df = clean_dataframe(df)
    >>> print(clean_df.head())
    """

    df_cleaned = df.dropna()
    df_cleaned = df_cleaned[df_cleaned["Speed"].between(0.5, 2.0, inclusive='neither')]
    return df_cleaned

def calculate_distance_from_fixed_point(df):
    """
    Calculates distance from a fixed point ('X_1', 'Y_1') to each row's ('X', 'Y') coordinates in the DataFrame.

    This function iterates over each row in the input DataFrame and calculates the Euclidean distance 
    from a fixed point ('X_1', 'Y_1') to each row's ('X', 'Y') coordinates.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame containing 'X', 'Y', 'Frame', 'Trial', and 'Condition' columns.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with additional column 'Distance' appended to the input DataFrame.

    Example:
    --------
    >>> df = pd.DataFrame({
    >>>     'X': [1, 2, 3],
    >>>     'Y': [4, 5, 6],
    >>>     'Frame': [1, 2, 3],
    >>>     'Trial': ['Trial1', 'Trial1', 'Trial2'],
    >>>     'Condition': ['Condition1', 'Condition1', 'Condition2']
    >>> })
    >>> df_processed = calculate_distance_from_fixed_point(df)
    >>> print(df_processed.head())
    """

    # Fixed point coordinates
    x_1 = 14
    y_1 = 2

    # Calculate distance for each row
    x_1, y_1 = 14, 2  # Fixed reference point

    # Handle NaN values before calculations
    df = df.copy()  # Avoid modifying original DataFrame
    df['Distance'] = np.where(df[['X', 'Y']].isnull().any(axis=1), np.nan,
                              np.hypot(df['X'] - x_1, df['Y'] - y_1))

    return df

import pandas as pd

def categorize_values(df: pd.DataFrame, column: str, value: float, width: float) -> pd.DataFrame:
    """
    Categorizes the values in the specified column of the DataFrame and adds the result
    as a new column called 'Preference Index'.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to analyze.
        value (float): The reference value for comparison.
        width (float): The width of the interval around the reference value.

    Returns:
        pd.DataFrame: The DataFrame with a new column 'Preference Index' indicating
                      1 if value is greater than value+width,
                      0 if between value+width and value-width,
                      -1 if less than value-width.
    """
    df['Preference Index'] = df[column].apply(lambda x: 1 if x > value + width else 
                                                    0 if value - width <= x <= value + width else 
                                                   -1)
    return df

# Example usage:
# df = pd.DataFrame({'column_name': [1, 5, 7, 10]})
# df = categorize_values(df, 'column_name', 5, 2)
# print(df)

import pandas as pd

def interpolate_missing_values(df):
    """
    Interpolates missing ('NaN') values for 'X' and 'Y' columns within each unique 
    combination of 'Trial' and 'Condition'. This will fill in the gaps in the trajectories 
    for each Trial within each Condition.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'X', 'Y', 'Trial', and 'Condition' columns.
    
    Returns:
        pd.DataFrame: A DataFrame with missing values in 'X' and 'Y' interpolated within each 'Trial' and 'Condition'.
    """
    # Ensure the necessary columns exist
    if not {'X', 'Y', 'Trial', 'Condition'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'X', 'Y', 'Trial', and 'Condition' columns.")
    
    # Create a copy to avoid modifying the original DataFrame
    df_copy = df.copy()

    # Ensure 'Trial' is treated as a string for correct grouping
    df_copy['Trial'] = df_copy['Trial'].astype(str)

    # Interpolate missing values for X and Y within each group defined by 'Trial' and 'Condition'
    df_copy['X'] = df_copy.groupby(['Trial', 'Condition'])['X'].transform(lambda group: group.interpolate(method='linear', limit_direction='both'))
    df_copy['Y'] = df_copy.groupby(['Trial', 'Condition'])['Y'].transform(lambda group: group.interpolate(method='linear', limit_direction='both'))

    # Return the DataFrame with interpolated values
    return df_copy

import numpy as np
from scipy.spatial import distance_matrix

def add_neighbor_distances(df):
    """
    Adds average and nearest neighbor distances to the dataframe, per row, based on 
    Condition, Trial, and Frame groupings.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing at least the columns: ['X', 'Y', 'Frame', 'Trial', 'Condition']

    Returns:
    --------
    pandas.DataFrame
        Original dataframe with added columns: 'AvgNeighborDist' and 'NearestNeighborDist'
    """
    # Initialize columns with NaNs
    df['AvgNeighborDist'] = np.nan
    df['NearestNeighborDist'] = np.nan

    # Group by condition, trial, and frame
    grouped = df.groupby(['Condition', 'Trial', 'Frame'])

    for (condition, trial, frame), group in grouped:
        coords = group[['X', 'Y']].to_numpy()
        indices = group.index

        if len(coords) < 2:
            continue  # Not enough points to compute distances

        dists = distance_matrix(coords, coords)
        np.fill_diagonal(dists, np.nan)  # Exclude self-distance

        # Compute per-individual distances
        avg_dists = np.nanmean(dists, axis=1)
        nearest_dists = np.nanmin(dists, axis=1)

        # Assign back to the dataframe using index
        df.loc[indices, 'AvgNeighborDist'] = avg_dists
        df.loc[indices, 'NearestNeighborDist'] = nearest_dists

    return df

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix, cKDTree

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix, cKDTree

def bootstrap_single_collective_conditions(df):
    """
    Computes simulated neighbor distances only for rows where Collective == 'Single'.
    This avoids modifying real group data.

    Parameters:
    -----------
    df : pd.DataFrame

    Returns:
    --------
    pd.DataFrame
        Original df, updated with AvgNeighborDist and NearestNeighborDist for 'Single' rows only.
    """
    df = df.copy()

    # Initialize new columns if not already present
    if 'AvgNeighborDist' not in df.columns:
        df['AvgNeighborDist'] = np.nan
    if 'NearestNeighborDist' not in df.columns:
        df['NearestNeighborDist'] = np.nan
    if 'Simulated' not in df.columns:
        df['Simulated'] = False

    # Only process 'Single' collective conditions
    single_df = df[df['Collective'] == 'Single']
    grouped_conditions = single_df['Condition'].unique()

    updated_single_rows = []

    for condition in grouped_conditions:
        cond_df = single_df[single_df['Condition'] == condition]

        for frame in cond_df['Frame'].unique():
            frame_df = cond_df[cond_df['Frame'] == frame]
            coords = frame_df[['X', 'Y']].to_numpy()

            if len(coords) < 2:
                continue  # Not enough data to compute distances

            # Distance calculations
            dists = distance_matrix(coords, coords)
            np.fill_diagonal(dists, np.nan)
            avg_dists = np.nanmean(dists, axis=1)

            tree = cKDTree(coords)
            dists_nn, _ = tree.query(coords, k=2)
            nearest_dists = dists_nn[:, 1]

            frame_df = frame_df.copy()
            frame_df['AvgNeighborDist'] = avg_dists
            frame_df['NearestNeighborDist'] = nearest_dists
            frame_df['Simulated'] = True

            updated_single_rows.append(frame_df)

    # Replace only the original 'Single' collective rows
    if updated_single_rows:
        updated_df = pd.concat(updated_single_rows, ignore_index=True)
        df_non_single = df[df['Collective'] != 'Single']
        df_single_updated = pd.concat([
        df[(df['Collective'] == 'Single') & (~df.index.isin(updated_df.index))],
        updated_df], ignore_index=True)
        df = pd.concat([df_non_single, df_single_updated], ignore_index=True)

    return df

