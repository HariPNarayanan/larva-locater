# =============================================================================
# collab_io.py
# =============================================================================
"""
collab_io.py
------------
Loading and preprocessing for the collaborator dataset.
 
Folder naming convention: <Genotype>_<Odour>_<Side>_<Concentration>
e.g. dORN_none_L_0  or  ORNGCaMPmcherry_none_L_0
 
Each CSV contains per-frame position data for one individual.
All column name mappings are passed as parameters so they can be
adjusted without editing this file if the collaborator's format changes.
"""
 
import os
import numpy as np
import pandas as pd
 
 
def load_collab(
    pathinfo: str,
    x_col: str    = "X",
    y_col: str    = "Y",
    vx_col: str   = "VX",
    frame_col: str = "frame",
    sep: str      = "_",
) -> pd.DataFrame:
    """
    Walk a directory tree, read every CSV, and return a single concatenated
    DataFrame with standardised column names.
 
    Folder naming convention expected (underscore-separated, 4 parts):
        <Genotype>_<Odour>_<Side>_<Concentration>
    e.g.  dORN_none_L_0
 
    The condition folder is the immediate parent of the trial folder,
    which is itself the immediate parent of the CSV files — matching
    the same two-level hierarchy as the main pipeline.
 
    Parameters
    ----------
    pathinfo : str
        Root directory to search recursively.
    x_col, y_col, vx_col, frame_col : str
        Column names as they appear in the raw CSV files.
    sep : str
        Separator character in the folder name. Default '_'.
 
    Returns
    -------
    pd.DataFrame with columns:
        Condition, Genotype, Odour, Side, Concentration,
        Trial, Individual, Frame, X, Y, VX
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
                parts = condition_folder.split(sep)
 
                if len(parts) == 4:
                    genotype, odour, side, concentration = parts
                else:
                    print(
                        f"Unexpected folder name '{condition_folder}' "
                        f"(expected 4 parts separated by '{sep}', got {len(parts)}). "
                        f"Metadata will be None for files in this folder."
                    )
                    genotype = odour = side = concentration = None
 
                data = {
                    "Condition":     condition_folder,
                    "Genotype":      genotype,
                    "Odour":         odour,
                    "Side":          side,
                    "Concentration": concentration,
                    "Trial":         os.path.basename(root),
                    "Individual":    f,
                    "Frame":         a[frame_col],
                    "X":             a[x_col],
                    "Y":             a[y_col],
                    "VX":            a[vx_col],
                }
                df_files.append(pd.DataFrame(data))
 
            except KeyError as e:
                print(f"Missing column in {file_path}: {e}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
 
    if not df_files:
        return pd.DataFrame()
 
    return pd.concat(df_files, ignore_index=True)
 
 
def preprocess_collab(
    pathinfo: str,
    max_frame: int = None,
    x_col: str     = "X (cm)",
    y_col: str     = "Y (cm)",
    vx_col: str    = "VX (cm/s)",
    frame_col: str = "frame",
    sep: str       = "_",
) -> pd.DataFrame:
    """
    Full preprocessing pipeline in one call:
        load → optional frame cutoff → interpolate missing X/Y
 
    Parameters
    ----------
    pathinfo : str
        Root data directory.
    max_frame : int or None
        If provided, rows with Frame >= max_frame are dropped.
    x_col, y_col, vx_col, frame_col : str
        Raw CSV column names.
    sep : str
        Folder name separator. Default '_'.
 
    Returns
    -------
    pd.DataFrame ready for collab_metrics and collab_figures functions.
    """
    df = load_collab(
        pathinfo,
        x_col=x_col, y_col=y_col, vx_col=vx_col,
        frame_col=frame_col, sep=sep,
    )
 
    if df.empty:
        return df
 
    if max_frame is not None:
        df = df[df["Frame"] < max_frame].copy()
 
    # Interpolate missing X, Y, VX within each (Trial, Condition) group
    for col in ("X", "Y", "VX"):
        if col in df.columns:
            df[col] = df.groupby(["Trial", "Condition"])[col].transform(
                lambda g: g.interpolate(method="linear", limit_direction="both")
            )
 
    return df.reset_index(drop=True)