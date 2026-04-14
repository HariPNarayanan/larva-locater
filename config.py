"""
config.py
---------
Single source of truth for arena constants, column name defaults,
and the shared palette/ordering logic that was previously copy-pasted
across every plotting function.
"""

import seaborn as sns

# ---------------------------------------------------------------------------
# Arena / experiment constants
# ---------------------------------------------------------------------------

TARGET_X: float = 14.0       # Odour source X position (cm)
TARGET_Y: float = 2.0        # Odour source Y position (cm)
SUCCESS_RADIUS: float = 2.0  # Radius considered "at target" (cm)

ARENA_WIDTH: float  = 30.0   # cm
ARENA_HEIGHT: float = 30.0   # cm

SPEED_MIN: float = 0.5       # Lower bound for valid speed filter (cm/s)
SPEED_MAX: float = 2.0       # Upper bound for valid speed filter (cm/s)

MAX_FRAME: int = 600         # Default frame cutoff

# ---------------------------------------------------------------------------
# Default column names
# Column name constants let you change a single string here if your
# raw data format ever changes, rather than hunting across all files.
# ---------------------------------------------------------------------------

COL_X           = "X"
COL_Y           = "Y"
COL_FRAME       = "Frame"
COL_SPEED       = "Speed"
COL_TRIAL       = "Trial"
COL_INDIVIDUAL  = "Individual"
COL_CONDITION   = "Condition"
COL_GENOTYPE    = "Genotype"
COL_STARVATION  = "Starvation"
COL_COLLECTIVE  = "Collective"
COL_CONCENTRATION = "Concentration"
COL_ODOUR       = "Odour"
COL_DISTANCE    = "Distance"

# ---------------------------------------------------------------------------
# Raw CSV column names (as exported by your tracking software)
# Centralised here so io.py never has magic strings.
# ---------------------------------------------------------------------------

RAW_X     = "X (cm)"
RAW_Y     = "Y (cm)"
RAW_SPEED = "SPEED#wcentroid (cm/s)"
RAW_VY    = "VY (cm/s)"
RAW_FRAME = "frame"

# ---------------------------------------------------------------------------
# Palette builder
# ---------------------------------------------------------------------------

def build_palette(
    df,
    condition_col: str = COL_CONDITION,
    starvation_col: str = COL_STARVATION,
    concentration_col: str = COL_CONCENTRATION,
):
    """
    Build a consistent color palette and ordered condition list from a dataframe.

    Replaces the copy-pasted palette logic that appeared in every plotting
    function. Call once at the top of any figure function:

        colors, order = config.build_palette(df)

    Color mapping
    -------------
    Fed       → Oranges palette
    5h        → Blues palette
    Everything else → Greens palette

    Ordering
    --------
    Conditions are grouped by Concentration (ascending), then within each
    concentration: Fed first, then 5h, then any others.

    Parameters
    ----------
    df : pd.DataFrame
    condition_col : str
    starvation_col : str
    concentration_col : str

    Returns
    -------
    condition_colors : dict  {condition_name: rgb_tuple}
    ordered_conditions : list[str]
    """

    # --- Separate conditions by starvation status ---
    is_fed   = df[starvation_col] == "Fed"
    is_5h    = df[starvation_col] == "5h"
    is_other = ~df[starvation_col].isin(["Fed", "5h"])

    fed_conditions   = sorted(df.loc[is_fed,   condition_col].unique())
    fiveh_conditions = sorted(df.loc[is_5h,    condition_col].unique())
    other_conditions = sorted(df.loc[is_other, condition_col].unique())

    # --- Build palettes (reversed so darkest shade = first condition) ---
    fed_palette   = list(reversed(sns.color_palette("Oranges", max(3, len(fed_conditions)))))
    fiveh_palette = list(reversed(sns.color_palette("Blues",   max(3, len(fiveh_conditions)))))
    other_palette = list(reversed(sns.color_palette("Greens",  max(3, len(other_conditions)))))

    condition_colors = {}
    for c, col in zip(fed_conditions,   fed_palette):   condition_colors[c] = col
    for c, col in zip(fiveh_conditions, fiveh_palette): condition_colors[c] = col
    for c, col in zip(other_conditions, other_palette): condition_colors[c] = col

    # --- Build ordered condition list ---
    ordered_conditions = []
    for conc in sorted(df[concentration_col].unique()):
        subset = df[df[concentration_col] == conc]

        fed   = sorted(subset.loc[subset[starvation_col] == "Fed",  condition_col].unique())
        fiveh = sorted(subset.loc[subset[starvation_col] == "5h",   condition_col].unique())
        other = sorted(subset.loc[~subset[starvation_col].isin(["Fed", "5h"]), condition_col].unique())

        ordered_conditions.extend(fed + fiveh + other)

    return condition_colors, ordered_conditions
