import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: SciPy's circular statistics is nicer, but we provide fallback.
try:
    from scipy.stats import circmean, circstd
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

def _angle_diff(a):
    """Normalize angle to [-pi, pi]. a can be numpy array."""
    a = (a + np.pi) % (2*np.pi) - np.pi
    return a

def rayleigh_p_value(angles):
    """
    Approximate Rayleigh test p-value for non-uniformity with mean direction != 0.
    Uses: Z = n * R^2, p ~ exp(-Z) * (1 + (2Z - Z**2)/(4n) - ...)
    Returns (R, Z, p)
    """
    n = len(angles)
    if n == 0:
        return np.nan, np.nan, np.nan
    C = np.sum(np.cos(angles))
    S = np.sum(np.sin(angles))
    R = np.sqrt(C**2 + S**2) / n
    Z = n * (R**2)
    # Approx p-value (approximation valid for moderate/large n)
    p = np.exp(-Z) * (1 + (2*Z - Z**2) / (4*n) - (24*Z - 132*Z**2 + 76*Z**3 - 9*Z**4) / (288 * n**2))
    p = min(max(p, 0.0), 1.0)
    return R, Z, p

def plot_odor_attraction_rose(df, odor=(15.,25.), frame_range=None, bins=36,
                             weight_by_speed=False, figsize_per_cond=(4,4)):
    """
    df: DataFrame with columns ['Individual','Condition','X','Y','Frame'] (Frame optional but recommended)
    odor: tuple (x_odor, y_odor)
    frame_range: (start_frame, end_frame) to subset. If None, use all rows.
    bins: number of angular bins for the rose plot (covering -pi..pi)
    weight_by_speed: if True, weight histogram counts by speed magnitude
    figsize_per_cond: single condition figure size (width,height)
    """
    # copy and check columns
    df = df.copy()
    required = {'Individual','Condition','X','Y'}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame must contain columns: {required}")
    if frame_range is not None:
        if 'Frame' not in df.columns:
            raise ValueError("Frame column not found while frame_range provided.")
        df = df[(df['Frame'] >= frame_range[0]) & (df['Frame'] <= frame_range[1])]

    # sort and compute velocities
    df = df.sort_values(['Individual', 'Frame'] if 'Frame' in df.columns else ['Individual'])
    df['dX'] = df.groupby('Individual')['X'].diff()
    df['dY'] = df.groupby('Individual')['Y'].diff()
    df = df.dropna(subset=['dX','dY']).copy()

    # movement angle and speed
    df['mov_angle'] = np.arctan2(df['dY'], df['dX'])
    df['speed'] = np.sqrt(df['dX']**2 + df['dY']**2)

    # vector to odor and bearing
    odor_x, odor_y = float(odor[0]), float(odor[1])
    df['to_odor_x'] = odor_x - df['X']
    df['to_odor_y'] = odor_y - df['Y']
    df['dist_to_odor'] = np.sqrt(df['to_odor_x']**2 + df['to_odor_y']**2)
    # avoid zero distance
    df = df[df['dist_to_odor'] > 1e-8].copy()
    df['bearing_to_odor'] = np.arctan2(df['to_odor_y'], df['to_odor_x'])

    # angular difference: movement direction minus bearing to odor (0 => toward odor)
    df['angle_diff'] = _angle_diff(df['mov_angle'] - df['bearing_to_odor'])

    # approach velocity (projection of movement vector onto unit vector toward odor)
    df['to_odor_ux'] = df['to_odor_x'] / df['dist_to_odor']
    df['to_odor_uy'] = df['to_odor_y'] / df['dist_to_odor']
    df['approach_vel'] = df['dX'] * df['to_odor_ux'] + df['dY'] * df['to_odor_uy']

    # Prepare plot per condition
    conditions = sorted(df['Condition'].unique())
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_cond[0]*n, figsize_per_cond[1]),
                             subplot_kw=dict(polar=True))
    if n == 1:
        axes = [axes]

    summary = []
    for ax, cond in zip(axes, conditions):
        sub = df[df['Condition']==cond]
        angles = sub['angle_diff'].values
        speeds = sub['speed'].values

        # histogram of angle diffs (centered at 0)
        weights = speeds if weight_by_speed else None
        counts, edges = np.histogram(angles, bins=bins, range=(-np.pi, np.pi), weights=weights)
        widths = np.diff(edges)
        # plot bars; place at edges[:-1]
        ax.bar(edges[:-1], counts, width=widths, align='edge', edgecolor='k', linewidth=0.3)
        ax.set_theta_zero_location("N")  # 0 at top for angle_diff=0 (toward odor)
        ax.set_theta_direction(-1)       # clockwise positive to match typical compass

        # compute mean resultant and rayleigh
        R, Z, p = rayleigh_p_value(angles)
        # mean angle (should be near 0 if attracted); compute using unit vector sum
        C = np.sum(np.cos(angles))
        S = np.sum(np.sin(angles))
        mean_angle = np.arctan2(S, C) if len(angles)>0 else np.nan
        # overlay mean resultant vector: point at mean_angle with length proportional to R
        ax.arrow(mean_angle, 0, 0, R * counts.max() * 0.9,
                 width=0.03 * counts.max(), length_includes_head=True, alpha=0.9, color='red')

        # approach velocity summary
        mean_approach = sub['approach_vel'].mean() if len(sub)>0 else np.nan
        prop_toward = np.mean(sub['approach_vel'] > 0) if len(sub)>0 else np.nan

        ax.set_title(f"Condition: {cond}\nR={R:.2f}, p={p:.3f}\nmean_approach={mean_approach:.3f}")
        summary.append({
            'Condition': cond,
            'n_steps': len(sub),
            'R': R, 'Z': Z, 'p_rayleigh': p,
            'mean_angle_diff_rad': mean_angle,
            'mean_approach_vel': mean_approach,
            'prop_steps_toward': prop_toward
        })

        # draw radial line at 0 for reference
        ax.plot([0,0],[0, counts.max()*1.02], color='black', linewidth=0.7, linestyle='--')

    plt.tight_layout()
    summary_df = pd.DataFrame(summary)
    return fig, summary_df

# Implement function that integrates distance-to-odour information and mean velocity towards odour - would be good to see if this is correlated, and at what frame regimes.

# Also plot 'success' timing (target_acquisition) - related to the rose?

# Bin by Distance to Odour instead of Frames to check for correlations between distance to odour and mean velocity to odour.