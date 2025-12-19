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

def compute_probability_given_radius_by_condition(
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
    plot=True,
    palette=None  # optional: dict {condition: color}
):
    """
    Computes P(r | condition) = Prob(success | cross r, condition)
    and plots one probability curve per condition.
    """
    
    df = df.copy()
    
    # -------------------------------------------------------------------------
    # Compute distances
    # -------------------------------------------------------------------------
    df['Distance'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    
    # -------------------------------------------------------------------------
    # Define radii to test
    # -------------------------------------------------------------------------
    if radius_steps is None:
        max_d = df['Distance'].max()
        radius_steps = np.linspace(max_d, success_radius, 50)[::-1]
    else:
        radius_steps = np.sort(np.array(radius_steps))[::-1]
    
    # -------------------------------------------------------------------------
    # Prepare output
    # -------------------------------------------------------------------------
    results = []
    
    # Conditions
    conditions = sorted(df[condition_col].dropna().unique())
    
    # Color palette
    if palette is None:
        # default: use matplotlib tab10
        cmap = plt.get_cmap("tab10")
        palette = {cond: cmap(i % 10) for i, cond in enumerate(conditions)}
    
    # -------------------------------------------------------------------------
    # Loop per condition
    # -------------------------------------------------------------------------
    for cond in conditions:
        df_c = df[df[condition_col] == cond]
        
        g = df_c.groupby(individual_col)
        
        # First time each individual reaches success radius
        first_success = g.apply(
            lambda d: d[d['Distance'] <= success_radius][frame_col].min()
        ).replace({np.inf: np.nan})
        
        # For each radius, compute probability
        for r in radius_steps:
            # First time each individual crosses radius r
            first_r = g.apply(
                lambda d: d[d['Distance'] <= r][frame_col].min()
            ).replace({np.inf: np.nan})
            
            crossed = first_r.dropna().index
            
            if len(crossed) == 0:
                P_r = np.nan
                succeeded = []
            else:
                succeeded = []
                for ind in crossed:
                    t_r = first_r[ind]
                    t_s = first_success[ind]
                    if not np.isnan(t_s) and t_s >= t_r:
                        succeeded.append(ind)
                
                P_r = len(succeeded) / len(crossed)
            
            results.append({
                'Condition': cond,
                'Radius': r,
                'NumCrossed': len(crossed),
                'NumSucceeded': len(succeeded),
                'P_success_given_r': P_r
            })
    
    results_df = pd.DataFrame(results)
    
    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    if plot:
        plt.figure(figsize=(10, 6))
        
        for cond in conditions:
            sub = results_df[results_df['Condition'] == cond]
            plt.plot(
                sub['Radius'],
                sub['P_success_given_r'],
                marker='o',
                label=cond,
                color=palette.get(cond, None)
            )
        
        plt.xlabel("Radius r (distance from target)")
        plt.ylabel("P(success | cross r)")
        plt.title("Probability of Success Given Radius (per condition)")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.legend(title="Condition")
        plt.tight_layout()
        plt.show()
    
    return results_df

def compute_probability_derivative_by_condition(
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
    palette=None,
    plot=True
):
    """
    Computes the derivative of P(success | cross r) w.r.t. radius for each condition.
    
    Returns:
        prob_df  - full probability table from first function
        deriv_df - dataframe of derivatives + identified steepest point per condition
    """
    
    # ---- 1. Get probability curves from your previous function ----
    prob_df = compute_probability_given_radius_by_condition(
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
        plot=False,       # <-- suppress plotting here
        palette=palette
    )
    
    # ---- 2. Compute derivatives per condition ----
    deriv_rows = []
    conditions = prob_df['Condition'].unique()
    
    for cond in conditions:
        sub = prob_df[prob_df['Condition'] == cond].sort_values('Radius')
        
        # Use numpy gradient for smooth derivative
        dP = np.gradient(sub['P_success_given_r'], sub['Radius'])
        
        for r, dp in zip(sub['Radius'], dP):
            deriv_rows.append({
                'Condition': cond,
                'Radius': r,
                'dP_dr': dp
            })
    
    deriv_df = pd.DataFrame(deriv_rows)

    # ---- 3. Identify steepest radius (decision threshold) ----
    threshold_rows = []
    
    for cond in conditions:
        sub = deriv_df[deriv_df['Condition'] == cond]
        max_idx = sub['dP_dr'].idxmax()
        
        threshold_rows.append({
            'Condition': cond,
            'ThresholdRadius': sub.loc[max_idx, 'Radius'],
            'MaxDerivative': sub.loc[max_idx, 'dP_dr']
        })
    
    threshold_df = pd.DataFrame(threshold_rows)

    # ---- 4. Plot if requested ----
    if plot:
        plt.figure(figsize=(10, 6))
        for cond in conditions:
            sub = deriv_df[deriv_df['Condition'] == cond]
            plt.plot(
                sub['Radius'],
                sub['dP_dr'],
                marker='o',
                label=f"{cond} (threshold: r={threshold_df[threshold_df['Condition']==cond]['ThresholdRadius'].iloc[0]:.2f})"
            )
        
        plt.xlabel("Radius r")
        plt.ylabel("dP/dr")
        plt.title("Derivative of Probability Curve (Decision Threshold Signal)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # ---- 5. Return full outputs ----
    return prob_df, deriv_df, threshold_df


import matplotlib.pyplot as plt

def plot_probability_and_derivative(
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
    concentration_col='Concentration',   # NEW: required for matching palette logic
    palette=None
):
    """
    Produces a two-panel plot:
        (1) Probability curves per condition
        (2) Derivative curves per condition with threshold highlighted

    Uses SAME palette logic as plot_prefindex_and_successrate_combined().
    Returns:
        prob_df, deriv_df, threshold_df
    """

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    # ---- Run probability + derivative computation ----
    prob_df, deriv_df, threshold_df = compute_probability_derivative_by_condition(
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

    # -------------------------------------------------------------------------
    # PALETTE: EXACT MATCH TO plot_prefindex_and_successrate_combined()
    # -------------------------------------------------------------------------
    if palette is None:

        fed_conditions = sorted(df.loc[df[condition_col].str.contains('Fed'), condition_col].unique())
        fiveh_conditions = sorted(df.loc[df[condition_col].str.contains('5h'), condition_col].unique())
        other_conditions = sorted(df.loc[~df[condition_col].str.contains('Fed|5h'), condition_col].unique())

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

        # This becomes the palette used in the plotting below
        palette = condition_colors.copy()

    # ---- Create combined figure ----
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    ax1, ax2 = axes

    # -------------------------------------------------------------------------
    # TOP PANEL — Probability curves
    # -------------------------------------------------------------------------
    for cond in conditions:
        sub = prob_df[prob_df['Condition'] == cond]
        ax1.plot(
            sub['Radius'], sub['P_success_given_r'], marker='o',
            label=cond, color=palette[cond]
        )

    ax1.set_ylabel("P(success | cross r)")
    ax1.set_title("Probability of Success Given Radius")
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(title="Condition")

    # -------------------------------------------------------------------------
    # BOTTOM PANEL — Derivative curves + threshold marker
    # -------------------------------------------------------------------------
    for cond in conditions:
        sub = deriv_df[deriv_df['Condition'] == cond]
        thr_row = threshold_df[threshold_df['Condition'] == cond].iloc[0]
        thr_r = thr_row['ThresholdRadius']
        thr_dp = thr_row['MaxDerivative']

        ax2.plot(
            sub['Radius'], sub['dP_dr'], marker='o',
            label=cond, color=palette[cond]
        )

        # Mark threshold
        ax2.scatter(thr_r, thr_dp, color=palette[cond], s=120,
                    edgecolor='black', zorder=10)
        ax2.axvline(thr_r, color=palette[cond], linestyle='--', alpha=0.5)
        ax2.text(
            thr_r, thr_dp,
            f" r={thr_r:.2f}",
            fontsize=10,
            color=palette[cond],
            ha='left', va='bottom'
        )

    ax2.set_xlabel("Radius r")
    ax2.set_ylabel("dP/dr")
    ax2.set_title("Derivative of Probability Curve (Decision Threshold)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return prob_df, deriv_df, threshold_df


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
            return f"{est:.3f} +/- {err:.3f}"

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

def plot_probability_and_bootstrap_logistic_with_dwell(
    df,
    target_x=14,
    target_y=2,
    success_radius=2,
    radius_steps=None,            # if float -> bin width; if None -> auto 20 bins
    frame_col='Frame',
    individual_col='Individual',
    x_col='X',
    y_col='Y',
    condition_col='Condition',
    concentration_col='Concentration',
    palette=None,
    n_boot=500,
    ci=95,
    save_csv=True,
    csv_path="/mnt/data/logistic_params_bootstrap.csv",
    random_state=1,
    frame_rate=1.0,               # frames per second (1 => 1 frame = 1 second)
    dwell_norm_percentile=95,     # percentile for dwell normalization (default: 95th)
    eps=1e-6                      # numeric epsilon to avoid exact 0/1 probs
):
    """
    Post-contact dwell model:
      - Detect contact events (outside -> inside success zone) per individual
      - For each contact record (entry_radius, dwell_seconds)
      - Aggregate by condition & radius-bin:
          - total_frames (opportunities)
          - contact_count
          - P_contact = contact_count / total_frames
          - mean_dwell_seconds (conditional on contact)
          - scaled_dwell = mean_dwell_seconds / dwell_norm_seconds (clipped to [0,1])
          - P_weighted = P_contact * scaled_dwell
      - Fit 4-param logistic to (Radius, P_weighted) per condition
      - Bootstrap aggregated (r, p_weighted) pairs to produce CI on params
    Returns dict with prob_df, params_df, comparison_table, bootstrap_samples, contacts_df, saved_params_csv
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.optimize import curve_fit
    import os

    rng = np.random.default_rng(random_state)

    # ---------------- helper functions ----------------
    def logistic_4param(r, A, L, k, r0):
        return A + (L - A) / (1.0 + np.exp(-k * (r - r0)))

    def fit_4param_logistic(r, p, bounds=None, p0=None, maxfev=20000):
        r = np.asarray(r, dtype=float)
        p = np.asarray(p, dtype=float)
        if bounds is None:
            bounds = ([-np.inf]*4, [np.inf]*4)
        popt, pcov = curve_fit(logistic_4param, r, p, p0=p0, bounds=bounds, maxfev=maxfev)
        return popt, pcov

    # ---------------- basic checks & prepare df ----------------
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Columns {x_col} and/or {y_col} not found in dataframe.")

    df = df.copy()
    # ensure sorted by frame within each individual/condition for contact detection
    if frame_col not in df.columns:
        # if no frame column, create one (but user said frame exists)
        df[frame_col] = np.arange(len(df))

    df = df.sort_values([condition_col, individual_col, frame_col]).reset_index(drop=True)

    # ---------------- compute per-frame radius & inside flag ----------------
    df['_radius_to_target'] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    df['_inside'] = (df['_radius_to_target'] <= success_radius).astype(int)

    # ---------------- detect contact events (outside -> inside transitions) ----------------
    contact_records = []
    # Group by condition and individual
    grouped = df.groupby([condition_col, individual_col], sort=False)

    for (cond, indiv), g in grouped:
        # g is already sorted by frame
        inside = g['_inside'].values
        radii = g['_radius_to_target'].values
        frames = g[frame_col].values

        # find indices where inside transitions from 0 -> 1
        prev = np.concatenate(([0], inside[:-1]))
        entries = np.where((prev == 0) & (inside == 1))[0]

        # for each entry, compute dwell: consecutive inside frames starting at entry
        for idx in entries:
            # entry frame's radius
            entry_radius = radii[idx]
            # count how many consecutive ones from idx onward
            j = idx
            while j < len(inside) and inside[j] == 1:
                j += 1
            dwell_frames = j - idx
            dwell_seconds = dwell_frames / float(frame_rate)

            contact_records.append({
                condition_col: cond,
                individual_col: indiv,
                'entry_frame': int(frames[idx]),
                'entry_radius': float(entry_radius),
                'dwell_frames': int(dwell_frames),
                'dwell_seconds': float(dwell_seconds)
            })

    contacts_df = pd.DataFrame(contact_records)
    # If no contacts, return gracefully
    if contacts_df.empty:
        raise RuntimeError("No contact events detected (no outside->inside transitions). Check success_radius / data.")

    # ---------------- build radius bins (Option A: same bins used for probability curves) ----------------
    dist_min = df['_radius_to_target'].min()
    dist_max = df['_radius_to_target'].max()

    if radius_steps is None:
        # create 20 bins between min and max (unless degenerate)
        if dist_max - dist_min <= 0:
            edges = np.array([dist_min, dist_max + 1e-6])
        else:
            n_bins = 20
            edges = np.linspace(dist_min, dist_max, n_bins + 1)
    else:
        if radius_steps <= 0:
            raise ValueError("radius_steps must be positive when provided.")
        start = min(0.0, dist_min)
        edges = np.arange(start, dist_max + radius_steps + 1e-8, radius_steps)
        if len(edges) < 2:
            edges = np.linspace(dist_min, dist_max, 21)

    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    df['_radius_bin'] = pd.cut(df['_radius_to_target'], bins=edges, labels=bin_centers, include_lowest=True).astype(float)
    contacts_df['_radius_bin'] = pd.cut(contacts_df['entry_radius'], bins=edges, labels=bin_centers, include_lowest=True).astype(float)

    # ---------------- aggregate frames (opportunities) per condition+bin ----------------
    agg_frames = df.groupby([condition_col, '_radius_bin']).agg(
        total_frames=(frame_col, 'count')
    ).reset_index().rename(columns={'_radius_bin': 'Radius'})

    # ---------------- aggregate contact counts and mean dwell per condition+bin ----------------
    agg_contacts = contacts_df.groupby([condition_col, '_radius_bin']).agg(
        contact_count=('dwell_frames', 'count'),
        mean_dwell_seconds=('dwell_seconds', 'mean')
    ).reset_index().rename(columns={'_radius_bin': 'Radius'})

    # merge
    merged = pd.merge(agg_frames, agg_contacts, on=[condition_col, 'Radius'], how='left')
    merged['contact_count'] = merged['contact_count'].fillna(0).astype(int)
    merged['mean_dwell_seconds'] = merged['mean_dwell_seconds'].astype(float)

    # ---------------- compute P_contact and scaled dwell ----------------
    # dwell_norm_seconds default = percentile of all contact dwell times
    all_dwell_seconds = contacts_df['dwell_seconds'].dropna().values
    if len(all_dwell_seconds) == 0:
        raise RuntimeError("No dwell times found in contacts (unexpected).")

    dwell_norm_seconds = np.percentile(all_dwell_seconds, dwell_norm_percentile)
    # avoid zero normalization
    if dwell_norm_seconds <= 0:
        dwell_norm_seconds = np.max(all_dwell_seconds) if len(all_dwell_seconds) > 0 else 1.0
        if dwell_norm_seconds <= 0:
            dwell_norm_seconds = 1.0

    merged['P_contact'] = np.where(merged['total_frames'] > 0,
                                   merged['contact_count'] / merged['total_frames'],
                                   np.nan)

    # mean_dwell_seconds may be NaN where contact_count == 0 -> set scaled_dwell = 0 there
    merged['scaled_dwell'] = np.where(
        merged['contact_count'] > 0,
        merged['mean_dwell_seconds'] / float(dwell_norm_seconds),
        0.0
    )
    merged['scaled_dwell'] = merged['scaled_dwell'].clip(0.0, 1.0)

    # Weighted probability to fit
    merged['P_weighted'] = merged['P_contact'].fillna(0.0) * merged['scaled_dwell']

    # Drop rows where total_frames == 0 (no opportunity)
    merged = merged[merged['total_frames'] > 0].copy()
    merged = merged.rename(columns={condition_col: 'Condition'})

    # Ensure columns
    prob_df = merged[['Condition', 'Radius', 'total_frames', 'contact_count', 'P_contact',
                      'mean_dwell_seconds', 'scaled_dwell', 'P_weighted']].sort_values(['Condition','Radius']).reset_index(drop=True)

    # ---------------- build palette if needed ----------------
    conditions = prob_df['Condition'].unique()
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
    # fallback for missing
    for i, cond in enumerate(sorted(conditions)):
        if cond not in palette:
            palette[cond] = list(sns.color_palette("husl", n_colors=len(conditions)))[i % len(conditions)]

    # ---------------- Fit logistic + bootstrap (robust) ----------------
    fit_summary = []
    bootstrap_results = {}

    for cond in conditions:
        sub = prob_df[prob_df['Condition'] == cond].sort_values('Radius')
        r = sub['Radius'].values
        p = sub['P_weighted'].values

        # drop NaNs (shouldn't be any) and ensure numeric
        mask = ~np.isnan(r) & ~np.isnan(p)
        r = np.asarray(r[mask], dtype=float)
        p = np.asarray(p[mask], dtype=float)

        if len(r) < 4:
            fit_summary.append({
                "Condition": cond,
                "A": np.nan, "L": np.nan, "k": np.nan, "r0": np.nan,
                "A_se": np.nan, "L_se": np.nan, "k_se": np.nan, "r0_se": np.nan,
                "R2": np.nan
            })
            bootstrap_results[cond] = pd.DataFrame(columns=['A','L','k','r0'])
            continue

        # Clip p to avoid exact 0/1
        p_safe = np.clip(p, eps, 1.0 - eps)

        # initial guesses
        A0 = float(np.min(p_safe))
        L0 = float(np.max(p_safe))
        k0 = 1.0
        r0_0 = float(np.median(r))

        # bounds: A and L in [eps, 1-eps], k reasonably bounded, r0 within observed radius
        lower_bounds = [eps, eps, -100.0, np.min(r)]
        upper_bounds = [1.0 - eps, 1.0 - eps, 100.0, np.max(r)]

        try:
            popt, pcov = fit_4param_logistic(r, p_safe, bounds=(lower_bounds, upper_bounds),
                                             p0=[A0, L0, k0, r0_0])
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

        # Bootstrap over aggregated (r, p_weighted) pairs (skip failed fits)
        boot_params = []
        n_pts = len(r)
        for b in range(n_boot):
            idx = rng.integers(0, n_pts, n_pts)
            r_bs = r[idx]
            p_bs = p_safe[idx]
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
                # skip failed bootstrap sample
                continue

        bootstrap_results[cond] = pd.DataFrame(boot_params, columns=['A','L','k','r0'])

    params_df = pd.DataFrame(fit_summary)

    # ---------------- compute CIs from bootstrap ----------------
    alpha = 100.0 - float(ci)
    lower_q = alpha/2.0
    upper_q = 100.0 - alpha/2.0

    for param in ['A','L','k','r0']:
        lowers, uppers = [], []
        for cond in params_df['Condition']:
            boot_df = bootstrap_results.get(cond, pd.DataFrame())
            if boot_df.empty:
                lowers.append(np.nan)
                uppers.append(np.nan)
            else:
                lowers.append(np.nanpercentile(boot_df[param], lower_q))
                uppers.append(np.nanpercentile(boot_df[param], upper_q))
        params_df[f"{param}_ci_lower"] = lowers
        params_df[f"{param}_ci_upper"] = uppers

    # ---------------- plotting: show contact statistics and fitted curves ----------------
    fig, axes = plt.subplots(2,1,figsize=(12,10), sharex=True)
    ax_stat, ax_fit = axes

    # empirical: P_contact and P_weighted points
    for cond in conditions:
        sub = prob_df[prob_df['Condition'] == cond]
        ax_stat.plot(sub['Radius'], sub['P_contact'], 'o-', label=f"{cond} (P_contact)", alpha=0.6, color=palette.get(cond))
        ax_stat.plot(sub['Radius'], sub['P_weighted'], 's--', label=f"{cond} (P_weighted)", alpha=0.9, color=palette.get(cond))
    ax_stat.set_ylabel("Probability")
    ax_stat.set_ylim(-0.02, 1.02)
    ax_stat.grid(True)
    ax_stat.legend(fontsize='small')

    # fitted curves with CI shading
    r_dense = np.linspace(prob_df['Radius'].min(), prob_df['Radius'].max(), 400)
    for cond in conditions:
        row = params_df[params_df['Condition'] == cond].iloc[0]
        A_fit, L_fit, k_fit, r0_fit = row[['A','L','k','r0']]
        if not np.isnan(A_fit):
            ax_fit.plot(r_dense, logistic_4param(r_dense, A_fit, L_fit, k_fit, r0_fit),
                        color=palette.get(cond), label=cond)
            boot_df = bootstrap_results.get(cond, pd.DataFrame())
            if not boot_df.empty:
                n_take = min(len(boot_df), 200)
                sample_idx = np.linspace(0, len(boot_df)-1, n_take).astype(int)
                y_samples = np.stack([logistic_4param(r_dense, *boot_df.iloc[i][['A','L','k','r0']].values)
                                      for i in sample_idx], axis=0)
                lower_band = np.nanpercentile(y_samples, lower_q, axis=0)
                upper_band = np.nanpercentile(y_samples, upper_q, axis=0)
                ax_fit.fill_between(r_dense, lower_band, upper_band, color=palette.get(cond), alpha=0.15)
    ax_fit.set_ylabel("Fitted P_weighted")
    ax_fit.set_ylim(-0.02, 1.02)
    ax_fit.grid(True)
    ax_fit.legend()
    ax_fit.set_xlabel("Radius (bin centers)")

    plt.tight_layout()
    plt.show()

    # ---------------- format comparison table ----------------
    def fmt_pm(est, lo, hi):
        if np.isnan(est) or np.isnan(lo) or np.isnan(hi):
            return "NA"
        err = (hi - lo) / 2.0
        return f"{est:.3f} +/- {err:.3f}"

    comparison_rows = []
    for _, r in params_df.iterrows():
        comparison_rows.append({
            "Condition": r['Condition'],
            "A": fmt_pm(r["A"], r["A_ci_lower"], r["A_ci_upper"]),
            "L": fmt_pm(r["L"], r["L_ci_lower"], r["L_ci_upper"]),
            "k": fmt_pm(r["k"], r["k_ci_lower"], r["k_ci_upper"]),
            "r0": fmt_pm(r["r0"], r["r0_ci_lower"], r["r0_ci_upper"]),
            "R2": f"{r['R2']:.3f}" if not np.isnan(r['R2']) else "NA"
        })
    comparison_df = pd.DataFrame(comparison_rows).sort_values('Condition')

    # ---------------- save CSVs ----------------
    saved_params_csv = None
    if save_csv:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
        params_df.to_csv(csv_path, index=False)
        comp_path = csv_path.replace(".csv", "_comparison.csv")
        comparison_df.to_csv(comp_path, index=False)
        dwell_path = csv_path.replace(".csv", "_postcontact_by_bin.csv")
        prob_df.to_csv(dwell_path, index=False)
        contacts_path = csv_path.replace(".csv", "_contacts.csv")
        contacts_df.to_csv(contacts_path, index=False)
        saved_params_csv = csv_path

    # ---------------- return everything ----------------
    return {
        "prob_df": prob_df,
        "params_df": params_df,
        "comparison_table": comparison_df,
        "bootstrap_samples": bootstrap_results,
        "contacts_df": contacts_df,
        "saved_params_csv": saved_params_csv,
        "dwell_norm_seconds": float(dwell_norm_seconds)
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
    plt.title("Histogram of Dwell Times (successful only): radius =" + str(success_radius))

    # ECDF
    plt.subplot(1,2,2)
    for cond, sub in dwell_df.groupby(condition_col):
        sorted_vals = np.sort(sub["DwellTime_seconds"])
        yvals = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        plt.plot(sorted_vals, yvals, label=cond)
    plt.xlabel("Post-success dwell time (seconds)")
    plt.ylabel("ECDF")
    plt.title("Empirical CDF of Dwell Times (successful only): radius =" + str(success_radius))
    plt.legend()

    plt.tight_layout()
    plt.show()

    return dwell_df

def plot_post_success_dwell_box_scatter(
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
    Unsuccessful individuals (dwell = 0) are excluded.

    Produces a boxplot + scatter per condition.

    Returns:
        dwell_df: rows = individuals, columns = [Condition, Individual, DwellTime_seconds]
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # --- Compute distances and inside status ---
    df = df.copy()
    df["dist"] = np.sqrt((df[x_col] - target_x)**2 + (df[y_col] - target_y)**2)
    df["inside"] = df["dist"] <= success_radius

    dwell_records = []

    # --- Dwell-time calculation (same logic as your function) ---
    for (cond, indiv), sub in df.groupby([condition_col, individual_col]):
        sub = sub.sort_values(frame_col)

        inside_frames = sub[sub["inside"]][frame_col].values
        if len(inside_frames) == 0:
            continue  # skip unsuccessful individuals

        first_entry = inside_frames[0]
        after_first = sub[sub[frame_col] >= first_entry]
        dwell = after_first["inside"].sum()

        if dwell > 0:
            dwell_records.append([cond, indiv, dwell])

    dwell_df = pd.DataFrame(
        dwell_records,
        columns=[condition_col, individual_col, "DwellTime_seconds"]
    )

    # --- Plot: box + scatter ---
    plt.figure(figsize=(8, 6))

    sns.boxplot(
        data=dwell_df,
        x=condition_col,
        y="DwellTime_seconds",
        showfliers=False,
        width=0.6
    )

    sns.stripplot(
        data=dwell_df,
        x=condition_col,
        y="DwellTime_seconds",
        color="black",
        alpha=0.7,
        jitter=True,
        size=5
    )

    plt.ylabel("Post-success dwell time (seconds)")
    plt.xlabel("Condition")
    plt.title(f"Post-success Dwell Time by Condition (radius = {success_radius})")

    plt.tight_layout()
    plt.show()
