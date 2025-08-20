def analyze_two_way_anova(
    df,
    factor_a='Genotype',
    factor_b='Starvation',
    value_col='Preference Index',
    alpha=0.05,
    verbose=True,
    trial_averages=True,
    condition_col='Condition',
    frame_col='Frame',
    return_data=False,
    plot=False,
    plot_transform=False,  # NEW flag to plot Yeo-Johnson effect
    palette='Set2'
):
    """
    Performs a 2-way ANOVA on the effect of two categorical variables on a continuous outcome.
    Optionally averages across trials (within each condition/frame) before running the analysis.
    Checks residual normality and applies Yeo-Johnson transformation if residuals are non-normal.
    Optionally plots a box+scatter of the data with significance annotations from post-hoc tests.
    
    New parameter:
    - plot_transform: If True and transformation is applied, plots data before and after Yeo-Johnson.
    """

    import pandas as pd
    import numpy as np
    from scipy.stats import shapiro, ttest_ind
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    from sklearn.preprocessing import PowerTransformer
    import warnings
    import seaborn as sns
    import matplotlib.pyplot as plt

    # --- Helper function for plotting before/after Yeo-Johnson ---
    def plot_yeo_johnson_comparison(original, transformed, feature_name):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].hist(original, bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title(f'Before Yeo-Johnson\n{feature_name}')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Frequency')

        axes[1].hist(transformed, bins=30, color='salmon', edgecolor='black')
        axes[1].set_title(f'After Yeo-Johnson\n{feature_name}')
        axes[1].set_xlabel('Value')
        axes[1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.show()

    # --- Trial averaging step ---
    if trial_averages:
        groupby_cols = [condition_col, frame_col, factor_a, factor_b]
        df = (
            df.groupby(groupby_cols, as_index=False)
              .mean(numeric_only=True)
        )

    # Keep only relevant columns and drop NA
    df_clean = df[[factor_a, factor_b, value_col]].dropna()
    transformed = False
    value_col_used = value_col

    def run_anova(data, dep_col):
        model = ols(f'Q("{dep_col}") ~ C(Q("{factor_a}")) * C(Q("{factor_b}"))', data=data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        residuals = model.resid
        shapiro_p = shapiro(residuals)[1]
        return model, anova_table, shapiro_p

    # Initial ANOVA
    model, anova_table, shapiro_p = run_anova(df_clean, value_col)

    if verbose:
        print("▶️ Initial 2-way ANOVA Results:")
        print(anova_table)
        print(f"\n▶️ Shapiro-Wilk p-value for residuals: {shapiro_p:.4f}")

    # Apply transformation if residuals are non-normal
    if shapiro_p < alpha:
        pt = PowerTransformer(method='yeo-johnson')
        transformed_values = pt.fit_transform(df_clean[[value_col]])
        df_clean[value_col + '_trans'] = transformed_values
        value_col_used = value_col + '_trans'
        transformed = True

        if plot_transform:
            plot_yeo_johnson_comparison(df_clean[value_col].values, transformed_values, value_col)

        model, anova_table, shapiro_p = run_anova(df_clean, value_col_used)

        if verbose:
            print("\n⚠️ Residuals not normal — applied Yeo-Johnson transformation.")
            print("▶️ Transformed 2-way ANOVA Results:")
            print(anova_table)
            print(f"\n▶️ Shapiro-Wilk p-value for transformed residuals: {shapiro_p:.4f}")

    # --- Simple effects ---
    def compute_simple_effects(df, factor_a, factor_b, value_col):
        simple_effects = {}
        # A within levels of B
        effects_a_within_b = {}
        for b_level in df[factor_b].unique():
            subset = df[df[factor_b] == b_level]
            groups = [group[value_col].values for _, group in subset.groupby(factor_a)]
            if len(groups) == 2:
                stat, pval = ttest_ind(*groups, equal_var=False)
                effects_a_within_b[b_level] = {'t': stat, 'p': pval}
            else:
                warnings.warn(f"More than 2 levels in {factor_a}; skipping post-hoc for {factor_b}={b_level}")

        # B within levels of A
        effects_b_within_a = {}
        for a_level in df[factor_a].unique():
            subset = df[df[factor_a] == a_level]
            groups = [group[value_col].values for _, group in subset.groupby(factor_b)]
            if len(groups) == 2:
                stat, pval = ttest_ind(*groups, equal_var=False)
                effects_b_within_a[a_level] = {'t': stat, 'p': pval}
            else:
                warnings.warn(f"More than 2 levels in {factor_b}; skipping post-hoc for {factor_a}={a_level}")

        return {
            f'{factor_a}_within_{factor_b}': effects_a_within_b,
            f'{factor_b}_within_{factor_a}': effects_b_within_a
        }

    from numpy import mean, var, sqrt
    import pandas as pd

    def cohen_d(x, y):
        """Compute Cohen's d for two independent samples."""
        nx, ny = len(x), len(y)
        dof = nx + ny - 2
        pooled_var = ((nx - 1) * var(x, ddof=1) + (ny - 1) * var(y, ddof=1)) / dof
        return (mean(x) - mean(y)) / sqrt(pooled_var)

    def compute_effect_sizes(df, factor_a, factor_b, value_col):
        """Compute Cohen's d for the same pairs tested in simple_effects."""
        effect_sizes = []

        # A within levels of B
        for b_level in df[factor_b].unique():
            subset = df[df[factor_b] == b_level]
            groups = [group[value_col].values for _, group in subset.groupby(factor_a)]
            if len(groups) == 2:
                d = cohen_d(groups[0], groups[1])
                effect_sizes.append({
                    'comparison': f"{factor_a} within {factor_b}={b_level}",
                    'cohens_d': d
                })

        # B within levels of A
        for a_level in df[factor_a].unique():
            subset = df[df[factor_a] == a_level]
            groups = [group[value_col].values for _, group in subset.groupby(factor_b)]
            if len(groups) == 2:
                d = cohen_d(groups[0], groups[1])
                effect_sizes.append({
                    'comparison': f"{factor_b} within {factor_a}={a_level}",
                    'cohens_d': d
                })

        return pd.DataFrame(effect_sizes)

    simple_effects = compute_simple_effects(df_clean, factor_a, factor_b, value_col_used)

    # --- Plot helper with significance annotations ---
    def plot_box_scatter(data, effects):
        plt.figure(figsize=(8, 6))

        sanity_check = (data[value_col_used].min(), data[value_col_used].max())

        ax = sns.boxplot(
            x=factor_a,
            y=value_col_used,
            hue=factor_b,
            data=data,
            palette=palette,
            showcaps=True,
            fliersize=0,
            boxprops=dict(alpha=0.6)
        )
        sns.stripplot(
            x=factor_a,
            y=value_col_used,
            hue=factor_b,
            data=data,
            dodge=True,
            jitter=True,
            color='black',
            alpha=0.6
        )

        # Remove duplicate legends
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(data[factor_b].unique())], labels[:len(data[factor_b].unique())], title=factor_b)

        # --- Significance annotations ---
        # Position annotations above the highest point
        y_max = data[value_col_used].max()

        # Map each genotype to its x-position
        x_positions = {gen: idx for idx, gen in enumerate(data[factor_a].unique())}
        hue_levels = list(data[factor_b].unique())

        def annotate_sig(ax, x1, x2, y, pval, h=0.05, show_ns=True):
            """Draw significance bar with stars or 'ns'."""
            if pval < 0.001:
                stars = "***"
            elif pval < 0.01:
                stars = "**"
            elif pval < 0.05:
                stars = "*"
            else:
                stars = "ns" if show_ns else ""

            if stars:
                ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
                ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', color='k', fontsize=11)

        # Control spacing
        base_offset = 0.2    # distance above max datapoint for first bar
        gap = 0.25           # gap between successive bars

        # 1️⃣ Genotype differences within each starvation condition
        for i, b_level in enumerate(hue_levels):
            pval = effects[f'{factor_a}_within_{factor_b}'][b_level]['p']
            x1 = x_positions[list(data[factor_a].unique())[0]] - 0.2 + i*0.4
            x2 = x_positions[list(data[factor_a].unique())[1]] - 0.2 + i*0.4
            annotate_sig(ax, x1, x2, y_max + base_offset + (i * gap), pval)

        # 2️⃣ Starvation differences within each genotype
        for j, a_level in enumerate(data[factor_a].unique()):
            pval = effects[f'{factor_b}_within_{factor_a}'][a_level]['p']
            x1 = j - 0.2
            x2 = j + 0.2
            annotate_sig(ax, x1, x2, y_max + base_offset + gap*len(hue_levels) + (j * gap), pval)

        plt.xlabel(factor_a)
        plt.ylabel(value_col)
        plt.title(f"{value_col} by {factor_a} and {factor_b} and {sanity_check}")
        plt.tight_layout()
        plt.show()


    if plot:
        plot_box_scatter(df_clean, simple_effects)

    result = {
        'anova_table': anova_table,
        'normality_p': shapiro_p,
        'transformed': transformed,
        'model': model,
        'simple_effects': simple_effects
    }

    if return_data:
        result['data_used'] = df_clean

    effect_sizes_df = compute_effect_sizes(df_clean, factor_a, factor_b, value_col_used)
    if verbose:
        print(effect_sizes_df)

    return result

def analyze_two_way_anova_permutation(
    df,
    factor_a='Genotype',
    factor_b='Starvation',
    value_col='Preference Index',
    alpha=0.05,
    verbose=True,
    trial_averages=True,
    condition_col='Condition',
    frame_col='Frame',
    return_data=False,
    plot=False,
    plot_perm_dists=False,  # NEW: plot permutation F distributions
    n_perm=10000,           # Number of permutations
    ss_type='III',          # 'III' (default) or 'II'
    random_state=None,
    palette='Set2'
):
    """
    Two-way ANOVA using permutation tests (Freedman–Lane) for robustness to non-normal data.
    Tests main effects and the interaction, mirrors the ergonomics of the original function.

    Parameters
    ----------
    df : pd.DataFrame
        Input data.
    factor_a, factor_b : str
        Column names of the two categorical factors.
    value_col : str
        Column name of the continuous response.
    alpha : float
        Significance threshold (used only for plotting stars).
    verbose : bool
        Print ANOVA tables, permutation p-values, and simple effects.
    trial_averages : bool
        If True, average within (condition, frame, factor_a, factor_b) before analysis.
    condition_col, frame_col : str
        Columns used for trial averaging.
    return_data : bool
        If True, include the cleaned/averaged data and permutation distributions in the return dict.
    plot : bool
        If True, draw box+strip plot with permutation-based significance annotations.
    plot_perm_dists : bool
        If True, plot histograms of permutation F-statistics with observed F marked.
    n_perm : int
        Number of permutations for each tested effect.
    ss_type : {'II','III'}
        Type of sums of squares used to compute F-statistics inside each permutation.
        'III' is recommended when the interaction is in the model.
    random_state : int or None
        Seed for reproducibility.
    palette : str or list
        Seaborn palette.

    Returns
    -------
    result : dict
        {
          'anova_table_parametric': (statsmodels Type-II/III ANOVA on original data; for reference),
          'perm_results': {
              'A': {'F_obs':..., 'p_perm':..., 'F_perm': np.array([...])},
              'B': {...},
              'A:B': {...}
          },
          'partial_eta_sq': {'A':..., 'B':..., 'A:B':...},
          'simple_effects': {
              'A_within_B': {b_level: {'stat':diff_means, 'p_perm':..., 'd':...}, ...},
              'B_within_A': {a_level: {'stat':diff_means, 'p_perm':..., 'd':...}, ...},
          },
          'data_used': df_clean (optional)
        }
    """
    import numpy as np
    import pandas as pd
    import warnings
    import seaborn as sns
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    rng = np.random.default_rng(random_state)

    # ---------- 0) Trial averaging (optional) ----------
    if trial_averages:
        groupby_cols = [condition_col, frame_col, factor_a, factor_b]
        df = (
            df.groupby(groupby_cols, as_index=False)
              .mean(numeric_only=True)
        )

    # Keep relevant columns and drop NA
    df_clean = df[[factor_a, factor_b, value_col]].dropna().copy()

    # Ensure categorical dtypes (important for stable design matrices)
    df_clean[factor_a] = df_clean[factor_a].astype('category')
    df_clean[factor_b] = df_clean[factor_b].astype('category')

    # ---------- 1) Build formulae ----------
    # Full model includes interaction
    full_formula = f'Q("{value_col}") ~ C(Q("{factor_a}")) * C(Q("{factor_b}"))'
    # Reduced models drop exactly one target term each:
    red_A  = f'Q("{value_col}") ~ C(Q("{factor_b}")) + C(Q("{factor_a}")):C(Q("{factor_b}"))'
    red_B  = f'Q("{value_col}") ~ C(Q("{factor_a}")) + C(Q("{factor_a}")):C(Q("{factor_b}"))'
    red_AB = f'Q("{value_col}") ~ C(Q("{factor_a}")) + C(Q("{factor_b}"))'

    # Helper for Type label
    ss_type_num = 3 if str(ss_type).upper() == 'III' else 2

    # ---------- 2) Fit full model on original data ----------
    full_model = ols(full_formula, data=df_clean).fit()
    anova_table = sm.stats.anova_lm(full_model, typ=ss_type_num)

    # Row labels in the ANOVA table
    lab_A  = f'C(Q("{factor_a}"))'
    lab_B  = f'C(Q("{factor_b}"))'
    lab_AB = f'C(Q("{factor_a}")):C(Q("{factor_b}"))'

    # Extract observed F statistics and SS for effect sizes
    def get_effect_row(label):
        if label not in anova_table.index:
            # If coding created slightly different labels, try to find by "endswith"
            for idx in anova_table.index:
                if idx.endswith(label):
                    return idx
            raise KeyError(f"Effect label '{label}' not found in ANOVA table. Found: {list(anova_table.index)}")
        return label

    row_A  = get_effect_row(lab_A)
    row_B  = get_effect_row(lab_B)
    row_AB = get_effect_row(lab_AB)

    F_obs = {
        'A':  float(anova_table.loc[row_A,  'F']),
        'B':  float(anova_table.loc[row_B,  'F']),
        'A:B':float(anova_table.loc[row_AB, 'F']),
    }

    # For partial eta^2 we need SS_effect and SS_residual
    SS_res = float(anova_table.loc['Residual', 'sum_sq'])
    SS_eff = {
        'A':  float(anova_table.loc[row_A,  'sum_sq']),
        'B':  float(anova_table.loc[row_B,  'sum_sq']),
        'A:B':float(anova_table.loc[row_AB, 'sum_sq']),
    }
    partial_eta_sq = {k: SS_eff[k] / (SS_eff[k] + SS_res) if (SS_eff[k] + SS_res) > 0 else np.nan
                      for k in SS_eff}

    # ---------- 3) Freedman–Lane permutation p-values ----------
    def perm_p_for_term(reduced_formula, term_label, F_obs_term):
        # Fit reduced model; get fitted and residuals
        red_model = ols(reduced_formula, data=df_clean).fit()
        yhat_red = red_model.fittedvalues.values
        resid    = red_model.resid.values

        F_perm = np.empty(n_perm, dtype=float)
        for b in range(n_perm):
            # Permute residuals, reconstruct response under H0
            y_perm = yhat_red + resid[rng.permutation(resid.shape[0])]
            # Refit full model on permuted response
            m_perm = ols(full_formula, data=df_clean.assign(**{value_col: y_perm})).fit()
            at_perm = sm.stats.anova_lm(m_perm, typ=ss_type_num)

            # Get the term F
            # Support slight label variations
            key = term_label
            if key not in at_perm.index:
                for idx in at_perm.index:
                    if idx.endswith(term_label):
                        key = idx
                        break
            F_perm[b] = float(at_perm.loc[key, 'F'])

        # p-value (upper tail)
        p_perm = (1.0 + np.sum(F_perm >= F_obs_term)) / (n_perm + 1.0)
        return F_perm, p_perm

    Fperm_A,  p_A  = perm_p_for_term(red_A,  row_A,  F_obs['A'])
    Fperm_B,  p_B  = perm_p_for_term(red_B,  row_B,  F_obs['B'])
    Fperm_AB, p_AB = perm_p_for_term(red_AB, row_AB, F_obs['A:B'])

    perm_results = {
        'A':   {'F_obs': F_obs['A'],   'p_perm': p_A,  'F_perm': Fperm_A},
        'B':   {'F_obs': F_obs['B'],   'p_perm': p_B,  'F_perm': Fperm_B},
        'A:B': {'F_obs': F_obs['A:B'], 'p_perm': p_AB, 'F_perm': Fperm_AB},
    }

    if verbose:
        print("▶️ Parametric ANOVA table (for reference):")
        print(anova_table)
        print("\n▶️ Permutation p-values (Freedman–Lane):")
        for k in ['A','B','A:B']:
            print(f"  {k}: F_obs={perm_results[k]['F_obs']:.3f}, p_perm={perm_results[k]['p_perm']:.4f}")

    # ---------- 4) Simple effects via permutation t-tests ----------
    def cohen_d(x, y):
        nx, ny = len(x), len(y)
        if nx < 2 or ny < 2:
            return np.nan
        vx = np.var(x, ddof=1)
        vy = np.var(y, ddof=1)
        dof = nx + ny - 2
        if dof <= 0:
            return np.nan
        pooled = ((nx - 1) * vx + (ny - 1) * vy) / dof
        if pooled <= 0:
            return np.nan
        return (np.mean(x) - np.mean(y)) / np.sqrt(pooled)

    def perm_t_two_sample(x, y, n_perm=10000, rng=None):
        # Test statistic: absolute difference in means (two-sided)
        x = np.asarray(x)
        y = np.asarray(y)
        obs = abs(x.mean() - y.mean())

        pooled = np.concatenate([x, y])
        nx = x.size
        n = pooled.size
        cnt = 0
        for _ in range(n_perm):
            idx = rng.permutation(n)
            x_star = pooled[idx[:nx]]
            y_star = pooled[idx[nx:]]
            stat = abs(x_star.mean() - y_star.mean())
            cnt += (stat >= obs)
        p = (1 + cnt) / (n_perm + 1.0)
        return obs, p

    # A within levels of B
    effects_a_within_b = {}
    for b_level in df_clean[factor_b].cat.categories:
        sub = df_clean[df_clean[factor_b] == b_level]
        groups = [g[value_col].values for _, g in sub.groupby(factor_a)]
        if len(groups) == 2:
            stat, pval = perm_t_two_sample(groups[0], groups[1], n_perm=n_perm, rng=rng)
            d = cohen_d(groups[0], groups[1])
            effects_a_within_b[b_level] = {'stat': stat, 'p_perm': pval, 'd': d}
        else:
            warnings.warn(f"More than 2 levels in {factor_a}; skipping simple effect at {factor_b}={b_level}")

    # B within levels of A
    effects_b_within_a = {}
    for a_level in df_clean[factor_a].cat.categories:
        sub = df_clean[df_clean[factor_a] == a_level]
        groups = [g[value_col].values for _, g in sub.groupby(factor_b)]
        if len(groups) == 2:
            stat, pval = perm_t_two_sample(groups[0], groups[1], n_perm=n_perm, rng=rng)
            d = cohen_d(groups[0], groups[1])
            effects_b_within_a[a_level] = {'stat': stat, 'p_perm': pval, 'd': d}
        else:
            warnings.warn(f"More than 2 levels in {factor_b}; skipping simple effect at {factor_a}={a_level}")

    simple_effects = {
        f'{factor_a}_within_{factor_b}': effects_a_within_b,
        f'{factor_b}_within_{factor_a}': effects_b_within_a
    }

    # ---------- 5) Plots ----------
    def annotate_sig(ax, x1, x2, y, pval, h=0.05, show_ns=True):
        if pval < 0.001: stars = "***"
        elif pval < 0.01: stars = "**"
        elif pval < 0.05: stars = "*"
        else: stars = "ns" if show_ns else ""
        if stars:
            ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c='k')
            ax.text((x1+x2)/2, y+h, stars, ha='center', va='bottom', color='k', fontsize=11)

    def plot_box_scatter_with_perm(df_plot, simple_eff):
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        ax = sns.boxplot(
            x=factor_a, y=value_col, hue=factor_b, data=df_plot,
            palette=palette, showcaps=True, fliersize=0, boxprops=dict(alpha=0.6))
        sns.stripplot(
            x=factor_a, y=value_col, hue=factor_b, data=df_plot,
            dodge=True, jitter=True, color='black', alpha=0.6)

        # De-duplicate legend
        handles, labels = ax.get_legend_handles_labels()
        k = df_plot[factor_b].nunique()
        ax.legend(handles[:k], labels[:k], title=factor_b)

        y_max = df_plot[value_col].max()
        base_offset, gap = 0.2, 0.25
        x_positions = {gen: idx for idx, gen in enumerate(df_plot[factor_a].cat.categories)}
        hue_levels = list(df_plot[factor_b].cat.categories)

        # 1) A within levels of B
        for i, b_level in enumerate(hue_levels):
            pval = simple_eff[f'{factor_a}_within_{factor_b}'][b_level]['p_perm']
            x1 = x_positions[df_plot[factor_a].cat.categories[0]] - 0.2 + i*0.4
            x2 = x_positions[df_plot[factor_a].cat.categories[1]] - 0.2 + i*0.4
            annotate_sig(ax, x1, x2, y_max + base_offset + (i * gap), pval)

        # 2) B within levels of A
        for j, a_level in enumerate(df_plot[factor_a].cat.categories):
            pval = simple_eff[f'{factor_b}_within_{factor_a}'][a_level]['p_perm']
            x1 = j - 0.2
            x2 = j + 0.2
            annotate_sig(ax, x1, x2, y_max + base_offset + gap*len(hue_levels) + (j * gap), pval)

        plt.xlabel(factor_a)
        plt.ylabel(value_col)
        plt.title(f"{value_col} by {factor_a} and {factor_b} (permutation p-values)")
        plt.tight_layout()
        plt.show()

    if plot:
        plot_box_scatter_with_perm(df_clean, simple_effects)

    if plot_perm_dists:
        import matplotlib.pyplot as plt
        fig_labels = [('A', Fperm_A, F_obs['A']),
                      ('B', Fperm_B, F_obs['B']),
                      ('A:B', Fperm_AB, F_obs['A:B'])]
        for name, dist, obs in fig_labels:
            plt.figure(figsize=(6,4))
            plt.hist(dist, bins=40, edgecolor='black')
            plt.axvline(obs, linestyle='--')
            plt.title(f'Permutation F for {name}')
            plt.xlabel('F-statistic')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.show()

    # ---------- 6) Package results ----------
    result = {
        'anova_table_parametric': anova_table,
        'perm_results': perm_results,
        'partial_eta_sq': partial_eta_sq,
        'simple_effects': simple_effects,
    }
    if return_data:
        # Include permutation distributions so you can inspect/save them
        result['data_used'] = df_clean

    return result
