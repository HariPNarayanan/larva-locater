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
