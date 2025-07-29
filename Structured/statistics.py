def run_tukey_posthoc(
    df,
    value_col,
    factor_a,
    factor_b,
    min_val=None,
    max_val=None,
    alpha=0.05,
    trial_averages=True,
    group_cols=['Starvation', 'Trial', 'Genotype'],  # Customizable if needed
):
    """
    Performs Tukey's HSD post-hoc test on the specified value column across the interaction
    groups of factor_a and factor_b, with optional group averaging.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe.
    value_col : str
        Name of the numeric column to analyze.
    factor_a : str
        Name of the first categorical factor.
    factor_b : str
        Name of the second categorical factor.
    min_val : float, optional
        Minimum value threshold for value_col (inclusive).
    max_val : float, optional
        Maximum value threshold for value_col (inclusive).
    alpha : float, optional, default=0.05
        Significance level for Tukey HSD test.
    trial_averages : bool, optional, default=False
        Whether to average the value_col over specified group_cols before performing the test.
    group_cols : list of str, optional
        List of column names to group by for averaging (e.g., ['Genotype', 'Starvation', 'Trial']).

    Returns:
    --------
    tukey_result : statsmodels.stats.multicomp.TukeyHSDResults
        The result object from Tukey HSD test.
    summary_df : pandas.DataFrame
        Summary DataFrame of Tukey test results.
    """
    import pandas as pd
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    df_clean = df.copy()
    df_clean = df_clean.dropna(subset=[value_col, factor_a, factor_b])

    # Apply value bounds
    if min_val is not None:
        df_clean = df_clean[df_clean[value_col] >= min_val]
    if max_val is not None:
        df_clean = df_clean[df_clean[value_col] <= max_val]

    # Optional averaging per group (e.g., trial-level)
    if trial_averages:
        if group_cols is None:
            raise ValueError("If trial_averages=True, you must provide group_cols.")
        group_cols = list(set(group_cols + [factor_a, factor_b]))  # ensure both factors are included
        df_clean = df_clean.groupby(group_cols)[value_col].mean().reset_index()

    # Create interaction label
    df_clean["Group"] = df_clean[factor_a].astype(str) + "_" + df_clean[factor_b].astype(str)

    # Run Tukey HSD
    tukey = pairwise_tukeyhsd(endog=df_clean[value_col], groups=df_clean["Group"], alpha=alpha)
    summary_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    return tukey, summary_df


def analyze_two_way_anova(
    df,
    factor_a='Genotype',
    factor_b='Starvation',
    value_col='Preference Index',
    alpha=0.05,
    verbose=True,
    trial_averages=True,
    group_cols=['Starvation', 'Trial', 'Genotype'],  # Customizable if needed
    return_data = False
):
    """
    Performs a 2-way ANOVA on the effect of two categorical variables on a continuous outcome.
    Optionally averages across trials before running the analysis.
    Checks residual normality and applies Yeo-Johnson transformation if residuals are non-normal.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    factor_a : str
        First categorical factor (e.g., 'Genotype').
    factor_b : str
        Second categorical factor (e.g., 'Starvation').
    value_col : str
        Column name for the dependent variable.
    alpha : float
        Significance threshold for normality test.
    verbose : bool
        Whether to print summaries.
    trial_averages : bool
        Whether to average over repeated trials (grouped by `group_cols`) before analysis.
    group_cols : list of str
        Columns to group by when averaging trials (default ['Condition', 'Trial']).

    Returns:
    --------
    dict with:
        - 'anova_table': ANOVA summary table
        - 'normality_p': Shapiro-Wilk p-value for residuals
        - 'transformed': bool, whether Yeo-Johnson transformation was applied
        - 'model': fitted statsmodels ANOVA model object
        - 'data_used': the preprocessed DataFrame actually used in analysis
    """

    import pandas as pd
    import numpy as np
    from scipy.stats import shapiro
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    from sklearn.preprocessing import PowerTransformer

    # Optional trial-averaging step
    # Inside the function

    # Ensure factor_a and factor_b are preserved during grouping
    if trial_averages:
        groupby_cols = list(set(group_cols + [factor_a, factor_b]))
        df = df.groupby(groupby_cols).mean(numeric_only=True).reset_index()

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

    # Apply transformation if needed
    if shapiro_p < alpha:
        pt = PowerTransformer(method='yeo-johnson')
        df_clean[value_col + '_trans'] = pt.fit_transform(df_clean[[value_col]])
        value_col_used = value_col + '_trans'
        transformed = True

        # Re-run ANOVA
        model, anova_table, shapiro_p = run_anova(df_clean, value_col_used)

        if verbose:
            print("\n⚠️ Residuals not normal — applied Yeo-Johnson transformation.")
            print("▶️ Transformed 2-way ANOVA Results:")
            print(anova_table)
            print(f"\n▶️ Shapiro-Wilk p-value for transformed residuals: {shapiro_p:.4f}")

    from scipy.stats import ttest_ind
    import warnings

    def compute_simple_effects(df, factor_a, factor_b, value_col):
        """
        Computes simple effects of factor A within levels of factor B and vice versa.
        Only supports 2-level factors for now.
        """
        simple_effects = {}

        # A within levels of B
        effects_a_within_b = {}
        for b_level in df[factor_b].unique():
            subset = df[df[factor_b] == b_level]
            groups = [group[value_col].values for name, group in subset.groupby(factor_a)]
            if len(groups) == 2:
                stat, pval = ttest_ind(*groups, equal_var=False)
                effects_a_within_b[b_level] = {'t': stat, 'p': pval}
            else:
                warnings.warn(f"More than 2 levels in {factor_a}; skipping post-hoc for {factor_b}={b_level}")
        
        # B within levels of A
        effects_b_within_a = {}
        for a_level in df[factor_a].unique():
            subset = df[df[factor_a] == a_level]
            groups = [group[value_col].values for name, group in subset.groupby(factor_b)]
            if len(groups) == 2:
                stat, pval = ttest_ind(*groups, equal_var=False)
                effects_b_within_a[a_level] = {'t': stat, 'p': pval}
            else:
                warnings.warn(f"More than 2 levels in {factor_b}; skipping post-hoc for {factor_a}={a_level}")

        return {
            f'{factor_a}_within_{factor_b}': effects_a_within_b,
            f'{factor_b}_within_{factor_a}': effects_b_within_a
        }    

    # Compute simple effects
    simple_effects = compute_simple_effects(df_clean, factor_a, factor_b, value_col_used)

    return {
        'anova_table': anova_table,
        'normality_p': shapiro_p,
        'transformed': transformed,
        'model': model,
        'simple_effects': simple_effects
     }

    if return_data:
        result['data_used'] = df_clean