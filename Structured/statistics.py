def run_two_way_anova(df, plot=True):
    """
    Runs a 2-way ANOVA on 'Speed' based on 'Genotype' and 'Starvation'.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with columns 'Speed', 'Genotype', and 'Starvation'.
    plot : bool, optional
        Whether to display a bar plot of mean speeds with error bars.

    Returns:
    --------
    anova_table : pandas.DataFrame
        The ANOVA table (type II).
    group_stats : pandas.DataFrame
        Mean and SEM of Speed for each Genotype-Starvation group.
    """
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Drop rows with missing values in relevant columns
    df_clean = df.dropna(subset=['Speed', 'Genotype', 'Starvation'])

    # Convert to categorical
    df_clean['Genotype'] = df_clean['Genotype'].astype('category')
    df_clean['Starvation'] = df_clean['Starvation'].astype('category')

    # 2-way ANOVA model
    model = ols('Speed ~ C(Genotype) + C(Starvation) + C(Genotype):C(Starvation)', data=df_clean).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Group stats
    group_stats = df_clean.groupby(['Genotype', 'Starvation'])['Speed'].agg(['mean', 'sem']).reset_index()
    group_stats.rename(columns={'mean': 'Mean_Speed', 'sem': 'SEM_Speed'}, inplace=True)

    # Plot
    if plot:
        plt.figure(figsize=(8,6))
        sns.barplot(
            data=group_stats,
            x='Genotype',
            y='Mean_Speed',
            hue='Starvation',
            ci=None,
            palette='viridis',
            capsize=0.1,
            errwidth=1,
            yerr=group_stats['SEM_Speed']
        )
        plt.ylabel("Speed (cm/s)")
        plt.title("Mean Speed by Genotype and Starvation")
        plt.tight_layout()
        plt.show()

    return anova_table, group_stats

def run_tukey_posthoc(df):
    """
    Performs Tukey's HSD post-hoc test on Speed across Genotype-Starvation combinations.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with 'Speed', 'Genotype', and 'Starvation' columns.

    Returns:
    --------
    tukey_result : statsmodels object
        The result of the Tukey HSD test.
    summary_df : pandas.DataFrame
        Summary table of significant group differences.
    """
    import pandas as pd
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Drop missing
    df_clean = df.dropna(subset=['Speed', 'Genotype', 'Starvation'])

    # Create interaction group label
    df_clean['Group'] = df_clean['Genotype'].astype(str) + "_" + df_clean['Starvation'].astype(str)

    # Tukey HSD
    tukey = pairwise_tukeyhsd(endog=df_clean['Speed'], groups=df_clean['Group'], alpha=0.05)

    # Convert to DataFrame
    summary_df = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

    return tukey, summary_df

def analyze_preference_index(
    df, 
    condition_col='Condition', 
    value_col='Preference Index', 
    alpha=0.05,
    verbose=True
):
    """
    Analyzes the effect of a categorical condition on the Preference Index.
    Automatically checks normality, applies Yeo-Johnson transformation if needed,
    and selects appropriate statistical test (ANOVA or Kruskal-Wallis).
    Includes post-hoc comparisons.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    condition_col : str
        Column name for categorical condition (e.g., 'Condition').
    value_col : str
        Column name for the dependent variable (e.g., 'Preference Index').
    alpha : float
        Significance threshold, default = 0.05.
    verbose : bool
        Whether to print the summaries.

    Returns:
    --------
    dict with:
        - 'test': str, name of test used
        - 'p': float, main test p-value
        - 'normality': dict, Shapiro-Wilk p-values per group
        - 'anova_table' or 'kruskal_stat': result of main test
        - 'posthoc': post-hoc comparison table (Tukey or Dunn)
        - 'transformed': bool, whether transformation was applied
    """

    import pandas as pd
    import numpy as np
    from scipy.stats import shapiro, kruskal
    from statsmodels.formula.api import ols
    import statsmodels.api as sm
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    from sklearn.preprocessing import PowerTransformer
    import scikit_posthocs as sp

    df = df[[condition_col, value_col]].dropna()

    # Step 1: Shapiro-Wilk normality test for each group
    normality_pvals = {}
    for cond in df[condition_col].unique():
        group = df[df[condition_col] == cond][value_col]
        stat, pval = shapiro(group)
        normality_pvals[cond] = pval

    normal_groups = all(p > alpha for p in normality_pvals.values())

    # Step 2: Transformation if needed
    transformed = False
    df_transformed = df.copy()
    if not normal_groups:
        pt = PowerTransformer(method='yeo-johnson')
        df_transformed[value_col + '_trans'] = pt.fit_transform(df[[value_col]])
        value_col_used = value_col + '_trans'
        transformed = True

        # Re-check normality (optional: rerun Shapiro here if you'd like)
    else:
        value_col_used = value_col

    # Step 3: Homogeneity check skipped for simplicity; go with test selection
    groups = [group[value_col_used].values for name, group in df_transformed.groupby(condition_col)]

    if normal_groups:
        # ANOVA
        model = ols(f'Q("{value_col_used}") ~ C(Q("{condition_col}"))', data=df_transformed).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_main = anova_table['PR(>F)'][0]
        test_type = 'ANOVA'

        # Post-hoc
        tukey = pairwise_tukeyhsd(df_transformed[value_col_used], df_transformed[condition_col])
        posthoc_result = pd.DataFrame(data=tukey.summary().data[1:], columns=tukey.summary().data[0])

        if verbose:
            print("▶️ Test Used: ANOVA")
            print(anova_table)
            print("\n▶️ Tukey Post-hoc:")
            print(posthoc_result)

        return {
            'test': test_type,
            'p': p_main,
            'normality': normality_pvals,
            'anova_table': anova_table,
            'posthoc': posthoc_result,
            'transformed': transformed
        }

    else:
        # Kruskal-Wallis
        stat, p_kruskal = kruskal(*groups)
        test_type = 'Kruskal-Wallis'

        # Post-hoc: Dunn's test
        posthoc_result = sp.posthoc_dunn(df, val_col=value_col, group_col=condition_col, p_adjust='holm')

        if verbose:
            print("▶️ Test Used: Kruskal-Wallis")
            print(f"H = {stat:.4f}, p = {p_kruskal:.4f}")
            print("\n▶️ Dunn Post-hoc (Holm-adjusted):")
            print(posthoc_result)

        return {
            'test': test_type,
            'p': p_kruskal,
            'normality': normality_pvals,
            'kruskal_stat': stat,
            'posthoc': posthoc_result,
            'transformed': transformed
        }

    # Summary logic reminder (included for future reference):
    """
    Analysis Flow:
    → Grouped by 'Condition':
        └── Shapiro-Wilk test for normality
            └── If not normal → Yeo-Johnson transform
                └── Recheck normality (optional)
                    ├── If normal → One-way ANOVA + Tukey
                    └── If still not normal → Kruskal-Wallis + Dunn (Holm-corrected)
    """
