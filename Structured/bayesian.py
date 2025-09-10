import pandas as pd
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt

def fit_bayesian_pi_model(df):
    """
    Fits a Bayesian mixed-effects model for Preference Index and
    visualizes fixed effect posteriors.

    Fixed effects:
        - Genotype
        - Starvation
        - Genotype x Starvation interaction
    Random effects:
        - Random intercept for Condition
        - Random intercept for Trial nested within Condition

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with at least the columns:
        ["Preference Index", "Genotype", "Starvation", "Condition", "Trial"]

    Returns
    -------
    results : bambi.Fitted
        Fitted Bambi model object
    """

    # Create nested trial identifier (Trial within Condition)
    df = df.copy()
    df["Condition_Trial"] = df["Condition"].astype(str) + "_" + df["Trial"].astype(str)

    # Build model
    model = bmb.Model(
    "Preference_Index ~ Genotype * Starvation + (1|Trial)",
    df
    )

    # Fit model (posterior sampling)
    results = model.fit(
    draws=2000,
    chains=4,
    cores=4,
    random_seed=42,
    target_accept=0.9   # or 0.95
    )

    # Print summary
    print(az.summary(results, var_names=["Intercept", "Genotype", "Starvation", "Genotype:Starvation"]))

    # Plot forest plot for fixed effects
    az.plot_forest(
        results,
        var_names=["Intercept", "Genotype", "Starvation", "Genotype:Starvation"],
        combined=True,
        hdi_prob=0.95
    )
    plt.title("Posterior estimates of fixed effects (95% HDI)")
    plt.show()

    return results
