"""
regression.py
============================================================================
Phase 4 -- OLS/WLS regression of signing ROI on contract + market structure.

INPUT   data/processed/roi_estimates.csv   (self-sufficient: carries search_roi,
        price_roi, market_saturation, contract_aav_millions, archetype,
        price_confidence_weight). signings_master.csv is the upstream source of
        these columns but is not needed again here.
OUTPUT  data/processed/regression_results.csv  (one row per coefficient per
        model) + printed statsmodels summaries (Model A and Model B side by side)
        + a Watson-sensitivity comparison.

MODEL SPECIFICATION RATIONALE
    n=10-11 constrains us to 3 predictors max to preserve meaningful degrees of
    freedom. market_saturation is included in both models: Phase 3 confirmed it
    cleanly separates price_roi outcomes (non-saturated avg +0.50 vs saturated
    avg -0.07). The full 6-variable model from the original spec is reserved for
    a larger future dataset. HC3 robust SEs correct for heteroskedasticity
    without assuming homogeneity. This is a DIRECTIONAL model -- flag p<0.10 as
    "directionally significant", NOT statistically significant. The small-sample
    caveat is non-negotiable in the output.

    Model A (primary):  search_roi ~ log(AAV) + market_saturation + C(archetype)
                        OLS, HC3.  n=12 (search_roi finite for all signings).
    Model B (secondary): price_roi ~ market_saturation + log(AAV) + C(archetype)
                        WLS weighted by price_confidence_weight, HC3.  n=10
                        (Mahomes-2020 & OBJ-2021 are COVID-excluded upstream).

    Categorical references (patsy, alphabetical): market_saturation baseline =
    'non_saturated' -> the reported term market_saturation[T.saturated] is the
    SATURATED-vs-non_saturated effect (the headline). archetype baseline = 'QB'
    -> C(archetype)[T.Skill] is the Skill-vs-QB effect.

    Watson sensitivity: deshaun_watson_2022 ($230M fully guaranteed, suspended
    the full 2022 season) is flagged controversy_flag=True. Each model is refit
    excluding Watson to check the result is not driven by that single atypical
    observation; both versions are reported.

USAGE   python regression.py
============================================================================
"""

import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col

HERE = os.path.dirname(os.path.abspath(__file__))
ROI_CSV = os.path.join(HERE, "data", "processed", "roi_estimates.csv")
OUT_CSV = os.path.join(HERE, "data", "processed", "regression_results.csv")

FORMULA_A = "search_roi ~ np.log(contract_aav_millions) + market_saturation + C(archetype)"
FORMULA_B = "price_roi ~ market_saturation + np.log(contract_aav_millions) + C(archetype)"
SAT_TERM = "market_saturation[T.saturated]"   # the headline coefficient

OUTPUT_COLUMNS = ["model", "variable", "coef", "std_err", "t_stat", "p_value",
                  "ci_lower", "ci_upper"]


# ===========================================================================
# Fitting
# ===========================================================================
def fit_search(df):
    """OLS, HC3 robust SEs."""
    return smf.ols(FORMULA_A, data=df).fit(cov_type="HC3")


def fit_price(df):
    """WLS weighted by data-confidence, HC3 robust SEs."""
    return smf.wls(FORMULA_B, data=df, weights=df["price_confidence_weight"]).fit(
        cov_type="HC3")


# ===========================================================================
# Extract a tidy coefficient table from a fitted result
# ===========================================================================
def coef_table(model_name, res):
    ci = res.conf_int()
    ci.columns = ["lo", "hi"]
    rows = []
    for var in res.params.index:
        rows.append({
            "model": model_name,
            "variable": var,
            "coef": round(float(res.params[var]), 5),
            "std_err": round(float(res.bse[var]), 5),
            "t_stat": round(float(res.tvalues[var]), 4),
            "p_value": round(float(res.pvalues[var]), 4),
            "ci_lower": round(float(ci.loc[var, "lo"]), 5),
            "ci_upper": round(float(ci.loc[var, "hi"]), 5),
        })
    return rows


def headline(res, label):
    """One-line market_saturation finding with a directional-significance flag."""
    if SAT_TERM not in res.params.index:
        return f"  {label}: market_saturation term absent (check categories)."
    coef, p = res.params[SAT_TERM], res.pvalues[SAT_TERM]
    if p < 0.10:
        sig = "DIRECTIONALLY significant (p<0.10)"
    else:
        sig = "not significant"
    return (f"  {label}: saturated-vs-non_saturated coef = {coef:+.3f}  "
            f"(p = {p:.3f}, {sig})")


# ===========================================================================
# Main
# ===========================================================================
def main():
    df = pd.read_csv(ROI_CSV)

    # Watson controversy flag (single atypical observation; see docstring).
    df["controversy_flag"] = df["player_id"] == "deshaun_watson_2022"

    # Analysis frames: drop rows whose dependent variable is NaN (COVID-excluded
    # price_roi etc.). Predictors are present for every signing.
    df_search = df.dropna(subset=["search_roi"]).copy()
    df_price = df.dropna(subset=["price_roi"]).copy()

    df_search_xw = df_search[~df_search["controversy_flag"]].copy()
    df_price_xw = df_price[~df_price["controversy_flag"]].copy()

    # Fit all four (primary + Watson-excluded sensitivity).
    a_full, a_xw = fit_search(df_search), fit_search(df_search_xw)
    b_full, b_xw = fit_price(df_price), fit_price(df_price_xw)

    models = [
        ("Model A", a_full), ("Model A (ex-Watson)", a_xw),
        ("Model B", b_full), ("Model B (ex-Watson)", b_xw),
    ]

    # ---- Printed output -------------------------------------------------
    print("\n" + "#" * 70)
    print("# SMALL-SAMPLE CAVEAT (non-negotiable):")
    print("#   n = 9-12. These are DIRECTIONAL estimates, not confirmatory")
    print("#   statistics. Treat p<0.10 as 'directionally significant' only.")
    print("#   Coefficients indicate direction/magnitude of association, not")
    print("#   causal effects. Robust HC3 SEs; interpret with caution.")
    print("#" * 70)

    print(f"\nModel A: {FORMULA_A}\n   OLS, HC3, n = {int(a_full.nobs)}")
    print(f"Model B: {FORMULA_B}\n   WLS (w=price_confidence_weight), HC3, "
          f"n = {int(b_full.nobs)}\n")

    print("=" * 70)
    print("PRIMARY MODELS -- Model A (search_roi)  vs  Model B (price_roi)")
    print("=" * 70)
    print(summary_col([a_full, b_full], stars=True,
                      model_names=["A: search_roi", "B: price_roi"],
                      info_dict={"N": lambda x: f"{int(x.nobs)}",
                                 "R2": lambda x: f"{x.rsquared:.3f}"}))

    print("\n" + "=" * 70)
    print("WATSON SENSITIVITY -- full vs Watson-excluded")
    print("=" * 70)
    print(summary_col([a_full, a_xw, b_full, b_xw], stars=True,
                      model_names=["A full", "A exW", "B full", "B exW"],
                      info_dict={"N": lambda x: f"{int(x.nobs)}",
                                 "R2": lambda x: f"{x.rsquared:.3f}"}))

    # ---- Headline: market_saturation in both models ---------------------
    print("\n" + "=" * 70)
    print("HEADLINE -- market_saturation (saturated vs non_saturated)")
    print("=" * 70)
    print(headline(a_full, "Model A (search_roi)"))
    print(headline(a_xw,   "Model A ex-Watson    "))
    print(headline(b_full, "Model B (price_roi) "))
    print(headline(b_xw,   "Model B ex-Watson    "))
    print("=" * 70 + "\n")

    # ---- Write tidy results --------------------------------------------
    rows = []
    for name, res in models:
        rows.extend(coef_table(name, res))
    out = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(out)} coefficient rows ({len(models)} models) -> {OUT_CSV}")


if __name__ == "__main__":
    main()
