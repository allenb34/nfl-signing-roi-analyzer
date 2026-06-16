"""
roi_calculator.py
============================================================================
Phase 3a -- ROI calculator for the NFL Player Signing ROI project.

INPUT
    data/processed/event_study_results.csv   (abnormal lifts per signing/metric)
    data/processed/signings_master.csv        (contract $ + descriptive fields)

OUTPUT
    data/processed/roi_estimates.csv          (12 rows, one per signing)

THREE ROI METRICS PER SIGNING
    search_roi           = abnormal_search_lift_pp / contract_aav_millions
                           -- PRIMARY signal: buzz generated per $M of average
                              annual value.
    attendance_roi       = abnormal_attendance_lift_pct_y1 / contract_aav_millions
                           -- VALID ONLY for non-capacity-saturated teams. Set
                              to NaN (with a reason) for saturated teams, whose
                              attendance can't respond to roster moves, and for
                              COVID-excluded comparisons.
    guarantee_efficiency = abnormal_search_lift_pp / contract_guaranteed_millions
                           -- buzz per *guaranteed* $M (risk-adjusted: guaranteed
                              money is the dollars actually at stake).

CLASSIFICATION COLUMNS
    market_saturation : 'saturated' / 'non_saturated'. The SATURATED_TEAMS set
                        operationalises the ">95% of stadium capacity" rule with
                        the team-level judgement from the brief (Seattle, KC,
                        Dallas run effectively sold-out; Raiders/Dolphins/Browns/
                        Vikings/Rams have headroom).
    signing_type      : 'departure' is derived STRUCTURALLY (team != attendance_
                        team -- the Wilson-2022 trade away from Seattle). The
                        dataset has no prior-team field, so 'free_agent' (joined
                        a NEW team) vs 'extension' (re-signed) is set from the
                        documented NEW_TEAM_SIGNINGS set below. Promote this to a
                        config column if you want it fully data-driven.

VALIDATION
    search_roi and guarantee_efficiency must be finite for ALL 12 signings.
    attendance_roi may be NaN ONLY for saturated teams or COVID-excluded y1
    comparisons; it must be finite for every other (non-saturated, non-COVID) row.

USAGE
    python roi_calculator.py
============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
EVENT_CSV = os.path.join(HERE, "data", "processed", "event_study_results.csv")
MASTER_CSV = os.path.join(HERE, "data", "processed", "signings_master.csv")
OUT_CSV = os.path.join(HERE, "data", "processed", "roi_estimates.csv")

# Capacity-saturated franchises (>95% full regardless of roster moves) -- their
# attendance VOLUME can't move, so attendance_roi is undefined for them.
SATURATED_TEAMS = {"Seattle Seahawks", "Kansas City Chiefs", "Dallas Cowboys"}

# Signings where the player joined a NEW team (trade or free agency). Everything
# else (and not a departure) is an extension / re-signing. Departures are
# detected structurally, so they need not be listed here.
NEW_TEAM_SIGNINGS = {"davante_adams_2022", "tyreek_hill_2022",
                     "odell_beckham_2021", "deshaun_watson_2022"}

# Relative regression weights for price_roi by data confidence: a 2015 annual
# average must NOT carry the same weight as a 2022 30-day window. Feed
# price_confidence_weight as sample weights in the downstream WLS regression.
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.5, "low": 0.25}

OUTPUT_COLUMNS = [
    "player_id", "player_name", "archetype", "attendance_team",
    "signing_type", "market_saturation",
    "contract_aav_millions", "contract_guaranteed_millions",
    "abnormal_search_lift_pp", "abnormal_attendance_lift_pct_y1",
    "search_roi", "attendance_roi", "attendance_roi_note",
    "guarantee_efficiency",
    # Phase 3b price block (populated once the SeatGeek worksheet is filled).
    "price_lift_pct", "price_roi", "price_roi_note",
    "price_data_confidence", "price_confidence_weight",
]


# ===========================================================================
# Step 1 -- Load inputs
# ===========================================================================
def load_inputs():
    for path, hint in [(EVENT_CSV, "event_study.py"),
                       (MASTER_CSV, "master_merge.py")]:
        if not os.path.exists(path):
            sys.exit(f"[FATAL] Missing {os.path.basename(path)} -- run {hint} first.")
    return pd.read_csv(EVENT_CSV), pd.read_csv(MASTER_CSV)


# ===========================================================================
# Step 2 -- Pull the abnormal lifts we need into a per-signing lookup
# ===========================================================================
def abnormal_lookup(events, metric):
    """player_id -> abnormal_lift_pct for one metric (NaN where excluded)."""
    sub = events[events["metric"] == metric].set_index("player_id")
    return sub["abnormal_lift_pct"]


def _truthy(v):
    """Parse a possibly-string/bool/NaN COVID flag to a real bool."""
    return str(v).strip().lower() in ("true", "1", "yes")


# ===========================================================================
# Step 3 -- Classification helpers
# ===========================================================================
def classify_signing_type(row):
    # Departure is encoded structurally: the signing team differs from the team
    # whose market we track (Wilson signed in Denver; we track Seattle).
    if str(row["team"]).strip() != str(row["attendance_team"]).strip():
        return "departure"
    if row["player_id"] in NEW_TEAM_SIGNINGS:
        return "free_agent"
    return "extension"


def market_saturation(attendance_team):
    return "saturated" if attendance_team in SATURATED_TEAMS else "non_saturated"


# ===========================================================================
# Step 4 -- Compute ROI rows
# ===========================================================================
def build_roi(events, master):
    search_abn = abnormal_lookup(events, "search_interest")
    att_y1_abn = abnormal_lookup(events, "attendance_y1")

    rows = []
    for _, m in master.iterrows():
        pid = m["player_id"]
        aav_m = m["aav_usd"] / 1e6
        gtd_m = m["guaranteed_usd"] / 1e6
        s_abn = search_abn.get(pid, np.nan)        # always present (search valid)
        a_abn = att_y1_abn.get(pid, np.nan)        # NaN if COVID-excluded

        saturation = market_saturation(m["attendance_team"])

        # search_roi & guarantee_efficiency: defined for all signings.
        search_roi = s_abn / aav_m if aav_m else np.nan
        guarantee_eff = s_abn / gtd_m if gtd_m else np.nan

        # attendance_roi: undefined for saturated teams and COVID-excluded y1.
        reasons = []
        if saturation == "saturated":
            reasons.append("saturated_capacity")
        if pd.isna(a_abn):
            reasons.append("covid_excluded")
        if reasons:
            attendance_roi = np.nan
            note = ";".join(reasons)
        else:
            attendance_roi = a_abn / aav_m if aav_m else np.nan
            note = ""

        # --- price_roi (Phase 3b): the saturated-market signal attendance can't
        # give. price_lift = pct_change_30d (immediate signing bump; for
        # low-confidence rows this is the annual-average proxy ratio). Excluded
        # for COVID-inherited rows (Mahomes-2020) and rows without price data
        # yet. Weighted by confidence for the downstream regression.
        price_lift = m.get("pct_change_30d", np.nan)
        # A row is price-COVID-excluded if either the inherited season flag is
        # set (covid_post_1 -> Mahomes-2020) OR a cell in the 30-day comparison
        # is tagged covid_distorted (OBJ-2021's pre window): the comparison is
        # COVID-invalid regardless of which path flagged it.
        price_covid = _truthy(m.get("price_covid_affected"))
        if (str(m.get("price_src_pre", "")).strip().lower() == "covid_distorted"
                or str(m.get("price_src_post", "")).strip().lower() == "covid_distorted"):
            price_covid = True
        price_conf = m.get("price_data_confidence")
        price_conf = price_conf if isinstance(price_conf, str) else None
        price_weight = CONFIDENCE_WEIGHT.get(price_conf, np.nan)

        if price_covid:
            price_roi, price_note = np.nan, "covid_excluded"
        elif pd.isna(price_lift):
            price_roi, price_note = np.nan, "no_price_data"
        elif aav_m:
            price_roi, price_note = price_lift / aav_m, ""
        else:
            price_roi, price_note = np.nan, "no_price_data"

        rows.append({
            "player_id": pid,
            "player_name": m["player_name"],
            "archetype": m["archetype"],
            "attendance_team": m["attendance_team"],
            "signing_type": classify_signing_type(m),
            "market_saturation": saturation,
            "contract_aav_millions": round(aav_m, 2),
            "contract_guaranteed_millions": round(gtd_m, 2),
            "abnormal_search_lift_pp": round(s_abn, 2) if pd.notna(s_abn) else np.nan,
            "abnormal_attendance_lift_pct_y1": round(a_abn, 2) if pd.notna(a_abn) else np.nan,
            "search_roi": round(search_roi, 3) if pd.notna(search_roi) else np.nan,
            "attendance_roi": round(attendance_roi, 3) if pd.notna(attendance_roi) else np.nan,
            "attendance_roi_note": note,
            "guarantee_efficiency": round(guarantee_eff, 3) if pd.notna(guarantee_eff) else np.nan,
            "price_lift_pct": round(price_lift, 1) if pd.notna(price_lift) else np.nan,
            "price_roi": round(price_roi, 3) if pd.notna(price_roi) else np.nan,
            "price_roi_note": price_note,
            "price_data_confidence": price_conf,
            "price_confidence_weight": price_weight,
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


# ===========================================================================
# Step 5 -- Validation
# ===========================================================================
def validate(df):
    print("\n" + "=" * 64)
    print("DATA VALIDATION REPORT -- roi_estimates.csv")
    print("=" * 64)
    ok = True

    print(f"Rows: {len(df)} (expect 12)")
    if len(df) != 12:
        ok = False
        print("  [FAIL] Row count != 12.")

    # search_roi & guarantee_efficiency must be finite for all signings.
    for col in ["search_roi", "guarantee_efficiency"]:
        bad = df.loc[~np.isfinite(df[col].astype(float)), "player_id"].tolist()
        if bad:
            ok = False
            print(f"  [FAIL] {col} not finite for: {bad}")
        else:
            print(f"  {col}: finite for all 12  OK")

    # attendance_roi: NaN permitted ONLY for saturated OR covid-excluded rows.
    expect_nan = (df["market_saturation"].eq("saturated") |
                  df["abnormal_attendance_lift_pct_y1"].isna())
    # (a) rows that should be finite but are NaN
    should_finite_bad = df.loc[~expect_nan & df["attendance_roi"].isna(), "player_id"].tolist()
    # (b) rows that should be NaN but are finite
    should_nan_bad = df.loc[expect_nan & df["attendance_roi"].notna(), "player_id"].tolist()
    if should_finite_bad or should_nan_bad:
        ok = False
        print(f"  [FAIL] attendance_roi NaN-rule violated: "
              f"unexpected_NaN={should_finite_bad} unexpected_value={should_nan_bad}")
    else:
        n_fin = df["attendance_roi"].notna().sum()
        print(f"  attendance_roi: NaN-rule OK ({n_fin} finite / "
              f"{12 - n_fin} NaN by saturation/COVID)")

    # price_roi NaN-rule: price_roi is NaN exactly when a reason is recorded
    # (covid_excluded or no_price_data) and finite exactly when the note is
    # blank. This holds before prices land (all 'no_price_data') and after.
    note_blank = df["price_roi_note"].fillna("") == ""
    bad_a = df.loc[note_blank & df["price_roi"].isna(), "player_id"].tolist()
    bad_b = df.loc[~note_blank & df["price_roi"].notna(), "player_id"].tolist()
    if bad_a or bad_b:
        ok = False
        print(f"  [FAIL] price_roi NaN-rule violated: "
              f"unexpected_NaN={bad_a} unexpected_value={bad_b}")
    else:
        n_price = df["price_roi"].notna().sum()
        pending = (df["price_roi_note"] == "no_price_data").sum()
        print(f"  price_roi: NaN-rule OK ({n_price} finite / {pending} pending "
              f"prices / {(df['price_roi_note'] == 'covid_excluded').sum()} COVID)")

    # Confidence weight must be assigned (finite) for every signing.
    if not np.isfinite(df["price_confidence_weight"].astype(float)).all():
        ok = False
        print("  [FAIL] price_confidence_weight missing for some signings.")

    # Classification coverage.
    print("\nsigning_type counts:   ", df["signing_type"].value_counts().to_dict())
    print("market_saturation counts:", df["market_saturation"].value_counts().to_dict())

    # player_id uniqueness.
    if df["player_id"].duplicated().any():
        ok = False
        print("[FAIL] Duplicate player_id.")

    print("=" * 64)
    print("VALIDATION:", "PASS" if ok else "FAIL (see above)")
    print("=" * 64 + "\n")
    return ok


def print_summary(df):
    print("Search-based ROI leaderboard (abnormal buzz pp per $M AAV):")
    g = df.sort_values("search_roi", ascending=False)
    print(f"  {'signing':<24}{'type':<11}{'AAV$M':>7}{'srch_roi':>10}{'gtd_eff':>9}{'att_roi':>9}")
    for _, r in g.iterrows():
        att = f"{r['attendance_roi']:.3f}" if pd.notna(r["attendance_roi"]) else "NaN"
        print(f"  {r['player_id']:<24}{r['signing_type']:<11}"
              f"{r['contract_aav_millions']:>7.1f}{r['search_roi']:>10.3f}"
              f"{r['guarantee_efficiency']:>9.3f}{att:>9}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading event-study results + master ...")
    events, master = load_inputs()
    print(f"  {master['player_id'].nunique()} signings.")

    df = build_roi(events, master)
    ok = validate(df)
    print_summary(df)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows -> {OUT_CSV}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
