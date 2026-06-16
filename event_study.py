"""
event_study.py
============================================================================
Phase 2 -- Event study calculator for the NFL Player Signing ROI project.

INPUT
    data/processed/signings_master.csv  (produced by master_merge.py)

WHAT IT COMPUTES
    For each signing and each METRIC, the abnormal lift: how much the metric
    moved from the pre-signing baseline to the post-signing window, relative to
    what the peer cohort did over the same kind of window.

    METRICS (one row each per signing):
      attendance_y1   -- avg home attendance, pre -> first post season
      attendance_y2   -- avg home attendance, pre -> second post season
      search_interest -- Google Trends interest, pre -> post window

    raw_lift_pct          = (post - pre) / pre * 100
    cohort_mean_lift_pct  = mean raw_lift across ALL valid signings (the
                            counterfactual "normal" move for this metric)
    abnormal_lift_pct     = raw_lift - cohort_mean_lift           (vs whole field)
    archetype_mean_lift   = mean raw_lift across same-archetype valid signings
    abnormal_vs_archetype = raw_lift - archetype_mean_lift        (vs QB / Skill peers)

COVID HANDLING (decision: exclude the COMPARISON, not the whole signing)
    A metric row is EXCLUDED (raw_lift = NaN, dropped from every benchmark) if
    either season in its pre->post comparison is COVID-flagged in the master.
    Consequence, by design:
      - russell_wilson_2019: y1 (2018->2019) is clean -> KEPT; y2 (->2020) EXCLUDED
      - patrick_mahomes_2020: y1 (->2020) EXCLUDED; y2 (2019->2021) clean -> KEPT
      - odell_beckham_2021: pre=2020 COVID -> BOTH attendance comparisons EXCLUDED
    COVID observations are flagged and excluded explicitly -- never imputed.

OUTPUT
    data/processed/event_study_results.csv  (long format: signing x metric)

USAGE
    python event_study.py
============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MASTER_CSV = os.path.join(HERE, "data", "processed", "signings_master.csv")
OUT_CSV = os.path.join(HERE, "data", "processed", "event_study_results.csv")

# Reliability ranking of an attendance cell's provenance (higher = better).
# Used to surface a per-estimate confidence so abnormal lifts built on
# capacity assumptions aren't read with the same weight as confirmed figures.
SRC_RANK = {"espn_confirmed": 3, "estimated": 2, "estimated_capacity": 1,
            "pfr_scrape": 2, "covid_void": 0, "covid_limited": 0,
            "na": 0, "missing": 0}

# Attendance volume is structurally inelastic for capacity-saturated franchises
# (Seattle, KC, Dallas consistently >95% full regardless of roster moves).
# For non-saturated teams (Raiders ~61k, Dolphins ~64k, Browns ~67k),
# attendance lift IS a valid signal -- segment these separately.
# The primary ROI signal is search interest lift; secondary market
# PRICE data (SeatGeek) is the correct attendance-substitute for
# capacity-saturated markets and will be collected in Phase 3b.
# NOTE: attendance rows are KEPT (not discarded); structural_note flags them.
SATURATED_TEAMS = {"Seattle Seahawks", "Kansas City Chiefs", "Dallas Cowboys"}

OUTPUT_COLUMNS = [
    "player_id", "player_name", "archetype", "metric",
    "value_pre", "value_post", "raw_lift_pct",
    "cohort_mean_lift_pct", "abnormal_lift_pct",
    "archetype_mean_lift_pct", "abnormal_vs_archetype_pct",
    "source_quality", "structural_note", "excluded", "excluded_reason",
]


# ===========================================================================
# Step 1 -- Load the master
# ===========================================================================
def load_master():
    if not os.path.exists(MASTER_CSV):
        sys.exit("[FATAL] signings_master.csv not found -- run master_merge.py first.")
    df = pd.read_csv(MASTER_CSV)
    # COVID flags may read back as bool or as 'True'/'False' strings; normalise.
    for col in ["covid_pre", "covid_post_1", "covid_post_2"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.lower() == "true"
    return df


# ===========================================================================
# Step 2 -- Turn each signing into one record per metric (the raw lift stage)
# ===========================================================================
def _worst_src(*tags):
    """Return the lower-confidence provenance tag among attendance cells."""
    return min(tags, key=lambda t: SRC_RANK.get(str(t), 0))


def build_metric_records(master):
    """Expand the wide master into long (signing x metric) records with the raw
    lift computed and COVID/no-data exclusions flagged (benchmarks come later)."""
    records = []
    for _, r in master.iterrows():
        # --- Each metric: (name, pre_value, post_value, covid_excluded?, src) ---
        metrics = [
            ("attendance_y1",
             r["avg_attendance_pre"], r["avg_attendance_post_1"],
             bool(r["covid_pre"]) or bool(r["covid_post_1"]),
             _worst_src(r["src_pre"], r["src_post_1"])),
            ("attendance_y2",
             r["avg_attendance_pre"], r["avg_attendance_post_2"],
             bool(r["covid_pre"]) or bool(r["covid_post_2"]),
             _worst_src(r["src_pre"], r["src_post_2"])),
            ("search_interest",
             r["avg_interest_pre"], r["avg_interest_post"],
             False, "pytrends"),
        ]
        saturated = r["attendance_team"] in SATURATED_TEAMS
        for name, pre, post, covid_excl, src in metrics:
            excluded, reason, raw_lift = False, "", np.nan

            if covid_excl:
                excluded, reason = True, "covid"
            elif pd.isna(pre) or pd.isna(post):
                excluded, reason = True, "no_data"
            elif float(pre) == 0:
                excluded, reason = True, "no_data"  # can't divide by zero baseline
            else:
                raw_lift = (float(post) - float(pre)) / float(pre) * 100.0

            # structural_note flags WHY an attendance lift may be uninformative
            # even when numerically valid: capacity-saturated teams can't show
            # volume response (use Phase 3b price data instead).
            if name.startswith("attendance"):
                note = ("capacity_saturated_use_price" if saturated
                        else "non_saturated_valid_signal")
            else:
                note = "primary_roi_signal"

            records.append({
                "player_id": r["player_id"],
                "player_name": r["player_name"],
                "archetype": r["archetype"],
                "metric": name,
                "value_pre": pre,
                "value_post": post,
                "raw_lift_pct": round(raw_lift, 2) if pd.notna(raw_lift) else np.nan,
                "source_quality": src,
                "structural_note": note,
                "excluded": excluded,
                "excluded_reason": reason,
            })
    return pd.DataFrame(records)


# ===========================================================================
# Step 3 -- Benchmarks: cohort mean and archetype mean per metric
# ===========================================================================
def add_benchmarks(df):
    """For each metric, the counterfactual is the mean raw_lift of all valid
    (non-excluded) signings for that metric. abnormal = raw_lift - benchmark.

    Benchmarks are computed on the full valid cohort (matching the project
    spec). Excluded rows contribute nothing and receive NaN benchmarks.
    """
    valid = df[~df["excluded"]]

    # cohort mean per metric, and archetype mean per (metric, archetype).
    cohort_mean = valid.groupby("metric")["raw_lift_pct"].mean()
    arch_mean = valid.groupby(["metric", "archetype"])["raw_lift_pct"].mean()

    def _cohort(row):
        return cohort_mean.get(row["metric"], np.nan) if not row["excluded"] else np.nan

    def _arch(row):
        if row["excluded"]:
            return np.nan
        return arch_mean.get((row["metric"], row["archetype"]), np.nan)

    df["cohort_mean_lift_pct"] = df.apply(_cohort, axis=1).round(2)
    df["archetype_mean_lift_pct"] = df.apply(_arch, axis=1).round(2)
    df["abnormal_lift_pct"] = (df["raw_lift_pct"] - df["cohort_mean_lift_pct"]).round(2)
    df["abnormal_vs_archetype_pct"] = (
        df["raw_lift_pct"] - df["archetype_mean_lift_pct"]).round(2)
    return df


# ===========================================================================
# Step 4 -- Validation + readable summary
# ===========================================================================
def validate(df):
    print("\n" + "=" * 64)
    print("DATA VALIDATION REPORT -- event_study_results.csv")
    print("=" * 64)
    ok = True

    n_sign = df["player_id"].nunique()
    print(f"Signings: {n_sign} (expect 12)   Metric rows: {len(df)} (expect 36)")
    if n_sign != 12 or len(df) != 36:
        ok = False
        print("  [FAIL] Unexpected signing/row count.")

    # Per-metric coverage: valid vs excluded.
    print("\nPer-metric coverage (valid / total) and exclusions:")
    for metric, g in df.groupby("metric"):
        valid = (~g["excluded"]).sum()
        reasons = g.loc[g["excluded"], "excluded_reason"].value_counts().to_dict()
        print(f"  {metric:<16} {valid}/{len(g)} valid   excluded={reasons or '{}'}")

    # Sanity: for each metric, abnormal_lift over the valid cohort must average
    # ~0 (since abnormal = raw - cohort_mean). Tolerance 0.05 absorbs the
    # residual from rounding abnormal_lift_pct to 2 decimals; a larger deviation
    # would signal a real benchmarking bug.
    print("\nAbnormal-lift zero-sum check (valid rows, should be ~0):")
    for metric, g in df[~df["excluded"]].groupby("metric"):
        m = g["abnormal_lift_pct"].mean()
        flag = "" if abs(m) < 0.05 else "  <-- NOT ~0 (bug?)"
        if abs(m) >= 0.05:
            ok = False
        print(f"  {metric:<16} mean abnormal = {m:+.4f}{flag}")

    # No valid row may carry a NaN raw_lift; no excluded row may carry a number.
    bad_valid = df[~df["excluded"] & df["raw_lift_pct"].isna()]
    bad_excl = df[df["excluded"] & df["raw_lift_pct"].notna()]
    if len(bad_valid) or len(bad_excl):
        ok = False
        print(f"\n[FAIL] raw_lift/excluded mismatch: "
              f"{bad_valid['player_id'].tolist()} / {bad_excl['player_id'].tolist()}")

    print("=" * 64)
    print("VALIDATION:", "PASS" if ok else "FAIL (see above)")
    print("=" * 64 + "\n")
    return ok


def print_leaderboard(df):
    """Human-readable abnormal-lift ranking per metric (valid rows only)."""
    for metric, g in df[~df["excluded"]].groupby("metric"):
        g = g.sort_values("abnormal_lift_pct", ascending=False)
        cm = g["cohort_mean_lift_pct"].iloc[0]
        print(f"\n{metric}  (cohort mean lift = {cm:+.1f}%)")
        print(f"  {'signing':<24}{'raw':>9}{'abn(coh)':>10}{'abn(arch)':>11}  src")
        for _, r in g.iterrows():
            print(f"  {r['player_id']:<24}{r['raw_lift_pct']:>8.1f}%"
                  f"{r['abnormal_lift_pct']:>9.1f}%{r['abnormal_vs_archetype_pct']:>10.1f}%"
                  f"  {r['source_quality']}")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading master ...")
    master = load_master()
    print(f"  {len(master)} signings loaded.")

    df = build_metric_records(master)
    df = add_benchmarks(df)
    df = df[OUTPUT_COLUMNS]  # enforce column order

    ok = validate(df)
    print_leaderboard(df)

    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {len(df)} metric rows -> {OUT_CSV}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
