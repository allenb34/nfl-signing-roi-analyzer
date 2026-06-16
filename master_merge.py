"""
master_merge.py
============================================================================
Phase 1 -- FINAL STEP: combine the three per-source CSVs into one master.

PURPOSE
    Join contract_data.csv + attendance_data.csv + trends_data.csv into a
    single analysis-ready table, signings_master.csv, with one row per signing.

KEY DESIGN
    signings_config.csv defines the canonical set of 12 player_id keys. We build
    the master spine from the config, then LEFT-JOIN each source onto it on
    player_id (validate='1:1'). This guarantees:
      - exactly the 12 expected rows (no source can add/drop signings),
      - any source missing a player_id surfaces as nulls + a loud warning,
      - each source's `data_source` is preserved as <source>_source for auditing.

OUTPUT
    data/processed/signings_master.csv

USAGE
    python master_merge.py
============================================================================
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "signings_config.csv")
RAW = os.path.join(HERE, "data", "raw")
OUT_DIR = os.path.join(HERE, "data", "processed")
OUT_CSV = os.path.join(OUT_DIR, "signings_master.csv")

CONTRACT_CSV = os.path.join(RAW, "contract_data.csv")
ATTENDANCE_CSV = os.path.join(RAW, "attendance_data.csv")
TRENDS_CSV = os.path.join(RAW, "trends_data.csv")
SEATGEEK_CSV = os.path.join(RAW, "seatgeek_data.csv")


def _require(path, builder_hint):
    """Load a source CSV or exit with a clear hint on which script to run."""
    if not os.path.exists(path):
        sys.exit(f"[FATAL] Missing {os.path.basename(path)} -- run "
                 f"{builder_hint} first.")
    return pd.read_csv(path)


# ===========================================================================
# Step 1 -- Build the master spine from the config (the canonical key list)
# ===========================================================================
def load_spine():
    """The descriptive columns every signing carries, keyed on player_id."""
    cfg = pd.read_csv(CONFIG, comment="#", skip_blank_lines=True)
    cfg.columns = [c.strip() for c in cfg.columns]
    spine_cols = ["player_id", "player_name", "signing_date", "team",
                  "attendance_team", "position", "archetype", "search_term"]
    return cfg[spine_cols].copy()


# ===========================================================================
# Step 2 -- Merge each source onto the spine, preserving provenance
# ===========================================================================
def merge_source(master, src_df, source_name, keep_cols):
    """Left-join selected columns of src_df onto master on player_id.

    Renames the source's `data_source` to <source_name>_source so the master
    records where every block came from. validate='1:1' raises if either side
    has duplicate player_ids (which would silently fan out the master).
    """
    df = src_df.copy()
    if "data_source" in df.columns:
        df = df.rename(columns={"data_source": f"{source_name}_source"})
        keep_cols = keep_cols + [f"{source_name}_source"]

    df = df[["player_id"] + keep_cols]

    # Flag any player_id present in the source but not the spine (orphan key).
    orphans = set(df["player_id"]) - set(master["player_id"])
    if orphans:
        print(f"  [WARN] {source_name}: player_id(s) not in config spine "
              f"(ignored by left-join): {sorted(orphans)}")

    # Flag spine keys the source is missing (will become nulls).
    missing = set(master["player_id"]) - set(df["player_id"])
    if missing:
        print(f"  [WARN] {source_name}: missing data for {sorted(missing)} "
              f"-> nulls in master.")

    return master.merge(df, on="player_id", how="left", validate="1:1")


# ===========================================================================
# Step 3 -- Validation of the assembled master
# ===========================================================================
def validate(df):
    print("\n" + "=" * 64)
    print("DATA VALIDATION REPORT -- signings_master.csv")
    print("=" * 64)
    ok = True

    print(f"Rows: {len(df)} (brief expects 12)   Columns: {df.shape[1]}")
    if len(df) != 12:
        ok = False
        print("  [FAIL] Row count != 12 after merge.")

    # player_id uniqueness -- the master primary key.
    dupes = df["player_id"][df["player_id"].duplicated()].tolist()
    if dupes:
        ok = False
        print(f"  [FAIL] Duplicate player_id(s): {dupes}")
    else:
        print("  player_id uniqueness: OK")

    # Per-source coverage: how many rows actually have data from each source.
    print("\nSource coverage (rows with data / 12):")
    for src, probe in [("contract", "total_value_usd"),
                       ("attendance", "avg_attendance_pre"),
                       ("trends", "avg_interest_pre"),
                       ("seatgeek", "pct_change_30d")]:
        if probe in df.columns:
            n = df[probe].notna().sum()
            note = "" if n == 12 else "   <-- some rows pending"
            print(f"  {src:<12} {n}/12{note}")

    # Null counts across the whole master (per column).
    print("\nNull counts per column (non-zero only):")
    nulls = df.isna().sum()
    for col, n in nulls[nulls > 0].items():
        print(f"  {col:<24} {n}")
    if nulls.sum() == 0:
        print("  (none)")

    # signing_date sanity.
    if pd.to_datetime(df["signing_date"], errors="coerce").isna().any():
        ok = False
        print("\n[FAIL] Unparseable signing_date in master.")

    print("=" * 64)
    print("VALIDATION:", "PASS" if ok else "FAIL (see above)")
    print("=" * 64 + "\n")
    return ok


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading sources ...")
    contract = _require(CONTRACT_CSV, "contract_data_collector.py")
    attendance = _require(ATTENDANCE_CSV, "attendance_collector.py")
    trends = _require(TRENDS_CSV, "trends_collector.py")

    master = load_spine()
    print(f"  spine: {len(master)} signings from config")

    # Contract block: financial terms (skip descriptive cols already on spine).
    master = merge_source(master, contract, "contract", keep_cols=[
        "contract_years", "total_value_usd", "aav_usd", "guaranteed_usd",
        "otc_url"])

    # Attendance block: per-cell value + source + COVID flag (attendance_team
    # already on spine). The covid_* flags drive the event study's exclusions.
    master = merge_source(master, attendance, "attendance", keep_cols=[
        "season_pre", "avg_attendance_pre", "src_pre", "covid_pre",
        "season_post_1", "avg_attendance_post_1", "src_post_1", "covid_post_1",
        "season_post_2", "avg_attendance_post_2", "src_post_2", "covid_post_2",
        "covid_affected"])

    # Trends block: buzz metrics (search_term/signing_date already on spine).
    master = merge_source(master, trends, "trends", keep_cols=[
        "window_weeks", "avg_interest_pre", "avg_interest_post",
        "peak_interest", "peak_date", "interest_lift_pct"])

    # SeatGeek price block (Phase 3b). Optional: merged when present so the
    # pipeline is wired end-to-end before prices are filled in. Columns are
    # price_*-namespaced upstream, so they don't collide with the attendance
    # block (src_pre / covid_affected etc.). pct_change_30d feeds price_roi;
    # price_covid_affected drives the same COVID exclusion as attendance.
    if os.path.exists(SEATGEEK_CSV):
        seatgeek = pd.read_csv(SEATGEEK_CSV)
        master = merge_source(master, seatgeek, "seatgeek", keep_cols=[
            "price_pre_30d", "price_post_30d", "price_post_season1",
            "pct_change_30d", "price_src_pre", "price_src_post",
            "price_src_season1", "price_covid_affected", "price_data_confidence"])
    else:
        print("  [info] seatgeek_data.csv not present yet -- price block skipped.")

    validate(master)

    os.makedirs(OUT_DIR, exist_ok=True)
    master.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(master)} rows x {master.shape[1]} cols -> {OUT_CSV}")


if __name__ == "__main__":
    main()
