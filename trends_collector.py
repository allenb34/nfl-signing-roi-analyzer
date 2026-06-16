"""
trends_collector.py
============================================================================
Phase 1 -- Data source #3 of 3 for the NFL Player Signing ROI pipeline.

PURPOSE
    For each of the 12 signings, pull Google Trends search interest in a window
    around the signing_date and summarise it into a pre/post buzz comparison:
        avg_interest_pre   -- mean weekly interest in the window BEFORE signing
        avg_interest_post  -- mean weekly interest in the window AFTER signing
        peak_interest      -- max interest across the window (the buzz spike)
        interest_lift_pct  -- % change of post vs pre (the "signing bump")

WHY search_term (not player_name)
    Russell Wilson appears in THREE rows. Google Trends can't tell the eras
    apart by name, so we anchor each row on its own signing_date window and use
    the clean `search_term` column ("Russell Wilson") -- the different
    timeframes disambiguate the three events. The "(Departure)" annotation lives
    only in player_name and is never sent to Google.

DATA SOURCE
    pytrends (unofficial Google Trends API). Unlike PFR, Trends is usually
    reachable, but Google rate-limits aggressively (HTTP 429). We retry with
    backoff and, if a row still fails, fall back to optional manual_trends_*
    config columns, else tag the row 'missing' so you can re-run just those.

OUTPUT
    data/raw/trends_data.csv, keyed on player_id (+ signing_date carried for
    reference) so it joins to the master via the shared primary key.

USAGE
    python trends_collector.py                 # default ±12-week window, US
    python trends_collector.py --window 8 --geo US
    python trends_collector.py --no-fetch      # offline: structure + manual only
============================================================================
"""

import argparse
import os
import sys
import time

# truststore: system trust store for TLS on this Windows/Python env.
import truststore
truststore.inject_into_ssl()

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "signings_config.csv")
OUTPUT_DIR = os.path.join(HERE, "data", "raw")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "trends_data.csv")

# --save-weekly persists the raw weekly series for the Page 2 event-study chart.
# It goes to data/processed/ (committed) since the dashboard reads it; the
# summary trends_data.csv stays in data/raw/ unchanged (this is additive).
PROCESSED_DIR = os.path.join(HERE, "data", "processed")
WEEKLY_CSV = os.path.join(PROCESSED_DIR, "trends_weekly.csv")
WEEKLY_COLUMNS = ["player_id", "signing_date", "week_date", "interest_value",
                  "window"]
# Google returns daily data for sub-9-month windows; we resample to weekly and
# keep +/-8 weeks around the signing for a focused, clean event-study line.
WEEKLY_WINDOW_WEEKS = 8

# Exact output schema for this source (player_id is the join key).
OUTPUT_COLUMNS = [
    "player_id",
    "search_term",
    "signing_date",
    "window_weeks",
    "avg_interest_pre",
    "avg_interest_post",
    "peak_interest",
    "peak_date",
    "interest_lift_pct",
    "data_source",        # 'pytrends' | 'manual' | 'missing'
]


# ===========================================================================
# Step 1 -- Load the signings registry
# ===========================================================================
def load_config(config_path):
    if not os.path.exists(config_path):
        sys.exit(f"[FATAL] Config not found: {config_path}")

    cfg = pd.read_csv(config_path, comment="#", skip_blank_lines=True)
    cfg.columns = [c.strip() for c in cfg.columns]

    required = {"player_id", "signing_date", "search_term"}
    missing = required - set(cfg.columns)
    if missing:
        sys.exit(f"[FATAL] Config missing required columns: {sorted(missing)}")

    for col in ["player_id", "signing_date", "search_term"]:
        cfg[col] = cfg[col].astype(str).str.strip()
    return cfg


# ===========================================================================
# Step 2 -- Fetch + summarise Google Trends for one signing
# ===========================================================================
def fetch_trends(search_term, signing_date, window_weeks, geo, pytrends,
                 max_retries=3):
    """Pull interest-over-time for one term in a window centred on signing_date.

    Returns (summary_stats_dict, raw_series) on success, or (None, None) on
    failure. raw_series is the full fetched interest series (daily resolution
    for sub-9-month windows) and is used for the weekly time-series export.

    The window is [signing_date - window_weeks, signing_date + window_weeks].
    Rows dated before signing_date form the 'pre' baseline; rows on/after form
    'post'.
    """
    sign_dt = pd.to_datetime(signing_date)
    start = (sign_dt - pd.Timedelta(weeks=window_weeks)).strftime("%Y-%m-%d")
    end = (sign_dt + pd.Timedelta(weeks=window_weeks)).strftime("%Y-%m-%d")
    timeframe = f"{start} {end}"

    # Retry loop with linear backoff to ride out Google's 429 rate limits.
    for attempt in range(1, max_retries + 1):
        try:
            pytrends.build_payload([search_term], cat=0, timeframe=timeframe,
                                   geo=geo, gprop="")
            iot = pytrends.interest_over_time()
            break
        except Exception as exc:  # pytrends raises various request errors
            wait = 5 * attempt
            print(f"    [trends] '{search_term}' attempt {attempt} failed "
                  f"({exc.__class__.__name__}); retrying in {wait}s ...")
            time.sleep(wait)
    else:
        print(f"    [trends] '{search_term}' gave up after {max_retries} tries.")
        return None, None

    if iot is None or iot.empty or search_term not in iot.columns:
        return None, None

    series = iot[search_term]
    pre = series[series.index < sign_dt]
    post = series[series.index >= sign_dt]
    if pre.empty or post.empty:
        # Window too narrow / no data either side -> unusable for pre/post.
        return None, None

    avg_pre = float(pre.mean())
    avg_post = float(post.mean())
    peak_val = int(series.max())
    peak_date = series.idxmax().strftime("%Y-%m-%d")
    lift = ((avg_post - avg_pre) / avg_pre * 100) if avg_pre > 0 else None

    stats = {
        "avg_interest_pre": round(avg_pre, 1),
        "avg_interest_post": round(avg_post, 1),
        "peak_interest": peak_val,
        "peak_date": peak_date,
        "interest_lift_pct": round(lift, 1) if lift is not None else None,
    }
    return stats, series


def weekly_records(player_id, signing_date, series):
    """Resample a (daily) interest series to weekly means and emit one record
    per week within +/-WEEKLY_WINDOW_WEEKS of the signing, tagged pre/post."""
    sign_dt = pd.to_datetime(signing_date)
    weekly = series.resample("W").mean().round().dropna()
    lo = sign_dt - pd.Timedelta(weeks=WEEKLY_WINDOW_WEEKS)
    hi = sign_dt + pd.Timedelta(weeks=WEEKLY_WINDOW_WEEKS)
    weekly = weekly[(weekly.index >= lo) & (weekly.index <= hi)]

    recs = []
    for week_dt, val in weekly.items():
        recs.append({
            "player_id": player_id,
            "signing_date": signing_date,
            "week_date": week_dt.strftime("%Y-%m-%d"),
            "interest_value": int(val),
            "window": "pre" if week_dt < sign_dt else "post",
        })
    return recs


# ===========================================================================
# Step 3 -- Build one output row per signing (fetch -> manual -> missing)
# ===========================================================================
def build_row(cfg_row, window_weeks, geo, pytrends, allow_fetch,
              save_weekly=False):
    """Return (summary_row, weekly_rows). weekly_rows is non-empty only when
    save_weekly and a live series was fetched."""
    term = cfg_row["search_term"]
    stats, series = None, None

    if allow_fetch and pytrends is not None:
        print(f"  - {cfg_row['player_id']}: querying Trends for '{term}' ...")
        stats, series = fetch_trends(term, cfg_row["signing_date"], window_weeks,
                                     geo, pytrends)
        time.sleep(2.0)  # spacing between players to ease rate limiting

    if stats is not None:
        data_source = "pytrends"
    else:
        # Optional manual fallback: add manual_trends_pre / manual_trends_post
        # columns to the config to hand-enter interest values if Trends fails.
        mp = cfg_row.get("manual_trends_pre")
        mq = cfg_row.get("manual_trends_post")
        if pd.notna(mp) and pd.notna(mq) and str(mp).strip() and str(mq).strip():
            avg_pre, avg_post = float(mp), float(mq)
            lift = ((avg_post - avg_pre) / avg_pre * 100) if avg_pre > 0 else None
            stats = {
                "avg_interest_pre": round(avg_pre, 1),
                "avg_interest_post": round(avg_post, 1),
                "peak_interest": None,
                "peak_date": None,
                "interest_lift_pct": round(lift, 1) if lift is not None else None,
            }
            data_source = "manual"
        else:
            stats = {k: None for k in ("avg_interest_pre", "avg_interest_post",
                                       "peak_interest", "peak_date",
                                       "interest_lift_pct")}
            data_source = "missing"

    summary_row = {
        "player_id": cfg_row["player_id"],
        "search_term": term,
        "signing_date": cfg_row["signing_date"],
        "window_weeks": window_weeks,
        **stats,
        "data_source": data_source,
    }

    weekly_rows = []
    if save_weekly and series is not None:
        weekly_rows = weekly_records(cfg_row["player_id"],
                                     cfg_row["signing_date"], series)
    return summary_row, weekly_rows


# ===========================================================================
# Step 4 -- Data validation
# ===========================================================================
def validate(df):
    print("\n" + "=" * 64)
    print("DATA VALIDATION REPORT -- trends_data.csv")
    print("=" * 64)
    ok = True

    print(f"Rows collected: {len(df)} (brief expects 12)")
    if len(df) != 12:
        print("  [WARN] Row count != 12 -- check signings_config.csv.")

    print("\nNull counts per column:")
    val_cols = ["avg_interest_pre", "avg_interest_post", "peak_interest",
                "interest_lift_pct"]
    for col, n in df.isna().sum().items():
        flag = "  <-- has nulls" if n else ""
        print(f"  {col:<22} {n}{flag}")
    n_missing = (df["data_source"] == "missing").sum()
    if n_missing:
        print(f"  [TODO] {n_missing} row(s) have no Trends data -- re-run "
              f"trends_collector.py (rate limit) or fill manual_trends_* cols.")

    # Interest values are Google-normalised 0..100; flag anything outside.
    for col in ["avg_interest_pre", "avg_interest_post", "peak_interest"]:
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = df.loc[vals.notna() & ((vals < 0) | (vals > 100)), "player_id"].tolist()
        if bad:
            ok = False
            print(f"[FAIL] {col} outside 0-100 (not normalised?) for: {bad}")

    # signing_date parseable.
    if pd.to_datetime(df["signing_date"], errors="coerce").isna().any():
        ok = False
        print("[FAIL] Unparseable signing_date present.")

    # player_id uniqueness.
    dupes = df["player_id"][df["player_id"].duplicated()].tolist()
    if dupes:
        ok = False
        print(f"[FAIL] Duplicate player_id(s): {dupes}")
    else:
        print("player_id uniqueness: OK")

    print("=" * 64)
    print("VALIDATION:", "PASS" if ok else "FAIL (see above)")
    print("=" * 64 + "\n")
    return ok


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Collect Google Trends buzz "
                                                 "per signing via pytrends.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--window", type=int, default=12,
                        help="Weeks before/after signing_date (default 12).")
    parser.add_argument("--geo", default="US",
                        help="Trends geo (default US; '' for worldwide).")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip pytrends; structure + manual_trends_* only.")
    parser.add_argument("--save-weekly", action="store_true",
                        help="Also persist the raw weekly series to "
                             "data/processed/trends_weekly.csv (for the "
                             "Page 2 event-study chart).")
    args = parser.parse_args()

    print("Loading signings registry ...")
    cfg = load_config(args.config)
    print(f"  {len(cfg)} signing(s) loaded.")

    # Lazily create the pytrends client only when we actually fetch.
    pytrends = None
    if not args.no_fetch:
        from pytrends.request import TrendReq
        # tz=360 = US central offset in minutes; hl = host language.
        pytrends = TrendReq(hl="en-US", tz=360)

    print("\nBuilding trends rows ...")
    results = [build_row(r, args.window, args.geo, pytrends,
                         allow_fetch=not args.no_fetch,
                         save_weekly=args.save_weekly)
               for _, r in cfg.iterrows()]
    rows = [summary for summary, _ in results]
    weekly_rows = [wr for _, wlist in results for wr in wlist]

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Tidy dtypes: peak_interest integer-nullable; the averages stay float.
    df["peak_interest"] = pd.to_numeric(df["peak_interest"],
                                        errors="coerce").astype("Int64")

    validate(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUTPUT_CSV}")

    # Additive weekly export (does not touch the summary above).
    if args.save_weekly:
        wdf = pd.DataFrame(weekly_rows, columns=WEEKLY_COLUMNS)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        wdf.to_csv(WEEKLY_CSV, index=False)
        n_sign = wdf["player_id"].nunique()
        print(f"Wrote {len(wdf)} weekly rows across {n_sign} signings "
              f"-> {WEEKLY_CSV}")


if __name__ == "__main__":
    main()
