"""
attendance_collector.py
============================================================================
Phase 1 -- Data source #2 of 3 for the NFL Player Signing ROI pipeline.

PURPOSE
    Season-average HOME attendance for three windows around each signing:
        season_pre     -- full season immediately before the signing
        season_post_1  -- first full season after the signing
        season_post_2  -- second full season after (where available)
    Season averages (not game-by-game) because the signings are offseason
    events; this is the unit the pre/post event study needs.

ATTENDANCE TEAM
    Read from `attendance_team`, NOT `team` -- for the Russell Wilson 2022
    DEPARTURE we measure SEATTLE (the team he left), not Denver.

PROVENANCE & COVID (per cell, from config)
    Pro-Football-Reference 403s automated clients, so attendance comes from the
    manual_att_* columns. Each cell also carries:
      - att_src_*  : espn_confirmed | estimated | estimated_capacity |
                     covid_void | covid_limited | na
      - covid_*    : True if that season is COVID-distorted (2020 closures /
                     reduced capacity). These observations are FLAGGED and later
                     EXCLUDED from the event study -- never imputed.
    A row-level `covid_affected` summarises whether any cell is COVID-flagged.

OUTPUT
    data/raw/attendance_data.csv, keyed on player_id.

USAGE
    python attendance_collector.py              # manual values + COVID flags
    python attendance_collector.py --scrape     # also attempt PFR (expect 403)
============================================================================
"""

import argparse
import os
import re
import sys
import time

# truststore: system trust store for TLS on this Windows/Python env.
import truststore
truststore.inject_into_ssl()

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "signings_config.csv")
OUTPUT_DIR = os.path.join(HERE, "data", "raw")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "attendance_data.csv")

# Output schema: per-cell value + source + covid flag, plus row-level summary.
OUTPUT_COLUMNS = [
    "player_id", "attendance_team",
    "season_pre", "avg_attendance_pre", "src_pre", "covid_pre",
    "season_post_1", "avg_attendance_post_1", "src_post_1", "covid_post_1",
    "season_post_2", "avg_attendance_post_2", "src_post_2", "covid_post_2",
    "covid_affected",
]

# Recognised provenance tags (anything else is flagged in validation).
VALID_SOURCES = {"espn_confirmed", "estimated", "estimated_capacity",
                 "covid_void", "covid_limited", "na", "pfr_scrape", "missing"}

# Full team name -> PFR franchise code (only used if --scrape is requested).
PFR_TEAM_CODES = {
    "Seattle Seahawks": "sea", "Denver Broncos": "den",
    "Las Vegas Raiders": "rai", "Miami Dolphins": "mia",
    "Los Angeles Rams": "ram", "Kansas City Chiefs": "kan",
    "Cleveland Browns": "cle", "Minnesota Vikings": "min",
    "Dallas Cowboys": "dal",
}
REQUEST_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}


# ===========================================================================
# Small helpers
# ===========================================================================
def _to_int(v):
    """Parse a cell to int; '' / NaN -> None. '0' stays 0 (a real COVID value)."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    try:
        return int(round(float(v)))
    except (ValueError, TypeError):
        return None


def _to_bool(v):
    """Parse a config truthy string to bool (blank -> False)."""
    return str(v).strip().lower() in ("true", "1", "yes")


def _clean_str(v, default="missing"):
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return default
    return str(v).strip()


# ===========================================================================
# Step 1 -- Load the signings registry
# ===========================================================================
def load_config(config_path):
    if not os.path.exists(config_path):
        sys.exit(f"[FATAL] Config not found: {config_path}")
    cfg = pd.read_csv(config_path, comment="#", skip_blank_lines=True)
    cfg.columns = [c.strip() for c in cfg.columns]
    required = {"player_id", "signing_date", "attendance_team"}
    missing = required - set(cfg.columns)
    if missing:
        sys.exit(f"[FATAL] Config missing required columns: {sorted(missing)}")
    for col in ["player_id", "signing_date", "attendance_team"]:
        cfg[col] = cfg[col].astype(str).str.strip()
    return cfg


# ===========================================================================
# Step 2 -- Derive the three season-year labels from the signing date
# ===========================================================================
def derive_seasons(cfg_row):
    """NFL seasons labelled by start year. post_1 = signing year, pre = year-1,
    post_2 = year+1. Exact for the 11 offseason signings; OBJ (Nov-2021) keeps
    post_1 = 2021 by decision (the 2020 COVID pre-flag handles his anomaly)."""
    year = pd.to_datetime(cfg_row["signing_date"]).year
    return year - 1, year, year + 1


# ===========================================================================
# Step 3 -- Best-effort PFR scrape (only when --scrape; expected to 403)
# ===========================================================================
def scrape_pfr_attendance(team_name, season_year, session):
    code = PFR_TEAM_CODES.get(team_name)
    if code is None:
        return None
    url = f"https://www.pro-football-reference.com/teams/{code}/{season_year}.htm"
    try:
        resp = session.get(url, headers=REQUEST_HEADERS, timeout=25)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        print(f"    [scrape] {team_name} {season_year}: HTTP {resp.status_code} "
              f"(PFR block) -> manual.")
        return None
    html = resp.text.replace("<!--", "").replace("-->", "")
    text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)
    m = re.search(r"Attendance[:\s]*([\d,]{4,})", text, re.I)
    return int(re.sub(r"[^0-9]", "", m.group(1))) if m else None


# ===========================================================================
# Step 4 -- Build one output row per signing
# ===========================================================================
def build_row(cfg_row, session, allow_scrape):
    s_pre, s_p1, s_p2 = derive_seasons(cfg_row)
    team = cfg_row["attendance_team"]

    # (config value column, config source column, config covid column)
    cells = {
        "pre":    ("manual_att_pre",    "att_src_pre",    "covid_pre",    s_pre),
        "post_1": ("manual_att_post_1", "att_src_post_1", "covid_post_1", s_p1),
        "post_2": ("manual_att_post_2", "att_src_post_2", "covid_post_2", s_p2),
    }

    out_vals, out_srcs, out_covid = {}, {}, {}
    for key, (vcol, scol, ccol, season) in cells.items():
        value = _to_int(cfg_row.get(vcol))
        src_cfg = _clean_str(cfg_row.get(scol), default="")
        covid = _to_bool(cfg_row.get(ccol))

        if value is not None:
            # Manual value present -> trust the config's per-cell source tag.
            src = src_cfg if src_cfg else "manual"
        elif src_cfg.lower() == "na":
            # Intentionally no data for this season (e.g. 2025 not used yet).
            src = "na"
        elif allow_scrape:
            sc = scrape_pfr_attendance(team, season, session)
            time.sleep(1.0)
            value, src = (sc, "pfr_scrape") if sc is not None else (None, "missing")
        else:
            src = "missing"

        out_vals[key], out_srcs[key], out_covid[key] = value, src, covid

    return {
        "player_id": cfg_row["player_id"],
        "attendance_team": team,
        "season_pre": s_pre, "avg_attendance_pre": out_vals["pre"],
        "src_pre": out_srcs["pre"], "covid_pre": out_covid["pre"],
        "season_post_1": s_p1, "avg_attendance_post_1": out_vals["post_1"],
        "src_post_1": out_srcs["post_1"], "covid_post_1": out_covid["post_1"],
        "season_post_2": s_p2, "avg_attendance_post_2": out_vals["post_2"],
        "src_post_2": out_srcs["post_2"], "covid_post_2": out_covid["post_2"],
        "covid_affected": any(out_covid.values()),
    }


# ===========================================================================
# Step 5 -- Data validation
# ===========================================================================
def validate(df):
    print("\n" + "=" * 64)
    print("DATA VALIDATION REPORT -- attendance_data.csv")
    print("=" * 64)
    ok = True

    print(f"Rows collected: {len(df)} (brief expects 12)")
    if len(df) != 12:
        ok = False
        print("  [FAIL] Row count != 12.")

    att_cols = ["avg_attendance_pre", "avg_attendance_post_1", "avg_attendance_post_2"]
    src_cols = ["src_pre", "src_post_1", "src_post_2"]
    covid_cols = ["covid_pre", "covid_post_1", "covid_post_2"]

    # Null counts (post_2 nulls are expected for the two 'na' 2024 signings).
    print("\nNull counts per column (non-zero only):")
    nz = df.isna().sum()
    for col, n in nz[nz > 0].items():
        print(f"  {col:<24} {n}")
    if nz.sum() == 0:
        print("  (none)")

    # Provenance tags must be recognised.
    for col in src_cols:
        bad = sorted(set(df[col]) - VALID_SOURCES)
        if bad:
            ok = False
            print(f"[FAIL] Unrecognised source tag in {col}: {bad}")

    # Season ordering pre < post_1 < post_2 must always hold.
    bad_order = df[~((df["season_pre"] < df["season_post_1"]) &
                     (df["season_post_1"] < df["season_post_2"]))]
    if len(bad_order):
        ok = False
        print(f"[FAIL] Season ordering broken: {bad_order['player_id'].tolist()}")
    else:
        print("\nSeason ordering (pre < post_1 < post_2): OK")

    # Attendance sanity 15k-100k -- but SKIP covid-flagged cells (0 / ~14k are
    # intentional) and nulls.
    for vcol, ccol in zip(att_cols, covid_cols):
        vals = pd.to_numeric(df[vcol], errors="coerce")
        consider = vals.notna() & ~df[ccol]
        bad = df.loc[consider & ((vals < 15000) | (vals > 100000)), "player_id"].tolist()
        if bad:
            print(f"  [WARN] {vcol} out of 15k-100k (non-COVID) for: {bad}")

    # COVID flag audit -- show exactly which observations are flagged & excluded.
    print("\nCOVID-flagged observations (excluded from event study):")
    any_covid = False
    for _, r in df.iterrows():
        for cell, ccol, vcol, scol in [
            ("pre", "covid_pre", "avg_attendance_pre", "src_pre"),
            ("post_1", "covid_post_1", "avg_attendance_post_1", "src_post_1"),
            ("post_2", "covid_post_2", "avg_attendance_post_2", "src_post_2")]:
            if r[ccol]:
                any_covid = True
                print(f"  {r['player_id']:<22} {cell:<7} "
                      f"value={r[vcol]}  src={r[scol]}")
    if not any_covid:
        print("  (none)")
    print(f"Rows with covid_affected=True: {int(df['covid_affected'].sum())} "
          f"(expected 3: wilson_2019, odell_beckham_2021, mahomes_2020)")

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
    parser = argparse.ArgumentParser(description="Collect season-average home "
                                                 "attendance per signing.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--scrape", action="store_true",
                        help="Also attempt PFR scrape for blank cells (expect 403).")
    args = parser.parse_args()

    print("Loading signings registry ...")
    cfg = load_config(args.config)
    print(f"  {len(cfg)} signing(s) loaded.")

    session = requests.Session()
    print("\nBuilding attendance rows ...")
    rows = [build_row(r, session, allow_scrape=args.scrape)
            for _, r in cfg.iterrows()]

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Clean nullable-int dtypes for seasons + attendance values.
    for col in ["season_pre", "season_post_1", "season_post_2",
                "avg_attendance_pre", "avg_attendance_post_1", "avg_attendance_post_2"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    validate(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
