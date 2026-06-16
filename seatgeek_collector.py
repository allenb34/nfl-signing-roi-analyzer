"""
seatgeek_collector.py
============================================================================
Phase 3b -- Secondary-market ticket PRICE collector (price_roi input).

WHY THIS EXISTS
    For capacity-saturated franchises (Seattle, KC, Dallas) attendance VOLUME
    can't respond to a signing -- the building is already full. The correct
    revenue signal there is secondary-market ticket PRICE. Captures:
      price_pre_30d        -- avg resale price, 30 days BEFORE the signing
      price_post_30d       -- avg resale price, 30 days AFTER the signing
      price_post_season1   -- avg resale price across the first full post season
    and derives pct_change_30d, the immediate signing price bump (the lift that
    feeds price_roi in roi_calculator.py).

SOURCE HIERARCHY (fill the worksheet using, in priority order)
    1. TicketIQ annual NFL resale price reports (team-season averages) --
       ticketiq.com/blog, search "[TEAM] average ticket price [YEAR]".
    2. Industry coverage (Forbes, SBJ, ESPN) on price moves around a signing --
       useful for the 30-day window estimates.       -> src tag industry_coverage
    3. Wayback Machine snapshots of StubHub/SeatGeek pages -- only viable for
       2019+ signings with enough crawl density.                -> src tag wayback
    For LOW-confidence rows (pre-2019), the 30-day window usually isn't
    recoverable: use the team's ANNUAL AVERAGE resale price as the pre/post
    proxy and tag the src as annual_avg_proxy.

    NOTE: SeatGeek's own public listings are current/upcoming only and cannot
    serve historical windows -- the scrape stub below stays disabled for the 12
    historical signings and exists only to enrich a FUTURE upcoming signing.

DATA CONFIDENCE (price_data_confidence; default by signing year, overridable)
      <= 2018  -> low     (team-season avg only, no real 30-day window)
      2019-2021 -> medium  (TicketIQ + some Wayback)
      >= 2022  -> high    (TicketIQ + SBJ coverage + Wayback viable)
    roi_calculator.py turns this into a regression weight so a 2015 annual
    average is NOT treated like a 2022 30-day window.

COVID INHERITANCE
    price_covid_affected mirrors the attendance post_1 COVID flag (covid_post_1
    in the config): a signing whose first post season is the 2020 COVID season
    (Mahomes-2020) is flagged and dropped from price_roi automatically.

OUTPUT
    data/raw/seatgeek_data.csv, keyed on player_id (price columns namespaced
    price_* / price_src_* so they never collide with the attendance block in
    the master merge).

USAGE
    python seatgeek_collector.py        # reads/creates the worksheet, validates
============================================================================
"""

import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "signings_config.csv")
MANUAL_DIR = os.path.join(HERE, "data", "manual")
WORKSHEET = os.path.join(MANUAL_DIR, "seatgeek_prices.csv")
OUTPUT_DIR = os.path.join(HERE, "data", "raw")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "seatgeek_data.csv")

WINDOW_DAYS = 30  # +/- window around the signing date for the price comparison

# Columns the user fills in the worksheet (windows + default confidence
# pre-filled as guidance; price_* and src_* are the entry fields).
WORKSHEET_COLUMNS = [
    "player_id", "attendance_team", "signing_date",
    "window_pre", "window_post", "season1_label", "price_data_confidence",
    "price_pre_30d", "price_post_30d", "price_post_season1",
    "src_pre", "src_post", "src_season1",
]

OUTPUT_COLUMNS = [
    "player_id", "attendance_team", "signing_date",
    "window_pre", "price_pre_30d", "price_src_pre",
    "window_post", "price_post_30d", "price_src_post",
    "season1_label", "price_post_season1", "price_src_season1",
    "pct_change_30d", "price_covid_affected", "price_data_confidence",
    "data_source",
]

# Recognised price provenance tags (the source hierarchy + housekeeping).
# covid_distorted: used on Mahomes-2020's 30-day cells (price left blank); the
# price_covid_affected flag already excludes the row downstream, this just keeps
# the provenance honest about WHY the cell is empty.
VALID_SOURCES = {"ticketiq", "industry_coverage", "wayback", "annual_avg_proxy",
                 "seatgeek_confirmed", "vendor", "estimated", "manual",
                 "seatgeek_scrape", "covid_distorted", "na", "missing", ""}

VALID_CONFIDENCE = {"low", "medium", "high"}


# ===========================================================================
# Helpers
# ===========================================================================
def _to_bool(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def _price(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    try:
        return round(float(v), 2)
    except (ValueError, TypeError):
        return None


def _src(v):
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return ""
    return str(v).strip()


def default_confidence(signing_date):
    """Default price-data confidence by signing year (overridable in worksheet).
    Matches the available-source reality: pre-2019 is annual-average only;
    2019-2021 has TicketIQ + partial Wayback; 2022+ is well covered."""
    yr = pd.to_datetime(signing_date).year
    if yr <= 2018:
        return "low"
    if yr <= 2021:
        return "medium"
    return "high"


# ===========================================================================
# Step 1 -- Load registry + compute price windows
# ===========================================================================
def load_config(config_path):
    if not os.path.exists(config_path):
        sys.exit(f"[FATAL] Config not found: {config_path}")
    cfg = pd.read_csv(config_path, comment="#", skip_blank_lines=True)
    cfg.columns = [c.strip() for c in cfg.columns]
    for col in ["player_id", "signing_date", "attendance_team"]:
        cfg[col] = cfg[col].astype(str).str.strip()
    return cfg


def compute_windows(signing_date):
    sign = pd.to_datetime(signing_date)
    pre_start = (sign - pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    post_end = (sign + pd.Timedelta(days=WINDOW_DAYS)).strftime("%Y-%m-%d")
    sign_str = sign.strftime("%Y-%m-%d")
    yr = sign.year
    return (f"{pre_start}..{sign_str}",
            f"{sign_str}..{post_end}",
            f"{yr} NFL season (Sep {yr}-Jan {yr + 1})")


# ===========================================================================
# Step 2 -- Worksheet: auto-generate if missing, else load
# ===========================================================================
def ensure_worksheet(cfg):
    """Create the manual price worksheet pre-filled with windows + a default
    confidence (and an annual_avg_proxy src hint on low-confidence rows) if it
    does not exist. Returns the loaded worksheet DataFrame."""
    if not os.path.exists(WORKSHEET):
        os.makedirs(MANUAL_DIR, exist_ok=True)
        rows = []
        for _, r in cfg.iterrows():
            w_pre, w_post, s1 = compute_windows(r["signing_date"])
            conf = default_confidence(r["signing_date"])
            # Low-confidence rows realistically only have an annual average, so
            # pre-seed the src hint (user overwrites if a 30-day pull exists).
            src_hint = "annual_avg_proxy" if conf == "low" else ""
            rows.append({
                "player_id": r["player_id"],
                "attendance_team": r["attendance_team"],
                "signing_date": r["signing_date"],
                "window_pre": w_pre, "window_post": w_post, "season1_label": s1,
                "price_data_confidence": conf,
                "price_pre_30d": "", "price_post_30d": "", "price_post_season1": "",
                "src_pre": src_hint, "src_post": src_hint, "src_season1": src_hint,
            })
        pd.DataFrame(rows, columns=WORKSHEET_COLUMNS).to_csv(WORKSHEET, index=False)
        print(f"  [worksheet] created template -> {WORKSHEET}")
        print(f"  [worksheet] fill price_* + src_* (confidence pre-filled), re-run.")

    ws = pd.read_csv(WORKSHEET)
    ws.columns = [c.strip() for c in ws.columns]
    ws["player_id"] = ws["player_id"].astype(str).str.strip()
    return ws


# ===========================================================================
# Step 3 -- Best-effort SeatGeek scrape stub (current listings only; disabled)
# ===========================================================================
def scrape_seatgeek(attendance_team, window_label):
    """Returns None by design -- SeatGeek public listings are current/upcoming
    only and cannot serve historical windows. Wire the SeatGeek API here for a
    future upcoming-season signing; historical rows always use the worksheet."""
    return None


# ===========================================================================
# Step 4 -- Build one output row per signing
# ===========================================================================
def build_row(cfg_row, ws_by_id):
    pid = cfg_row["player_id"]
    w_pre, w_post, s1 = compute_windows(cfg_row["signing_date"])
    ws = ws_by_id.get(pid, {})

    # COVID inheritance from the attendance post_1 season flag in the config.
    price_covid = _to_bool(cfg_row.get("covid_post_1"))

    # Confidence: worksheet override if present, else default by year.
    conf = _src(ws.get("price_data_confidence")) or default_confidence(
        cfg_row["signing_date"])

    cells = {}
    for key, pcol, scol in [("pre", "price_pre_30d", "src_pre"),
                            ("post", "price_post_30d", "src_post"),
                            ("season1", "price_post_season1", "src_season1")]:
        val = _price(ws.get(pcol))
        src = _src(ws.get(scol))
        src = (src or "manual") if val is not None else (src or "missing")
        cells[key] = (val, src)

    pre_v, post_v = cells["pre"][0], cells["post"][0]
    pct_change = (round((post_v - pre_v) / pre_v * 100, 1)
                  if (pre_v not in (None, 0) and post_v is not None) else None)

    srcs = {cells[k][1] for k in cells}
    if srcs <= {"missing", ""}:
        data_source = "missing"
    elif "seatgeek_scrape" in srcs:
        data_source = "seatgeek_scrape"
    else:
        data_source = "manual"

    return {
        "player_id": pid,
        "attendance_team": cfg_row["attendance_team"],
        "signing_date": cfg_row["signing_date"],
        "window_pre": w_pre, "price_pre_30d": pre_v, "price_src_pre": cells["pre"][1],
        "window_post": w_post, "price_post_30d": post_v, "price_src_post": cells["post"][1],
        "season1_label": s1, "price_post_season1": cells["season1"][0],
        "price_src_season1": cells["season1"][1],
        "pct_change_30d": pct_change,
        "price_covid_affected": price_covid,
        "price_data_confidence": conf,
        "data_source": data_source,
    }


# ===========================================================================
# Step 5 -- Validation
# ===========================================================================
def validate(df):
    print("\n" + "=" * 64)
    print("DATA VALIDATION REPORT -- seatgeek_data.csv")
    print("=" * 64)
    ok = True

    print(f"Rows collected: {len(df)} (brief expects 12)")
    if len(df) != 12:
        ok = False
        print("  [FAIL] Row count != 12.")

    price_cols = ["price_pre_30d", "price_post_30d", "price_post_season1"]
    src_cols = ["price_src_pre", "price_src_post", "price_src_season1"]

    filled = int(df[price_cols].notna().sum().sum())
    print(f"\nPrice cells filled: {filled}/36")
    if filled < 36:
        print(f"  [TODO] {36 - filled} price cell(s) empty -- fill "
              f"data/manual/seatgeek_prices.csv (windows + confidence pre-filled).")

    # Provenance + confidence tags must be recognised.
    for col in src_cols:
        bad = sorted(set(df[col].fillna("")) - VALID_SOURCES)
        if bad:
            ok = False
            print(f"[FAIL] Unrecognised source tag in {col}: {bad}")
    bad_conf = sorted(set(df["price_data_confidence"]) - VALID_CONFIDENCE)
    if bad_conf:
        ok = False
        print(f"[FAIL] Unrecognised price_data_confidence: {bad_conf}")

    # Price sanity: NFL resale averages roughly $20-$3000; flag outliers.
    for col in price_cols:
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = df.loc[vals.notna() & ((vals < 20) | (vals > 3000)), "player_id"].tolist()
        if bad:
            print(f"  [WARN] {col} outside $20-$3000 for: {bad}")

    # COVID inheritance audit.
    covid_rows = df.loc[df["price_covid_affected"], "player_id"].tolist()
    print(f"\nprice_covid_affected=True: {covid_rows or '[]'} "
          f"(expected ['patrick_mahomes_2020'])")

    # Confidence distribution.
    print("price_data_confidence:", df["price_data_confidence"].value_counts().to_dict())

    if df["player_id"].duplicated().any():
        ok = False
        print("[FAIL] Duplicate player_id.")
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
    print("Loading signings registry ...")
    cfg = load_config(DEFAULT_CONFIG)
    print(f"  {len(cfg)} signing(s) loaded.")

    print("\nPreparing price worksheet ...")
    ws = ensure_worksheet(cfg)
    ws_by_id = {row["player_id"]: row.to_dict() for _, row in ws.iterrows()}

    print("\nBuilding SeatGeek rows ...")
    rows = [build_row(r, ws_by_id) for _, r in cfg.iterrows()]
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    validate(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
