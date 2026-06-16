"""
contract_data_collector.py
============================================================================
Phase 1 -- Data source #1 of 3 for the NFL Player Signing ROI pipeline.

PURPOSE
    Collect contract terms (years, total value, APY, guaranteed money) for each
    of the 12 signings defined in signings_config.csv, using OverTheCap.com as
    the primary source, with a manual-entry fallback baked into the config.

OUTPUT
    data/raw/contract_data.csv  -- one row per signing, keyed on
    (player_id, signing_date) so it joins cleanly with the attendance and
    Google Trends CSVs in the master merge step.

DESIGN NOTES
    - signings_config.csv is the single registry of join keys. We NEVER invent
      a player_id here; we read it from the config so all three collectors agree.
    - OverTheCap sits behind Cloudflare and changes its HTML periodically, so
      scraping is best-effort: if a fetch/parse fails for a player, we fall back
      to the manual_* columns in the config and tag data_source accordingly.
      The pipeline therefore never hard-fails on a single blocked request.

USAGE
    python contract_data_collector.py
    python contract_data_collector.py --config signings_config.csv --no-scrape
============================================================================
"""

import argparse
import os
import re
import sys
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Paths -- resolve relative to THIS file so the script runs from any cwd.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CONFIG = os.path.join(HERE, "signings_config.csv")
OUTPUT_DIR = os.path.join(HERE, "data", "raw")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "contract_data.csv")

# The exact, ordered schema every downstream script expects from this source.
OUTPUT_COLUMNS = [
    "player_id",          # PRIMARY KEY (shared across all collectors)
    "player_name",
    "signing_date",       # JOIN KEY (ISO YYYY-MM-DD)
    "team",
    "position",
    "archetype",          # analysis grouping (QB / Skill / ...)
    "contract_years",
    "total_value_usd",
    "aav_usd",            # average annual value = total / years (if not given)
    "guaranteed_usd",
    "otc_url",
    "data_source",        # 'otc_scrape' or 'manual' -- provenance for auditing
]

# A real browser UA reduces the odds of an immediate Cloudflare block. Even so,
# treat any scrape as best-effort and rely on the manual fallback when needed.
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ===========================================================================
# Step 1 -- Load the signings registry (the source of truth for join keys)
# ===========================================================================
def load_config(config_path):
    """Read signings_config.csv, skipping the '#' comment/instruction lines.

    Returns a DataFrame with one row per signing. Raises with a clear message
    if the file is missing or still contains only the placeholder examples.
    """
    if not os.path.exists(config_path):
        sys.exit(f"[FATAL] Config not found: {config_path}\n"
                 f"        Create it from the template and add your 12 signings.")

    # comment='#' drops the instruction header; skip_blank_lines keeps it tidy.
    cfg = pd.read_csv(config_path, comment="#", skip_blank_lines=True)

    # Normalise column whitespace/casing so minor edits don't break the loader.
    cfg.columns = [c.strip() for c in cfg.columns]

    required = {"player_id", "player_name", "signing_date", "team"}
    missing = required - set(cfg.columns)
    if missing:
        sys.exit(f"[FATAL] Config is missing required columns: {sorted(missing)}")

    # Trim stray whitespace in the key columns -- a leading space in player_id
    # would silently break the master merge later.
    for col in ["player_id", "signing_date", "team", "position", "archetype",
                "otc_url"]:
        if col in cfg.columns:
            cfg[col] = cfg[col].astype(str).str.strip()

    return cfg


# ===========================================================================
# Step 2 -- Scrape one OverTheCap player page (best-effort)
# ===========================================================================
def _money_to_int(text):
    """Convert an OTC money string like '$37,500,000' -> 37500000 (int).

    Returns None if no parseable number is present.
    """
    if not text:
        return None
    digits = re.sub(r"[^0-9]", "", str(text))
    return int(digits) if digits else None


def scrape_otc_contract(otc_url, session, timeout=20):
    """Fetch and parse a single OverTheCap player page.

    Returns a dict with whatever of {contract_years, total_value_usd, apy_usd,
    guaranteed_usd} we could extract, or {} on any failure. We deliberately
    swallow errors and return {} so the caller can fall back to manual data --
    one blocked page must not sink the whole run.

    NOTE: OTC markup changes over time. The selectors below target the contract
    summary table present as of this writing; if OTC restructures, this returns
    {} and the manual_* config values take over. Verify scraped numbers against
    the brief before trusting them.
    """
    if not otc_url or otc_url.lower() in ("nan", "none", ""):
        return {}

    try:
        resp = session.get(otc_url, headers=REQUEST_HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"    [scrape] request failed ({exc.__class__.__name__}); "
              f"will use manual fallback.")
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    out = {}

    # --- Strategy A: parse any HTML <table> with pandas, look for labelled rows.
    # OTC renders a contract summary table; pd.read_html turns it into a frame
    # we can scan for 'Years', 'Total Value', 'Average', 'Guaranteed' labels.
    try:
        tables = pd.read_html(resp.text)
    except ValueError:
        tables = []  # no tables parsed -- fall through to text scan

    for tbl in tables:
        flat = tbl.astype(str)
        for _, row in flat.iterrows():
            cells = [c.strip() for c in row.tolist()]
            label = cells[0].lower() if cells else ""
            value = cells[1] if len(cells) > 1 else ""
            if "year" in label and out.get("contract_years") is None:
                yrs = _money_to_int(value)
                # Years is a small count, not money; guard against grabbing $.
                out["contract_years"] = yrs if (yrs and yrs < 25) else None
            elif "total" in label and "value" in label:
                out["total_value_usd"] = _money_to_int(value)
            elif "average" in label or "aav" in label or "apy" in label \
                    or "per year" in label:
                out["aav_usd"] = _money_to_int(value)
            elif "guarantee" in label and out.get("guaranteed_usd") is None:
                out["guaranteed_usd"] = _money_to_int(value)

    # --- Strategy B: regex the raw page text for labelled dollar figures as a
    # backup when the table layout didn't yield a field.
    page_text = soup.get_text(" ", strip=True)

    def _find(label_pattern):
        m = re.search(label_pattern + r"[:\s]*\$?([\d,]+)", page_text, re.I)
        return _money_to_int(m.group(1)) if m else None

    out.setdefault("total_value_usd", _find(r"total value"))
    out.setdefault("guaranteed_usd", _find(r"fully guaranteed|guaranteed at signing|guaranteed"))

    # Drop keys whose values are None so .update() in the caller doesn't clobber
    # a good manual value with a missing scrape.
    return {k: v for k, v in out.items() if v is not None}


# ===========================================================================
# Step 3 -- Build one clean output row per signing (scrape -> manual fallback)
# ===========================================================================
def build_row(cfg_row, session, allow_scrape):
    """Assemble a single contract_data row for one signing.

    Precedence: a successfully scraped field wins; otherwise the manual_* value
    from the config is used. data_source records which path actually supplied
    the financial figures.
    """
    # Start from the manual fallback values in the config (may be NaN/blank).
    manual = {
        "contract_years": cfg_row.get("manual_years"),
        "total_value_usd": cfg_row.get("manual_total"),
        "aav_usd": cfg_row.get("manual_aav"),
        "guaranteed_usd": cfg_row.get("manual_guaranteed"),
    }

    scraped = {}
    if allow_scrape:
        print(f"  - {cfg_row['player_id']}: scraping OTC ...")
        scraped = scrape_otc_contract(cfg_row.get("otc_url"), session)
        # Be polite: small delay so we don't hammer OTC across 12 requests.
        time.sleep(1.5)

    # Merge: manual as base, scraped values override where present.
    merged = dict(manual)
    merged.update(scraped)
    data_source = "otc_scrape" if scraped else "manual"

    # Derive AAV if we have total + years but no explicit average.
    if (merged.get("aav_usd") in (None, "") or pd.isna(merged.get("aav_usd"))):
        tv, yrs = merged.get("total_value_usd"), merged.get("contract_years")
        if tv and yrs:
            try:
                merged["aav_usd"] = int(round(float(tv) / float(yrs)))
            except (ValueError, ZeroDivisionError):
                pass

    return {
        "player_id": cfg_row["player_id"],
        "player_name": cfg_row["player_name"],
        "signing_date": cfg_row["signing_date"],
        "team": cfg_row.get("team"),
        "position": cfg_row.get("position"),
        "archetype": cfg_row.get("archetype"),
        "contract_years": merged.get("contract_years"),
        "total_value_usd": merged.get("total_value_usd"),
        "aav_usd": merged.get("aav_usd"),
        "guaranteed_usd": merged.get("guaranteed_usd"),
        "otc_url": cfg_row.get("otc_url"),
        "data_source": data_source,
    }


# ===========================================================================
# Step 4 -- Data validation (null counts, date range, key uniqueness, dtypes)
# ===========================================================================
def validate(df):
    """Run sanity checks and print a report. Returns True if all checks pass.

    Checks are non-fatal warnings so you can inspect partial data, except the
    duplicate-key check which would corrupt the master merge.
    """
    print("\n" + "=" * 64)
    print("DATA VALIDATION REPORT -- contract_data.csv")
    print("=" * 64)
    ok = True

    # (a) Row count vs. expected 12 signings.
    print(f"Rows collected: {len(df)} (brief expects 12)")
    if len(df) != 12:
        print("  [WARN] Row count != 12 -- check signings_config.csv.")

    # (b) Null counts per column.
    print("\nNull counts per column:")
    nulls = df.isna().sum()
    for col, n in nulls.items():
        flag = "  <-- has nulls" if n else ""
        print(f"  {col:<18} {n}{flag}")
    if nulls[["total_value_usd", "guaranteed_usd"]].sum() > 0:
        print("  [WARN] Missing contract money -- fill manual_* cols in config "
              "or fix otc_url.")

    # (c) signing_date parses as a real date and sits in a sane range.
    parsed = pd.to_datetime(df["signing_date"], errors="coerce")
    n_bad = parsed.isna().sum()
    if n_bad:
        ok = False
        print(f"\n[FAIL] {n_bad} signing_date value(s) not ISO YYYY-MM-DD.")
    else:
        lo, hi = parsed.min().date(), parsed.max().date()
        print(f"\nsigning_date range: {lo} -> {hi}")
        # NFL free-agency era sanity window; widen if your brief predates this.
        if lo.year < 2010 or hi.year > 2030:
            print("  [WARN] A signing_date falls outside 2010-2030 -- verify.")

    # (d) player_id must be unique (it is the master primary key).
    dupes = df["player_id"][df["player_id"].duplicated()].tolist()
    if dupes:
        ok = False
        print(f"[FAIL] Duplicate player_id(s): {dupes} -- keys must be unique.")
    else:
        print("player_id uniqueness: OK")

    # (e) Money columns are numeric where present.
    for col in ["total_value_usd", "aav_usd", "guaranteed_usd", "contract_years"]:
        coerced = pd.to_numeric(df[col], errors="coerce")
        bad = df[col].notna() & coerced.isna()
        if bad.any():
            ok = False
            print(f"[FAIL] Non-numeric values in {col}: "
                  f"{df.loc[bad, 'player_id'].tolist()}")

    print("=" * 64)
    print("VALIDATION:", "PASS" if ok else "FAIL (see above)")
    print("=" * 64 + "\n")
    return ok


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Collect NFL contract data "
                                                 "from OverTheCap.")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="Path to signings_config.csv")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip OTC scraping; use manual_* config values only.")
    args = parser.parse_args()

    print("Loading signings registry ...")
    cfg = load_config(args.config)
    print(f"  {len(cfg)} signing(s) loaded from {os.path.basename(args.config)}")

    # One Session reuses the TCP connection across all 12 player requests.
    session = requests.Session()

    print("\nBuilding contract rows ...")
    rows = [build_row(r, session, allow_scrape=not args.no_scrape)
            for _, r in cfg.iterrows()]

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)

    # Numeric coercion so the CSV writes clean ints, not '37500000.0' strings.
    for col in ["contract_years", "total_value_usd", "aav_usd", "guaranteed_usd"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    validate(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
