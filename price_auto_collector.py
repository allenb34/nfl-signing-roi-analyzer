"""
price_auto_collector.py
============================================================================
Phase 3b helper -- attempt to AUTO-FILL the SeatGeek price worksheet
(data/manual/seatgeek_prices.csv) from three public sources, in priority order,
so the manual lookup is reduced to only what genuinely can't be automated.

COLUMN MAPPING (this writes to the worksheet's EXISTING columns so the
downstream chain seatgeek_collector -> master_merge -> roi_calculator keeps
working unchanged):
    price_pre_30d       <- Source 1 TicketIQ  (pre-signing SEASON average)
    price_post_season1  <- Source 1 TicketIQ  (post-signing SEASON average)
    price_post_30d      <- Source 2 Wayback   (nearest 30-day-window snapshot)
    src_pre / src_post / src_season1  <- per-cell provenance, tagged with the
                                         source that actually filled that cell.
Note: for high-confidence rows the pre cell is a SEASON average (TicketIQ has no
true 30-day pre figure); src=ticketiq documents that. The post_30d cell is the
real announcement-window number from Wayback. pct_change_30d (computed later by
seatgeek_collector) = (price_post_30d - price_pre_30d)/price_pre_30d.

SOURCES (priority order, each best-effort; a miss falls through to the next)
    1. TicketIQ blog  -- team-season average resale prices.
    2. Wayback CDX    -- nearest archived SeatGeek listings page within +/-30d.
    3. Web search     -- DuckDuckGo HTML; regex a keyword-anchored $ figure.

HONESTY GUARDRAILS
    - A $ figure is accepted ONLY when it sits next to a price keyword
      (average / get-in / starting at) and falls in $20-$3000. Otherwise the
      cell is left empty rather than filled with a wrong number.
    - COVID rows (price_covid_affected via config covid_post_1) are SKIPPED
      entirely: prices left blank, all src cells set to covid_distorted.
    - Existing non-empty values are PRESERVED, never overwritten.
    - Any price cell still empty after all sources -> src = manual_required, and
      the row is flagged in the coverage report.

USAGE
    python price_auto_collector.py --dry-run   # print URLs only, no fetching
    python price_auto_collector.py             # live fetch, write worksheet
    python price_auto_collector.py --limit 3   # only first 3 non-COVID rows
============================================================================
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import timedelta

import truststore
truststore.inject_into_ssl()

import pandas as pd
import requests
from bs4 import BeautifulSoup

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG = os.path.join(HERE, "signings_config.csv")
WORKSHEET = os.path.join(HERE, "data", "manual", "seatgeek_prices.csv")

DELAY = 1.5          # polite delay between live requests (s)
TIMEOUT = 15         # per-request timeout (s)
PRICE_MIN, PRICE_MAX = 20, 3000   # plausible NFL resale avg bounds ($)
KEYWORDS = ("average", "avg", "get-in", "get in", "starting at", "starting from")
AUTO_SOURCES = {"ticketiq", "wayback", "web_search"}   # values subject to the guard

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"),
    "Accept-Language": "en-US,en;q=0.9",
}

# attendance_team -> SeatGeek URL slug (for Wayback CDX lookups).
SEATGEEK_SLUG = {
    "Seattle Seahawks": "seattle-seahawks", "Denver Broncos": "denver-broncos",
    "Las Vegas Raiders": "las-vegas-raiders", "Miami Dolphins": "miami-dolphins",
    "Los Angeles Rams": "los-angeles-rams", "Kansas City Chiefs": "kansas-city-chiefs",
    "Cleveland Browns": "cleveland-browns", "Minnesota Vikings": "minnesota-vikings",
    "Dallas Cowboys": "dallas-cowboys",
}


# ===========================================================================
# Shared helpers
# ===========================================================================
def _is_empty(v):
    return v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == ""


def extract_keyword_price(text):
    """Return the first $ figure that sits next to a price keyword and is in
    range, else None. Refuses to guess from un-anchored dollar amounts."""
    if not text:
        return None
    low = text.lower()
    for m in re.finditer(r"\$\s?([\d,]{2,7})", text):
        ctx = low[max(0, m.start() - 70): m.end() + 25]
        if any(k in ctx for k in KEYWORDS):
            val = int(m.group(1).replace(",", ""))
            if PRICE_MIN <= val <= PRICE_MAX:
                return val
    return None


def _get(url, session):
    try:
        r = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r
        print(f"      HTTP {r.status_code}")
    except requests.RequestException as exc:
        print(f"      {exc.__class__.__name__}")
    return None


# ===========================================================================
# URL construction (also used by --dry-run)
# ===========================================================================
def ticketiq_urls(team, pre_year, post_year):
    """Candidate TicketIQ pages to scan for this team's season averages."""
    q = team.lower().replace(" ", "+")
    return [
        "https://www.ticketiq.com/blog/nfl-ticket-prices/",
        f"https://www.ticketiq.com/?s={q}+{post_year}",
    ]


def wayback_cdx_url(slug, signing_dt):
    start = (signing_dt - timedelta(days=30)).strftime("%Y%m%d")
    end = (signing_dt + timedelta(days=30)).strftime("%Y%m%d")
    return (f"http://web.archive.org/cdx/search/cdx?url=seatgeek.com/{slug}-tickets"
            f"&output=json&from={start}&to={end}&limit=8")


def web_search_url(player_name, team, year):
    q = f"{player_name} {team} ticket prices {year} resale average".replace(" ", "+")
    return f"https://html.duckduckgo.com/html/?q={q}"


# ===========================================================================
# Source 1 -- TicketIQ team-season averages
# ===========================================================================
def source_ticketiq(team, pre_year, post_year, session):
    """Return (pre_season_avg, post_season_avg); either may be None."""
    pre_val = post_val = None
    for url in ticketiq_urls(team, pre_year, post_year):
        r = _get(url, session)
        time.sleep(DELAY)
        if r is None:
            continue
        text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
        # Look for a keyword-anchored price in the neighbourhood of each year.
        for year, slot in ((pre_year, "pre"), (post_year, "post")):
            idx = text.find(str(year))
            if idx == -1:
                continue
            window = text[max(0, idx - 200): idx + 200]
            val = extract_keyword_price(window)
            if val is not None:
                if slot == "pre" and pre_val is None:
                    pre_val = val
                elif slot == "post" and post_val is None:
                    post_val = val
        if pre_val and post_val:
            break
    return pre_val, post_val


# ===========================================================================
# Source 2 -- Wayback nearest SeatGeek snapshot in the 30-day window
# ===========================================================================
def source_wayback(slug, signing_dt, session):
    """Return the avg listing price from the nearest archived SeatGeek page, or
    None. Picks the snapshot whose timestamp is closest to the signing date."""
    if slug is None:
        return None
    r = _get(wayback_cdx_url(slug, signing_dt), session)
    time.sleep(DELAY)
    if r is None:
        return None
    try:
        rows = json.loads(r.text)
    except json.JSONDecodeError:
        return None
    if not rows or len(rows) < 2:
        return None

    header, *records = rows
    ts_i, orig_i = header.index("timestamp"), header.index("original")
    target = signing_dt.strftime("%Y%m%d%H%M%S")
    best = min(records, key=lambda rec: abs(int(rec[ts_i]) - int(target[:len(rec[ts_i])] or 0))
               if rec[ts_i].isdigit() else 10 ** 18)

    snap = f"http://web.archive.org/web/{best[ts_i]}/{best[orig_i]}"
    r2 = _get(snap, session)
    time.sleep(DELAY)
    if r2 is None:
        return None
    text = BeautifulSoup(r2.text, "html.parser").get_text(" ", strip=True)
    return extract_keyword_price(text)


# ===========================================================================
# Source 3 -- Web-search snippet fallback
# ===========================================================================
def source_web_search(player_name, team, year, session):
    r = _get(web_search_url(player_name, team, year), session)
    time.sleep(DELAY)
    if r is None:
        return None
    text = BeautifulSoup(r.text, "html.parser").get_text(" ", strip=True)
    return extract_keyword_price(text)


# ===========================================================================
# Per-row processing
# ===========================================================================
def process_row(ws_row, meta, session, dry_run):
    """Fill empty price cells for one signing. Mutates and returns ws_row (a
    dict). Returns also a per-cell source dict for the report."""
    pid = ws_row["player_id"]
    team = ws_row["attendance_team"]
    signing_dt = pd.to_datetime(ws_row["signing_date"])
    year = signing_dt.year
    pre_year, post_year = year - 1, year
    player_name = meta[pid]["player_name"]
    slug = SEATGEEK_SLUG.get(team)

    # --- COVID rows: skip every source, tag covid_distorted. ---
    if meta[pid]["covid"]:
        for s in ("src_pre", "src_post", "src_season1"):
            ws_row[s] = "covid_distorted"
        print(f"  {pid}: COVID -> skipped (covid_distorted)")
        return ws_row

    if dry_run:
        print(f"  {pid} ({team}):")
        print(f"    S1 TicketIQ : {ticketiq_urls(team, pre_year, post_year)}")
        print(f"    S2 Wayback  : {wayback_cdx_url(slug, signing_dt) if slug else '(no slug)'}")
        print(f"    S3 WebSearch: {web_search_url(player_name, team, year)}")
        return ws_row

    print(f"  {pid} ({team}): fetching ...")

    # --- Source 1: TicketIQ -> pre cell + season1 cell. ---
    need_pre = _is_empty(ws_row["price_pre_30d"])
    need_s1 = _is_empty(ws_row["price_post_season1"])
    if need_pre or need_s1:
        pre_val, post_val = source_ticketiq(team, pre_year, post_year, session)
        if need_pre and pre_val is not None:
            ws_row["price_pre_30d"], ws_row["src_pre"] = pre_val, "ticketiq"
        if need_s1 and post_val is not None:
            ws_row["price_post_season1"], ws_row["src_season1"] = post_val, "ticketiq"

    # --- Source 2: Wayback -> 30-day post cell. ---
    if _is_empty(ws_row["price_post_30d"]):
        wb = source_wayback(slug, signing_dt, session)
        if wb is not None:
            ws_row["price_post_30d"], ws_row["src_post"] = wb, "wayback"

    # --- Source 3: web search -> any cell still empty. ---
    for pcol, scol in [("price_pre_30d", "src_pre"),
                       ("price_post_30d", "src_post"),
                       ("price_post_season1", "src_season1")]:
        if _is_empty(ws_row[pcol]):
            val = source_web_search(player_name, team, year, session)
            if val is not None:
                ws_row[pcol], ws_row[scol] = val, "web_search"

    # --- Anything still empty -> manual_required. ---
    for pcol, scol in [("price_pre_30d", "src_pre"),
                       ("price_post_30d", "src_post"),
                       ("price_post_season1", "src_season1")]:
        if _is_empty(ws_row[pcol]):
            ws_row[scol] = "manual_required"
    return ws_row


# ===========================================================================
# Cross-row duplicate guard (artifact rejection)
# ===========================================================================
def reject_cross_row_duplicates(df):
    """Independent per-team/per-year resale averages don't coincide to the
    dollar. So an auto-sourced price that shows up for TWO OR MORE different
    signings is almost certainly a scraped PAGE ARTIFACT (ad copy, footer,
    repeated search snippet), not a real figure. Blank those cells and downgrade
    them to manual_required. Returns the list of rejected (player_id, col, value).
    """
    rejected = []
    for pcol, scol in [("price_pre_30d", "src_pre"),
                       ("price_post_30d", "src_post"),
                       ("price_post_season1", "src_season1")]:
        auto = df[df[scol].isin(AUTO_SOURCES) & df[pcol].map(lambda v: not _is_empty(v))]
        vals = pd.to_numeric(auto[pcol], errors="coerce")
        dup_vals = set(vals.value_counts()[lambda s: s > 1].index)
        for idx in auto.index:
            if pd.to_numeric(df.at[idx, pcol], errors="coerce") in dup_vals:
                rejected.append((df.at[idx, "player_id"], pcol, df.at[idx, pcol]))
                df.at[idx, pcol] = ""
                df.at[idx, scol] = "manual_required"
    return df, rejected


# ===========================================================================
# Coverage report
# ===========================================================================
def report(df, meta):
    print("\n" + "=" * 64)
    print("COVERAGE REPORT -- seatgeek_prices.csv (auto-fill)")
    print("=" * 64)
    noncovid = df[~df["player_id"].map(lambda p: meta[p]["covid"])]
    n = len(noncovid)
    print(f"Non-COVID rows: {n} (COVID skipped: {len(df) - n})")
    for col in ["price_pre_30d", "price_post_30d", "price_post_season1"]:
        filled = int(noncovid[col].map(lambda v: not _is_empty(v)).sum())
        print(f"  {col:<20} {filled}/{n} filled")

    # Rows still missing ALL three prices -> need a manual lookup.
    def _missing_all(r):
        return all(_is_empty(r[c]) for c in
                   ["price_pre_30d", "price_post_30d", "price_post_season1"])
    stuck = noncovid[noncovid.apply(_missing_all, axis=1)]["player_id"].tolist()
    print(f"\nmanual_required (no price from any source): {len(stuck)}")
    for p in stuck:
        print(f"  - {p}")
    print("=" * 64 + "\n")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Auto-fill the SeatGeek price worksheet.")
    ap.add_argument("--dry-run", action="store_true", help="Print URLs only.")
    ap.add_argument("--limit", type=int, default=None, help="Limit non-COVID rows.")
    args = ap.parse_args()

    if not os.path.exists(WORKSHEET):
        sys.exit("[FATAL] Worksheet missing -- run seatgeek_collector.py first.")

    cfg = pd.read_csv(CONFIG, comment="#", skip_blank_lines=True)
    cfg.columns = [c.strip() for c in cfg.columns]
    meta = {str(r["player_id"]).strip(): {
        # Use the clean search_term (no "(Departure)" annotation) for queries.
        "player_name": (str(r["search_term"]).strip()
                        if not _is_empty(r.get("search_term"))
                        else str(r["player_name"]).strip()),
        "covid": str(r.get("covid_post_1")).strip().lower() in ("true", "1", "yes"),
    } for _, r in cfg.iterrows()}

    ws = pd.read_csv(WORKSHEET)
    ws.columns = [c.strip() for c in ws.columns]
    ws["player_id"] = ws["player_id"].astype(str).str.strip()

    session = requests.Session()
    print(f"{'DRY RUN: ' if args.dry_run else ''}Processing "
          f"{len(ws)} worksheet rows ...\n")

    processed = 0
    new_rows = []
    for _, row in ws.iterrows():
        d = row.to_dict()
        is_covid = meta[d["player_id"]]["covid"]
        if args.limit is not None and not is_covid and processed >= args.limit:
            new_rows.append(d)
            continue
        new_rows.append(process_row(d, meta, session, args.dry_run))
        if not is_covid:
            processed += 1

    out = pd.DataFrame(new_rows, columns=ws.columns)

    if args.dry_run:
        print("\n[dry-run] No fetches performed, worksheet not modified.")
        return

    # Artifact guard: drop any auto value shared across signings before writing.
    out, rejected = reject_cross_row_duplicates(out)
    if rejected:
        print(f"\n[guard] rejected {len(rejected)} cross-row duplicate value(s) "
              f"as likely artifacts:")
        for pid, col, val in rejected:
            print(f"  {pid:<22} {col:<20} {val}  -> manual_required")

    out.to_csv(WORKSHEET, index=False)
    print(f"Updated worksheet -> {WORKSHEET}")
    report(out, meta)


if __name__ == "__main__":
    main()
