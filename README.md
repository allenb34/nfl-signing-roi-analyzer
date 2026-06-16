# NFL Player Signing ROI Analyzer

## Project Summary
An end-to-end data pipeline and interactive dashboard that quantifies the return on investment of 12 high-profile NFL player signings (2015–2024) across three demand signals — search engagement, secondary-market ticket price, and stadium attendance — normalized per dollar of contract value. The analysis isolates the *announcement effect* of each signing via a pre/post event study, then regresses ROI on contract size, player archetype, and market structure.

## Key Finding
**Saturated-market signings generate ~4× higher search-engagement ROI per dollar** (Model A: `market_saturation` coefficient = **+4.18**, p = 0.004, HC3-robust OLS). Where a team's stadium is already capacity-bound (Seattle, Kansas City, Dallas), attendance volume can't respond to a signing — so *attention* and *price*, not turnstile counts, carry the business signal. The secondary-market **price ROI** result is directionally consistent (saturated markets show *lower* price ROI, coef = −0.59) but **exploratory only**: n = 10 and the effect is sensitive to a single observation (Deshaun Watson).

> ⚠️ **Small-sample caveat:** n = 10–12. All coefficients are *directional* (magnitude and sign of association), **not causal estimates**. Treat p < 0.10 as directionally significant. High R² reflects overfitting risk in a small sample.

## Data Sources
| Source | Metric | Method | Coverage |
|---|---|---|---|
| OverTheCap | Contract terms (AAV, total, guaranteed) | Manual entry / scrape | 12 / 12 |
| ESPN | Avg home attendance (season) | Manual entry (PFR blocks scrapers) | 12 / 12 |
| Google Trends | Search interest (pre/post window) | `pytrends` (live) | 12 / 12 |
| TicketIQ / SBJ / Wayback | Secondary-market ticket price | Manual entry (confidence-tiered) | 10 / 12 (2 COVID-excluded) |

## Pipeline (run in order)
| # | Script | Output |
|---|---|---|
| 1 | `contract_data_collector.py` | `data/raw/contract_data.csv` |
| 2 | `attendance_collector.py` | `data/raw/attendance_data.csv` |
| 3 | `trends_collector.py` | `data/raw/trends_data.csv` |
| 4 | `seatgeek_collector.py` *(+ `price_auto_collector.py` triage helper)* | `data/raw/seatgeek_data.csv` |
| 5 | `master_merge.py` | `data/processed/signings_master.csv` |
| 6 | `event_study.py` | `data/processed/event_study_results.csv` |
| 7 | `roi_calculator.py` | `data/processed/roi_estimates.csv` |
| 8 | `regression.py` | `data/processed/regression_results.csv` |
| 9 | `app.py` | Streamlit dashboard |

`signings_config.csv` is the single source of truth for join keys (`player_id`, `signing_date`) and all manual inputs — every collector reads from it.

## How to Run Locally
```bash
# Dashboard only (reads pre-computed CSVs in data/processed/):
pip install -r requirements.txt
python -m streamlit run app.py

# To regenerate the pipeline from scratch, additionally install:
pip install statsmodels requests beautifulsoup4 lxml pytrends truststore
python contract_data_collector.py
python attendance_collector.py
python trends_collector.py
python seatgeek_collector.py
python master_merge.py
python event_study.py
python roi_calculator.py
python regression.py
```
*Windows note:* the scraping collectors call `truststore.inject_into_ssl()` for system-trust-store TLS.

## Methodology Notes
- **Event study window.** ROI numerators use a window centered on the signing date to isolate the announcement effect (±12 weeks for search; ±30 days for ticket price). Attendance uses season-average pre vs. first post season.
- **COVID exclusions.** Observations in COVID-distorted seasons are **flagged and excluded, never imputed.** Exclusion is *comparison-level*: e.g., Russell Wilson 2019's 2018→2019 attendance comparison is kept while its →2020 comparison is dropped. Price rows tagged `covid_distorted` (or inheriting the season COVID flag) are excluded entirely — Patrick Mahomes 2020 and Odell Beckham Jr. 2021.
- **Market saturation.** Teams whose home attendance runs **> 95% of stadium capacity** regardless of roster moves are classified *saturated* (Seattle, Kansas City, Dallas). For these markets, attendance is inelastic and secondary-market price is the correct revenue proxy.
- **Confidence weighting.** Ticket-price ROI is weighted in the WLS regression by a data-confidence tier (high = 1.0, medium = 0.5, low = 0.25) so a 2015 annual-average proxy isn't treated like a 2022 30-day window.
- **Robust inference.** HC3 heteroskedasticity-robust standard errors throughout. Each model is refit excluding Watson as a sensitivity check; both versions are reported.

---
*Phases 1–5: data collection → merge → event study → ROI → regression → dashboard.*
