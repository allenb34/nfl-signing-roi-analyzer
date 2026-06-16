"""
app.py -- NFL Player Signing ROI dashboard (Phase 5, Streamlit).
============================================================================
PRESENTATION LAYER ONLY. Reads the pipeline's CSV outputs and renders them;
it never refits a model. The regression tables come straight from the 16-row
regression_results.csv produced by regression.py.

Run locally:   python -m streamlit run app.py
Deploy:        Streamlit Community Cloud (requirements.txt provided).

Five pages: League Overview / Signing Deep-Dive / Seahawks Module /
Regression Explorer / Methodology.
============================================================================
"""

import os

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
PROC = os.path.join(HERE, "data", "processed")
# NOTE: trends_data.csv lives in data/processed/ (committed) -- data/raw/ is
# gitignored and not present on the Streamlit Cloud deployment.

# ---- Palette ---------------------------------------------------------------
NEUTRAL_BG = "#0E1117"
CARD_BG = "#1A1D24"
TEXT = "#FAFAFA"
ACCENT = "#5B8FF9"        # neutral blue
MUTED = "#A5ACAF"
SAT_COLOR = "#5B8FF9"     # saturated markets
NONSAT_COLOR = "#F6A623"  # non-saturated markets
P_COLORS = {"p<0.05": "#3DCC91", "0.05-0.10": "#F6A623", "p>0.10": "#6B7280"}
# Seahawks colorway -- USED ONLY on the Seahawks module page.
SEA_NAVY, SEA_GREEN, SEA_SILVER = "#002244", "#69BE28", "#A5ACAF"

SEAHAWKS_IDS = ["russell_wilson_2015", "russell_wilson_2019",
                "russell_wilson_2022", "geno_smith_2023", "dk_metcalf_2022"]
WATSON_ID = "deshaun_watson_2022"


# ===========================================================================
# Data loading (pure; cached in the app)
# ===========================================================================
def load_data():
    """Return merged per-signing frame + the regression results table."""
    roi = pd.read_csv(os.path.join(PROC, "roi_estimates.csv"))
    reg = pd.read_csv(os.path.join(PROC, "regression_results.csv"))
    master = pd.read_csv(os.path.join(PROC, "signings_master.csv"))
    trends = pd.read_csv(os.path.join(PROC, "trends_data.csv"))

    # Bring the trends summary + a couple of master price/attendance fields onto
    # the ROI frame so each page has one tidy row per signing.
    keep_master = ["player_id", "avg_interest_pre", "avg_interest_post",
                   "peak_interest", "peak_date", "interest_lift_pct",
                   "pct_change_30d", "signing_date", "price_post_30d",
                   "price_pre_30d"]
    df = roi.merge(master[keep_master], on="player_id", how="left")
    df["controversy_flag"] = df["player_id"] == WATSON_ID
    return df, reg


def load_weekly():
    """Weekly Trends time-series for the Page 2 event-study chart (Phase 6)."""
    w = pd.read_csv(os.path.join(PROC, "trends_weekly.csv"))
    w["week_date"] = pd.to_datetime(w["week_date"])
    return w


def _cached_load():
    # @st.cache_data wrapper (kept separate so load_data stays import-testable).
    return load_data()


# ===========================================================================
# Pure figure builders (no Streamlit calls -> unit-testable)
# ===========================================================================
def fig_roi_scatter(df, ycol, title):
    """Scatter of AAV vs an ROI metric, coloured by market saturation."""
    d = df.dropna(subset=[ycol]).copy()
    d["sat_label"] = d["market_saturation"].map(
        {"saturated": "Saturated", "non_saturated": "Non-saturated"})
    fig = px.scatter(
        d, x="contract_aav_millions", y=ycol, color="sat_label",
        size="price_confidence_weight", size_max=22, text="player_name",
        color_discrete_map={"Saturated": SAT_COLOR, "Non-saturated": NONSAT_COLOR},
        labels={"contract_aav_millions": "Contract AAV ($M)", ycol: title,
                "sat_label": "Market"},
        hover_data={"player_name": True, "contract_aav_millions": ":.1f",
                    ycol: ":.3f", "price_confidence_weight": False,
                    "sat_label": False},
    )
    fig.update_traces(textposition="top center",
                      textfont=dict(size=9, color=MUTED))
    fig.add_hline(y=0, line_dash="dot", line_color=MUTED, opacity=0.5)
    fig.update_layout(template="plotly_dark", height=460,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08, x=0),
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


def fig_event_study(wk, signing_date):
    """Weekly search-interest event-study line for one signing: a continuous
    line over the ±8-week window, signing date marked, pre/post shaded, peak
    annotated. Reads the true weekly series (trends_weekly.csv, Phase 6)."""
    wk = wk.sort_values("week_date")
    sign_dt = pd.to_datetime(signing_date)
    xmin, xmax = wk["week_date"].min(), wk["week_date"].max()
    peak = wk.loc[wk["interest_value"].idxmax()]

    fig = go.Figure()
    # Shaded windows: pre = gray, post = light blue.
    fig.add_vrect(x0=xmin, x1=sign_dt, fillcolor=MUTED, opacity=0.10, line_width=0)
    fig.add_vrect(x0=sign_dt, x1=xmax, fillcolor=ACCENT, opacity=0.12, line_width=0)
    # The interest line.
    fig.add_trace(go.Scatter(
        x=wk["week_date"], y=wk["interest_value"], mode="lines+markers",
        line=dict(color=ACCENT, width=2.5), marker=dict(size=5),
        hovertemplate="%{x|%Y-%m-%d}: %{y}<extra></extra>"))
    # Signing date marker.
    fig.add_vline(x=sign_dt, line_dash="dash", line_color="#FFFFFF",
                  annotation_text="Signing announced", annotation_position="top",
                  annotation_font_color="#FFFFFF")
    # Peak point + annotation (date and value).
    fig.add_trace(go.Scatter(
        x=[peak["week_date"]], y=[peak["interest_value"]], mode="markers",
        marker=dict(size=12, color=SEA_GREEN, symbol="star"), showlegend=False,
        hoverinfo="skip"))
    fig.add_annotation(
        x=peak["week_date"], y=peak["interest_value"], ax=0, ay=-32,
        showarrow=True, arrowhead=2, arrowcolor=SEA_GREEN,
        text=f"peak {int(peak['interest_value'])} "
             f"({peak['week_date'].strftime('%b %d, %Y')})",
        font=dict(color=SEA_GREEN, size=11))
    fig.update_layout(template="plotly_dark", height=360, showlegend=False,
                      title="Weekly search interest (±8 weeks around signing)",
                      yaxis_title="Google Trends interest (0-100)", xaxis_title="",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10))
    return fig


def fig_coefficients(reg, model_names):
    """Horizontal coefficient plot with CI whiskers, coloured by p-value bucket.
    Intercept omitted for readability."""
    d = reg[reg["model"].isin(model_names) & (reg["variable"] != "Intercept")].copy()
    d["label"] = d["variable"].str.replace("np.log(contract_aav_millions)",
                                           "log(AAV)", regex=False) \
                              .str.replace("market_saturation[T.saturated]",
                                           "saturated", regex=False) \
                              .str.replace("C(archetype)[T.Skill]", "Skill",
                                           regex=False)
    d["label"] = d["label"] + "  [" + d["model"].str.replace("Model ", "") + "]"

    def bucket(p):
        return "p<0.05" if p < 0.05 else ("0.05-0.10" if p < 0.10 else "p>0.10")
    d["pb"] = d["p_value"].map(bucket)

    fig = go.Figure()
    for pb, grp in d.groupby("pb"):
        fig.add_bar(
            y=grp["label"], x=grp["coef"], orientation="h", name=pb,
            marker_color=P_COLORS[pb],
            error_x=dict(type="data", symmetric=False,
                         array=grp["ci_upper"] - grp["coef"],
                         arrayminus=grp["coef"] - grp["ci_lower"],
                         color=MUTED, thickness=1.2),
        )
    fig.add_vline(x=0, line_dash="dot", line_color=MUTED)
    fig.update_layout(template="plotly_dark", height=420, barmode="group",
                      xaxis_title="Coefficient (95% CI)",
                      legend=dict(orientation="h", y=1.1, title=""),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=30, b=10))
    return fig


# ===========================================================================
# Small UI helpers
# ===========================================================================
def inject_css(seahawks=False):
    accent = SEA_GREEN if seahawks else ACCENT
    edge = SEA_NAVY if seahawks else CARD_BG
    st.markdown(f"""
    <style>
      .block-container {{ padding-top: 2.2rem; max-width: 1100px; }}
      .card {{ background:{CARD_BG}; border-radius:12px; padding:16px 18px;
               border-left:4px solid {accent}; margin-bottom:12px; }}
      .callout {{ background:linear-gradient(90deg,{edge},{CARD_BG});
                  border-radius:12px; padding:16px 20px; border:1px solid {accent};
                  margin:10px 0; font-size:1.02rem; }}
      .pill {{ display:inline-block; padding:2px 10px; border-radius:999px;
               background:{accent}; color:#0E1117; font-weight:600;
               font-size:0.78rem; }}
      .warn {{ background:#3a2a00; border-left:4px solid #F6A623;
               border-radius:10px; padding:14px 18px; margin:8px 0; }}
      h1,h2,h3 {{ color:{TEXT}; }}
    </style>""", unsafe_allow_html=True)


def metric_tile(col, label, value, note=""):
    with col:
        st.markdown(f"<div class='card'><div style='color:{MUTED};"
                    f"font-size:0.8rem'>{label}</div>"
                    f"<div style='font-size:1.5rem;font-weight:700'>{value}</div>"
                    f"<div style='color:{MUTED};font-size:0.75rem'>{note}</div>"
                    f"</div>", unsafe_allow_html=True)


def _fmt(v, pct=False, dp=2):
    if pd.isna(v):
        return "--"
    return f"{v:+.{dp}f}%" if pct else f"{v:.{dp}f}"


# ===========================================================================
# Pages
# ===========================================================================
def page_overview(df, reg):
    st.title("NFL Player Signing ROI")
    st.caption("Search engagement ROI per $1M AAV -- 12 NFL signings, 2015-2024")

    st.plotly_chart(fig_roi_scatter(df, "search_roi", "Search ROI (abn pp / $M)"),
                    use_container_width=True)

    st.markdown(
        "<div class='callout'>📈 <b>Model A headline:</b> saturated-market "
        "signings generate <b>~4x higher buzz ROI per dollar</b> "
        "(coef = <b>+4.18</b>, p = 0.004).</div>", unsafe_allow_html=True)

    st.subheader("Secondary-market price ROI")
    st.plotly_chart(fig_roi_scatter(df, "price_roi", "Price ROI (Δ% / $M)"),
                    use_container_width=True)
    st.markdown(
        "<div class='warn'>⚠️ <b>Price ROI is an exploratory finding only</b> -- "
        "n = 10, and the saturation effect is sensitive to Deshaun Watson "
        "(p flips 0.048 → 0.192 when excluded).</div>", unsafe_allow_html=True)


def page_deep_dive(df, reg):
    st.title("Signing Deep-Dive")
    names = df.sort_values("signing_date")["player_name"].tolist()
    pick = st.selectbox("Select a signing", names)
    row = df[df["player_name"] == pick].iloc[0]

    if row["controversy_flag"]:
        st.markdown(
            "<div class='warn'>⚠️ <b>Controversy flag.</b> Deshaun Watson: "
            "$230M fully guaranteed, suspended the full 2022 season. ROI figures "
            "likely reflect initial hype before the controversy fully repriced "
            "the market -- treat as atypical.</div>", unsafe_allow_html=True)

    st.markdown(f"<span class='pill'>{row['archetype']}</span> &nbsp; "
                f"<span class='pill'>{row['market_saturation']}</span> &nbsp; "
                f"<span class='pill'>{row['signing_type']}</span>",
                unsafe_allow_html=True)
    st.write("")

    weekly = st.cache_data(load_weekly)()
    wk = weekly[weekly["player_id"] == row["player_id"]]

    left, right = st.columns([3, 2])
    with left:
        if wk.empty:
            st.info("No weekly Trends series for this signing.")
        else:
            st.plotly_chart(fig_event_study(wk, row["signing_date"]),
                            use_container_width=True)
    with right:
        st.markdown("**Revenue-lift breakdown**")
        st.markdown(
            f"<div class='card'>Abnormal search lift &nbsp; "
            f"<b>{_fmt(row['abnormal_search_lift_pp'], dp=1)} pp</b><br>"
            f"30-day price change &nbsp; <b>{_fmt(row['pct_change_30d'], pct=True, dp=1)}"
            f"</b><br>Attendance y1 (abn) &nbsp; "
            f"<b>{_fmt(row['abnormal_attendance_lift_pct_y1'], pct=True, dp=1)}</b>"
            f"</div>", unsafe_allow_html=True)

    st.subheader("ROI estimates")
    c1, c2, c3 = st.columns(3)
    metric_tile(c1, "Search ROI", _fmt(row["search_roi"], dp=3),
                "abnormal buzz pp / $M AAV")
    metric_tile(c2, "Price ROI", _fmt(row["price_roi"], dp=3),
                row["price_roi_note"] if isinstance(row["price_roi_note"], str)
                and row["price_roi_note"] else "Δ% / $M AAV")
    metric_tile(c3, "Attendance ROI", _fmt(row["attendance_roi"], dp=3),
                row["attendance_roi_note"] if isinstance(row["attendance_roi_note"],
                str) and row["attendance_roi_note"] else "saturated→NaN")


def page_seahawks(df, reg):
    inject_css(seahawks=True)
    st.title("🐦 Seahawks Module")
    st.caption("Five Seattle events -- Russell Wilson ×3, Geno Smith, DK Metcalf")

    sea = df[df["player_id"].isin(SEAHAWKS_IDS)].set_index("player_id")

    st.subheader("Wilson: extensions vs. departure")
    cols = st.columns(3)
    for col, pid, tag in zip(cols,
                             ["russell_wilson_2015", "russell_wilson_2019",
                              "russell_wilson_2022"],
                             ["2015 extension", "2019 extension", "2022 departure"]):
        r = sea.loc[pid]
        with col:
            st.markdown(
                f"<div class='card' style='border-left-color:{SEA_GREEN}'>"
                f"<div style='color:{SEA_SILVER};font-size:0.8rem'>{tag}</div>"
                f"<div style='font-size:0.95rem;margin:4px 0'>Search ROI "
                f"<b>{_fmt(r['search_roi'], dp=2)}</b></div>"
                f"<div style='font-size:0.95rem'>Price ROI "
                f"<b>{_fmt(r['price_roi'], dp=2)}</b></div>"
                f"<div style='font-size:0.95rem'>Abn search "
                f"<b>{_fmt(r['abnormal_search_lift_pp'], dp=1)} pp</b></div></div>",
                unsafe_allow_html=True)

    st.markdown(
        f"<div class='callout' style='border-color:{SEA_GREEN}'>"
        "The Wilson <b>departure</b> generated <b>+32pp abnormal search "
        "interest</b> -- comparable to his 2019 extension -- while secondary "
        "ticket prices <b>fell ~11%</b>. Fan attention is not equivalent to "
        "business value.</div>", unsafe_allow_html=True)

    st.subheader("Geno Smith")
    g = sea.loc["geno_smith_2023"]
    st.markdown(
        f"<div class='card' style='border-left-color:{SEA_GREEN}'>"
        f"Geno's buzz <b>peaked during the 2022 playoff run (Jan)</b>, not the "
        f"March contract (peak date {g['peak_date']}). The signing <b>confirmed "
        f"value rather than created it</b> -- abnormal search ROI "
        f"{_fmt(g['search_roi'], dp=2)}.</div>", unsafe_allow_html=True)


def _reg_table(reg, model_name):
    d = reg[reg["model"] == model_name][
        ["variable", "coef", "std_err", "p_value", "ci_lower", "ci_upper"]].copy()
    d["variable"] = d["variable"].str.replace("np.log(contract_aav_millions)",
                                              "log(AAV)", regex=False) \
        .str.replace("market_saturation[T.saturated]", "saturated", regex=False) \
        .str.replace("C(archetype)[T.Skill]", "Skill", regex=False)
    d["sig"] = d["p_value"].map(lambda p: "***" if p < .01 else
                                "**" if p < .05 else "*" if p < .10 else "")
    return d.rename(columns={"variable": "term"}).set_index("term")


def page_regression(df, reg):
    st.title("Regression Explorer")
    st.markdown(
        "<div class='warn'><b>n = 12.</b> Coefficients indicate direction and "
        "magnitude of association, <b>not causal effects</b>. R²=0.91 reflects "
        "overfitting risk in a small sample; treat p&lt;0.10 as "
        "<i>directionally</i> significant only.</div>", unsafe_allow_html=True)

    exclude = st.toggle("Exclude Watson (sensitivity)", value=False)
    b_model = "Model B (ex-Watson)" if exclude else "Model B"
    a_model = "Model A (ex-Watson)" if exclude else "Model A"

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{a_model} — search_roi** (OLS, HC3)")
        st.dataframe(_reg_table(reg, a_model).style.format(
            {"coef": "{:.3f}", "std_err": "{:.3f}", "p_value": "{:.3f}",
             "ci_lower": "{:.3f}", "ci_upper": "{:.3f}"}), use_container_width=True)
    with c2:
        st.markdown(f"**{b_model} — price_roi** (WLS, HC3)")
        st.dataframe(_reg_table(reg, b_model).style.format(
            {"coef": "{:.3f}", "std_err": "{:.3f}", "p_value": "{:.3f}",
             "ci_lower": "{:.3f}", "ci_upper": "{:.3f}"}), use_container_width=True)

    st.subheader("Coefficients (95% CI, by p-value)")
    st.plotly_chart(fig_coefficients(reg, [a_model, b_model]),
                    use_container_width=True)


def page_methodology(df, reg):
    st.title("Methodology")

    st.subheader("Pipeline")
    stages = ["Contract\n(OverTheCap)", "Attendance\n(ESPN)",
              "Search\n(Google Trends)", "Price\n(TicketIQ)",
              "→ master_merge", "→ event_study", "→ roi_calculator",
              "→ regression"]
    cols = st.columns(len(stages))
    for col, s in zip(cols, stages):
        col.markdown(f"<div class='card' style='text-align:center;font-size:0.72rem;"
                     f"min-height:64px'>{s}</div>", unsafe_allow_html=True)

    st.subheader("Data sources & coverage")
    cov = pd.DataFrame({
        "Source": ["Contract", "Attendance", "Search interest", "Ticket price"],
        "Coverage": ["12/12", "12/12", "12/12", "10/12 (2 COVID-excluded)"],
        "Confidence": ["manual / OTC", "ESPN + capacity est.",
                       "pytrends (live)", "TicketIQ; conf. tiers low/med/high"],
    })
    st.dataframe(cov, use_container_width=True, hide_index=True)

    st.subheader("COVID exclusion logic")
    st.markdown(
        "- Observations in COVID-distorted seasons are **flagged and excluded, "
        "never imputed**.\n"
        "- Attendance: comparison-level exclusion (e.g. Wilson-2019's 2018→2019 "
        "stays; its →2020 drops).\n"
        "- Price: a `covid_distorted` cell or inherited `covid_post_1` flag "
        "excludes the row (Mahomes-2020, OBJ-2021).")

    st.subheader("Market saturation")
    st.markdown(
        "Teams whose home attendance runs **>95% of capacity** regardless of "
        "roster moves are *saturated* — attendance can't respond, so price is "
        "the signal. Classified saturated: **Seattle, Kansas City, Dallas**.")

    with open(os.path.join(PROC, "roi_estimates.csv"), "rb") as f:
        st.download_button("⬇️ Download roi_estimates.csv", f.read(),
                           "roi_estimates.csv", "text/csv")


# ===========================================================================
# Main (runs only under Streamlit, where __name__ == '__main__')
# ===========================================================================
PAGES = {
    "League Overview": page_overview,
    "Signing Deep-Dive": page_deep_dive,
    "Seahawks Module": page_seahawks,
    "Regression Explorer": page_regression,
    "Methodology": page_methodology,
}


def main():
    st.set_page_config(page_title="NFL Signing ROI", page_icon="🏈",
                       layout="wide", initial_sidebar_state="expanded")
    df, reg = st.cache_data(load_data)()
    inject_css(seahawks=False)
    st.sidebar.title("🏈 Signing ROI")
    st.sidebar.caption("12 NFL signings · 2015–2024")
    choice = st.sidebar.radio("Page", list(PAGES.keys()))
    st.sidebar.markdown("---")
    st.sidebar.caption("Presentation layer only — models are fit in "
                       "regression.py, not here.")
    PAGES[choice](df, reg)


if __name__ == "__main__":
    main()
