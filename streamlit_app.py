"""Streamlit app: market-wide valuation explorer for all tickers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta
import inspect

from utilities.sidebar_style import apply_sidebar_style
from utilities.style_utils import apply_google_font
from utilities.data_access import load_valuation_universe
from utilities.valuation_analysis import (
    calculate_cdf,
    calculate_historical_stats,
    generate_valuation_histogram,
    get_metric_column,
    get_valuation_status,
    remove_outliers_iqr,
)

# ---------------------------------------------------------------------------
# Page configuration and shared styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Market Valuation Explorer",
    page_icon="📈",
    layout="wide",
)

apply_google_font()
apply_sidebar_style()

VALUATION_COLUMNS = ["PE", "PB", "PS", "EV_EBITDA"]
DEFAULT_LOOKBACK_YEARS = 5
MARKET_CAP_BILLION_TO_DATA_SCALE = 1
PLOTLY_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
DATAFRAME_SUPPORTS_WIDTH = "width" in inspect.signature(st.dataframe).parameters


def render_plotly_chart(fig: go.Figure, *, config: dict | None = None):
    """Render Plotly charts with forward/backward compatibility for sizing."""

    kwargs: dict = {}
    if config is not None:
        kwargs["config"] = config

    if PLOTLY_SUPPORTS_WIDTH:
        return st.plotly_chart(fig, width="stretch", **kwargs)
    return st.plotly_chart(fig, use_container_width=True, **kwargs)


@st.cache_data(ttl=1800)
def load_market_data(
    years: int = DEFAULT_LOOKBACK_YEARS,
    *,
    min_market_cap: float | None = None,
) -> pd.DataFrame:
    """Load and pre-process valuation data for the full ticker universe."""

    df = load_valuation_universe(years=years, min_market_cap=min_market_cap)
    if df.empty:
        return df

    df = df.sort_values(["TICKER", "TRADE_DATE"]).reset_index(drop=True)

    if min_market_cap is not None:
        latest_caps = df.groupby("TICKER").tail(1)

        cap_source = "CUR_MKT_CAP" if "CUR_MKT_CAP" in latest_caps.columns else "MKT_CAP"
        if cap_source not in latest_caps.columns:
            return df

        eligible_tickers = latest_caps[latest_caps[cap_source] >= min_market_cap]["TICKER"].unique()
        df = df[df["TICKER"].isin(eligible_tickers)]

    return df


def build_group_aggregates(
    df: pd.DataFrame,
    *,
    metric_cols: list[str],
    group_col: str,
) -> pd.DataFrame:
    """Create median aggregates per grouping value across time."""

    if group_col not in df.columns:
        return pd.DataFrame(columns=df.columns)

    aggregate = (
        df.groupby(["TRADE_DATE", group_col])[metric_cols]
        .median()
        .reset_index()
    )

    if aggregate.empty:
        return aggregate

    aggregate["TICKER"] = aggregate[group_col]
    aggregate["GroupValue"] = aggregate[group_col]
    aggregate["IsAggregate"] = True

    for col in df.columns:
        if col not in aggregate.columns:
            aggregate[col] = np.nan

    return aggregate


def prepare_summary_table(
    df: pd.DataFrame,
    *,
    metric_col: str,
    group_col: str,
    top_ticker_set: set[str] | None,
) -> pd.DataFrame:
    """Create the summary statistics table for the selected cohort."""

    rows: list[dict] = []

    for ticker, ticker_df in df.groupby("TICKER"):
        series = ticker_df[metric_col].dropna()
        if len(series) < 20:
            continue

        hist_stats = calculate_historical_stats(df, ticker, metric_col)
        if not hist_stats:
            continue

        cdf_val = calculate_cdf(df, ticker, metric_col)
        status, _ = get_valuation_status(hist_stats.get("z_score"))

        group_value = ticker_df[group_col].iloc[0] if group_col in ticker_df.columns else "Unclassified"
        is_aggregate = bool(ticker_df.get("IsAggregate", pd.Series([False])).iloc[0])

        if not is_aggregate and top_ticker_set is not None and ticker not in top_ticker_set:
            continue

        rows.append(
            {
                "Ticker": ticker,
                "Group": group_value,
                "Current": hist_stats.get("current"),
                "Mean": hist_stats.get("mean"),
                "CDF (%)": cdf_val,
                "Z-Score": hist_stats.get("z_score"),
                "Status": status,
                "IsAggregate": is_aggregate,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return table

    # Determine ordering: aggregates per group first (sorted by current), followed by member tickers
    aggregates = table[table["IsAggregate"]].copy()
    if not aggregates.empty:
        aggregates["SortValue"] = aggregates["Current"].fillna(-np.inf)
        ordered_groups = aggregates.sort_values("SortValue", ascending=False)["Group"].tolist()
    else:
        ordered_groups = []

    # Append any groups without aggregates to maintain coverage
    for group in sorted(table["Group"].unique()):
        if group not in ordered_groups:
            ordered_groups.append(group)

    ordered_rows: list[pd.DataFrame] = []
    for group in ordered_groups:
        group_slice = table[table["Group"] == group]
        if group_slice.empty:
            continue

        agg_row = group_slice[group_slice["IsAggregate"]]
        if not agg_row.empty:
            ordered_rows.append(agg_row)

        member_rows = (
            group_slice[~group_slice["IsAggregate"]]
            .sort_values("Current", ascending=False, na_position="last")
        )
        if not member_rows.empty:
            ordered_rows.append(member_rows)

    if ordered_rows:
        table = pd.concat(ordered_rows, ignore_index=True)
    else:
        table = table.sort_values("Current", ascending=False, na_position="last").reset_index(drop=True)

    return table


# ---------------------------------------------------------------------------
# Sidebar controls and dataset load
# ---------------------------------------------------------------------------

with st.sidebar:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with st.sidebar:
    st.markdown("### Settings")
    metric_choice = st.selectbox(
        "Valuation Metric",
        ["P/B", "P/E", "P/S", "EV/EBITDA"],
        index=0,
    )
    metric_column = get_metric_column(metric_choice)

    min_market_cap_bn = st.number_input(
        "Minimum market cap (bn)",
        min_value=0.0,
        value=4000.0,
        step=50.0,
        help="All calculations will use tickers with market cap above this threshold.",
    )

min_market_cap_value: float | None = None
if min_market_cap_bn and min_market_cap_bn > 0:
    min_market_cap_value = float(min_market_cap_bn * MARKET_CAP_BILLION_TO_DATA_SCALE)

market_df = load_market_data(min_market_cap=min_market_cap_value)

if market_df.empty:
    threshold_msg = (
        f"No tickers meet the current market cap floor of {min_market_cap_bn:,.0f} bn."
        if min_market_cap_value is not None
        else "Valuation data not available."
    )
    st.error(f"{threshold_msg} Please adjust the settings or refresh the data pipeline.")
    st.stop()

if "Industry_L2" in market_df.columns:
    grouping_column = "Industry_L2"
    grouping_label = "Industry (Level 2)"
elif "Sector" in market_df.columns:
    grouping_column = "Sector"
    grouping_label = "Sector"
else:
    grouping_column = "Sector"
    grouping_label = "Sector"

with st.sidebar:
    available_groups = ["All Market"] + sorted(
        g for g in market_df[grouping_column].dropna().unique().tolist() if g
    )
    selected_group = st.selectbox(
        f"Focus {grouping_label}",
        available_groups,
    )
    st.caption(f"Grouping fixed at {grouping_label} for performance.")

    only_vn30 = st.checkbox("Only VN30 constituents", value=False)
    max_entities = st.slider(
        "Max entities in distribution chart",
        min_value=10,
        max_value=80,
        value=40,
        step=5,
    )

    axis_scale = st.radio(
        "Distribution Scale",
        ("Linear", "Log"),
        index=0,
        horizontal=True,
    )

# ---------------------------------------------------------------------------
# Data filtering & derived dataset
# ---------------------------------------------------------------------------

filtered_df = market_df.copy()
if only_vn30 and "VNI_Flag" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["VNI_Flag"].str.upper() == "Y"]

if "MKT_CAP" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["MKT_CAP"].notna()]
else:
    st.warning("Market cap data unavailable; minimum threshold ignored.")

if filtered_df.empty:
    st.warning("No data matches the selected filters.")
    st.stop()

filtered_df = filtered_df.assign(IsAggregate=False)
filtered_df["GroupValue"] = filtered_df[grouping_column]

focus_label = "All Market" if selected_group == "All Market" else selected_group
if selected_group != "All Market":
    filtered_df = filtered_df[filtered_df[grouping_column] == selected_group]

if filtered_df.empty:
    st.warning("No tickers available for the selected grouping.")
    st.stop()

included_company_count = filtered_df["TICKER"].nunique()

cap_source = "CUR_MKT_CAP" if "CUR_MKT_CAP" in filtered_df.columns else "MKT_CAP"
latest_caps_bn: float | None = None
median_caps_bn: float | None = None

if cap_source in filtered_df.columns:
    latest_caps = (
        filtered_df[["TICKER", "TRADE_DATE", cap_source]]
        .dropna(subset=[cap_source])
        .sort_values("TRADE_DATE")
        .groupby("TICKER")
        .tail(1)
    )

    if not latest_caps.empty:
        latest_caps_bn = latest_caps[cap_source].min() / MARKET_CAP_BILLION_TO_DATA_SCALE
        median_caps_bn = latest_caps[cap_source].median() / MARKET_CAP_BILLION_TO_DATA_SCALE

metric_columns_present = [col for col in VALUATION_COLUMNS if col in filtered_df.columns]
if metric_column not in metric_columns_present:
    st.error(f"Metric column '{metric_column}' not found in dataset.")
    st.stop()

group_aggregates = build_group_aggregates(
    filtered_df,
    metric_cols=metric_columns_present,
    group_col=grouping_column,
)

if selected_group == "All Market":
    dist_df = group_aggregates.copy()
else:
    dist_df = pd.concat(
        [
            group_aggregates[group_aggregates[grouping_column] == selected_group],
            filtered_df,
        ],
        ignore_index=True,
        sort=False,
    )

focus_df = pd.concat(
    [filtered_df, group_aggregates],
    ignore_index=True,
    sort=False,
)

latest_date = focus_df["TRADE_DATE"].max()
st.title("Market Valuation Explorer")
label_suffix = grouping_label if selected_group == "All Market" else selected_group
st.markdown("Full-universe valuation analytics with sector-level context and drilldowns.")
st.caption(f"Latest trading day: {latest_date.strftime('%Y-%m-%d')}")
st.caption(
    f"Companies included in calculations: {included_company_count:,} "
    f"(market cap ≥ {min_market_cap_bn:,.0f} bn)"
)
if latest_caps_bn is not None:
    st.caption(
        f"Smallest eligible market cap: {latest_caps_bn:,.1f} bn | "
        f"Median eligible market cap: {median_caps_bn:,.1f} bn"
    )

# ---------------------------------------------------------------------------
# Distribution view
# ---------------------------------------------------------------------------

if dist_df.empty:
    st.info("No aggregates available for the selected view.")
    st.stop()

log_allowed = axis_scale == "Log" and (dist_df[metric_column] > 0).any()
if axis_scale == "Log" and not log_allowed:
    st.info("Log scale unavailable because the selection includes zero or negative values. Using linear scale.")
    axis_scale = "Linear"

latest_idx = dist_df.groupby("TICKER")["TRADE_DATE"].idxmax()
latest_snapshot = dist_df.loc[latest_idx, ["TICKER", metric_column]].dropna(subset=[metric_column])
ordered = latest_snapshot.sort_values(metric_column, ascending=False)["TICKER"].tolist()

if focus_label in ordered:
    ordered.remove(focus_label)
ordered = [focus_label] + ordered
ordered = ordered[:max_entities]

st.subheader("Valuation Distribution")
fig_distribution = go.Figure()
valid_tickers: list[str] = []

for ticker in ordered:
    ticker_series = dist_df.loc[dist_df["TICKER"] == ticker, metric_column].dropna()
    if len(ticker_series) < 15:
        continue

    clean_series = remove_outliers_iqr(ticker_series, multiplier=2.5)
    if len(clean_series) < 10:
        clean_series = ticker_series

    # Avoid zero/negative values when log scale is selected.
    if axis_scale == "Log":
        clean_series = clean_series[clean_series > 0]
        ticker_series = ticker_series[ticker_series > 0]
        if len(clean_series) < 15 or len(ticker_series) < 15:
            continue

    p5 = clean_series.quantile(0.05)
    p25 = clean_series.quantile(0.25)
    p50 = clean_series.quantile(0.50)
    p75 = clean_series.quantile(0.75)
    p95 = clean_series.quantile(0.95)

    current_value = ticker_series.iloc[-1]
    percentile = (clean_series <= current_value).mean() * 100 if len(clean_series) else None

    fig_distribution.add_trace(
        go.Candlestick(
            x=[ticker],
            open=[round(p25, 2) if p25 is not None else None],
            high=[round(p95, 2) if p95 is not None else None],
            low=[round(p5, 2) if p5 is not None else None],
            close=[round(p75, 2) if p75 is not None else None],
            name=ticker,
            showlegend=False,
            increasing_line_color="#D3D3D3",
            decreasing_line_color="#D3D3D3",
            hovertext=f"{ticker}<br>Median: {p50:.2f}",
        )
    )

    fig_distribution.add_trace(
        go.Scatter(
            x=[ticker],
            y=[current_value],
            mode="markers",
            marker=dict(
                size=9,
                color="#478B81" if ticker != focus_label else "#1F4E5F",
                symbol="circle",
            ),
            name=f"{ticker} current",
            showlegend=False,
            hovertemplate=(
                f"<b>{ticker}</b><br>Current: {current_value:.2f}<br>"
                f"Percentile: {percentile:.1f}%<extra></extra>"
            ),
        )
    )

    valid_tickers.append(ticker)

yaxis_config = dict(fixedrange=True)
if axis_scale == "Log":
    yaxis_config["type"] = "log"

fig_distribution.update_layout(
    title=f"{metric_choice} distribution for {focus_label}",
    xaxis_title="Ticker / Aggregate",
    yaxis_title=f"{metric_choice}",
    height=520,
    hovermode="x unified",
    xaxis=dict(
        categoryorder="array",
        categoryarray=valid_tickers,
        rangeslider=dict(visible=False),
        fixedrange=True,
    ),
    yaxis=yaxis_config,
    dragmode=False,
)

render_plotly_chart(fig_distribution, config={"displayModeBar": False})

# ---------------------------------------------------------------------------
# Drilldown charts
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Ticker Drilldown")
st.caption("Analyse historical trends and distribution percentiles for a selected entity.")

col_ticker, col_range, _ = st.columns([2, 2, 6])

available_tickers = sorted(filtered_df["TICKER"].unique().tolist())
aggregate_options = sorted(group_aggregates[grouping_column].dropna().unique().tolist())

with col_ticker:
    entity_choices = ["Ticker"] + ([grouping_label] if aggregate_options else [])
    entity_type = st.radio(
        "Entity Type",
        entity_choices,
        index=0,
        horizontal=True,
    )

    if entity_type == "Ticker":
        default_index = available_tickers.index(focus_label) if focus_label in available_tickers else 0
        selected_ticker = st.selectbox(
            "Select Ticker",
            available_tickers,
            index=default_index if available_tickers else 0,
        )
    else:
        default_index = aggregate_options.index(focus_label) if focus_label in aggregate_options else 0
        selected_ticker = st.selectbox(
            f"Select {grouping_label}",
            aggregate_options,
            index=default_index if aggregate_options else 0,
        )

with col_range:
    period_choice = st.selectbox(
        "Time horizon",
        ["1 Year", "2 Years", "3 Years", "5 Years", "All Time"],
        index=2,
    )

if period_choice == "1 Year":
    start_date = latest_date - timedelta(days=365)
elif period_choice == "2 Years":
    start_date = latest_date - timedelta(days=730)
elif period_choice == "3 Years":
    start_date = latest_date - timedelta(days=1095)
elif period_choice == "5 Years":
    start_date = latest_date - timedelta(days=1825)
else:
    start_date = focus_df["TRADE_DATE"].min()

col_trend, col_hist = st.columns([6, 6])

with col_trend:
    ticker_history = focus_df[
        (focus_df["TICKER"] == selected_ticker)
        & (focus_df["TRADE_DATE"] >= start_date)
    ].sort_values("TRADE_DATE")

    if ticker_history.empty:
        st.info("Insufficient history for the selected ticker.")
    else:
        hist_stats = calculate_historical_stats(focus_df, selected_ticker, metric_column)
        if hist_stats:
            fig_trend = go.Figure()
            fig_trend.add_trace(
                go.Scatter(
                    x=ticker_history["TRADE_DATE"],
                    y=ticker_history[metric_column],
                    mode="lines",
                    name=f"{metric_choice}",
                    line=dict(color="#478B81", width=2),
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=[ticker_history["TRADE_DATE"].min(), ticker_history["TRADE_DATE"].max()],
                    y=[hist_stats["mean"], hist_stats["mean"]],
                    mode="lines",
                    name="Mean",
                    line=dict(color="black", dash="solid", width=2),
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=[ticker_history["TRADE_DATE"].min(), ticker_history["TRADE_DATE"].max()],
                    y=[hist_stats["upper_1sd"], hist_stats["upper_1sd"]],
                    mode="lines",
                    name="+1 SD",
                    line=dict(color="red", dash="dash", width=1),
                )
            )
            fig_trend.add_trace(
                go.Scatter(
                    x=[ticker_history["TRADE_DATE"].min(), ticker_history["TRADE_DATE"].max()],
                    y=[hist_stats["lower_1sd"], hist_stats["lower_1sd"]],
                    mode="lines",
                    name="-1 SD",
                    line=dict(color="green", dash="dash", width=1),
                )
            )
            if hist_stats.get("current") is not None:
                fig_trend.add_trace(
                    go.Scatter(
                        x=[ticker_history["TRADE_DATE"].max()],
                        y=[hist_stats["current"]],
                        mode="markers",
                        name="Current",
                        marker=dict(size=8, color="grey"),
                    )
                )

            fig_trend.update_layout(
                title=f"{selected_ticker} — {metric_choice} trend",
                xaxis_title="Date",
                yaxis_title=f"{metric_choice}",
                height=420,
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )

            render_plotly_chart(fig_trend)

            stat_left, stat_right = st.columns(2)
            with stat_left:
                st.metric(
                    "Current",
                    f"{hist_stats['current']:.2f}" if hist_stats['current'] is not None else "N/A",
                )
                st.metric("Mean", f"{hist_stats['mean']:.2f}" if hist_stats['mean'] is not None else "N/A")
            with stat_right:
                st.metric("Std Dev", f"{hist_stats['std']:.2f}" if hist_stats['std'] is not None else "N/A")
                z_score = hist_stats.get("z_score")
                if z_score is not None:
                    status, _ = get_valuation_status(z_score)
                    st.metric("Z-Score", f"{z_score:.2f}", delta=status)
        else:
            st.info("Not enough observations to compute statistics.")

with col_hist:
    histogram_payload = generate_valuation_histogram(focus_df, selected_ticker, metric_column)
    if histogram_payload is None:
        st.info("Not enough data to compute a distribution histogram for this selection.")
    else:
        bar_colors = ["#D9D9D9"] * len(histogram_payload["counts"])
        if histogram_payload["current_bin_idx"] is not None:
            bar_colors[histogram_payload["current_bin_idx"]] = "#478B81"

        fig_hist = go.Figure(
            data=[
                go.Bar(
                    x=histogram_payload["bin_labels"],
                    y=histogram_payload["counts"],
                    marker_color=bar_colors,
                    text=histogram_payload["counts"],
                    textposition="auto",
                    showlegend=False,
                )
            ]
        )
        fig_hist.update_layout(
            title=dict(
                text=(
                    f"{selected_ticker} — {metric_choice} distribution"
                    f"<br><sub>Current: {histogram_payload['current_value']:.2f}"
                    f" | Percentile: {histogram_payload['percentile']:.1f}%</sub>"
                ),
                x=0.5,
                xanchor="center",
            ),
            xaxis_title=f"{metric_choice} range",
            yaxis_title="Observations",
            height=420,
            bargap=0.12,
            hovermode="x",
        )

        render_plotly_chart(fig_hist)

        dist_left, dist_right = st.columns(2)
        with dist_left:
            st.metric("Current", f"{histogram_payload['current_value']:.2f}")
            st.metric("Median", f"{histogram_payload['median']:.2f}")
        with dist_right:
            st.metric("Percentile (CDF)", f"{histogram_payload['percentile']:.1f}%")
            st.metric("Data points", histogram_payload["n_total"])

# ---------------------------------------------------------------------------
# Summary table & roll-up metrics
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Valuation Summary Table")

summary_df = prepare_summary_table(
    focus_df,
    metric_col=metric_column,
    group_col=grouping_column,
    top_ticker_set=None,
)

if summary_df.empty:
    st.info("Insufficient data to build the summary table.")
else:
    table_df = summary_df.copy()

    status_palette = {
        "Very Cheap": "#90EE90",
        "Cheap": "#B8E6B8",
        "Fair": "#FFFFCC",
        "Expensive": "#FFD4A3",
        "Very Expensive": "#FFB3B3",
    }

    display_columns = ["Ticker", "Group", "Current", "Mean", "CDF (%)", "Z-Score", "Status"]
    display_df = table_df[display_columns].copy()

    number_formats = {
        "Current": "{:.2f}",
        "Mean": "{:.2f}",
        "CDF (%)": "{:.1f}",
        "Z-Score": "{:.2f}",
    }

    def style_table(data: pd.DataFrame) -> pd.DataFrame:
        styles = pd.DataFrame("", index=data.index, columns=data.columns)
        for idx in data.index:
            if table_df.loc[idx, "IsAggregate"]:
                styles.loc[idx, :] = "background-color: #E8E8E8"
            status_color = status_palette.get(data.loc[idx, "Status"], "")
            if status_color:
                styles.loc[idx, "Status"] = f"background-color: {status_color}"
        return styles

    styled_table = (
        display_df.style
        .format(number_formats, na_rep="N/A")
        .apply(style_table, axis=None)
        .hide(axis="index")
    )

    if DATAFRAME_SUPPORTS_WIDTH:
        st.dataframe(styled_table, width="stretch")
    else:
        st.dataframe(styled_table, use_container_width=True)

    st.markdown("---")
    col_under, col_fair, col_over = st.columns(3)
    with col_under:
        st.metric(
            "Undervalued",
            int((summary_df["Status"].isin(["Very Cheap", "Cheap"])).sum()),
        )
    with col_fair:
        st.metric("Fairly valued", int((summary_df["Status"] == "Fair").sum()))
    with col_over:
        st.metric(
            "Overvalued",
            int((summary_df["Status"].isin(["Expensive", "Very Expensive"])).sum()),
        )
