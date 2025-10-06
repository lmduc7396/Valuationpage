"""
Valuation Analysis Page
Provides comprehensive valuation metrics analysis for Vietnamese banking sector
"""

import streamlit as st
import inspect

# Page configuration
st.set_page_config(
    page_title="Valuation Analysis",
    page_icon="",
    layout="wide"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sys
from datetime import timedelta

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import and apply Google Fonts
from utilities.style_utils import apply_google_font
from utilities.sidebar_style import apply_sidebar_style
apply_google_font()

# Apply consistent sidebar styling
apply_sidebar_style()

# Import utilities
from utilities.valuation_analysis import (
    get_metric_column,
    calculate_historical_stats,
    prepare_statistics_table,
    get_sector_and_components,
    get_valuation_status,
    generate_valuation_histogram
)
from utilities.data_access import load_valuation_banking


@st.cache_data(ttl=1800)
def load_valuation_data() -> pd.DataFrame:
    df = load_valuation_banking()
    if df.empty:
        return df
    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])
    return df

# Title and description
st.title("Banking Sector Valuation Analysis")
st.markdown("Comprehensive valuation metrics analysis with distribution charts, historical trends, and statistical measures")

# Manual refresh control
with st.sidebar:
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Load data
df = load_valuation_data()

if df.empty:
    st.error("Valuation data not found. Please run the valuation preparation pipeline.")
    st.stop()

# Build sector and sub-sector aggregate series (median of component banks by day)
try:
    base = df.copy()
    # Only use individual banks to form aggregates (3-char tickers)
    base_banks = base[base['TICKER'].astype(str).str.len() == 3]
    # Ensure needed columns
    needed_cols = {'TRADE_DATE', 'TICKER', 'Type', 'PE', 'PB'}
    if needed_cols.issubset(set(base_banks.columns)) and not base_banks.empty:
        # Sub-sector medians per date
        sub_agg = (
            base_banks
            .groupby(['TRADE_DATE', 'Type'])[['PE', 'PB']]
            .median()
            .reset_index()
        )
        # Create rows where TICKER is the Type label (e.g., 'SOCB', 'Private_1', ...)
        sub_agg['TICKER'] = sub_agg['Type']

        # Overall sector median per date
        sector_agg = (
            base_banks
            .groupby(['TRADE_DATE'])[['PE', 'PB']]
            .median()
            .reset_index()
        )
        sector_agg['TICKER'] = 'Sector'
        sector_agg['Type'] = 'Sector'

        # Combine back with original
        agg_df = pd.concat([base, sub_agg, sector_agg], ignore_index=True, sort=False)
        # Drop duplicates if any
        agg_df = agg_df.drop_duplicates(subset=['TICKER', 'TRADE_DATE'])
        df = agg_df
except Exception as e:
    st.warning(f"Could not compute sector aggregates: {e}")

# Sidebar metric selection
with st.sidebar:
    st.markdown("### Settings")
    metric_type = st.radio(
        "Valuation Metric:",
        ["P/E", "P/B"],
        index=1,  # Default to P/B (index 1)
        help="This selection will update all charts and tables"
    )
    metric_col = get_metric_column(metric_type)

# Get latest date in data
latest_date = df['TRADE_DATE'].max()

# Show latest data in smaller text
st.caption(f"Latest data: {latest_date.strftime('%Y-%m-%d')}")

# Chart 1: Valuation Distribution Candle Chart
st.subheader("Valuation Distribution by Bank")

# Sector selection above the chart
sector_options = ["Sector", "SOCB", "Private_1", "Private_2", "Private_3"]
selected_sector = st.selectbox(
    "Select Sector:",
    sector_options,
    help="Shows selected sector plus all component banks"
)

# Get tickers to display
display_tickers = get_sector_and_components(df, selected_sector)

# Create candle chart
fig_candle = go.Figure()

# Prepare data for each ticker
valid_tickers = []
for ticker in display_tickers:
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()
    
    if len(ticker_data) < 20:  # Skip if insufficient data
        continue
    
    valid_tickers.append(ticker)
    
    # Calculate percentiles with smart outlier handling
    # First, identify extreme outliers (e.g., P/E > 100 when median is 20)
    median_val = ticker_data.median()
    
    # Only exclude extreme outliers (values more than 5x the median)
    if metric_type == "P/E":
        # For P/E, be more aggressive with outlier removal
        upper_limit = min(100, median_val * 5) if median_val > 0 else 100
        clean_data = ticker_data[ticker_data <= upper_limit]
    else:
        # For P/B, be more lenient
        upper_limit = median_val * 4 if median_val > 0 else 10
        clean_data = ticker_data[ticker_data <= upper_limit]
    
    # Ensure we still have enough data
    if len(clean_data) < 20:
        clean_data = ticker_data  # Use original if too much was filtered
    
    # Calculate percentiles for candle
    p5 = clean_data.quantile(0.05)
    p25 = clean_data.quantile(0.25)
    p50 = clean_data.quantile(0.50)
    p75 = clean_data.quantile(0.75)
    p95 = clean_data.quantile(0.95)
    
    # Get current value
    current_val = ticker_data.iloc[-1] if len(ticker_data) > 0 else None
    
    # Add candlestick with light grey color
    fig_candle.add_trace(go.Candlestick(
        x=[ticker],
        open=[round(p25, 2)],
        high=[round(p95, 2)],  # Use p95 for upper wick
        low=[round(p5, 2)],    # Use p5 for lower wick
        close=[round(p75, 2)],
        name=ticker,
        showlegend=False,
        increasing_line_color='lightgrey',
        decreasing_line_color='lightgrey',
        hovertext=f"{ticker}<br>Median: {p50:.2f}"
    ))
    
    # Add current value as scatter point with smaller size and custom color
    if current_val and not pd.isna(current_val):
        # Calculate percentile
        percentile = np.sum(clean_data <= current_val) / len(clean_data) * 100
        
        fig_candle.add_trace(go.Scatter(
            x=[ticker],
            y=[current_val],
            mode='markers',
            marker=dict(size=8, color='#478B81', symbol='circle'),
            name=f"{ticker} Current",
            showlegend=False,
            hovertemplate=(
                f"<b>{ticker}</b><br>" +
                f"Current: {current_val:.2f}<br>" +
                f"Percentile: {percentile:.1f}%<br>" +
                f"Median: {p50:.2f}<br>" +
                "<extra></extra>"
            )
        ))

# Update layout
fig_candle.update_layout(
    title=f"{metric_type} Distribution - {selected_sector}",
    xaxis_title="Bank",
    yaxis_title=f"{metric_type} Ratio",
    height=500,
    hovermode='x unified',
    xaxis=dict(
        categoryorder='array',
        categoryarray=valid_tickers,  # Maintain order
        rangeslider=dict(visible=False),  # Disable range slider
        fixedrange=True  # Disable zoom and pan
    ),
    yaxis=dict(
        fixedrange=True  # Disable zoom and pan on y-axis too
    ),
    dragmode=False  # Disable all drag interactions
)

render_plotly_chart(fig_candle, config={'displayModeBar': False, 'staticPlot': False})

# Combined Ticker Selection for Charts 2 and 3
st.markdown("---")
st.subheader("Individual Bank/Sector Analysis")
st.caption("Select a ticker to view both historical trend and distribution analysis")

# Common ticker and date range selection
col_select1, col_select2, col_select3 = st.columns([2, 2, 6])

with col_select1:
    # Ticker selection for both charts
    all_tickers = sorted(df['TICKER'].unique())
    selected_ticker = st.selectbox(
        "Select Bank/Sector:",
        all_tickers,
        index=all_tickers.index('Sector') if 'Sector' in all_tickers else 0,
        key="common_ticker_select"
    )

with col_select2:
    # Date range selection for time series
    date_range = st.selectbox(
        "Time Period:",
        ["1 Year", "2 Years", "3 Years", "5 Years", "All Time"],
        index=2  # Default to 3 years
    )
    
    # Calculate date filter
    if date_range == "1 Year":
        start_date = latest_date - timedelta(days=365)
    elif date_range == "2 Years":
        start_date = latest_date - timedelta(days=730)
    elif date_range == "3 Years":
        start_date = latest_date - timedelta(days=1095)
    elif date_range == "5 Years":
        start_date = latest_date - timedelta(days=1825)
    else:
        start_date = df['TRADE_DATE'].min()

# Display charts side by side
col_chart1, col_chart2 = st.columns([6, 6])

# Chart 2: Historical Valuation Time Series
with col_chart1:
    # Filter data for selected ticker and date range
    ticker_df = df[(df['TICKER'] == selected_ticker) & (df['TRADE_DATE'] >= start_date)].copy()
    ticker_df = ticker_df.sort_values('TRADE_DATE')
    
    if len(ticker_df) > 0:
        # Calculate statistics
        hist_stats = calculate_historical_stats(df, selected_ticker, metric_col)
        
        if hist_stats:
            # Create figure
            fig_ts = go.Figure()
            
            # Add main valuation line with custom color
            fig_ts.add_trace(go.Scatter(
                x=ticker_df['TRADE_DATE'],
                y=ticker_df[metric_col],
                mode='lines',
                name=f'{metric_type} Ratio',
                line=dict(color='#478B81', width=2)
            ))
            
            # Add mean line
            fig_ts.add_trace(go.Scatter(
                x=[ticker_df['TRADE_DATE'].min(), ticker_df['TRADE_DATE'].max()],
                y=[hist_stats['mean'], hist_stats['mean']],
                mode='lines',
                name='Mean',
                line=dict(color='black', width=2, dash='solid')
            ))
            
            # Add +1 SD line
            fig_ts.add_trace(go.Scatter(
                x=[ticker_df['TRADE_DATE'].min(), ticker_df['TRADE_DATE'].max()],
                y=[hist_stats['upper_1sd'], hist_stats['upper_1sd']],
                mode='lines',
                name='+1 SD',
                line=dict(color='red', width=1, dash='dash')
            ))
            
            # Add -1 SD line
            fig_ts.add_trace(go.Scatter(
                x=[ticker_df['TRADE_DATE'].min(), ticker_df['TRADE_DATE'].max()],
                y=[hist_stats['lower_1sd'], hist_stats['lower_1sd']],
                mode='lines',
                name='-1 SD',
                line=dict(color='green', width=1, dash='dash')
            ))
            
            # Add current value marker as small grey dot
            if hist_stats['current'] is not None:
                fig_ts.add_trace(go.Scatter(
                    x=[ticker_df['TRADE_DATE'].max()],
                    y=[hist_stats['current']],
                    mode='markers',
                    name='Current',
                    marker=dict(size=8, color='grey', symbol='circle')
                ))
            
            # Update layout
            fig_ts.update_layout(
                title=f"{selected_ticker} - {metric_type} Trend",
                xaxis_title="Date",
                yaxis_title=f"{metric_type} Ratio",
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            render_plotly_chart(fig_ts)
            
            # Show statistics below the chart
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Current", f"{hist_stats['current']:.2f}" if hist_stats['current'] else "N/A")
                st.metric("Mean", f"{hist_stats['mean']:.2f}")
            with col_stat2:
                st.metric("Std Dev", f"{hist_stats['std']:.2f}")
                z_score = hist_stats.get('z_score')
                if z_score is not None:
                    status, color = get_valuation_status(z_score)
                    st.metric("Z-Score", f"{z_score:.2f}", delta=status)
    else:
        st.warning(f"No data available for {selected_ticker}")

# Chart 3: Valuation Distribution Histogram
with col_chart2:
    # Generate histogram data for the same selected ticker
    hist_data = generate_valuation_histogram(df, selected_ticker, metric_col)
    
    if hist_data:
        # Create histogram figure
        fig_hist = go.Figure()
        
        # Create bar colors - highlight current bin
        bar_colors = ['#E0E0E0'] * len(hist_data['counts'])
        if hist_data['current_bin_idx'] is not None:
            bar_colors[hist_data['current_bin_idx']] = '#478B81'
        
        # Add bars
        fig_hist.add_trace(go.Bar(
            x=hist_data['bin_labels'],
            y=hist_data['counts'],
            marker_color=bar_colors,
            text=hist_data['counts'],
            textposition='auto',
            showlegend=False,
            hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
        ))
        
        # Update layout
        fig_hist.update_layout(
            title=dict(
                text=f"{selected_ticker} - {metric_type} Distribution<br>" +
                     f"<sub>Current: {hist_data['current_value']:.2f} (CDF: {hist_data['percentile']:.1f}%)</sub>",
                x=0.5,
                xanchor='center'
            ),
            xaxis_title=f"{metric_type} Range",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
            hovermode='x',
            bargap=0.1
        )
        
        # Display histogram
        render_plotly_chart(fig_hist)
        
        # Show distribution statistics below the histogram
        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            st.metric("Current Value", f"{hist_data['current_value']:.2f}")
            st.metric("Percentile (CDF)", f"{hist_data['percentile']:.1f}%")
        with col_dist2:
            st.metric("Median", f"{hist_data['median']:.2f}")
            st.metric("Data Points", hist_data['n_total'])
    else:
        st.info(f"Insufficient data to generate histogram for {selected_ticker}")
    

# Table: Valuation Statistics Table
st.markdown("---")
st.subheader("Valuation Statistics Summary")

# Prepare statistics table
stats_df = prepare_statistics_table(df, metric_col)

if not stats_df.empty:
    # Create interactive table using Plotly
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Keep original dataframe for processing
    table_df = stats_df.copy()
    
    # Remove Type column for display
    if 'Type' in table_df.columns:
        table_df = table_df.drop('Type', axis=1)
    
    # Prepare formatted values for display
    formatted_current = table_df['Current'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    formatted_mean = table_df['Mean'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    formatted_cdf = table_df['CDF (%)'].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "N/A")
    formatted_zscore = table_df['Z-Score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    
    # Prepare colors for status column
    status_colors = []
    for status in table_df['Status']:
        if status == "Very Cheap":
            status_colors.append('#90EE90')
        elif status == "Cheap":
            status_colors.append('#B8E6B8')
        elif status == "Fair":
            status_colors.append('#FFFFCC')
        elif status == "Expensive":
            status_colors.append('#FFD4A3')
        elif status == "Very Expensive":
            status_colors.append('#FFB3B3')
        else:
            status_colors.append('white')
    
    # Identify sector rows for highlighting
    row_colors = []
    for ticker in table_df['Ticker']:
        if ticker in ['Sector', 'SOCB', 'Private_1', 'Private_2', 'Private_3']:
            row_colors.append('#E8E8E8')
        else:
            row_colors.append('white')
    
    # Create the main table figure
    fig_table = go.Figure(data=[go.Table(
        header=dict(
            values=['Ticker', 'Current', 'Mean', 'CDF (%)', 'Z-Score', 'Status'],
            fill_color='#478B81',
            font=dict(color='white', size=12),
            align='left',
            height=30
        ),
        cells=dict(
            values=[
                table_df['Ticker'],
                formatted_current,
                formatted_mean,
                formatted_cdf,
                formatted_zscore,
                table_df['Status']
            ],
            fill_color=[
                row_colors,  # Ticker column
                row_colors,  # Current column
                row_colors,  # Mean column
                row_colors,  # CDF column
                row_colors,  # Z-Score column
                status_colors  # Status column with custom colors
            ],
            align='left',
            height=25,
            font=dict(size=13)
        )
    )])
    
    fig_table.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    # Display the main table
    render_plotly_chart(fig_table)
    
    # Summary statistics
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cheap_count = len(stats_df[stats_df['Status'].isin(['Very Cheap', 'Cheap'])])
        st.metric("Undervalued Banks", cheap_count)
    
    with col2:
        fair_count = len(stats_df[stats_df['Status'] == 'Fair'])
        st.metric("Fairly Valued Banks", fair_count)
    
    with col3:
        expensive_count = len(stats_df[stats_df['Status'].isin(['Expensive', 'Very Expensive'])])
        st.metric("Overvalued Banks", expensive_count)
else:
    st.warning("Insufficient data to generate statistics table")
PLOTLY_SUPPORTS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters


def render_plotly_chart(fig, *, config=None):
    chart_kwargs = {}
    if PLOTLY_SUPPORTS_WIDTH:
        chart_kwargs["width"] = "stretch"
    else:
        chart_kwargs["use_container_width"] = True

    if config is not None:
        chart_kwargs["config"] = config

    return st.plotly_chart(fig, **chart_kwargs)
