"""
Utility functions for valuation analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional

def get_metric_column(metric_type: str) -> str:
    """Map UI metric labels to dataframe column names."""

    mapping = {
        "P/E": "PE",
        "P/B": "PB",
        "P/S": "PS",
        "EV/EBITDA": "EV_EBITDA",
    }
    return mapping.get(metric_type, metric_type)

def remove_outliers_iqr(data: pd.Series, multiplier: float = 1.5) -> pd.Series:
    """Remove outliers using IQR method"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

def calculate_distribution_stats(df: pd.DataFrame, ticker: str, metric_col: str) -> Dict:
    """
    Calculate distribution statistics for candle chart
    Returns percentiles and current value
    """
    # Get historical data for the ticker
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()
    
    if len(ticker_data) < 20:  # Need minimum data points
        return None
    
    # Remove outliers for cleaner distribution
    clean_data = remove_outliers_iqr(ticker_data, multiplier=2.0)
    
    # Get current value (most recent)
    current_value = df[df['TICKER'] == ticker][metric_col].iloc[-1] if len(df[df['TICKER'] == ticker]) > 0 else None
    
    # Calculate percentiles
    stats_dict = {
        'p5': clean_data.quantile(0.05),
        'p25': clean_data.quantile(0.25),
        'p50': clean_data.quantile(0.50),  # Median
        'p75': clean_data.quantile(0.75),
        'p95': clean_data.quantile(0.95),
        'current': current_value,
        'count': len(clean_data)
    }
    
    # Calculate current value percentile
    if current_value is not None and not pd.isna(current_value):
        percentile_rank = (clean_data <= current_value).sum() / len(clean_data) * 100
        stats_dict['percentile'] = percentile_rank
    else:
        stats_dict['percentile'] = None
    
    return stats_dict

def calculate_historical_stats(df: pd.DataFrame, ticker: str, metric_col: str) -> Dict:
    """
    Calculate historical statistics for time series chart
    Returns mean, std dev, and current z-score
    """
    # Get historical data
    ticker_data = df[df['TICKER'] == ticker][[metric_col, 'TRADE_DATE']].copy()
    ticker_data = ticker_data.dropna()
    
    if len(ticker_data) < 30:  # Need minimum data points
        return None
    
    # Sort by date
    ticker_data = ticker_data.sort_values('TRADE_DATE')
    
    # Remove outliers
    clean_values = remove_outliers_iqr(ticker_data[metric_col], multiplier=3.0)
    
    # Calculate statistics
    mean_val = clean_values.mean()
    std_val = clean_values.std()
    current_val = ticker_data[metric_col].iloc[-1] if len(ticker_data) > 0 else None
    
    stats_dict = {
        'mean': mean_val,
        'std': std_val,
        'upper_1sd': mean_val + std_val,
        'lower_1sd': mean_val - std_val,
        'upper_2sd': mean_val + 2 * std_val,
        'lower_2sd': mean_val - 2 * std_val,
        'current': current_val
    }
    
    # Calculate z-score
    if current_val is not None and not pd.isna(current_val):
        z_score = (current_val - mean_val) / std_val if std_val > 0 else 0
        stats_dict['z_score'] = z_score
    else:
        stats_dict['z_score'] = None
    
    return stats_dict

def calculate_cdf(df: pd.DataFrame, ticker: str, metric_col: str) -> float:
    """
    Calculate cumulative distribution function (percentile rank)
    Returns value between 0 and 100
    """
    # Get historical data
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()
    
    if len(ticker_data) < 20:
        return None
    
    # Get current value
    current_value = ticker_data.iloc[-1] if len(ticker_data) > 0 else None
    
    if current_value is None or pd.isna(current_value):
        return None
    
    # Remove outliers
    clean_data = remove_outliers_iqr(ticker_data, multiplier=3.0)
    
    # Calculate CDF (percentile)
    cdf_value = (clean_data <= current_value).sum() / len(clean_data) * 100
    
    return cdf_value

def get_valuation_status(z_score: float) -> Tuple[str, str]:
    """
    Get valuation status based on z-score
    Returns (status, color)
    """
    if z_score is None:
        return ("N/A", "gray")
    elif z_score < -1.5:
        return ("Very Cheap", "darkgreen")
    elif z_score < -0.5:
        return ("Cheap", "green")
    elif z_score < 0.5:
        return ("Fair", "yellow")
    elif z_score < 1.5:
        return ("Expensive", "orange")
    else:
        return ("Very Expensive", "red")

def prepare_statistics_table(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    """
    Prepare statistics table with all tickers
    """
    results = []
    
    # Get unique tickers
    tickers = df['TICKER'].unique()
    
    for ticker in tickers:
        # Get ticker type
        ticker_type = df[df['TICKER'] == ticker]['Type'].iloc[0] if len(df[df['TICKER'] == ticker]) > 0 else "Unknown"
        
        # Skip if insufficient data
        ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()
        if len(ticker_data) < 20:
            continue
        
        # Calculate statistics
        hist_stats = calculate_historical_stats(df, ticker, metric_col)
        if hist_stats is None:
            continue
        
        cdf_value = calculate_cdf(df, ticker, metric_col)
        
        # Get status
        status, color = get_valuation_status(hist_stats.get('z_score'))
        
        # Determine if this is a sector aggregate
        is_sector = ticker in ['Sector', 'SOCB', 'Private_1', 'Private_2', 'Private_3']
        
        results.append({
            'Ticker': ticker,
            'Type': ticker_type,
            'Current': hist_stats.get('current', None),
            'Mean': hist_stats.get('mean', None),
            'CDF (%)': cdf_value,
            'Z-Score': hist_stats.get('z_score', None),
            'Status': status,
            'IsSector': is_sector
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort properly: Sector first, then each sub-sector with its banks
    if not results_df.empty:
        # Separate sectors and individual banks
        sectors_df = results_df[results_df['IsSector'] == True].copy()
        banks_df = results_df[results_df['IsSector'] == False].copy()
        
        # Define the exact order we want
        sector_order = ['Sector', 'SOCB', 'Private_1', 'Private_2', 'Private_3']
        
        # Build the final dataframe in the correct order
        final_rows = []
        
        # First add "Sector" if it exists
        if 'Sector' in sectors_df['Ticker'].values:
            final_rows.append(sectors_df[sectors_df['Ticker'] == 'Sector'])
        
        # Then for each sub-sector type, add the sector row followed by its banks
        for sector_type in ['SOCB', 'Private_1', 'Private_2', 'Private_3']:
            # Add the sector aggregate row if it exists
            if sector_type in sectors_df['Ticker'].values:
                final_rows.append(sectors_df[sectors_df['Ticker'] == sector_type])
            
            # Add the component banks for this sector type
            component_banks = banks_df[banks_df['Type'] == sector_type]
            if not component_banks.empty:
                # Sort banks by current value (descending)
                component_banks = component_banks.sort_values('Current', ascending=False)
                final_rows.append(component_banks)
        
        # Add any remaining banks that don't belong to the standard sectors
        other_banks = banks_df[~banks_df['Type'].isin(['SOCB', 'Private_1', 'Private_2', 'Private_3'])]
        if not other_banks.empty:
            other_banks = other_banks.sort_values('Current', ascending=False)
            final_rows.append(other_banks)
        
        # Combine all rows
        if final_rows:
            results_df = pd.concat(final_rows, ignore_index=True)
        
        # Drop helper columns
        results_df = results_df.drop(['IsSector'], axis=1, errors='ignore')
    
    return results_df

def get_sector_and_components(df: pd.DataFrame, sector: str) -> List[str]:
    """
    Get list of tickers for a sector and its components
    """
    if sector == "Sector":
        # Return overall sector plus all individual banks
        all_tickers = ['Sector'] + sorted(df[df['TICKER'].str.len() == 3]['TICKER'].unique().tolist())
        return all_tickers
    else:
        # Return sector aggregate plus its component banks
        component_banks = df[df['Type'] == sector]['TICKER'].unique().tolist()
        # Remove the sector itself if it's in the list
        component_banks = [t for t in component_banks if t != sector]
        # Add sector at the beginning
        return [sector] + sorted(component_banks)

def generate_valuation_histogram(df: pd.DataFrame, ticker: str, metric_col: str, n_bins: int = 8) -> Dict:
    """
    Generate histogram data for a ticker's valuation metric
    Returns bin edges, counts, current value bin, and formatted data for visualization
    """
    # Get historical data for the ticker
    ticker_data = df[df['TICKER'] == ticker][metric_col].dropna()
    
    if len(ticker_data) < 20:  # Need minimum data points
        return None
    
    # Remove outliers for cleaner distribution
    clean_data = remove_outliers_iqr(ticker_data, multiplier=3.0)
    
    # Get current value (most recent)
    current_value = ticker_data.iloc[-1] if len(ticker_data) > 0 else None
    
    if current_value is None or pd.isna(current_value):
        return None
    
    # Create histogram bins
    counts, bin_edges = np.histogram(clean_data, bins=n_bins)
    
    # Find which bin the current value belongs to
    current_bin_idx = None
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= current_value < bin_edges[i + 1]:
            current_bin_idx = i
            break
    # Handle edge case where current value equals the last edge
    if current_value == bin_edges[-1] and len(counts) > 0:
        current_bin_idx = len(counts) - 1
    
    # Create bin centers for plotting
    bin_centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    
    # Format bin labels
    bin_labels = []
    for i in range(len(bin_edges) - 1):
        label = f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}"
        bin_labels.append(label)
    
    # Calculate percentile
    percentile = (clean_data <= current_value).sum() / len(clean_data) * 100
    
    histogram_data = {
        'ticker': ticker,
        'current_value': current_value,
        'current_bin_idx': current_bin_idx,
        'bin_edges': bin_edges.tolist(),
        'bin_centers': bin_centers,
        'bin_labels': bin_labels,
        'counts': counts.tolist(),
        'percentile': percentile,
        'n_total': len(clean_data),
        'min_value': float(clean_data.min()),
        'max_value': float(clean_data.max()),
        'median': float(clean_data.median())
    }
    
    return histogram_data
