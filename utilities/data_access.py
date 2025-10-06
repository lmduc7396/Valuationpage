"""High-level data loading helpers for reading curated datasets from the warehouse."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from utilities.db import get_connection

# Ensure environment variables from .env are available before any DB calls
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'Data'


@lru_cache(maxsize=1)
def _keycode_mapping() -> dict:
    key_items_path = DATA_DIR / 'Key_items.xlsx'
    if not key_items_path.exists():
        return {}

    key_items = pd.read_excel(key_items_path)
    return dict(zip(key_items['KeyCode'], key_items['Name']))


def _rename_metrics(df: pd.DataFrame) -> pd.DataFrame:
    mapping = _keycode_mapping()
    available = {k: v for k, v in mapping.items() if k in df.columns}
    if available:
        df = df.rename(columns=available)
    return df


def _load_dataframe(query: str, params: Optional[list] = None) -> pd.DataFrame:
    with get_connection(db="target") as conn:
        return pd.read_sql(query, conn, params=params)


def load_banking_metrics(period: str, *, rename: bool = True) -> pd.DataFrame:
    query = "SELECT * FROM dbo.BankingMetrics WHERE PERIOD_TYPE = %s"
    df = _load_dataframe(query, params=[period])

    if rename:
        df = _rename_metrics(df)

    if 'BANK_TYPE' in df.columns and 'Type' not in df.columns:
        df['Type'] = df['BANK_TYPE']

    if 'DATE_STRING' in df.columns:
        label = 'Date_Quarter' if period.upper() == 'Q' else 'Year'
        df[label] = df['DATE_STRING']

    return df


def load_banking_forecast(*, rename: bool = True) -> pd.DataFrame:
    df = _load_dataframe("SELECT * FROM dbo.BankingForecast")
    if rename:
        df = _rename_metrics(df)
    if 'BANK_TYPE' in df.columns and 'Type' not in df.columns:
        df['Type'] = df['BANK_TYPE']
    if 'DATE_STRING' in df.columns and 'Year' not in df.columns:
        df['Year'] = df['DATE_STRING']
    return df


def load_valuation_banking() -> pd.DataFrame:
    """Load last 5 years of PE/PB for banking tickers only.

    - Source: dbo.Market_Data
    - Columns: TICKER, TRADE_DATE, PE, PB, Type
    - Filters: TRADE_DATE >= GETDATE() - 5 years; TICKER limited to those present in BankingMetrics
    """
    query = """
        SELECT md.TICKER,
               md.TRADE_DATE,
               md.PE,
               md.PB,
               bm.BANK_TYPE AS Type
        FROM dbo.Market_Data AS md
        INNER JOIN (
            SELECT TICKER, MAX(BANK_TYPE) AS BANK_TYPE
            FROM dbo.BankingMetrics
            GROUP BY TICKER
        ) AS bm
            ON md.TICKER = bm.TICKER
        WHERE md.TRADE_DATE >= DATEADD(year, -5, CAST(GETDATE() AS date))
          AND (md.PE IS NOT NULL OR md.PB IS NOT NULL)
    """

    df = _load_dataframe(query)
    return df


def load_valuation_universe(
    years: int = 5,
    *,
    min_market_cap: float | None = None,
) -> pd.DataFrame:
    """Load valuation metrics for all tickers with sector metadata.

    Args:
        years: Number of trailing years to include (default 5).
        min_market_cap: Optional minimum market cap threshold in billions of VND (matching
            `Market_Data.MKT_CAP`).

    Returns:
        DataFrame with valuation ratios and Sector_Map classifications.
    """

    if years <= 0:
        raise ValueError("years must be positive")

    start_date = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years)).date()

    params: list

    if min_market_cap is not None:
        query = """
            ;WITH LatestCaps AS (
                SELECT md.TICKER,
                       MAX(md.TRADE_DATE) AS LatestDate
                FROM dbo.Market_Data AS md
                WHERE md.TRADE_DATE >= %s
                GROUP BY md.TICKER
            ),
            EligibleTickers AS (
                SELECT md.TICKER
                FROM LatestCaps AS lc
                INNER JOIN dbo.Market_Data AS md
                    ON md.TICKER = lc.TICKER
                   AND md.TRADE_DATE = lc.LatestDate
                WHERE COALESCE(md.MKT_CAP, 0) >= %s
            )
            SELECT md.TICKER,
                   md.TRADE_DATE,
                   md.PE,
                   md.PB,
                   md.PS,
                   md.EV_EBITDA,
                   md.MKT_CAP,
                   sm.Sector,
                   sm.L1,
                   sm.L2,
                   sm.L3,
                   sm.VNI
            FROM dbo.Market_Data AS md
            LEFT JOIN dbo.Sector_Map AS sm
                ON md.TICKER = sm.Ticker
            WHERE md.TRADE_DATE >= %s
              AND (md.PE IS NOT NULL
                   OR md.PB IS NOT NULL
                   OR md.PS IS NOT NULL
                   OR md.EV_EBITDA IS NOT NULL)
              AND md.TICKER IN (SELECT TICKER FROM EligibleTickers)
        """
        params = [start_date, min_market_cap, start_date]
    else:
        query = """
            SELECT md.TICKER,
                   md.TRADE_DATE,
                   md.PE,
                   md.PB,
                   md.PS,
                   md.EV_EBITDA,
                   md.MKT_CAP,
                   sm.Sector,
                   sm.L1,
                   sm.L2,
                   sm.L3,
                   sm.VNI
            FROM dbo.Market_Data AS md
            LEFT JOIN dbo.Sector_Map AS sm
                ON md.TICKER = sm.Ticker
            WHERE md.TRADE_DATE >= %s
              AND (md.PE IS NOT NULL
                   OR md.PB IS NOT NULL
                   OR md.PS IS NOT NULL
                   OR md.EV_EBITDA IS NOT NULL)
        """
        params = [start_date]

    df = _load_dataframe(query, params=params)

    df = _load_dataframe(query, params=params)
    if df.empty:
        return df

    df['TRADE_DATE'] = pd.to_datetime(df['TRADE_DATE'])

    rename_map = {
        'L1': 'Industry_L1',
        'L2': 'Industry_L2',
        'L3': 'Industry_L3',
        'VNI': 'VNI_Flag',
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    for col in ['PE', 'PB', 'PS', 'EV_EBITDA', 'MKT_CAP']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if 'VNI_Flag' in df.columns:
        df['VNI_Flag'] = df['VNI_Flag'].fillna('N')

    for col in ['Sector', 'Industry_L1', 'Industry_L2', 'Industry_L3']:
        if col in df.columns:
            df[col] = df[col].fillna('Unclassified')

    return df


def load_sector_map() -> pd.DataFrame:
    """Fetch the latest sector mapping for all tickers."""

    return _load_dataframe("SELECT * FROM dbo.Sector_Map")


def load_earnings_quality(period: str) -> pd.DataFrame:
    table = 'EarningsQualityQuarterly' if period.upper() == 'Q' else 'EarningsQualityYearly'
    df = _load_dataframe(f"SELECT * FROM dbo.{table}")
    return df


def load_comments() -> pd.DataFrame:
    df = _load_dataframe("SELECT * FROM dbo.Banking_Comments")
    if 'DATE' in df.columns and 'QUARTER' not in df.columns:
        df = df.rename(columns={'DATE': 'QUARTER'})
    return df


def load_quarterly_analysis() -> pd.DataFrame:
    return _load_dataframe("SELECT * FROM dbo.QuarterlyAnalysis")
