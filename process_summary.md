# Valuation App Update Summary

## Work Completed
- Added a sidebar control for minimum market-cap filtering and surfaced the number of companies used in calculations.
- Refactored downstream filtering so all calculations honour the selected market-cap floor across charts, tables, and drilldowns.
- Replaced deprecated Streamlit sizing arguments with compatibility helpers to avoid keyword warnings while keeping responsive layouts.
- Iteratively optimised `load_valuation_universe`/`load_market_data` to push the market-cap filter into the SQL query and post-load guardrails, ensuring only qualifying tickers remain.
- Added contextual captions showing the smallest and median market caps (in billions) among included companies for quick validation.

## Persisting Issue
- Market-cap filtering still fails to narrow the universe: with the threshold at 4,000 bn the app reports 728 eligible tickers, and with the threshold raised to 10,000 bn the count *increases* to 730. The smallest “eligible” market cap shows at 7,586.2 bn while the median reaches 1,361,745.0 bn, implying the SQL predicate never bites despite expectations of <200 companies passing the filter.
- Increasing the threshold to 10,000 bn also triggers `pandas.errors.DatabaseError` (redacted by Streamlit). The traceback points to `utilities/data_access.py:load_valuation_universe` when executing the parameterised query.

### Hypotheses Under Review
- **Unit mismatch**: `Market_Data.MKT_CAP` may be stored in raw VND or millions, while the UI sends “billions” without conversion. If so, the condition `COALESCE(latest.CUR_MKT_CAP, 0) >= threshold` will accept nearly every ticker.
- **Snapshot source change**: Replacing the `MarketCap` table with an `OUTER APPLY` on `Market_Data` could be surfacing stale or synthetic tickers (pref shares, aggregates) whose latest rows all satisfy the check. We need to confirm whether `MarketCap` remains the authoritative daily snapshot and whether we should reintroduce it or restrict tickers.
- **SQL execution failure**: The redacted DatabaseError might stem from the new correlated subquery (e.g., plan timeout, implicit conversion, or overflow when binding large float parameters). Capturing the raw SQL error via SSMS or elevated Streamlit logs is required before adjusting the query structure.

### Next Steps
- Validate the stored units for `Market_Data.MKT_CAP` and reconcile them with the UI multiplier (`MARKET_CAP_BILLION_TO_DATA_SCALE`).
- Compare results by running the filter directly in SQL Server using both the `MarketCap` snapshot and the `OUTER APPLY` approach to confirm which dataset provides accurate counts.
- Retrieve the full SQL Server error message (log scrape or direct query run) to determine whether the failure necessitates query rewrites or index hints.
