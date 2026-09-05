-- Forecast accuracy against the demand history that produced it.
--
-- Joins the run results in DuckDB to the underlying sales history in Parquet.
-- Neither source answers this alone: forecast_runs knows how the model
-- performed but not what the series looked like, and the Parquet knows the
-- series but nothing about the model.
--
-- The question is whether intermittency predicts forecast quality. zero_day_pct
-- is the share of stocked days with no sales, which is the standard definition
-- of intermittent demand and a more direct measure than average volume.
--
-- Parameters: {parquet_path}

WITH history AS (
    SELECT
        item_id,
        store_id,
        COUNT(*)                                          AS days_stocked,
        AVG(sales)                                        AS hist_avg_daily,
        100.0 * COUNT(*) FILTER (WHERE sales = 0)
              / COUNT(*)                                  AS zero_day_pct,
        MAX(sales)                                        AS peak_day_sales,
        AVG(sell_price)                                   AS avg_price
    FROM read_parquet('{parquet_path}')
    GROUP BY item_id, store_id
),

joined AS (
    SELECT
        r.item_id,
        r.store_id,
        r.mase,
        r.mape,
        r.rmse,
        r.avg_daily_demand AS forecast_avg_daily,
        h.hist_avg_daily,
        h.zero_day_pct,
        h.days_stocked,
        h.peak_day_sales,
        ROUND(h.avg_price, 2) AS avg_price,
        CASE
            WHEN h.zero_day_pct >= 60 THEN 'highly intermittent (60%+ zero days)'
            WHEN h.zero_day_pct >= 30 THEN 'intermittent (30-60%)'
            WHEN h.zero_day_pct >= 10 THEN 'occasional gaps (10-30%)'
            ELSE 'continuous (<10% zero days)'
        END AS demand_pattern
    FROM forecast_runs r
    INNER JOIN history h
        ON r.item_id = h.item_id
       AND r.store_id = h.store_id
    WHERE r.mase IS NOT NULL
)

SELECT
    demand_pattern,
    COUNT(*)                                          AS runs,
    ROUND(AVG(zero_day_pct), 1)                       AS avg_zero_day_pct,
    ROUND(MEDIAN(hist_avg_daily), 2)                  AS median_daily_units,
    ROUND(MEDIAN(mase), 3)                            AS median_mase,
    COUNT(*) FILTER (WHERE mase < 1.0)                AS beat_naive,
    ROUND(100.0 * COUNT(*) FILTER (WHERE mase < 1.0)
          / COUNT(*), 0)                              AS pct_beat_naive,
    ROUND(MEDIAN(mape), 1)                            AS median_mape,
    ROUND(MEDIAN(rmse), 2)                            AS median_rmse
FROM joined
GROUP BY demand_pattern
ORDER BY avg_zero_day_pct;
