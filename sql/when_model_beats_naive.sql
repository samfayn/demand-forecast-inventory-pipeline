-- When does the model actually beat a seasonal naive baseline?
--
-- A 200-run sweep produced a median MASE near 1.0 with a range from roughly
-- 0.04 to 4.26. A median that close to the baseline hides the fact that the
-- model is excellent on some series and much worse than naive on others.
-- This segments the runs to find where the difference lies.
--
-- Demand volume and variability (coefficient of variation) are computed from
-- what each run recorded, then bucketed with NTILE so the comparison is
-- across equal-sized groups rather than arbitrary thresholds.
--
-- Parameters: none (reads forecast_runs)

WITH scored AS (
    SELECT
        item_id,
        store_id,
        avg_daily_demand,
        std_daily_demand,
        mase,
        mape,
        rmse,
        -- coefficient of variation: variability relative to demand level,
        -- which is the scale-free way to compare a 0.5-unit/day item against
        -- a 20-unit/day one
        CASE WHEN avg_daily_demand > 0
             THEN std_daily_demand / avg_daily_demand
        END AS coef_variation
    FROM forecast_runs
    WHERE mase IS NOT NULL
),

bucketed AS (
    SELECT
        *,
        NTILE(4) OVER (ORDER BY avg_daily_demand) AS volume_quartile,
        NTILE(4) OVER (ORDER BY coef_variation)   AS variability_quartile
    FROM scored
    WHERE coef_variation IS NOT NULL
)

SELECT
    volume_quartile,
    ROUND(MIN(avg_daily_demand), 2)                            AS min_demand,
    ROUND(MAX(avg_daily_demand), 2)                            AS max_demand,
    COUNT(*)                                                   AS runs,
    ROUND(MEDIAN(coef_variation), 2)                           AS median_cv,
    ROUND(MEDIAN(mase), 3)                                     AS median_mase,
    COUNT(*) FILTER (WHERE mase < 1.0)                         AS beat_naive,
    ROUND(100.0 * COUNT(*) FILTER (WHERE mase < 1.0)
          / COUNT(*), 0)                                       AS pct_beat_naive,
    ROUND(MEDIAN(mape), 1)                                     AS median_mape,
    ROUND(MEDIAN(rmse), 2)                                     AS median_rmse
FROM bucketed
GROUP BY volume_quartile
ORDER BY volume_quartile;
