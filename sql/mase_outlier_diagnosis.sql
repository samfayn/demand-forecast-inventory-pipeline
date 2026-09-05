-- MASE outliers, and whether they are real failures or measurement artifacts.
--
-- MASE scales test-set error by the mean absolute error of a seasonal naive
-- forecast computed on the *training* window. When that training window is
-- nearly flat, the denominator approaches zero and the ratio explodes for
-- reasons that have nothing to do with model quality. The same mechanism
-- works in reverse: a volatile training period followed by a quiet holdout
-- produces a flattering MASE near zero.
--
-- A 596-run sweep produced a median MASE of 0.97 but a mean of 1.65 and a
-- maximum of 368. That gap between median and mean is the signature of a few
-- extreme values, and this query checks whether they are genuine forecast
-- failures or degenerate scale factors that should be excluded before
-- reporting an average.
--
-- Read it as: if the extreme rows sit on near-zero demand, the metric broke
-- down rather than the model. If they sit on healthy demand, the model
-- genuinely failed and is worth investigating.
--
-- Parameters: none (reads forecast_runs)

WITH flagged AS (
    SELECT
        item_id,
        store_id,
        ROUND(avg_daily_demand, 3) AS avg_daily_demand,
        ROUND(std_daily_demand, 3) AS std_daily_demand,
        ROUND(mase, 3)             AS mase,
        ROUND(mape, 1)             AS mape,
        ROUND(rmse, 3)             AS rmse,
        CASE
            WHEN mase >= 10  THEN 'extreme high (>=10)'
            WHEN mase >= 2   THEN 'high (2-10)'
            WHEN mase >= 1   THEN 'loses to naive (1-2)'
            WHEN mase >= 0.5 THEN 'beats naive (0.5-1)'
            WHEN mase >= 0.2 THEN 'strong (0.2-0.5)'
            ELSE 'suspiciously low (<0.2)'
        END AS mase_band
    FROM forecast_runs
    WHERE mase IS NOT NULL
)

SELECT
    mase_band,
    COUNT(*)                                    AS runs,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct_of_runs,
    ROUND(MEDIAN(avg_daily_demand), 3)          AS median_demand,
    ROUND(MIN(avg_daily_demand), 3)             AS min_demand,
    ROUND(MAX(avg_daily_demand), 3)             AS max_demand,
    ROUND(MEDIAN(rmse), 3)                      AS median_rmse,
    -- if the extreme bands sit almost entirely on near-dead series, the
    -- metric is breaking down rather than the model failing
    COUNT(*) FILTER (WHERE avg_daily_demand < 0.1) AS runs_under_point_one_per_day
FROM flagged
GROUP BY mase_band
ORDER BY
    CASE mase_band
        WHEN 'suspiciously low (<0.2)' THEN 1
        WHEN 'strong (0.2-0.5)'        THEN 2
        WHEN 'beats naive (0.5-1)'     THEN 3
        WHEN 'loses to naive (1-2)'    THEN 4
        WHEN 'high (2-10)'             THEN 5
        ELSE 6
    END;