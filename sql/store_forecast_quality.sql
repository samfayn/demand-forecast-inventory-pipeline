-- Best and worst forecast items within each store.
--
-- Ranks every run inside its store partition by MASE, then keeps only the
-- rows at either end of that ranking. Useful operationally: the worst rows
-- are where an automated reorder point is least trustworthy and a human
-- should look before the system places an order.
--
-- Parameters: {n_per_store} — how many best/worst rows to keep per store

WITH ranked AS (
    SELECT
        store_id,
        item_id,
        ROUND(avg_daily_demand, 2) AS avg_daily_demand,
        ROUND(mase, 3)             AS mase,
        ROUND(rmse, 2)             AS rmse,
        ROUND(safety_stock, 1)     AS safety_stock,
        RANK() OVER (PARTITION BY store_id ORDER BY mase ASC)  AS best_rank,
        RANK() OVER (PARTITION BY store_id ORDER BY mase DESC) AS worst_rank
    FROM forecast_runs
    WHERE mase IS NOT NULL
)

SELECT
    store_id,
    CASE WHEN best_rank <= {n_per_store} THEN 'best' ELSE 'worst' END AS end_of_range,
    item_id,
    avg_daily_demand,
    mase,
    rmse,
    safety_stock,
    CASE WHEN mase < 1.0 THEN 'beats naive' ELSE 'loses to naive' END AS verdict
FROM ranked
WHERE best_rank <= {n_per_store} OR worst_rank <= {n_per_store}
ORDER BY store_id, mase;
