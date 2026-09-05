-- ABC classification of the catalog by revenue contribution.
--
-- Standard inventory-management segmentation: rank every item/store pair by
-- annual revenue, then cut the ranked list where cumulative revenue crosses
-- 80% (A items) and 95% (B items). A items justify tight forecasting and
-- frequent review; C items usually do not.
--
-- Runs directly against the Parquet file rather than a loaded table — DuckDB
-- reads it in place, so this scans 46M rows without an import step.
--
-- Parameters: {parquet_path}

WITH item_revenue AS (
    SELECT
        item_id,
        store_id,
        cat_id,
        COUNT(*)                              AS days_stocked,
        AVG(sales)                            AS avg_daily_units,
        SUM(sales * sell_price)               AS total_revenue
    FROM read_parquet('{parquet_path}')
    GROUP BY item_id, store_id, cat_id
),

ranked AS (
    SELECT
        *,
        SUM(total_revenue) OVER ()                       AS catalog_revenue,
        SUM(total_revenue) OVER (
            ORDER BY total_revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                AS running_revenue,
        ROW_NUMBER() OVER (ORDER BY total_revenue DESC)  AS revenue_rank
    FROM item_revenue
),

classified AS (
    SELECT
        *,
        running_revenue / catalog_revenue AS cumulative_share,
        CASE
            WHEN running_revenue / catalog_revenue <= 0.80 THEN 'A'
            WHEN running_revenue / catalog_revenue <= 0.95 THEN 'B'
            ELSE 'C'
        END AS abc_class
    FROM ranked
)

SELECT
    abc_class,
    COUNT(*)                                              AS item_store_pairs,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1)    AS pct_of_catalog,
    ROUND(SUM(total_revenue), 0)                          AS revenue,
    ROUND(100.0 * SUM(total_revenue)
          / SUM(SUM(total_revenue)) OVER (), 1)           AS pct_of_revenue,
    ROUND(AVG(avg_daily_units), 2)                        AS avg_daily_units
FROM classified
GROUP BY abc_class
ORDER BY abc_class;
