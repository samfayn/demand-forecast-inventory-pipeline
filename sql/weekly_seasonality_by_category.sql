-- Day-of-week demand pattern by category.
--
-- Prophet's weekly seasonality term is only worth fitting where a weekly
-- pattern actually exists. This measures how much each category's demand
-- swings across the week, expressed as an index where 100 equals that
-- category's own daily average, so categories with very different volumes
-- stay comparable.
--
-- The final column is the peak-to-trough spread: a category at 130 on
-- Saturday and 80 on Wednesday has a 50-point spread and a genuine weekly
-- rhythm; one sitting near 100 all week does not.
--
-- Parameters: {parquet_path}

WITH daily AS (
    SELECT
        cat_id,
        DAYNAME(date)   AS day_name,
        DAYOFWEEK(date) AS day_number,
        AVG(sales)      AS avg_units
    FROM read_parquet('{parquet_path}')
    GROUP BY cat_id, DAYNAME(date), DAYOFWEEK(date)
),

indexed AS (
    SELECT
        cat_id,
        day_name,
        day_number,
        avg_units,
        AVG(avg_units) OVER (PARTITION BY cat_id) AS category_mean
    FROM daily
)

SELECT
    cat_id,
    day_name,
    ROUND(avg_units, 3)                                     AS avg_units,
    ROUND(100.0 * avg_units / category_mean, 1)             AS demand_index,
    ROUND(100.0 * (MAX(avg_units) OVER (PARTITION BY cat_id)
                 - MIN(avg_units) OVER (PARTITION BY cat_id))
          / category_mean, 1)                               AS weekly_spread
FROM indexed
ORDER BY cat_id, day_number;
