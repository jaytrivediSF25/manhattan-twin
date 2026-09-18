"""Street closures from construction, used to flag contaminated days.

A closed block changes travel times for reasons that have nothing to do with a
toll. Anything not modelled is attributed to behaviour by construction, so
closure intensity enters as a control and heavy-closure days are available as
an exclusion for robustness.

The dataset is a permit record, not a time series: each row carries a work
start and end date, so it has to be expanded into daily counts before it can be
joined to anything.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from .socrata import RAW, Dataset, fetch

log = logging.getLogger(__name__)

CLOSURES = Dataset("data.cityofnewyork.us", "i6b5-j7bu", "street_closures", "work_start_date")
OUT = RAW / "closures_manhattan.parquet"

SELECT = "segmentid,onstreetname,borough_code,work_start_date,work_end_date"


def pull(force: bool = False) -> pl.DataFrame:
    """Fetch Manhattan closure permits overlapping the study window."""
    if OUT.exists() and not force:
        log.info("closures: cached")
        return pl.read_parquet(OUT)
    df = fetch(
        CLOSURES,
        select=SELECT,
        where=("borough_code='M' AND work_end_date >= '2023-01-01T00:00:00' "
               "AND work_start_date < '2026-09-01T00:00:00'"),
    )
    if df.is_empty():
        log.warning("closures: no rows returned")
        return df
    df = df.with_columns(
        [pl.col(c).str.to_datetime(strict=False).dt.date() for c in ("work_start_date", "work_end_date")]
    )
    df.write_parquet(OUT)
    log.info("closures: %d permits -> %s", df.height, OUT.name)
    return df


def daily_intensity(start: date = date(2023, 1, 1), end: date = date(2026, 9, 1)) -> pl.DataFrame:
    """Expand permits into a count of concurrently closed segments per day."""
    df = pull()
    if df.is_empty():
        return pl.DataFrame()
    days = pl.DataFrame({"service_date": pl.date_range(start, end, "1d", eager=True)})
    # A permit contributes to every day between its start and end, so the join
    # is a range overlap rather than an equality.
    out = (
        days.join(df, how="cross")
        .filter(
            (pl.col("service_date") >= pl.col("work_start_date"))
            & (pl.col("service_date") <= pl.col("work_end_date"))
        )
        .group_by("service_date")
        .agg(pl.col("segmentid").n_unique().alias("closed_segments"))
        .sort("service_date")
    )
    return out.with_columns(
        (pl.col("closed_segments") > pl.col("closed_segments").quantile(0.9)).alias("heavy_closure_day")
    )
