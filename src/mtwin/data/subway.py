"""MTA subway hourly ridership -- the mode-shift observable.

Aggregated server-side to (timestamp, station complex) before download. The raw
table is split by payment method and fare class, which multiplies row count by
roughly an order of magnitude for information this project never uses.

Seam: wujg-7c2s covers 2020-2024, 5wq4-mkjj covers 2025 onward.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from . import registry as reg
from .socrata import cached_monthly_pull, load_months

log = logging.getLogger(__name__)
SUBDIR = "subway_hourly"

SELECT = (
    "transit_timestamp, station_complex_id, station_complex, borough, "
    "latitude, longitude, sum(ridership) as ridership, sum(transfers) as transfers"
)
GROUP = "transit_timestamp, station_complex_id, station_complex, borough, latitude, longitude"


def _typed(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in ("ridership", "transfers", "latitude", "longitude") if c in df.columns]
        + [pl.col("transit_timestamp").str.to_datetime(strict=False)]
    )


def pull(start: date = reg.DATA_START, end: date = reg.DATA_END, borough: str = "Manhattan") -> None:
    seam = date(2025, 1, 1)
    where = f"borough='{borough}'"
    # Grouped queries cannot order by :id, so paging is ordered by the group
    # keys instead -- see fetch() for why.
    order = "transit_timestamp, station_complex_id"
    if start < seam:
        cached_monthly_pull(reg.SUBWAY_2024, start, min(end, seam), subdir=SUBDIR,
                            select=SELECT, extra_where=where, group=GROUP,
                            order=order, transform=_typed)
    if end > seam:
        cached_monthly_pull(reg.SUBWAY_2025, max(start, seam), end, subdir=SUBDIR,
                            select=SELECT, extra_where=where, group=GROUP,
                            order=order, transform=_typed)


def load() -> pl.DataFrame:
    return load_months(SUBDIR)
