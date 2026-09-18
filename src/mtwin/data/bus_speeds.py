"""MTA bus route segment speeds -- the project's primary surface observable.

Buses run the tolled avenues, are exempt from the toll themselves, and follow
fixed routes, which makes them an untreated probe of surface network speed.
The trade-off is that segment speed blends running speed with dwell time, so
this observable is triangulated against taxi-implied speeds rather than
trusted alone.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from . import registry as reg
from .socrata import cached_monthly_pull, load_months

log = logging.getLogger(__name__)

SUBDIR = "bus_segment_speeds"

# Keep only what the panel needs; the georeference blobs duplicate the lat/lon
# columns and roughly double row size on disk.
SELECT = ",".join(
    [
        "timestamp", "day_of_week", "hour_of_day", "route_id", "direction",
        "borough", "route_type", "stop_order",
        "timepoint_stop_id", "timepoint_stop_name",
        "timepoint_stop_latitude", "timepoint_stop_longitude",
        "next_timepoint_stop_id", "next_timepoint_stop_name",
        "next_timepoint_stop_latitude", "next_timepoint_stop_longitude",
        "road_distance", "average_travel_time", "average_road_speed",
        "bus_trip_count",
    ]
)

NUMERIC = [
    "hour_of_day", "stop_order", "road_distance", "average_travel_time",
    "average_road_speed", "bus_trip_count",
    "timepoint_stop_latitude", "timepoint_stop_longitude",
    "next_timepoint_stop_latitude", "next_timepoint_stop_longitude",
]


def _typed(df: pl.DataFrame) -> pl.DataFrame:
    """Socrata returns every value as a string; cast to real types once, here."""
    if df.is_empty():
        return df
    exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC if c in df.columns]
    if "timestamp" in df.columns:
        exprs.append(pl.col("timestamp").str.to_datetime(strict=False).alias("timestamp"))
    return df.with_columns(exprs)


def pull(start: date = reg.DATA_START, end: date = reg.DATA_END, borough: str = "Manhattan") -> None:
    """Fetch both sides of the 2024/2025 dataset seam into one cache directory."""
    where = f"borough='{borough}'"
    # 58t6-89vi ends 2024-12-01; kufs-yh3x picks up 2025-01-01.
    seam = date(2025, 1, 1)
    if start < seam:
        cached_monthly_pull(
            reg.BUS_SEG_2324, start, min(end, seam),
            subdir=SUBDIR, select=SELECT, extra_where=where, transform=_typed,
        )
    if end > seam:
        cached_monthly_pull(
            reg.BUS_SEG_2025, max(start, seam), end,
            subdir=SUBDIR, select=SELECT, extra_where=where, transform=_typed,
        )


def load() -> pl.DataFrame:
    """Load the stitched segment-speed panel, with a stable segment key."""
    df = load_months(SUBDIR)
    if df.is_empty():
        return df
    return df.with_columns(
        (pl.col("timepoint_stop_id").cast(pl.Utf8) + "->" + pl.col("next_timepoint_stop_id").cast(pl.Utf8))
        .alias("segment_id")
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    pull()
