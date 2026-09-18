"""NYC DOT link speeds -- the diversion channel, not a primary observable.

Manhattan has only ~25 links in this feed and every one is limited-access:
FDR Drive, West Side Highway / 12th / 11th Ave, the Lincoln, Hugh L. Carey and
Queens Midtown tunnels, and the Brooklyn / Manhattan bridge approaches. There
is no surface-street coverage at all.

That would be fatal for a primary observable, but it is exactly what makes
these links useful: FDR Drive and the West Side Highway are *excluded
roadways*, untolled routes running through the middle of the cordon. They are
where a driver diverts to avoid the charge, so this feed measures rerouting
directly.

Storage note: `link_points` and `encoded_poly_line` are long text blobs that
dominate row size. Geometry is persisted once to a separate table and dropped
from the time series.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import polars as pl

from . import registry as reg
from .socrata import RAW, cached_monthly_pull, fetch, load_months

log = logging.getLogger(__name__)

SUBDIR = "dot_speeds"
GEOMETRY = RAW / "dot_link_geometry.parquet"

# Deliberately excludes link_points / encoded_poly_line*; see module docstring.
SELECT = "id,speed,travel_time,status,data_as_of,link_id,owner,transcom_id,borough,link_name"

# Untolled routes running through the cordon -- the diversion channel proper.
EXCLUDED_ROADWAY_PATTERNS = ["FDR", "Westside Hwy", "West Side Hwy", "12th Ave", "11th ave", "11th Ave"]


def _typed(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    return df.with_columns(
        [
            pl.col("speed").cast(pl.Float64, strict=False),
            pl.col("travel_time").cast(pl.Float64, strict=False),
            pl.col("data_as_of").str.to_datetime(strict=False),
        ]
    )


def pull_geometry(boroughs: tuple[str, ...] = ("Manhattan", "Brooklyn", "Queens", "Bronx")) -> Path:
    """Persist one row of link geometry per link_id, once."""
    if GEOMETRY.exists():
        log.info("dot geometry: cached")
        return GEOMETRY
    quoted = ",".join(f"'{b}'" for b in boroughs)
    # A single recent day is enough to enumerate the active link inventory.
    df = fetch(
        reg.DOT_SPEEDS,
        select="link_id,link_name,borough,owner,link_points",
        where=(
            f"borough in({quoted}) AND data_as_of >= '2026-03-04T08:00:00' "
            "AND data_as_of < '2026-03-04T12:00:00'"
        ),
    )
    if df.is_empty():
        raise RuntimeError("dot geometry: no rows returned")
    df = df.unique(subset=["link_id"])
    df.write_parquet(GEOMETRY)
    log.info("dot geometry: %d links -> %s", df.height, GEOMETRY.name)
    return GEOMETRY


def pull(start: date = reg.DATA_START, end: date = reg.DATA_END,
         boroughs: tuple[str, ...] = ("Manhattan", "Brooklyn", "Queens", "Bronx")) -> None:
    pull_geometry(boroughs)
    quoted = ",".join(f"'{b}'" for b in boroughs)
    cached_monthly_pull(
        reg.DOT_SPEEDS, start, end,
        subdir=SUBDIR, select=SELECT, extra_where=f"borough in({quoted})",
        date_field="data_as_of", transform=_typed,
    )


def load() -> pl.DataFrame:
    return load_months(SUBDIR)


def load_geometry() -> pl.DataFrame:
    return pl.read_parquet(GEOMETRY) if GEOMETRY.exists() else pl.DataFrame()


def excluded_roadway_links() -> pl.DataFrame:
    """The in-cordon untolled links (FDR, West Side Hwy corridor)."""
    geo = load_geometry()
    if geo.is_empty():
        return geo
    pattern = "|".join(EXCLUDED_ROADWAY_PATTERNS)
    return geo.filter(
        (pl.col("borough") == "Manhattan") & pl.col("link_name").str.contains(f"(?i){pattern}")
    )
