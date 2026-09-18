"""MTA Bridges & Tunnels hourly crossings -- the only open pre-period entry counts.

Covers 2019 onward, which is what the CRZ entries dataset lacks. The overlap
with congestion-zone entry points is narrow: of the twelve CRZ detection
groups, only Hugh L. Carey Tunnel and Queens Midtown Tunnel appear here.
Holland and Lincoln are Port Authority facilities, and the East River bridges
(Brooklyn, Manhattan, Williamsburg, Queensboro) are free crossings that were
never metered before tolling began.

Quantitative pre/post volume work is therefore restricted to those two tunnels
plus the excluded roadways, and the gap must be stated explicitly.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from . import registry as reg
from .socrata import cached_monthly_pull, load_months

log = logging.getLogger(__name__)

SUBDIR = "bt_crossings"

SELECT = "transit_timestamp,date,hour,facility_id,facility,direction,payment_method,vehicle_class,vehicle_class_description,vehicle_class_category,traffic_count"

# Facilities that are also CRZ entry points, i.e. the usable pre-period series.
CRZ_LINKED = ["Hugh L. Carey Tunnel", "Queens Midtown Tunnel"]


def _typed(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in ("hour", "traffic_count") if c in df.columns]
    exprs += [
        pl.col(c).str.to_datetime(strict=False)
        for c in ("transit_timestamp", "date")
        if c in df.columns
    ]
    return df.with_columns(exprs)


def pull(start: date = reg.DATA_START, end: date = reg.DATA_END, crz_only: bool = True) -> None:
    """Pull hourly crossings, by default only for the two CRZ-linked tunnels."""
    where = None
    if crz_only:
        quoted = ",".join(f"'{f}'" for f in CRZ_LINKED)
        where = f"facility in({quoted})"
    cached_monthly_pull(
        reg.BT_CROSSINGS, start, end,
        subdir=SUBDIR, select=SELECT, extra_where=where,
        date_field="transit_timestamp", transform=_typed,
    )


def load() -> pl.DataFrame:
    return load_months(SUBDIR)
