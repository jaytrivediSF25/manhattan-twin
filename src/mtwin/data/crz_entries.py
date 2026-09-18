"""MTA Congestion Relief Zone vehicle entries.

Two properties matter. First, 10-minute resolution: the toll steps between peak
and overnight rates at a sharp clock threshold, so this supports a bunching /
regression-discontinuity-in-time design that needs no model and no pre-period.
Second, `excluded_roadway_entries` is reported separately from `crz_entries`,
which measures the diversion channel (FDR / West Side Highway) directly.

The dataset begins at policy launch, so it has no pre-period of its own.
"""

from __future__ import annotations

import logging
from datetime import date

import polars as pl

from . import registry as reg
from .socrata import cached_monthly_pull, load_months

log = logging.getLogger(__name__)

SUBDIR = "crz_entries"

SELECT = ",".join(
    [
        "toll_date", "toll_hour", "toll_10_minute_block", "minute_of_hour",
        "hour_of_day", "day_of_week_int", "day_of_week", "toll_week",
        "time_period", "vehicle_class", "detection_group", "detection_region",
        "crz_entries", "excluded_roadway_entries",
    ]
)

# The twelve tolling gantry groups, split by whether an MTA Bridges & Tunnels
# pre-period series exists for them. Only two do; Holland and Lincoln are Port
# Authority facilities and the East River bridges are free crossings, so
# neither appears in B&T data. This is the binding limitation on any
# pre/post *volume* comparison.
HAS_PRE_PERIOD = {"Hugh L. Carey Tunnel", "Queens Midtown Tunnel"}

NUMERIC = ["minute_of_hour", "hour_of_day", "day_of_week_int", "crz_entries", "excluded_roadway_entries"]
TIMESTAMPS = ["toll_date", "toll_hour", "toll_10_minute_block", "toll_week"]


def _typed(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df
    exprs = [pl.col(c).cast(pl.Float64, strict=False) for c in NUMERIC if c in df.columns]
    exprs += [pl.col(c).str.to_datetime(strict=False) for c in TIMESTAMPS if c in df.columns]
    return df.with_columns(exprs)


def pull(start: date = reg.POLICY_START, end: date = reg.DATA_END) -> None:
    cached_monthly_pull(
        reg.CRZ_ENTRIES, date(start.year, start.month, 1), end,
        subdir=SUBDIR, select=SELECT, date_field="toll_date", transform=_typed,
    )


def load() -> pl.DataFrame:
    return load_months(SUBDIR)
