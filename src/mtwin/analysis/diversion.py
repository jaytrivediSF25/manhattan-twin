"""The diversion channel: untolled roads inside the cordon.

FDR Drive and the West Side Highway run through the middle of the charging zone
but are *excluded roadways* -- a driver who stays on them is not charged. They
are therefore the obvious substitute for a tolled surface route, and the place
rerouting should show up first.

Two independent measurements:

  1. `excluded_roadway_entries` in the MTA CRZ data, reported separately from
     charged entries, which gives the untolled share of all zone entries.
  2. DOT link speeds on the FDR / West Side Highway corridors: if traffic moved
     onto them, they should have got slower.

They are independent instruments for the same behaviour, so agreement between
them is worth more than either alone.
"""

from __future__ import annotations

import polars as pl

from ..data import crz_entries, dot_speeds


def excluded_share() -> pl.DataFrame:
    """Monthly untolled share of all zone entries."""
    df = crz_entries.load()
    if df.is_empty():
        return df
    return (
        df.with_columns(pl.col("toll_date").dt.strftime("%Y-%m").alias("ym"))
        .group_by("ym")
        .agg(
            [
                pl.col("crz_entries").sum().alias("charged"),
                pl.col("excluded_roadway_entries").sum().alias("excluded"),
            ]
        )
        .with_columns(
            (pl.col("excluded") / (pl.col("charged") + pl.col("excluded")) * 100).alias("excluded_pct")
        )
        .sort("ym")
    )


def excluded_roadway_speeds() -> pl.DataFrame:
    """Monthly mean speed on the untolled in-cordon corridors vs other links."""
    spd = dot_speeds.load()
    if spd.is_empty():
        return spd
    excl = dot_speeds.excluded_roadway_links()
    ids = excl["link_id"].to_list() if not excl.is_empty() else []
    return (
        spd.filter(pl.col("speed").is_between(1, 70))
        .with_columns(
            [
                pl.col("data_as_of").dt.strftime("%Y-%m").alias("ym"),
                pl.col("link_id").is_in(ids).alias("is_excluded_roadway"),
                pl.col("data_as_of").dt.hour().alias("hour"),
            ]
        )
        .filter(pl.col("hour").is_between(7, 19))
        .group_by(["ym", "is_excluded_roadway"])
        .agg([pl.col("speed").mean().alias("mean_mph"), pl.len().alias("n")])
        .sort(["ym", "is_excluded_roadway"])
    )
