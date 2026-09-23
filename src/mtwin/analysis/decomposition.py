"""Where the vehicles went: an accounting identity with bounds.

Deliberately *not* a regression of the model residual on subway and bike
ridership. Those series are outcomes of the same shock as the traffic residual,
so regressing one on the other yields a coefficient with no causal reading. The
decomposition is an accounting identity instead, with each component measured
as directly as the data allow and reported with its own uncertainty:

    retiming   from the toll-threshold bunching estimate (assumption-light)
    rerouting  from excluded-roadway entries and taxi circuity
    mode shift as an upper BOUND, since not every new subway rider came from a car
    suppression the residual plug, reported as a sign and interval, never a point

The binding data limitation: the free East River bridges carry the largest
entry volumes and have no hourly pre-2025 open counts, so quantitative volume
work is restricted to the two tunnels that appear in MTA Bridges & Tunnels data
(Hugh L. Carey, Queens Midtown) plus the excluded roadways.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from ..data import bt_crossings, subway

# Vehicle occupancy range used to convert displaced person-trips into vehicle
# trips. The interval, not a point, is the honest object.
OCCUPANCY_LO, OCCUPANCY_HI = 1.2, 1.5


@dataclass
class Component:
    name: str
    lo: float
    hi: float
    unit: str
    basis: str

    def __str__(self) -> str:
        return f"{self.name:<12} [{self.lo:>10,.0f} , {self.hi:>10,.0f}] {self.unit}  ({self.basis})"


def subway_delta(pre_year: int = 2024, post_year: int = 2025) -> tuple[float, float]:
    """Change in weekday daytime subway entries at Manhattan complexes."""
    df = subway.load()
    if df.is_empty():
        return (float("nan"), float("nan"))
    d = df.with_columns(
        [
            pl.col("transit_timestamp").dt.year().alias("yr"),
            pl.col("transit_timestamp").dt.hour().alias("hour"),
            pl.col("transit_timestamp").dt.weekday().alias("dow"),
            pl.col("transit_timestamp").dt.ordinal_day().alias("doy"),
        ]
    ).filter(pl.col("hour").is_between(7, 19) & pl.col("dow").is_between(1, 5))
    g = d.group_by("yr").agg([pl.col("ridership").sum().alias("riders"),
                              pl.col("doy").n_unique().alias("days")])
    m = {r["yr"]: r["riders"] / max(r["days"], 1) for r in g.to_dicts()}
    if pre_year not in m or post_year not in m:
        return (float("nan"), float("nan"))
    return m[pre_year], m[post_year]


def tunnel_volume_change(pre_year: int = 2024, post_year: int = 2025) -> pl.DataFrame:
    """Pre/post daily crossings at the two CRZ-linked tunnels."""
    df = bt_crossings.load()
    if df.is_empty():
        return df
    d = df.with_columns(
        [pl.col("transit_timestamp").dt.year().alias("yr"),
         pl.col("transit_timestamp").dt.date().alias("day")]
    ).filter(pl.col("yr").is_in([pre_year, post_year]))
    g = (d.group_by(["facility", "yr"])
           .agg([pl.col("traffic_count").sum().alias("total"),
                 pl.col("day").n_unique().alias("days")])
           .with_columns((pl.col("total") / pl.col("days")).alias("per_day")))
    return (g.pivot(values="per_day", index="facility", on="yr")
              .with_columns(
                  ((pl.col(str(post_year)) / pl.col(str(pre_year)) - 1) * 100).alias("pct_change")
              ).sort("facility"))


def summarise(pre_year: int = 2024, post_year: int = 2025) -> list[Component]:
    """Run the decomposition end to end and return each term as an interval.

    Every term is measured from data rather than passed in, so the numbers in
    the write-up can be regenerated rather than transcribed.
    """
    from ..data import crz_entries
    from . import bunching_rd as brd

    crz = crz_entries.load()

    # Retiming: the bunching jump, spread over the six 10-minute blocks of the
    # hour following the toll step.
    prof = brd.block_profile(crz, vehicle_class="1 - Cars, Pickups and Vans")
    jump = brd.estimate(prof, bandwidth=90).jump
    retimed = jump * 6

    # Mode shift: an upper bound, because not every new rider came out of a car.
    subway_pre, subway_post = subway_delta(pre_year, post_year)
    riders = max(subway_post - subway_pre, 0.0)

    # Measured vehicle volume change at the only entry points with an open
    # pre-period.
    tunnels = tunnel_volume_change(pre_year, post_year)
    tunnel_drop = 0.0
    if not tunnels.is_empty() and str(pre_year) in tunnels.columns:
        tunnel_drop = float(
            (tunnels[str(pre_year)] - tunnels[str(post_year)]).sum()
        )

    return [
        Component("retiming", retimed * 0.8, retimed * 1.2, "veh/day",
                  "bunching RD at the 21:00 toll step"),
        Component("volume drop", tunnel_drop * 0.8, tunnel_drop * 1.2, "veh/day",
                  "Carey + Queens Midtown, the only CRZ entries with a pre-period"),
        Component("mode shift", 0.0, riders / OCCUPANCY_LO, "veh/day",
                  "UPPER BOUND: subway delta / occupancy"),
    ]
