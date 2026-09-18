"""Panel construction for the exposure design.

Two panels, one per observable:

  taxi_od_panel    OD pair x month, from TLC yellow cells (dwell-free speed,
                   plus circuity, which is the direct rerouting measure)
  bus_segment_panel  bus segment x month, from MTA segment speeds (untreated
                   probe, but blends running time with dwell time)

Both carry the same exposure grouping so the estimator can be run unchanged
across them: `crz` (treated), `above60` (spillover), `outer` (never-taker).
"""

from __future__ import annotations

import polars as pl

from ..data import bus_speeds, tlc, weather
from ..data.registry import POLICY_START
from ..network.zones import zone_centroids

# Weekday daytime only: the toll's peak window, and the hours where congestion
# actually binds.
HOUR_LO, HOUR_HI = 7, 19
MIN_TRIPS_CELL = 5       # per OD-date-hour cell, before monthly aggregation
MIN_TRIPS_MONTH = 30     # per OD-month, per the pre-registered threshold


def _period_index(d: pl.Expr) -> pl.Expr:
    return d.dt.year() * 12 + d.dt.month()


POLICY_PERIOD = POLICY_START.year * 12 + POLICY_START.month


def zone_groups() -> pl.DataFrame:
    """Exposure group per taxi zone."""
    z = zone_centroids()
    return z.with_columns(
        pl.when(pl.col("in_crz")).then(pl.lit("crz"))
        .when(pl.col("borough") == "Manhattan").then(pl.lit("above60"))
        .when(pl.col("borough").is_in(["Brooklyn", "Queens", "Bronx"])).then(pl.lit("outer"))
        .otherwise(pl.lit("drop")).alias("grp")
    ).select(["zone_id", "grp", "lat", "lon", "straddles"])


def taxi_od_panel(*, drop_straddle: bool = False) -> pl.DataFrame:
    """OD-pair x month panel of taxi-implied speed and circuity."""
    cells = tlc.load_yellow(subdir=tlc.SUBDIR_CITY)
    zg = zone_groups()
    if drop_straddle:
        zg = zg.filter(~pl.col("straddles"))

    d = cells.filter(
        pl.col("hour").is_between(HOUR_LO, HOUR_HI)
        & pl.col("dow").is_between(1, 5)
        & (pl.col("n_trips") >= MIN_TRIPS_CELL)
    )
    d = (
        d.join(zg.select(pl.col("zone_id").alias("pu"), pl.col("grp").alias("pu_grp")), on="pu")
         .join(zg.select(pl.col("zone_id").alias("do_"), pl.col("grp").alias("do_grp")), on="do_")
    )
    # A pair is treated only when both ends sit inside the cordon; mixed pairs
    # are partially exposed and are held out of the main contrast.
    d = d.with_columns(
        pl.when((pl.col("pu_grp") == "crz") & (pl.col("do_grp") == "crz")).then(pl.lit("crz"))
        .when((pl.col("pu_grp") == "above60") & (pl.col("do_grp") == "above60")).then(pl.lit("above60"))
        .when((pl.col("pu_grp") == "outer") & (pl.col("do_grp") == "outer")).then(pl.lit("outer"))
        .otherwise(pl.lit("mixed")).alias("grp")
    ).filter(pl.col("grp") != "mixed")

    panel = (
        d.group_by(["pu", "do_", "grp", _period_index(pl.col("service_date")).alias("period")])
        .agg(
            [
                (pl.col("mean_miles") * pl.col("n_trips")).sum().alias("veh_miles"),
                (pl.col("mean_seconds") * pl.col("n_trips")).sum().alias("veh_seconds"),
                pl.col("n_trips").sum().alias("n_trips"),
            ]
        )
        .filter(pl.col("n_trips") >= MIN_TRIPS_MONTH)
        .with_columns(
            [
                (pl.col("veh_miles") / (pl.col("veh_seconds") / 3600)).alias("speed_mph"),
                (pl.col("veh_miles") / pl.col("n_trips")).alias("mean_miles"),
                (pl.col("pu").cast(pl.Utf8) + "_" + pl.col("do_").cast(pl.Utf8)).alias("unit"),
            ]
        )
    )
    return _add_design_cols(panel)


def bus_segment_panel(boroughs=("Manhattan", "Brooklyn", "Queens", "Bronx")) -> pl.DataFrame:
    """Bus segment x month panel of segment speed."""
    df = bus_speeds.load(boroughs=boroughs)
    if df.is_empty():
        return df
    df = df.with_columns(
        ((pl.col("timepoint_stop_latitude") + pl.col("next_timepoint_stop_latitude")) / 2).alias("mid_lat")
    )
    df = df.with_columns(
        pl.when((pl.col("borough") == "Manhattan") & (pl.col("mid_lat") < 40.7648)).then(pl.lit("crz"))
        .when(pl.col("borough") == "Manhattan").then(pl.lit("above60"))
        .otherwise(pl.lit("outer")).alias("grp")
    )
    d = df.filter(
        pl.col("hour_of_day").is_between(HOUR_LO, HOUR_HI)
        & ~pl.col("day_of_week").is_in(["Saturday", "Sunday"])
    )
    panel = (
        d.group_by(["segment_id", "grp", _period_index(pl.col("timestamp")).alias("period")])
        .agg(
            [
                (pl.col("road_distance") * pl.col("bus_trip_count")).sum().alias("veh_miles"),
                (pl.col("average_travel_time") * pl.col("bus_trip_count")).sum().alias("veh_minutes"),
                pl.col("bus_trip_count").sum().alias("n_trips"),
            ]
        )
        .with_columns(
            [
                (pl.col("veh_miles") / (pl.col("veh_minutes") / 60)).alias("speed_mph"),
                pl.col("segment_id").alias("unit"),
            ]
        )
    )
    return _add_design_cols(panel)


def _add_design_cols(panel: pl.DataFrame) -> pl.DataFrame:
    """Attach the treatment interactions and event-time index."""
    return panel.with_columns(
        [
            pl.col("speed_mph").log().alias("log_speed"),
            (pl.col("period") >= POLICY_PERIOD).cast(pl.Float64).alias("post"),
            (pl.col("grp") == "crz").cast(pl.Float64).alias("is_crz"),
            (pl.col("grp") == "above60").cast(pl.Float64).alias("is_above60"),
            (pl.col("period") - POLICY_PERIOD).alias("rel_period"),
        ]
    ).with_columns(
        [
            (pl.col("is_crz") * pl.col("post")).alias("crz_post"),
            (pl.col("is_above60") * pl.col("post")).alias("above60_post"),
        ]
    ).filter(pl.col("speed_mph").is_finite() & (pl.col("speed_mph") > 0))


# --- Reservoir state panel ---------------------------------------------------

def reservoir_panel(min_trips: int = 20) -> pl.DataFrame:
    """Reservoir x hour state: speed, trip production, and implied accumulation.

    The macroscopic relation P = n * v (production = accumulation x space-mean
    speed) is an identity, and trip completions O are proportional to P for a
    fixed mean trip length. Taxi trips are a sample of completions, so

        n  is proportional to  trips / speed

    up to a per-reservoir constant that folds in taxi market share and mean trip
    length. That constant is absorbed by the reservoir fixed effect everywhere
    it matters, which is what makes the fundamental diagram estimable here
    without ever observing vehicle density.
    """
    from ..models.reservoir import zone_to_reservoir

    cells = tlc.load_yellow(subdir=tlc.SUBDIR_CITY)
    z2r = zone_to_reservoir().select(pl.col("zone_id").alias("pu"), "reservoir")

    d = (
        cells.join(z2r, on="pu")
        .filter(pl.col("n_trips") >= 3)
        .group_by(["reservoir", "service_date", "hour"])
        .agg(
            [
                (pl.col("mean_miles") * pl.col("n_trips")).sum().alias("veh_miles"),
                (pl.col("mean_seconds") * pl.col("n_trips")).sum().alias("veh_seconds"),
                pl.col("n_trips").sum().alias("trips"),
            ]
        )
        .filter(pl.col("trips") >= min_trips)
        .with_columns((pl.col("veh_miles") / (pl.col("veh_seconds") / 3600)).alias("speed_mph"))
    )
    d = d.with_columns(
        [
            (pl.col("trips") / pl.col("speed_mph")).alias("accum_proxy"),
            (pl.col("veh_miles")).alias("production_proxy"),
            _period_index(pl.col("service_date")).alias("period"),
            pl.col("service_date").dt.weekday().alias("dow"),
        ]
    )
    wx = weather.load()
    if not wx.is_empty():
        d = d.join(wx.select(["service_date", "rain_day", "snow_day", "tmax_c"]),
                   on="service_date", how="left")
    return d.filter(pl.col("speed_mph").is_finite() & (pl.col("speed_mph") > 0))
