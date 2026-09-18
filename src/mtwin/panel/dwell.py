"""Separating bus running time from dwell time.

A bus segment's travel time is the sum of two things that respond very
differently to a traffic policy:

    travel_time = dwell + distance / running_speed

Dwell -- time stopped at kerbs, plus acceleration and deceleration around each
stop -- does not care how congested the road is. Only the running component
does. Blending them attenuates any traffic effect measured on bus speed, which
is why in-cordon bus speed reads ~5.3 mph while taxis on the same streets read
~6.5 mph.

The split is identified off the cross-segment relationship between travel time
and distance. Pooling Manhattan segments gives

    travel_time = 2.51 min + 6.61 min/mile

so the intercept is fixed overhead per segment and the slope implies a running
speed near 9 mph. On a typical 0.7-mile segment that makes dwell roughly a
third of total time, and a measured effect on blended speed understates the
effect on running speed by about that factor.

The estimate is deliberately simple and reported with a sensitivity sweep
rather than presented as exact: what matters for the conclusion is the order of
the attenuation, not its third digit.
"""

from __future__ import annotations

import numpy as np
import polars as pl

# Segments shorter than this cannot support the subtraction: fixed overhead is
# most of their travel time and the residual running time is mostly noise.
MIN_DISTANCE_MI = 0.25
# Running speed must stay physically plausible after the subtraction.
MAX_RUNNING_MPH = 45.0


def estimate_overhead(df: pl.DataFrame, by: str | None = None) -> float | pl.DataFrame:
    """Fixed per-segment overhead in minutes, from travel time against distance.

    Fitted on the faster half of observations so the slope reflects running
    conditions rather than heavy congestion, which would otherwise be absorbed
    into the intercept and overstate dwell.
    """
    d = df.filter(pl.col("road_distance") > 0)
    fast = d.filter(pl.col("average_road_speed") >= pl.col("average_road_speed").median())

    def _fit(frame: pl.DataFrame) -> float:
        x = frame["road_distance"].to_numpy()
        y = frame["average_travel_time"].to_numpy()
        if len(x) < 50:
            return float("nan")
        _slope, intercept = np.polyfit(x, y, 1)
        return float(max(intercept, 0.0))

    if by is None:
        return _fit(fast)
    rows = [{by: k, "overhead_min": _fit(fast.filter(pl.col(by) == k))}
            for k in sorted(fast[by].unique().to_list())]
    return pl.DataFrame(rows)


def add_running_speed(df: pl.DataFrame, overhead_min: float) -> pl.DataFrame:
    """Attach running-only speed, with dwell subtracted from travel time."""
    return (
        df.filter(pl.col("road_distance") >= MIN_DISTANCE_MI)
        .with_columns(
            (pl.col("average_travel_time") - overhead_min).alias("running_time")
        )
        .filter(pl.col("running_time") > 0.1)
        .with_columns(
            (pl.col("road_distance") / (pl.col("running_time") / 60)).alias("running_mph")
        )
        .filter(pl.col("running_mph").is_between(1.0, MAX_RUNNING_MPH))
    )


def dwell_share(df: pl.DataFrame, overhead_min: float) -> float:
    """Fraction of total travel time that is fixed overhead."""
    d = df.filter(pl.col("road_distance") >= MIN_DISTANCE_MI)
    return float(overhead_min / d["average_travel_time"].mean())


def attenuation_factor(share: float) -> float:
    """How much a blended-speed effect understates the running-speed effect.

    With dwell constant over time, a proportional change in total time maps to
    a larger proportional change in running time by 1 / (1 - dwell share).
    """
    return 1.0 / max(1.0 - share, 1e-6)
