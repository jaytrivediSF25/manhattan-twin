"""Bunching / regression-discontinuity-in-time at the toll threshold.

The congestion toll steps down from the peak rate to the overnight rate at
exactly 21:00, every day of the week. Entry counts are published in 10-minute
blocks, so a driver who delays entry to catch the cheaper rate shows up as
missing mass just before 21:00 and excess mass just after.

This is the project's de-risking result. It uses only post-period data, needs
no twin, no control group and no parallel-trends assumption, and it answers a
prior question to everything else: do drivers visibly respond to this price at
all? If nothing bunches at a $6.75 step with a sharp clock boundary, the
premise that there is a measurable behavioural response is in trouble.

Empirically verified schedule (from the data, not assumed):
    weekdays  peak 05:00-20:59, overnight 21:00-04:59
    weekends  peak 09:00-20:59, overnight 21:00-08:59
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

# The toll step common to every day of the week.
THRESHOLD_MIN = 21 * 60
BLOCK = 10  # minutes per published block

# Blocks inside the donut are excluded when fitting the counterfactual, since
# they are exactly where the behavioural response lives.
DONUT_BLOCKS = 6  # +/- 60 minutes


@dataclass
class RDResult:
    threshold_min: int
    jump: float                 # discontinuity in entries per 10-min block
    jump_pct: float             # jump relative to the left-limit level
    left_limit: float
    right_limit: float
    placebo_p: float            # share of placebo thresholds with a larger |jump_pct|
    bandwidth_min: int
    grid_min: np.ndarray
    observed: np.ndarray
    fitted: np.ndarray


def block_profile(
    df: pl.DataFrame,
    *,
    vehicle_class: str | None = "1 - Cars, Pickups and Vans",
    weekdays_only: bool = True,
    detection_groups: list[str] | None = None,
) -> pl.DataFrame:
    """Mean entries per 10-minute block of the day, averaged over days."""
    d = df
    if vehicle_class:
        d = d.filter(pl.col("vehicle_class") == vehicle_class)
    if weekdays_only:
        d = d.filter(~pl.col("day_of_week").is_in(["Saturday", "Sunday"]))
    if detection_groups:
        d = d.filter(pl.col("detection_group").is_in(detection_groups))

    d = d.with_columns(
        (pl.col("hour_of_day").cast(pl.Int32) * 60 + pl.col("minute_of_hour").cast(pl.Int32))
        .alias("minute_of_day")
    )
    # Sum across gantries within a day, then average across days, so the result
    # is "entries per 10-minute block on a typical day".
    per_day = d.group_by(["toll_date", "minute_of_day"]).agg(
        pl.col("crz_entries").sum().alias("entries")
    )
    return (
        per_day.group_by("minute_of_day")
        .agg(pl.col("entries").mean().alias("entries"))
        .sort("minute_of_day")
    )


def _local_linear_rd(
    minutes: np.ndarray, entries: np.ndarray, threshold: int, bandwidth: int
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit separate linear trends either side of the threshold.

    A global polynomial cannot track the steep evening decay in entries and
    leaves systematic bias on both sides of the cut, so the counterfactual is
    estimated locally instead. The estimand is the discontinuity in the fitted
    level at the threshold.
    """
    rel = minutes - threshold
    win = np.abs(rel) <= bandwidth
    left = win & (rel < 0)
    right = win & (rel >= 0)
    if left.sum() < 3 or right.sum() < 3:
        raise ValueError("insufficient points on one side of the threshold")

    bl = np.polyfit(rel[left] / 60.0, entries[left], 1)
    br = np.polyfit(rel[right] / 60.0, entries[right], 1)
    left_limit = float(np.polyval(bl, 0.0))
    right_limit = float(np.polyval(br, 0.0))

    fitted = np.full_like(entries, np.nan, dtype=float)
    fitted[left] = np.polyval(bl, rel[left] / 60.0)
    fitted[right] = np.polyval(br, rel[right] / 60.0)
    return left_limit, right_limit, fitted, win


def estimate(
    profile: pl.DataFrame,
    threshold: int = THRESHOLD_MIN,
    *,
    bandwidth: int = 90,
    n_placebo: int = 200,
) -> RDResult:
    """Estimate the discontinuity in entry rate at the toll threshold."""
    minutes = profile["minute_of_day"].to_numpy().astype(float)
    entries = profile["entries"].to_numpy().astype(float)

    left_limit, right_limit, fitted, _ = _local_linear_rd(minutes, entries, threshold, bandwidth)
    jump = right_limit - left_limit
    jump_pct = jump / left_limit if left_limit else float("nan")

    # Placebo inference: the same local-linear jump estimated at every other
    # time of day. A real toll response should sit in the tail of that
    # distribution; a fitting artefact will not.
    placebo: list[float] = []
    for cand in range(int(minutes.min()) + bandwidth, int(minutes.max()) - bandwidth, BLOCK):
        if abs(cand - threshold) < 120:
            continue
        try:
            l, r, _, _ = _local_linear_rd(minutes, entries, cand, bandwidth)
        except ValueError:
            continue
        if l:
            placebo.append(abs((r - l) / l))
    p = float(np.mean([x >= abs(jump_pct) for x in placebo])) if placebo else float("nan")

    return RDResult(
        threshold_min=threshold,
        jump=jump,
        jump_pct=jump_pct,
        left_limit=left_limit,
        right_limit=right_limit,
        placebo_p=p,
        bandwidth_min=bandwidth,
        grid_min=minutes,
        observed=entries,
        fitted=fitted,
    )


# --- The full toll schedule --------------------------------------------------
# Verified against the data, not assumed. Three clock thresholds exist, and only
# one of them is a price cut:
#
#   21:00  every day      $9.00 -> $2.25   price FALLS
#   05:00  weekdays       $2.25 -> $9.00   price RISES
#   09:00  weekends       $2.25 -> $9.00   price RISES
#
# The sign prediction is what makes this a test rather than a description. At a
# price cut, drivers wait and entries jump up after the threshold. At a price
# rise they go early, so entries are pulled forward and the jump is NEGATIVE.
# A fitting artefact at an hour boundary has no reason to flip sign with the
# direction of the price change.

PEAK_TOLL, OVERNIGHT_TOLL = 9.00, 2.25

THRESHOLDS = {
    "21:00 price cut (all days)": {"minute": 21 * 60, "weekdays": None, "direction": -1},
    "05:00 price rise (weekdays)": {"minute": 5 * 60, "weekdays": True, "direction": +1},
    "09:00 price rise (weekends)": {"minute": 9 * 60, "weekdays": False, "direction": +1},
}


def price_semi_elasticity(jump_pct: float, direction: int) -> float:
    """Entry response per log point of toll, signed so cuts and rises compare.

    direction = -1 for a price cut, +1 for a rise. Returns d(log entries) /
    d(log toll), which is negative for a normal demand response either way.
    """
    dlog_price = np.log(OVERNIGHT_TOLL / PEAK_TOLL) * (-direction)
    return jump_pct / dlog_price if dlog_price else float("nan")


def threshold_profiles(
    df: pl.DataFrame,
    *,
    vehicle_class: str | None = "1 - Cars, Pickups and Vans",
) -> dict[str, pl.DataFrame]:
    """One block profile per threshold, on the days that threshold applies to.

    The 05:00 and 09:00 steps exist only on weekdays and weekends respectively,
    so each has to be estimated on its own day set. Shared with the figure code
    so the panel drawn and the number reported come from the same profile.
    """
    out: dict[str, pl.DataFrame] = {}
    for name, spec in THRESHOLDS.items():
        wd = spec["weekdays"]
        sub = df if wd is None else df.filter(
            ~pl.col("day_of_week").is_in(["Saturday", "Sunday"]) if wd
            else pl.col("day_of_week").is_in(["Saturday", "Sunday"])
        )
        out[name] = block_profile(sub, vehicle_class=vehicle_class, weekdays_only=False)
    return out


def all_thresholds(
    df: pl.DataFrame,
    *,
    vehicle_class: str | None = "1 - Cars, Pickups and Vans",
    bandwidth: int = 90,
) -> pl.DataFrame:
    """Estimate every toll threshold and report the implied elasticity."""
    rows = []
    profiles = threshold_profiles(df, vehicle_class=vehicle_class)
    for name, spec in THRESHOLDS.items():
        try:
            r = estimate(profiles[name], threshold=spec["minute"], bandwidth=bandwidth)
        except ValueError:
            continue
        rows.append(
            {
                "threshold": name,
                "price_direction": "cut" if spec["direction"] < 0 else "rise",
                "jump_pct": r.jump_pct,
                "jump_per_block": r.jump,
                "placebo_p": r.placebo_p,
                "semi_elasticity": price_semi_elasticity(r.jump_pct, spec["direction"]),
                "sign_as_predicted": (r.jump_pct > 0) == (spec["direction"] < 0),
            }
        )
    return pl.DataFrame(rows)


def habituation(
    df: pl.DataFrame,
    *,
    vehicle_class: str | None = "1 - Cars, Pickups and Vans",
    bandwidth: int = 90,
) -> pl.DataFrame:
    """Re-estimate the 21:00 jump quarter by quarter.

    Nineteen months of 10-minute data is long enough to ask whether the timing
    response fades as drivers habituate to the toll -- a question shorter panels
    could not address.
    """
    d = df.with_columns(
        (pl.col("toll_date").dt.year().cast(pl.Utf8) + "Q"
         + pl.col("toll_date").dt.quarter().cast(pl.Utf8)).alias("quarter")
    )
    rows = []
    for q in sorted(d["quarter"].unique().to_list()):
        sub = d.filter(pl.col("quarter") == q)
        prof = block_profile(sub, vehicle_class=vehicle_class, weekdays_only=False)
        try:
            r = estimate(prof, bandwidth=bandwidth, n_placebo=60)
        except ValueError:
            continue
        rows.append({"quarter": q, "jump_pct": r.jump_pct, "jump_per_block": r.jump,
                     "placebo_p": r.placebo_p})
    return pl.DataFrame(rows)
