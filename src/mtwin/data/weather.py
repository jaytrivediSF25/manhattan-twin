"""NOAA Central Park daily weather -- a control fed into the model.

Anything not modelled is attributed to behaviour by construction, so weather
enters as an input rather than being left to contaminate the residual.
Station USW00094728 is NY CITY CENTRAL PARK in GHCN-Daily.
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx
import polars as pl

from .socrata import RAW

log = logging.getLogger(__name__)

STATION = "USW00094728"
URL = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{STATION}.csv"
OUT = RAW / "weather_central_park.parquet"

KEEP = ["DATE", "PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND"]


def pull(force: bool = False) -> Path:
    if OUT.exists() and not force:
        log.info("weather: cached")
        return OUT
    with httpx.Client(timeout=300.0, follow_redirects=True) as c:
        r = c.get(URL)
        r.raise_for_status()
        OUT.write_bytes(r.content)  # staged as CSV bytes, rewritten below
    df = pl.read_csv(OUT, infer_schema_length=10000)
    cols = [c for c in KEEP if c in df.columns]
    df = df.select(cols).with_columns(pl.col("DATE").str.to_date(strict=False))
    df.write_parquet(OUT)
    log.info("weather: %d days -> %s", df.height, OUT.name)
    return OUT


def load() -> pl.DataFrame:
    if not OUT.exists():
        return pl.DataFrame()
    df = pl.read_parquet(OUT)
    # GHCN columns arrive as strings when a station has sparse coverage.
    # GHCN pads its fixed-width fields, and polars casts "     8" to null rather
    # than 8, so whitespace must be stripped before the cast or every reading
    # silently disappears.
    num = [c for c in ("PRCP", "SNOW", "SNWD", "TMAX", "TMIN", "AWND") if c in df.columns]
    df = df.with_columns(
        [pl.col(c).cast(pl.Utf8).str.strip_chars().cast(pl.Float64, strict=False) for c in num]
    )
    # GHCN reports tenths of degrees C and tenths of mm.
    return df.with_columns(
        [
            (pl.col("TMAX") / 10).alias("tmax_c"),
            (pl.col("TMIN") / 10).alias("tmin_c"),
            (pl.col("PRCP") / 10).alias("prcp_mm"),
            pl.col("DATE").alias("service_date"),
        ]
    ).with_columns(
        [
            (pl.col("prcp_mm") > 2.5).alias("rain_day"),
            (pl.col("SNOW").fill_null(0) > 0).alias("snow_day"),
        ]
    )
