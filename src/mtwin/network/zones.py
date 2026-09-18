"""Taxi-zone geography and congestion-zone classification.

Cordon membership is a real classification task rather than a lookup: the
charge boundary is 60th Street, and several taxi zones straddle it. Zones are
assigned by centroid latitude against the 60th Street line, and the zones whose
centroids fall near that line are kept as an explicit robustness set, because
an estimate that depends on how they are coded is not a robust estimate.

Two coordinate caveats are baked in here. The shapefile is in EPSG:2263 (NY
State Plane, feet) and must be reprojected; and after reprojection to EPSG:4326
DuckDB returns axis order (lat, lon), so latitude is ST_X, not ST_Y.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import polars as pl

SHAPEFILE = Path("/tmp/tz/taxi_zones/taxi_zones.shp")

# 60th Street, the charge boundary.
CORDON_LAT = 40.7648
# Zones whose centroid falls within this many degrees of the line are reported
# separately; roughly +/- 350 m.
STRADDLE_BAND = 0.003

# Reachable only from Queens, so not a congestion-zone entry despite latitude.
NOT_CORDON = {202}  # Roosevelt Island


def zone_centroids(shapefile: Path = SHAPEFILE) -> pl.DataFrame:
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    df = con.execute(
        f"""
        SELECT CAST(LocationID AS INTEGER) AS zone_id, zone, borough,
               ST_X(ST_Centroid(ST_Transform(geom,'EPSG:2263','EPSG:4326'))) AS lat,
               ST_Y(ST_Centroid(ST_Transform(geom,'EPSG:2263','EPSG:4326'))) AS lon
        FROM ST_Read('{shapefile}')
        """
    ).pl()
    con.close()
    return df.with_columns(
        [
            (
                (pl.col("borough") == "Manhattan")
                & (pl.col("lat") < CORDON_LAT)
                & ~pl.col("zone_id").is_in(list(NOT_CORDON))
            ).alias("in_crz"),
            ((pl.col("lat") - CORDON_LAT).abs() < STRADDLE_BAND).alias("straddles"),
        ]
    )


def crz_zone_ids() -> list[int]:
    df = zone_centroids()
    return sorted(df.filter(pl.col("in_crz"))["zone_id"].to_list())


def straddle_zone_ids() -> list[int]:
    df = zone_centroids()
    return sorted(df.filter(pl.col("straddles") & (pl.col("borough") == "Manhattan"))["zone_id"].to_list())
