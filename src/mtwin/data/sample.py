"""Committed derived tables, so the analyses run without the 4 GB raw pull.

`make data` fetches 43 months from eight sources -- several hours and roughly
4 GB -- which puts every number in the README out of reach of anyone who has
just cloned the repo. `data/sample/` holds the aggregates the analyses actually
consume, tracked in git at about 7 MB, so the headline results can be checked
from a fresh clone.

These are *derived* tables, not a subsample of rows. Each one is the full-period
aggregate at the resolution its consumer already reduces to, so the numbers it
produces are identical to the raw pull rather than an approximation of it. The
size comes from dropping the dimension nothing downstream reads: CRZ entries are
summed over the twelve detection groups, bus segment speeds are aggregated to
segment x month. Where a column is a deterministic function of the others it is
recomputed on load instead of stored.

Selection is explicit rather than magic:

    MTWIN_USE_SAMPLE=1   force the sample tables
    MTWIN_USE_SAMPLE=0   force `data/raw/`, and fail if it is missing
    unset                use `data/raw/` if it exists, else the sample tables

The unset case falls back only when `data/raw/` is absent *entirely*. A partial
or in-progress pull therefore never gets silently topped up with committed
aggregates, which would produce a number that is neither one thing nor the other.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import polars as pl

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[3]
SAMPLE = ROOT / "data" / "sample"
RAW = ROOT / "data" / "raw"

_TRUTHY = {"1", "true", "yes", "on"}
_announced = False


def enabled() -> bool:
    """Whether to read the committed sample tables instead of `data/raw/`."""
    flag = os.environ.get("MTWIN_USE_SAMPLE")
    if flag is not None:
        return flag.strip().lower() in _TRUTHY
    if RAW.exists():
        return False
    _announce()
    return True


def _announce() -> None:
    """Say so once. A silent switch of data source is how wrong numbers travel."""
    global _announced
    if not _announced:
        _announced = True
        log.warning("data/raw/ is absent; reading committed sample tables from %s", SAMPLE)


def load(name: str) -> pl.DataFrame:
    path = SAMPLE / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It ships with the repo; regenerate it with "
            f"`uv run python -m src.mtwin.data.sample` from a full data/raw/ pull."
        )
    return pl.read_parquet(path)


# --- Export ------------------------------------------------------------------
# Run from a machine that has the full raw pull. Everything below reads
# data/raw/ directly (never the sample path) so an export cannot be built from
# a previous export.

# zstd at the top level costs a few seconds once and buys ~20% over the default,
# which matters when the file is committed rather than cached.
_WRITE = {"compression": "zstd", "compression_level": 22}


def _export_crz_blocks() -> Path:
    """CRZ entries summed over detection groups.

    `bunching_rd.block_profile` sums gantries within a day and `diversion`
    sums them within a month, so the detection-group dimension is discarded by
    every consumer before it is used. Summing it away here is lossless for them
    and cuts the table twelve-fold.
    """
    from . import crz_entries

    df = crz_entries.load()
    slim = (
        df.group_by(["toll_date", "day_of_week", "hour_of_day", "minute_of_hour", "vehicle_class"])
        .agg(
            [
                pl.col("crz_entries").sum(),
                pl.col("excluded_roadway_entries").sum(),
            ]
        )
        .with_columns(
            [
                pl.col("hour_of_day").cast(pl.Int8),
                pl.col("minute_of_hour").cast(pl.Int8),
                pl.col("crz_entries").cast(pl.Int32),
                pl.col("excluded_roadway_entries").cast(pl.Int32),
            ]
        )
        .sort(["toll_date", "vehicle_class", "hour_of_day", "minute_of_hour"])
    )
    return _write(slim, "crz_blocks")


def _export_reservoir_panel() -> Path:
    """Reservoir x date x hour state.

    `veh_miles` and `veh_seconds` are dropped: the panel already exposes the
    two quantities derived from them that anything reads (`speed_mph` and
    `production_proxy`, the latter being `veh_miles` under its modelling name).
    """
    from ..panel.build import reservoir_panel

    return _write(reservoir_panel().drop(["veh_miles", "veh_seconds"]), "reservoir_panel")


def _export_bus_segment_panel() -> Path:
    """Bus segment x month aggregate, before the design columns.

    The treatment interactions, event-time index and log speed are all
    deterministic functions of these columns, so they are recomputed by
    `panel.build` on load rather than stored twice. `borough` is carried so the
    `boroughs=` argument still selects.
    """
    from ..data import bus_speeds
    from ..panel.build import bus_segment_panel

    panel = bus_segment_panel(boroughs=("Manhattan", "Brooklyn", "Queens", "Bronx"))
    # grp collapses Brooklyn/Queens/Bronx into "outer", so the borough label has
    # to come back from the source frame.
    seg_borough = (
        bus_speeds.load(boroughs=("Manhattan", "Brooklyn", "Queens", "Bronx"))
        .select(["segment_id", "borough"])
        .unique(subset=["segment_id"])
    )
    base = (
        panel.select(["segment_id", "grp", "route_id", "period", "veh_miles", "veh_minutes", "n_trips"])
        .join(seg_borough, on="segment_id", how="left")
        .sort(["segment_id", "period"])
    )
    return _write(base, "bus_segment_panel")


def _export_zone_centroids() -> Path:
    """Taxi-zone centroids, cordon flags and polygon areas.

    The taxi-zone shapefile is a manual download that `network.zones` expects at
    a hard-coded path outside the repo, and it is what classifies zones into
    reservoirs and exposure groups. 263 rows of derived geometry removes that
    dependency for everything except a fresh reprojection.
    """
    import duckdb

    from ..network.zones import SHAPEFILE, zone_centroids

    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    areas = con.execute(
        f"""
        SELECT CAST(LocationID AS INTEGER) AS zone_id,
               ST_Area(geom) / 1e6 * 0.09290304 AS km2
        FROM ST_Read('{SHAPEFILE}')
        """
    ).pl()
    con.close()
    return _write(zone_centroids().join(areas, on="zone_id", how="left").sort("zone_id"), "zone_centroids")


def _write(df: pl.DataFrame, name: str) -> Path:
    SAMPLE.mkdir(parents=True, exist_ok=True)
    path = SAMPLE / f"{name}.parquet"
    df.write_parquet(path, **_WRITE)
    log.info("%-22s %8d rows  %6.2f MB", name, df.height, path.stat().st_size / 1e6)
    return path


def export() -> list[Path]:
    """Rebuild every sample table from `data/raw/`."""
    if enabled():
        raise RuntimeError("refusing to export while MTWIN_USE_SAMPLE is on; the source must be data/raw/")
    return [
        _export_zone_centroids(),
        _export_crz_blocks(),
        _export_reservoir_panel(),
        _export_bus_segment_panel(),
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    total = sum(p.stat().st_size for p in export())
    log.info("%-22s %8s  %6.2f MB", "TOTAL", "", total / 1e6)
