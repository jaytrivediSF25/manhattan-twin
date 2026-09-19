"""NYC TLC trip records -- OD-level travel times and the circuity measure.

Taxis are a *treated* probe: they pay the per-trip CRZ fee on top of the 2019
congestion surcharge, so the mix of observed trips shifts discontinuously on
the treatment date. That rules them out as a standalone speed index, but they
supply something bus data cannot -- origin-destination structure, and metered
trip distance.

Metered distance is what makes rerouting directly measurable. Circuity
    c_od,t = mean_miles_od,t / network_shortest_path_od
separates two observationally similar stories, since
    delta log(travel_time) = delta log(circuity) - delta log(speed).
Rerouting raises circuity and speed together; genuine decongestion leaves
circuity flat while speed rises.

THE CELL SCHEMA BELOW IS LOCKED. Changing it means re-downloading every month,
so it deliberately carries more moments than the headline analysis needs
(p25 and median alongside means, plus fee columns for mix diagnostics).
"""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import duckdb
import httpx
import polars as pl

from . import registry as reg
from .socrata import RAW, month_windows

log = logging.getLogger(__name__)

CDN = "https://d37ci6vzurychx.cloudfront.net/trip-data"
SUBDIR_YELLOW = "tlc_yellow_cells"
SUBDIR_CITY = "tlc_yellow_cells_city"
SUBDIR_FHV = "tlc_fhv_cells"
LOOKUP = RAW / "taxi_zone_lookup.csv"

# Scratch space for raw monthly parquet. FHV files are ~500 MB and are deleted
# immediately after aggregation; yellow files are ~65 MB and may be retained.
SCRATCH = RAW / "_tlc_scratch"

MANHATTAN_ZONES = [
    4, 12, 13, 24, 41, 42, 43, 45, 48, 50, 68, 74, 75, 79, 87, 88, 90, 100,
    103, 104, 105, 107, 113, 114, 116, 120, 125, 127, 128, 137, 140, 141, 142,
    143, 144, 148, 151, 152, 153, 158, 161, 162, 163, 164, 166, 170, 186, 194,
    202, 209, 211, 224, 229, 230, 231, 232, 233, 234, 236, 237, 238, 239, 243,
    244, 246, 249, 261, 262, 263,
]

# Data-hygiene thresholds. Each removes a specific known contaminant rather
# than being a generic outlier rule:
#   store_and_fwd_flag='Y'  -> delayed transmission, unreliable timestamps
#   RatecodeID != 1         -> airport flat fares, negotiated fares, group rides
#   duration/distance/speed -> impossible or meter-fault records
MIN_SECONDS, MAX_SECONDS = 90, 3600
MIN_MILES, MAX_MILES = 0.3, 15.0
MIN_MPH, MAX_MPH = 1.0, 45.0


def _month_url(service: str, when: date) -> str:
    return f"{CDN}/{service}_tripdata_{when:%Y-%m}.parquet"


def _download(url: str, dest: Path, max_retries: int = 4) -> bool:
    """Stream a monthly parquet to disk. Returns False if not yet published.

    FHV files are ~500 MB, long enough that a transient read timeout part-way
    through is routine. Without a retry one dropped connection aborts the whole
    multi-month pull, so failures back off and resume, and a partial file is
    deleted rather than left to look like a completed download.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    delay = 5.0
    for attempt in range(max_retries):
        try:
            with httpx.stream("GET", url, timeout=600.0, follow_redirects=True) as r:
                if r.status_code in (403, 404):
                    log.info("tlc: %s not published yet", url.rsplit("/", 1)[-1])
                    return False
                r.raise_for_status()
                with dest.open("wb") as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
            return True
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            dest.unlink(missing_ok=True)
            log.warning("tlc download %s (attempt %d/%d): %s",
                        url.rsplit("/", 1)[-1], attempt + 1, max_retries, exc)
            time.sleep(delay)
            delay *= 2
    log.error("tlc: giving up on %s", url.rsplit("/", 1)[-1])
    return False


def _aggregate_sql(path: Path, pickup: str, dropoff: str, distance_col: str,
                   zones_filter: list[int] | None = None) -> str:
    """Reduce a month of trips to the locked (PU, DO, date, hour) cell schema.

    `zones_filter=None` keeps every NYC zone, which is what the exposure design
    needs: outer-borough OD pairs are the never-taker control, and restricting
    to Manhattan would discard them.
    """
    if zones_filter:
        zone_clause = (
            f'AND "PULocationID" IN ({",".join(str(z) for z in zones_filter)})\n'
            f'          AND "DOLocationID" IN ({",".join(str(z) for z in zones_filter)})'
        )
    else:
        zone_clause = ""
    return f"""
    WITH trips AS (
        SELECT
            CAST("PULocationID" AS INTEGER)             AS pu,
            CAST("DOLocationID" AS INTEGER)             AS do_,
            CAST({pickup} AS DATE)                      AS service_date,
            EXTRACT(hour FROM {pickup})                 AS hour,
            EXTRACT(dow  FROM {pickup})                 AS dow,
            CAST({distance_col} AS DOUBLE)              AS miles,
            date_diff('second', {pickup}, {dropoff})    AS seconds
        FROM read_parquet('{path}')
        WHERE "PULocationID" <> "DOLocationID"
          {zone_clause}
          AND {dropoff} > {pickup}
    )
    SELECT
        pu, do_, service_date, hour, any_value(dow) AS dow,
        count(*)                                         AS n_trips,
        median(miles / (seconds / 3600.0))               AS median_mph,
        quantile_cont(miles / (seconds / 3600.0), 0.25)  AS p25_mph,
        avg(miles / (seconds / 3600.0))                  AS mean_mph,
        avg(miles)                                       AS mean_miles,
        median(miles)                                    AS median_miles,
        avg(seconds)                                     AS mean_seconds,
        median(seconds)                                  AS median_seconds
    FROM trips
    WHERE seconds BETWEEN {MIN_SECONDS} AND {MAX_SECONDS}
      AND miles   BETWEEN {MIN_MILES} AND {MAX_MILES}
      AND (miles / (seconds / 3600.0)) BETWEEN {MIN_MPH} AND {MAX_MPH}
    GROUP BY pu, do_, service_date, hour
    """


def _hygiene_filter(path: Path) -> str:
    """Columns present only in yellow records; applied as a pre-filter view."""
    return f"""
    CREATE OR REPLACE TEMP VIEW clean AS
    SELECT * FROM read_parquet('{path}')
    WHERE (store_and_fwd_flag IS NULL OR store_and_fwd_flag <> 'Y')
      AND (RatecodeID IS NULL OR CAST(RatecodeID AS INTEGER) = 1)
    """


def pull_yellow(start: date = reg.DATA_START, end: date = reg.DATA_END,
                keep_raw: bool = True, zones_filter: list[int] | None = None,
                subdir: str = SUBDIR_YELLOW) -> list[Path]:
    """Download, aggregate, and cache yellow-taxi cells month by month."""
    out_dir = RAW / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    con = duckdb.connect()

    for win_start, _ in month_windows(start, end):
        out = out_dir / f"{win_start:%Y-%m}.parquet"
        written.append(out)
        if out.exists():
            log.info("tlc yellow %s: cached", f"{win_start:%Y-%m}")
            continue

        raw = SCRATCH / f"yellow_{win_start:%Y-%m}.parquet"
        if not raw.exists() and not _download(_month_url("yellow", win_start), raw):
            continue

        con.execute(_hygiene_filter(raw))
        sql = _aggregate_sql(
            raw, "tpep_pickup_datetime", "tpep_dropoff_datetime", "trip_distance",
            zones_filter=zones_filter,
        ).replace(f"read_parquet('{raw}')", "clean")
        df = con.execute(sql).pl()
        df.write_parquet(out)
        log.info("tlc yellow %s: %d cells -> %s/%s", f"{win_start:%Y-%m}", df.height, subdir, out.name)

        if not keep_raw:
            raw.unlink(missing_ok=True)
    con.close()
    return written


def load_yellow(subdir: str = SUBDIR_YELLOW) -> pl.DataFrame:
    files = sorted((RAW / subdir).glob("*.parquet"))
    frames = [pl.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def pull_fhv(start: date = reg.DATA_START, end: date = reg.DATA_END) -> list[Path]:
    """High-volume FHV (Uber/Lyft) cells, streamed and discarded month by month.

    FHV files are ~500 MB each, so ~21 GB for the full span -- they are deleted
    straight after aggregation rather than cached. The payoff is outer-borough
    coverage: yellow taxis barely leave Manhattan, which left the never-taker
    group in the taxi exposure design at 9-27 OD pairs per month from a churning
    set. FHV covers the outer boroughs properly.

    FHV records carry their own column names (pickup_datetime, trip_miles) and
    have no RatecodeID or store_and_fwd_flag, so the yellow hygiene filter does
    not apply; shared rides are dropped instead, since pooled detours corrupt
    the implied speed.
    """
    out_dir = RAW / SUBDIR_FHV
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    con = duckdb.connect()

    for win_start, _ in month_windows(start, end):
        out = out_dir / f"{win_start:%Y-%m}.parquet"
        written.append(out)
        if out.exists():
            log.info("tlc fhv %s: cached", f"{win_start:%Y-%m}")
            continue

        raw = SCRATCH / f"fhvhv_{win_start:%Y-%m}.parquet"
        if not raw.exists() and not _download(_month_url("fhvhv", win_start), raw):
            continue
        try:
            con.execute(
                f"""CREATE OR REPLACE TEMP VIEW clean AS
                    SELECT * FROM read_parquet('{raw}')
                    WHERE shared_match_flag IS NULL OR shared_match_flag <> 'Y'"""
            )
            sql = _aggregate_sql(
                raw, "pickup_datetime", "dropoff_datetime", "trip_miles", zones_filter=None
            ).replace(f"read_parquet('{raw}')", "clean")
            df = con.execute(sql).pl()
            df.write_parquet(out)
            log.info("tlc fhv %s: %d cells -> %s", f"{win_start:%Y-%m}", df.height, out.name)
        finally:
            # Always remove the raw file, even on failure: 500 MB per month
            # exhausts the disk within a few iterations otherwise.
            raw.unlink(missing_ok=True)
    con.close()
    return written


def load_fhv() -> pl.DataFrame:
    return load_yellow(subdir=SUBDIR_FHV)


def pull(start: date = reg.DATA_START, end: date = reg.DATA_END) -> None:
    pull_yellow(start, end)
