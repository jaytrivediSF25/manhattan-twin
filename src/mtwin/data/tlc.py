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
import shutil
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
SPILL = SCRATCH / "_duckdb_spill"

# Measured against the CDN from this machine: ~1.2 MB/s sustained, so a 460 MB
# month takes ~8 minutes. Six concurrent range requests moved no more total data
# than one stream, so the ceiling is the local uplink, not CloudFront throttling
# -- parallel fetching buys nothing here and surviving a stall is what matters.
CONNECT_TIMEOUT = 30.0
# A healthy transfer delivers a 1 MB chunk roughly every second, with jitter
# observed to peak near 11 s. The former blanket 600 s read timeout could not
# tell a dead socket from a slow one until ten minutes had been thrown away;
# 90 s sits far above the observed jitter and far below that, and because
# downloads now resume, noticing a stall early costs nothing.
READ_TIMEOUT = 90.0

# FHV cells land at ~220 MB per month, not the ~50 MB a Manhattan-only cut would
# give: zones_filter=None keeps all ~260 zones, so the OD grid is far larger.
# The full span is therefore ~9 GB of output on a disk with ~12 GB free, close
# enough that the loop stops cleanly while a margin remains rather than dying
# part-way through a write and leaving a truncated file that looks cached.
MIN_FREE_BYTES = 2 << 30

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


def _download(url: str, dest: Path, max_retries: int = 6) -> bool:
    """Stream a monthly parquet to disk, resuming a partial file. False = absent.

    The CDN advertises `Accept-Ranges: bytes` and answers ranged GETs with 206,
    so a dropped connection resumes from what is already on disk. That is the
    whole fix for the FHV pull: at ~1.2 MB/s a month is an ~8-minute transfer,
    and the previous wrapper deleted the partial file and restarted from byte 0,
    so a timeout 400 MB in discarded six minutes of work and re-rolled the same
    dice. Four such restarts burned half an hour and still failed.

    Content-Length is the completion test rather than "the stream ended", since
    a truncated body otherwise looks like a finished download and only surfaces
    later as an unreadable parquet.
    """
    name = url.rsplit("/", 1)[-1]
    dest.parent.mkdir(parents=True, exist_ok=True)
    timeout = httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT)
    total: int | None = None
    stalled = 0
    delay = 5.0

    # Bounded independently of `stalled` so that a connection dribbling a few
    # bytes per attempt cannot loop forever while still counting as progress.
    for _ in range(max_retries * 10):
        if stalled >= max_retries:
            break
        have = dest.stat().st_size if dest.exists() else 0
        try:
            if total is None:
                head = httpx.head(url, timeout=timeout, follow_redirects=True)
                if head.status_code in (403, 404):
                    log.info("tlc: %s not published yet", name)
                    return False
                head.raise_for_status()
                total = int(head.headers["content-length"])

            # A leftover larger than the published file is stale, not partial.
            if have > total:
                have = 0
            if have == total:
                return True

            headers = {"Range": f"bytes={have}-"} if have else {}
            mode = "ab" if have else "wb"
            with httpx.stream("GET", url, headers=headers, timeout=timeout,
                              follow_redirects=True) as r:
                if r.status_code in (403, 404):
                    log.info("tlc: %s not published yet", name)
                    return False
                # A server that ignores Range replies 200 with the whole body;
                # appending that to a partial file would silently corrupt it.
                if have and r.status_code != 206:
                    have, mode = 0, "wb"
                r.raise_for_status()
                with dest.open(mode) as fh:
                    for chunk in r.iter_bytes(1 << 20):
                        fh.write(chunk)
        except (httpx.RequestError, httpx.HTTPStatusError, KeyError, ValueError, OSError) as exc:
            # The partial file is deliberately kept -- it is what the next
            # attempt resumes from.
            log.warning("tlc download %s (%.0f/%.0f MB): %s",
                        name, have / 1e6, (total or 0) / 1e6, exc)

        now = dest.stat().st_size if dest.exists() else 0
        if total is not None and now == total:
            return True
        if now > have:
            # An attempt that moved bytes is progress, not a failure. On a link
            # this slow a month can legitimately need several resumes, and
            # charging them to the retry budget would abandon a transfer that is
            # already 90% done.
            stalled, delay = 0, 5.0
        else:
            stalled += 1
            time.sleep(delay)
            delay = min(delay * 2, 120.0)

    log.error("tlc: giving up on %s", name)
    return False


def _duck() -> duckdb.DuckDBPyConnection:
    """A DuckDB connection sized for a 16 GB laptop, not for the whole machine.

    DuckDB defaults memory_limit to ~80% of RAM (12.7 GiB here) and an in-memory
    database has no temp_directory, so a heavy month has nowhere to spill: the
    OS kills the process outright, which no `except` can catch and which takes
    every remaining month with it. A hard limit with a real spill directory
    converts that into a merely slow month. Threads are cut to the physical core
    count because each one holds its own partition of the group-by hash table,
    so thread count multiplies peak memory more than it buys throughput on an
    aggregation this short (~7 s per month, against ~8 minutes of download).
    """
    SPILL.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute(f"SET temp_directory='{SPILL}'")
    return con


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
    con = _duck()

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
    con = _duck()
    done = failed = 0

    try:
        for win_start, _ in month_windows(start, end):
            stamp = f"{win_start:%Y-%m}"
            out = out_dir / f"{stamp}.parquet"
            written.append(out)
            if out.exists():
                log.info("tlc fhv %s: cached", stamp)
                continue

            # Checked per month rather than once: a raw file and a month of
            # cells together need ~700 MB, and running the disk to zero mid-write
            # is the one failure this loop cannot recover from.
            free = shutil.disk_usage(out_dir).free
            if free < MIN_FREE_BYTES:
                log.error("tlc fhv %s: only %.1f GB free, stopping before the "
                          "disk fills", stamp, free / 1e9)
                break

            raw = SCRATCH / f"fhvhv_{stamp}.parquet"
            try:
                if not _download(_month_url("fhvhv", win_start), raw):
                    failed += 1
                    continue
                con.execute(
                    f"""CREATE OR REPLACE TEMP VIEW clean AS
                        SELECT * FROM read_parquet('{raw}')
                        WHERE shared_match_flag IS NULL OR shared_match_flag <> 'Y'"""
                )
                sql = _aggregate_sql(
                    raw, "pickup_datetime", "dropoff_datetime", "trip_miles",
                    zones_filter=None,
                ).replace(f"read_parquet('{raw}')", "clean")
                # COPY streams straight to disk. Going through .pl() instead
                # materialised the whole ~6 M-row result as Arrow and again as
                # Polars on top of DuckDB's own copy, which is the largest
                # avoidable memory spike in the loop. The temp name plus rename
                # is what makes `out.exists()` above a trustworthy cache test:
                # a killed write can never leave a truncated file under the real
                # name for a later run to skip over.
                # Level 15 over the default buys only ~5% (197 -> 189 MB), but
                # disk is the binding constraint on this span while the CPU is
                # idle through the ~8-minute download, so ~13 s of extra
                # compression per month is free in wall-clock terms.
                tmp = out.with_suffix(".parquet.part")
                con.execute(
                    f"COPY ({sql}) TO '{tmp}' "
                    "(FORMAT PARQUET, COMPRESSION ZSTD, COMPRESSION_LEVEL 15)"
                )
                n = con.execute(f"SELECT count(*) FROM read_parquet('{tmp}')").fetchone()[0]
                tmp.replace(out)
                done += 1
                log.info("tlc fhv %s: %d cells -> %s (%.0f MB, %.1f GB free)",
                         stamp, n, out.name, out.stat().st_size / 1e6,
                         shutil.disk_usage(out_dir).free / 1e9)
            except (duckdb.Error, OSError) as exc:
                # One bad month must not abort the remaining span, matching the
                # Socrata puller. These two cover what the aggregation can
                # actually raise -- a corrupt or truncated parquet, an
                # out-of-memory spill failure, a full disk -- while a genuine
                # bug still surfaces as a crash. No file is written, so a re-run
                # retries this month instead of treating the gap as settled.
                failed += 1
                out.with_suffix(".parquet.part").unlink(missing_ok=True)
                log.warning("tlc fhv %s: FAILED, skipping (%s: %s)",
                            stamp, type(exc).__name__, exc)
            finally:
                # Always remove the raw file, even on failure: 500 MB per month
                # exhausts the disk within a few iterations otherwise.
                raw.unlink(missing_ok=True)
    finally:
        con.close()
        shutil.rmtree(SPILL, ignore_errors=True)

    log.info("tlc fhv: %d months written, %d skipped", done, failed)
    return written


def load_fhv() -> pl.DataFrame:
    return load_yellow(subdir=SUBDIR_FHV)


def pull(start: date = reg.DATA_START, end: date = reg.DATA_END) -> None:
    pull_yellow(start, end)
