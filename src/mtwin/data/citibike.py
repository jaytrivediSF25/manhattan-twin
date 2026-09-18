"""Citi Bike trips, aggregated to daily counts inside and outside the cordon.

Included for completeness in the mode-shift accounting, with the expectation
that it is small: bike trips displace a far smaller share of vehicle trips than
transit does, so this enters the decomposition as a footnote rather than a
headline term.

Monthly archives are large relative to what is needed, so each is reduced to
daily counts and the raw file deleted.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date

import httpx
import polars as pl

from .socrata import RAW, month_windows

log = logging.getLogger(__name__)

BASE = "https://s3.amazonaws.com/tripdata"
SUBDIR = "citibike_daily"
CORDON_LAT = 40.7648


def _url(when: date) -> str:
    return f"{BASE}/{when:%Y%m}-citibike-tripdata.zip"


def pull(start: date = date(2023, 1, 1), end: date = date(2026, 8, 1)) -> None:
    out_dir = RAW / SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for win_start, _ in month_windows(start, end):
        out = out_dir / f"{win_start:%Y-%m}.parquet"
        if out.exists():
            log.info("citibike %s: cached", f"{win_start:%Y-%m}")
            continue
        try:
            with httpx.Client(timeout=600.0, follow_redirects=True) as c:
                r = c.get(_url(win_start))
                if r.status_code != 200:
                    log.info("citibike %s: not published", f"{win_start:%Y-%m}")
                    continue
                blob = r.content
        except httpx.RequestError as exc:
            log.warning("citibike %s: %s", f"{win_start:%Y-%m}", exc)
            continue

        frames = []
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            for name in z.namelist():
                if not name.endswith(".csv") or name.startswith("__"):
                    continue
                with z.open(name) as fh:
                    try:
                        df = pl.read_csv(fh.read(), infer_schema_length=5000, ignore_errors=True)
                    except (pl.exceptions.PolarsError, UnicodeDecodeError, ValueError) as exc:
                        # Citi Bike has shipped malformed months before; skip the
                        # bad file rather than abort the whole ingest.
                        log.warning("citibike %s/%s: %s", f"{win_start:%Y-%m}", name, exc)
                        continue
                cols = {c.lower(): c for c in df.columns}
                lat, started = cols.get("start_lat"), cols.get("started_at")
                if not lat or not started:
                    continue
                frames.append(
                    df.select(
                        [
                            pl.col(started).str.to_datetime(strict=False).dt.date().alias("service_date"),
                            (pl.col(lat).cast(pl.Float64, strict=False) < CORDON_LAT).alias("in_crz"),
                        ]
                    )
                )
        if not frames:
            continue
        (
            pl.concat(frames, how="diagonal_relaxed")
            .drop_nulls("service_date")
            .group_by(["service_date", "in_crz"])
            .agg(pl.len().alias("trips"))
            .sort("service_date")
            .write_parquet(out)
        )
        log.info("citibike %s: written", f"{win_start:%Y-%m}")


def load() -> pl.DataFrame:
    files = sorted((RAW / SUBDIR).glob("*.parquet"))
    frames = [pl.read_parquet(f) for f in files]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
