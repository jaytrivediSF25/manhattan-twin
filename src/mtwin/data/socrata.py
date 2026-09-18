"""Shared Socrata (SODA 2.1) paged fetch with on-disk Parquet caching.

Two hard-won constraints drive this module's design:

1. Unfiltered aggregates over large datasets (notably NYC DOT speeds `i4gi-tjb9`)
   time out server-side. Every pull must be constrained by a date window.
2. Offset paging without a stable `$order` can silently duplicate or skip rows.
   We always order by `:id`, the Socrata internal row identifier.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator, Sequence

import httpx
import polars as pl

log = logging.getLogger(__name__)

# httpx logs every request at INFO, which drowns out pull progress.
logging.getLogger("httpx").setLevel(logging.WARNING)

RAW = Path(__file__).resolve().parents[3] / "data" / "raw"
PAGE_SIZE = 50_000
MAX_RETRIES = 5


@dataclass(frozen=True)
class Dataset:
    """A Socrata dataset endpoint."""

    domain: str
    dataset_id: str
    name: str
    date_field: str | None = None

    @property
    def url(self) -> str:
        return f"https://{self.domain}/resource/{self.dataset_id}.json"


def _client() -> httpx.Client:
    headers = {"User-Agent": "manhattan-twin/0.1 (research)"}
    # An app token is optional but lifts the shared anonymous rate limit.
    token = os.environ.get("SOCRATA_APP_TOKEN")
    if token:
        headers["X-App-Token"] = token
    return httpx.Client(headers=headers, timeout=120.0, follow_redirects=True)


def _get(client: httpx.Client, url: str, params: dict) -> list[dict]:
    """GET one page, retrying on transient errors and throttling."""
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            r = client.get(url, params=params)
            if r.status_code == 200:
                return r.json()
            # 429 = throttled, 5xx = transient server error. Both are retryable.
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"{r.status_code} from {url}: {r.text[:300]}")
            log.warning("socrata %s (attempt %d), backing off %.0fs", r.status_code, attempt + 1, delay)
        except httpx.RequestError as exc:
            log.warning("socrata request error %s (attempt %d), backing off %.0fs", exc, attempt + 1, delay)
        time.sleep(delay)
        delay *= 2
    raise RuntimeError(f"socrata: giving up on {url} after {MAX_RETRIES} attempts")


def fetch(
    ds: Dataset,
    *,
    select: str | None = None,
    where: str | None = None,
    page_size: int = PAGE_SIZE,
    max_rows: int | None = None,
) -> pl.DataFrame:
    """Page through a Socrata query and return all rows as a DataFrame.

    Ordering by `:id` makes offset paging stable; without it Socrata gives no
    ordering guarantee and pages can overlap.
    """
    rows: list[dict] = []
    offset = 0
    with _client() as client:
        while True:
            params = {"$limit": page_size, "$offset": offset, "$order": ":id"}
            if select:
                params["$select"] = select
            if where:
                params["$where"] = where
            page = _get(client, ds.url, params)
            rows.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
            if max_rows and len(rows) >= max_rows:
                break
    if not rows:
        return pl.DataFrame()
    # All Socrata JSON values arrive as strings; typing is the caller's job.
    return pl.DataFrame(rows, infer_schema_length=None)


def month_windows(start: date, end: date) -> Iterator[tuple[date, date]]:
    """Yield [month_start, next_month_start) pairs covering [start, end)."""
    cur = date(start.year, start.month, 1)
    while cur < end:
        nxt = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
        yield cur, min(nxt, end)
        cur = nxt


def fetch_by_month(
    ds: Dataset,
    start: date,
    end: date,
    *,
    select: str | None = None,
    extra_where: str | None = None,
    date_field: str | None = None,
) -> Iterator[tuple[date, pl.DataFrame]]:
    """Fetch a date range one month at a time.

    Monthly chunks keep each query inside Socrata's server-side time budget and
    make partial progress resumable.
    """
    field = date_field or ds.date_field
    if not field:
        raise ValueError(f"{ds.name}: a date_field is required for windowed fetch")
    for win_start, win_end in month_windows(start, end):
        clause = (
            f"{field} >= '{win_start.isoformat()}T00:00:00' "
            f"AND {field} < '{win_end.isoformat()}T00:00:00'"
        )
        if extra_where:
            clause = f"({clause}) AND ({extra_where})"
        yield win_start, fetch(ds, select=select, where=clause)


def cached_monthly_pull(
    ds: Dataset,
    start: date,
    end: date,
    *,
    subdir: str,
    select: str | None = None,
    extra_where: str | None = None,
    date_field: str | None = None,
    transform=None,
) -> list[Path]:
    """Fetch month by month to Parquet, skipping months already on disk.

    Re-running is a no-op once every month file exists, which makes the ETL
    cheap to resume after an interruption.
    """
    out_dir = RAW / subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for win_start, _ in month_windows(start, end):
        path = out_dir / f"{win_start:%Y-%m}.parquet"
        written.append(path)
        if path.exists():
            log.info("%s %s: cached", ds.name, f"{win_start:%Y-%m}")
            continue
        field = date_field or ds.date_field
        win_end = date(win_start.year + (win_start.month == 12), (win_start.month % 12) + 1, 1)
        clause = (
            f"{field} >= '{win_start.isoformat()}T00:00:00' "
            f"AND {field} < '{win_end.isoformat()}T00:00:00'"
        )
        if extra_where:
            clause = f"({clause}) AND ({extra_where})"
        df = fetch(ds, select=select, where=clause)
        if transform is not None and not df.is_empty():
            df = transform(df)
        # Write even when empty so a genuinely empty month is not refetched forever.
        df.write_parquet(path)
        log.info("%s %s: %d rows -> %s", ds.name, f"{win_start:%Y-%m}", df.height, path.name)
    return written


def load_months(subdir: str) -> pl.DataFrame:
    """Concatenate all cached month files for a source."""
    files = sorted((RAW / subdir).glob("*.parquet"))
    if not files:
        return pl.DataFrame()
    frames = [pl.read_parquet(f) for f in files]
    frames = [f for f in frames if not f.is_empty()]
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
