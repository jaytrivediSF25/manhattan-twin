# manhattan-twin

A physics-constrained digital twin of the Manhattan congestion zone, using the
January 2025 launch of congestion pricing as a natural experiment.

**Findings are in [`RESULTS.md`](RESULTS.md).** Short version: drivers visibly
retime in response to the toll (+28.6% entry jump at the 21:00 threshold for
cars, +3.8% for flat-fee taxis); the fundamental diagram is invariant across the
shock; and the physics-constrained twin loses to its own ablation, so the
residual-as-behaviour framing does not survive its placebo tests.

## Layout

```
src/mtwin/
  data/      ETL, one module per source, Parquet cache keyed by (source, month)
  network/   taxi-zone geometry and cordon classification
  panel/     analysis panels: taxi OD, bus segment, reservoir state
  analysis/  bunching RD, exposure DiD, diversion, decomposition
  models/    reservoir partition, MFD, the twin, experiments
  figures/   publication figures
```

## Running it

```bash
uv sync
uv run python -m src.mtwin.data bus      # also: crz, bt, dot, subway, weather
uv run python -c "from src.mtwin.data import tlc; tlc.pull_yellow()"
uv run python -c "from src.mtwin.models.experiments import run_experiments; print(run_experiments())"
```

Data lands in `data/` (gitignored, ~4 GB). Every pull is cached per month, so
re-running is a no-op.

## Notes

- Python 3.12 (torch wheels do not yet cover 3.14). MPS is used when available.
- Socrata: unfiltered aggregates time out on the large tables, and offset paging
  needs a stable `$order`; grouped queries cannot order by `:id`.
- The pre-period ends 2024-05, before the June 2024 pause announcement.
