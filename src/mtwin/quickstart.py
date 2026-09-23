"""Reproduce the headline numbers from the committed sample tables.

Everything here runs off `data/sample/` in under a minute, which is the point:
`make data` is a multi-hour, ~4 GB pull, so without this there is no way for a
reader to check a single number in the README against the code that produced it.

What is *not* here is the twin itself (`make experiments`) and the rolling-origin
backtest. Both train a differentiable rollout for hundreds of epochs and belong
to a different time budget; the sample reservoir panel is sufficient for them, so
they run unchanged once you are willing to wait.
"""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import polars as pl

from .analysis import bunching_rd as brd
from .analysis.diversion import excluded_share
from .analysis.exposure_did import fit
from .data import crz_entries
from .models import mfd
from .panel.build import bus_segment_panel, reservoir_panel

# The classes the README tabulates, in its order. Cars face the time-varying
# toll; TLC taxi/FHV pay a flat per-trip fee and are the falsification.
VEHICLE_CLASSES = [
    "1 - Cars, Pickups and Vans",
    "2 - Single-Unit Trucks",
    "3 - Multi-Unit Trucks",
    "TLC Taxi/FHV",
    "4 - Buses",
    "5 - Motorcycles",
]


def _head(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


def bunching(crz: pl.DataFrame) -> None:
    _head("1. Bunching at the 21:00 toll step, by vehicle class")
    print(f"{'vehicle class':<28} {'jump':>8} {'placebo p':>10}")
    for vc in VEHICLE_CLASSES:
        prof = brd.block_profile(crz, vehicle_class=vc)
        r = brd.estimate(prof)
        print(f"{vc:<28} {r.jump_pct:>+8.1%} {r.placebo_p:>10.3f}")

    _head("2. All three toll thresholds (sign should follow price direction)")
    t = brd.all_thresholds(crz)
    print(f"{'threshold':<30} {'jump':>8} {'semi-elast':>11} {'p':>7}  sign ok")
    for row in t.to_dicts():
        print(f"{row['threshold']:<30} {row['jump_pct']:>+8.1%} "
              f"{row['semi_elasticity']:>11.3f} {row['placebo_p']:>7.3f}  {row['sign_as_predicted']}")

    _head("3. Habituation: the 21:00 jump, quarter by quarter")
    for row in brd.habituation(crz).to_dicts():
        print(f"  {row['quarter']}  {row['jump_pct']:>+7.1%}")


def diversion() -> None:
    _head("4. Untolled (excluded-roadway) share of zone entries")
    s = excluded_share()
    y = s["excluded_pct"].to_numpy()
    print(f"  {s['ym'][0]} .. {s['ym'][-1]}  range {y.min():.1f}% - {y.max():.1f}%  "
          f"first {y[0]:.1f}%  last {y[-1]:.1f}%")


def physics() -> None:
    _head("5. Fundamental-diagram invariance (post minus pre at matched accumulation)")
    panel = reservoir_panel()
    pre = pl.col("service_date") < date(2024, 6, 1)
    post = pl.col("service_date") >= date(2025, 1, 5)
    summ = mfd.invariance_summary(mfd.build_curves(panel, pre, post, per_period_scale=True))
    for row in summ.sort("reservoir").to_dicts():
        print(f"  {row['reservoir']:<18} {row['mean_shift_pct']:>+7.1%}  "
              f"({row['bins_compared']} bins)")


def speed_effect() -> None:
    _head("6. Exposure DiD: in-cordon bus segments vs outer-borough never-takers")
    bp = bus_segment_panel()
    r = fit(bp, weight_col="n_trips")
    print(r.summary())

    # The placebo is the load-bearing part of this result, so it runs here too:
    # pre-period data only, pretending the policy started a year early.
    placebo = (
        bp.filter(pl.col("period") < 2025 * 12 + 1)
        .with_columns((pl.col("is_crz") * (pl.col("period") >= 2024 * 12 + 1)).alias("crz_post"))
        .with_columns((pl.col("is_above60") * (pl.col("period") >= 2024 * 12 + 1)).alias("above60_post"))
    )
    p = fit(placebo, weight_col="n_trips")
    print(f"\nplacebo (policy pretended to start 2024-01): "
          f"{p.coef[0]:+.4f} (t = {p.coef[0] / p.se[0]:.2f})")


def main() -> None:
    # Forced rather than left to the fallback, so this target exercises the
    # sample path on a developer machine that does have the full pull -- which
    # is the only way a drift between the two would be caught.
    os.environ.setdefault("MTWIN_USE_SAMPLE", "1")
    print("manhattan-twin quickstart -- running from data/sample/")
    crz = crz_entries.load()
    print(f"CRZ blocks: {crz.height:,} rows, "
          f"{crz['toll_date'].min():%Y-%m-%d} .. {crz['toll_date'].max():%Y-%m-%d}")

    bunching(crz)
    diversion()
    physics()
    speed_effect()

    _head("7. Figures")
    from .figures.make_figures import main as figures

    figures()


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
