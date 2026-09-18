"""Synthetic control for the cordon speed series.

Why not a cross-city donor pool. The published work on this policy used Google
Maps data with other US cities as controls, which is the right design. It is not
reproducible from open data for 2023-2026: no other transit agency publishes
stop-to-stop bus speeds, and Chicago's Traffic Tracker archive -- the closest
public equivalent -- ends in May 2018. Uber Movement was discontinued in 2023.
Every donor available here is therefore inside New York and shares city-wide
shocks with the treated units, which is exactly the weakness the cross-city
design was meant to remove. That limitation is stated rather than hidden.

What this does instead is stronger than the naive difference-in-differences: the
donor pool is weighted to reproduce the cordon's *pre-policy trajectory*, rather
than assuming a simple average of outer-borough segments is the right
counterfactual. Weights are non-negative and sum to one, so the synthetic
control stays inside the convex hull of observed donors and cannot extrapolate.

Inference is placebo-in-space: the same estimator is run pretending each donor
was treated, and the real effect is read against that distribution.

Donor count is capped well below the number of pre-policy periods. With a few
hundred segment-level donors and ~24 pre-periods, non-negative least squares
reproduces the pre-period *exactly* -- a zero-RMSE fit that has memorised the
training window and predicts nothing out of sample. Donors are therefore
aggregated to route level and the pool is capped, so the pre-period fit is
informative rather than mechanical.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.optimize import nnls


@dataclass
class SCResult:
    periods: np.ndarray
    treated: np.ndarray
    synthetic: np.ndarray
    weights: dict[str, float]
    pre_rmse: float
    post_effect: float
    placebo_effects: np.ndarray
    p_value: float

    def summary(self) -> str:
        top = sorted(self.weights.items(), key=lambda kv: -kv[1])[:5]
        lines = [
            f"pre-period fit RMSE : {self.pre_rmse:.4f}",
            f"post-period effect  : {self.post_effect:+.4f} log points",
            f"placebo-in-space p  : {self.p_value:.3f}  (n={len(self.placebo_effects)} donors)",
            "top donor weights   : " + ", ".join(f"{k} {v:.2f}" for k, v in top if v > 0.01),
        ]
        return "\n".join(lines)


def _fit_weights(Y_pre_treated: np.ndarray, Y_pre_donors: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one that best match the pre-period.

    The sum-to-one constraint is imposed by augmenting the system with a heavily
    weighted row of ones, which keeps the problem a single non-negative least
    squares solve.
    """
    big = 1e6
    A = np.vstack([Y_pre_donors, np.ones((1, Y_pre_donors.shape[1])) * big])
    b = np.concatenate([Y_pre_treated, [big]])
    w, _ = nnls(A, b)
    return w / w.sum() if w.sum() > 0 else w


def run(
    panel: pl.DataFrame,
    *,
    treated_group: str = "crz",
    donor_group: str = "outer",
    policy_period: int | None = None,
    min_periods: int | None = None,
    donor_level: str = "route",
    max_donors: int = 12,
) -> SCResult:
    """Build a synthetic cordon from weighted donor segments."""
    from ..panel.build import POLICY_PERIOD

    policy_period = policy_period or POLICY_PERIOD

    # Treated series: trip-weighted mean log speed across in-cordon segments.
    treated = (
        panel.filter(pl.col("grp") == treated_group)
        .group_by("period")
        .agg(
            ((pl.col("log_speed") * pl.col("n_trips")).sum() / pl.col("n_trips").sum())
            .alias("y")
        )
        .sort("period")
    )
    periods = treated["period"].to_numpy()
    if min_periods is None:
        min_periods = len(periods)

    # Donors: aggregated to route level and capped, so the pool stays far
    # smaller than the number of pre-periods being fitted.
    donors = panel.filter(pl.col("grp") == donor_group)
    if donor_level == "route" and "route_id" in donors.columns:
        donors = (
            donors.group_by(["route_id", "period"])
            .agg(
                [
                    ((pl.col("log_speed") * pl.col("n_trips")).sum()
                     / pl.col("n_trips").sum()).alias("log_speed"),
                    pl.col("n_trips").sum().alias("n_trips"),
                ]
            )
            .rename({"route_id": "unit"})
        )
    counts = donors.group_by("unit").agg(
        [pl.len().alias("n"), pl.col("n_trips").sum().alias("vol")]
    )
    keep = (
        counts.filter(pl.col("n") >= min_periods)
        .sort("vol", descending=True)
        .head(max_donors)["unit"]
        .to_list()
    )
    donors = donors.filter(pl.col("unit").is_in(keep))
    wide = donors.pivot(values="log_speed", index="period", on="unit").sort("period")
    wide = wide.filter(pl.col("period").is_in(periods.tolist()))
    donor_names = [c for c in wide.columns if c != "period"]
    Y = wide.select(donor_names).to_numpy()

    ok = ~np.isnan(Y).any(axis=0)
    Y, donor_names = Y[:, ok], [n for n, k in zip(donor_names, ok) if k]
    y = treated["y"].to_numpy()

    pre = periods < policy_period
    post = ~pre

    w = _fit_weights(y[pre], Y[pre])
    synth = Y @ w
    pre_rmse = float(np.sqrt(np.mean((y[pre] - synth[pre]) ** 2)))
    effect = float(np.mean(y[post] - synth[post]))

    # Placebo in space: treat each donor as if it had been treated.
    placebos = []
    for j in range(Y.shape[1]):
        others = np.delete(Y, j, axis=1)
        wj = _fit_weights(Y[pre, j], others[pre])
        sj = others @ wj
        if np.sqrt(np.mean((Y[pre, j] - sj[pre]) ** 2)) <= 2 * pre_rmse:
            placebos.append(float(np.mean(Y[post, j] - sj[post])))
    placebos = np.array(placebos)
    p = float(np.mean(np.abs(placebos) >= abs(effect))) if placebos.size else float("nan")

    return SCResult(periods, y, synth, dict(zip(donor_names, w)), pre_rmse, effect, placebos, p)
