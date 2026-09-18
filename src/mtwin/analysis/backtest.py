"""Calendar-matched rolling-origin backtest: what model error looks like without a policy.

A single placebo gives one number at one horizon in one season. That is not a
null band -- it cannot say whether a post-policy deviation is unusual, because
there is nothing to compare its size against.

This repeats the identical train/forecast protocol at quarterly origins across
the whole pre-policy span, producing a *distribution* of horizon-h errors. The
policy-period error is then read against that distribution.

Calendar matching matters and is easy to get wrong. Training January-December
and forecasting January-August must be mirrored by every placebo origin, or
seasonality and forecast horizon are confounded and the band measures the wrong
thing: a forecast that starts in November faces the annual congestion peak and
will look bad for reasons that have nothing to do with model quality.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import torch

from ..models.experiments import FEATURES, _slice, build_tensors
from ..models.twin import Twin, TwinConfig, evaluate, fit
from ..panel.build import reservoir_panel


@dataclass
class BacktestResult:
    origins: list[date]
    errors: np.ndarray          # horizon-h RMSE at each placebo origin
    policy_error: float
    band_lo: float
    band_hi: float

    @property
    def outside_band(self) -> bool:
        return not (self.band_lo <= self.policy_error <= self.band_hi)

    def summary(self) -> str:
        return "\n".join(
            [
                f"placebo origins     : {len(self.origins)}",
                f"null band (10-90pct): [{self.band_lo:.4f}, {self.band_hi:.4f}]",
                f"median placebo error: {np.median(self.errors):.4f}",
                f"policy-period error : {self.policy_error:.4f}",
                f"outside the band    : {'YES' if self.outside_band else 'NO'}",
            ]
        )


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    return date(d.year + m // 12, m % 12 + 1, 1)


def run(
    *,
    train_months: int = 12,
    horizon_months: int = 8,
    epochs: int = 600,
    k_jam: float = 120.0,
    policy_start: date = date(2025, 1, 1),
    first_origin: date = date(2023, 1, 1),
    seed: int = 0,
) -> BacktestResult:
    """Fit at each quarterly origin and collect horizon-h forecast error."""
    panel = reservoir_panel()
    T = build_tensors(panel, k_jam=k_jam)
    R, F = len(T["res_names"]), len(FEATURES)

    def _err(train_start: date) -> float | None:
        train_end = _add_months(train_start, train_months)
        test_end = _add_months(train_end, horizon_months)
        tr = _slice(T, train_start, train_end)
        te = _slice(T, train_end, test_end)
        if tr["feats"].shape[0] < 60 or te["feats"].shape[0] < 30:
            return None
        torch.manual_seed(seed)
        m = Twin(TwinConfig(R, F, closure="monotone", epochs=epochs))
        fit(m, tr["feats"], T["res_onehot"], T["n_jam"], tr["y_speed"], tr["y_trips"],
            tr["mask"], epochs=epochs)
        return evaluate(m, te["feats"], T["res_onehot"], T["n_jam"], te["y_speed"], te["mask"])

    # Placebo origins: every quarter whose whole forecast window predates the
    # policy, so no placebo is contaminated by the treatment.
    origins, errors = [], []
    o = first_origin
    while _add_months(o, train_months + horizon_months) <= policy_start:
        e = _err(o)
        if e is not None:
            origins.append(o)
            errors.append(e)
        o = _add_months(o, 3)

    # The real thing: train the 12 months ending at the policy date, forecast
    # across it, on the same calendar footing as the placebos.
    policy_error = _err(_add_months(policy_start, -train_months)) or float("nan")

    errors = np.array(errors)
    lo, hi = (np.percentile(errors, [10, 90]) if errors.size else (np.nan, np.nan))
    return BacktestResult(origins, errors, policy_error, float(lo), float(hi))
