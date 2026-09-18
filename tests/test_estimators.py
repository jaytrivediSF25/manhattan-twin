"""Regression tests for the estimator bugs found during the build.

Each of these returned a plausible-looking number while being wrong, which is
the failure mode worth pinning down: a crash announces itself, a biased
coefficient does not.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.mtwin.analysis.exposure_did import event_study, fit
from src.mtwin.models.twin import MonotoneMFD
import torch


def _panel(effect: float = 0.05, level_gap: float = 2.0, unbalanced: bool = False,
           seed: int = 0) -> pl.DataFrame:
    """Synthetic two-way panel with a known treatment effect."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(80):
        treated = u < 40
        a = rng.normal(level_gap if treated else 0.0, 0.3)
        # Treated units carry far more weight, which is what exposed the
        # weighted-demeaning bug.
        w = 500.0 if treated else 5.0
        for t in range(24):
            if unbalanced and ((u + t) % 5 == 0):
                continue          # drop cells to break panel balance
            post = t >= 12
            y = a + 0.01 * t + (effect if (treated and post) else 0.0) + rng.normal(0, 0.02)
            rows.append(
                dict(unit=f"u{u}", period=t, log_speed=y, n_trips=w,
                     crz_post=float(treated and post), above60_post=0.0,
                     is_crz=float(treated), rel_period=t - 12)
            )
    return pl.DataFrame(rows)


def test_fit_recovers_known_effect():
    r = fit(_panel())
    assert abs(r.coef[0] - 0.05) < 0.01


@pytest.mark.parametrize("unbalanced", [False, True])
def test_weighted_fit_matches_unweighted(unbalanced):
    """Weights must not shift the coefficient when the effect is homogeneous.

    The original code multiplied y and X by sqrt(w) and then demeaned with
    UNWEIGHTED group means, which fails to absorb the fixed effects. The group
    level difference then leaked into the estimate.
    """
    df = _panel(unbalanced=unbalanced)
    unweighted = fit(df).coef[0]
    weighted = fit(df, weight_col="n_trips").coef[0]
    assert abs(weighted - unweighted) < 0.01
    assert abs(weighted - 0.05) < 0.015


def test_event_study_pre_period_is_flat():
    """Pre-period coefficients must sit near zero.

    With the weighted-demeaning bug these came back offset by a large constant
    -- every event-time estimate shifted by the between-group level difference,
    which looked like an enormous effect present before treatment.
    """
    df = _panel(level_gap=2.0, unbalanced=True)
    es = event_study(df, weight_col="n_trips", omit=-1)
    pre = [c for n, c in zip(es.names, es.coef) if int(n[4:-1]) < 0]
    post = [c for n, c in zip(es.names, es.coef) if int(n[4:-1]) >= 0]
    assert abs(np.mean(pre)) < 0.02, "pre-period should be flat"
    assert abs(np.mean(post) - 0.05) < 0.02


def test_mfd_is_monotone_by_construction():
    """The fundamental diagram must decrease for ANY parameter value.

    Monotonicity is structural here rather than penalised, so it has to hold
    even at randomly perturbed parameters where a penalty would not yet bind.
    """
    m = MonotoneMFD(3)
    with torch.no_grad():
        m.raw.normal_(0, 3.0)
    x = torch.linspace(0, 1, 200)
    for r in range(3):
        v = m(x, torch.full((200,), r, dtype=torch.long))
        assert torch.all(v[1:] <= v[:-1] + 1e-6), "MFD must be non-increasing"
        assert abs(float(v[0]) - 1.0) < 1e-3, "f(0) must be 1"
        assert float(v[-1]) < 0.02, "f(1) must be ~0"


def test_twin_forward_does_not_read_speed_from_inputs():
    """The twin must integrate accumulation, not be handed it.

    The accumulation proxy is trips/speed, so feeding it in as a covariate and
    predicting speed makes speed appear on both sides of the equation. The
    rollout signature takes only covariates, reservoir identity and capacity.
    """
    import inspect

    from src.mtwin.models.twin import Twin

    params = list(inspect.signature(Twin.forward).parameters)
    assert params == ["self", "feats", "res_onehot", "n_jam"]
