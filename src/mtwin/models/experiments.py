"""Training data assembly and the three headline experiments."""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import torch

from ..panel.build import reservoir_panel
from .reservoir import reservoir_capacity
from .twin import Twin, TwinConfig, evaluate, fit

FEATURES = ["sin_h", "cos_h", "sin_dow", "cos_dow", "sin_mo", "cos_mo", "rain"]


def build_tensors(panel: pl.DataFrame, k_jam: float = 120.0, device=None):
    """Reshape the reservoir panel into dense (day, hour, reservoir) tensors."""
    device = device or torch.device("cpu")
    res_names = sorted(panel["reservoir"].unique().to_list())
    r_index = {r: i for i, r in enumerate(res_names)}

    d = panel.with_columns(
        [
            (2 * np.pi * pl.col("hour") / 24).sin().alias("sin_h"),
            (2 * np.pi * pl.col("hour") / 24).cos().alias("cos_h"),
            (2 * np.pi * pl.col("dow") / 7).sin().alias("sin_dow"),
            (2 * np.pi * pl.col("dow") / 7).cos().alias("cos_dow"),
            (2 * np.pi * pl.col("service_date").dt.month() / 12).sin().alias("sin_mo"),
            (2 * np.pi * pl.col("service_date").dt.month() / 12).cos().alias("cos_mo"),
            pl.col("rain_day").fill_null(False).cast(pl.Float64).alias("rain"),
        ]
    )

    days = sorted(d["service_date"].unique().to_list())
    day_index = {v: i for i, v in enumerate(days)}
    D, H, R = len(days), 24, len(res_names)

    feats = np.zeros((D, H, len(FEATURES)), dtype=np.float32)
    y_speed = np.zeros((D, H, R), dtype=np.float32)
    y_trips = np.zeros((D, H, R), dtype=np.float32)
    mask = np.zeros((D, H, R), dtype=np.float32)

    di = np.array([day_index[v] for v in d["service_date"].to_list()])
    hi = d["hour"].to_numpy().astype(int)
    ri = np.array([r_index[v] for v in d["reservoir"].to_list()])

    for k, f in enumerate(FEATURES):
        feats[di, hi, k] = d[f].to_numpy()
    y_speed[di, hi, ri] = np.log(d["speed_mph"].to_numpy())
    y_trips[di, hi, ri] = np.log(d["trips"].to_numpy())
    mask[di, hi, ri] = 1.0

    caps = reservoir_capacity(k_jam=k_jam)
    n_jam = np.array([caps.get(r, 2e4) for r in res_names], dtype=np.float32)

    t = lambda a: torch.tensor(a, device=device)
    return {
        "feats": t(feats), "y_speed": t(y_speed), "y_trips": t(y_trips), "mask": t(mask),
        "n_jam": t(n_jam), "res_onehot": torch.eye(R, device=device),
        "days": days, "res_names": res_names,
    }


def _slice(T: dict, lo: date, hi: date) -> dict:
    idx = [i for i, d in enumerate(T["days"]) if lo <= d < hi]
    ix = torch.tensor(idx, device=T["feats"].device)
    out = dict(T)
    for k in ("feats", "y_speed", "y_trips", "mask"):
        out[k] = T[k].index_select(0, ix)
    out["days"] = [T["days"][i] for i in idx]
    return out


def run_experiments(k_jam: float = 120.0, epochs: int = 400, seed: int = 0) -> dict:
    """Train on the clean pre-period, then run the three headline experiments."""
    torch.manual_seed(seed)
    panel = reservoir_panel()
    T = build_tensors(panel, k_jam=k_jam)

    # Pre-period stops before the June 2024 pause announcement, which is itself
    # a behavioural shock and would contaminate a "pre-policy" baseline.
    PRE = _slice(T, date(2023, 1, 1), date(2024, 6, 1))
    POST = _slice(T, date(2025, 1, 5), date(2026, 8, 1))
    # Placebo: same horizon, no policy, to calibrate what model drift alone
    # looks like at this forecast distance.
    PLACEBO = _slice(T, date(2024, 6, 1), date(2024, 12, 1))

    def args(S):
        return (S["feats"], T["res_onehot"], T["n_jam"], S["y_speed"], S["y_trips"], S["mask"])

    R = len(T["res_names"])
    F = len(FEATURES)
    res = {"k_jam": k_jam}

    # --- three arms ---------------------------------------------------------
    # "monotone" vs "free" is the fair test: same rollout, same conservation,
    # only the monotonicity constraint differs. "none" is the architecture
    # comparison and is reported separately so the two are not conflated.
    twins = {}
    for arm in ("monotone", "free", "none"):
        torch.manual_seed(seed)
        m = Twin(TwinConfig(R, F, closure=arm, epochs=epochs))
        fit(m, *args(PRE), epochs=epochs)
        twins[arm] = m
        for label, S in (("pre", PRE), ("placebo", PLACEBO), ("post", POST)):
            res[f"{arm}_rmse_{label}"] = evaluate(
                m, S["feats"], T["res_onehot"], T["n_jam"], S["y_speed"], S["mask"]
            )
        res[f"{arm}_params"] = sum(p.numel() for p in m.parameters() if p.requires_grad)

    twin = twins["monotone"]

    # --- frozen physics, re-estimated behaviour -----------------------------
    import copy

    frozen = copy.deepcopy(twin)  # frozen-physics uses the monotone arm
    before = torch.cat([p.detach().flatten() for p in frozen.demand.parameters()]).clone()
    fit(frozen, *args(POST), freeze_physics=True, epochs=epochs)
    after = torch.cat([p.detach().flatten() for p in frozen.demand.parameters()])
    res["behaviour_shift_l2"] = float(torch.norm(after - before) / torch.norm(before))
    res["rmse_post_frozen"] = evaluate(frozen, POST["feats"], T["res_onehot"], T["n_jam"], POST["y_speed"], POST["mask"])

    # Same re-estimation against the placebo window: how much does the
    # behaviour block move when no policy happened?
    torch.manual_seed(seed)
    frozen_pl = copy.deepcopy(twin)
    b0 = torch.cat([p.detach().flatten() for p in frozen_pl.demand.parameters()]).clone()
    fit(frozen_pl, *args(PLACEBO), freeze_physics=True, epochs=epochs)
    b1 = torch.cat([p.detach().flatten() for p in frozen_pl.demand.parameters()])
    res["behaviour_shift_placebo_l2"] = float(torch.norm(b1 - b0) / torch.norm(b0))

    res["n_pre_days"] = len(PRE["days"])
    res["n_post_days"] = len(POST["days"])
    return res
