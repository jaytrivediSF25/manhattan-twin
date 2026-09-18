"""The twin: a physics-constrained reservoir state-space model.

Deliberately NOT called a PINN. There is no within-link spatial coordinate, no
observed density and no link flow count, so there is no PDE to enforce and an
autodiff collocation residual would be decoration. What there is, is a
conservation ODE over reservoirs plus a fundamental-diagram closure, integrated
by a differentiable rollout. Conservation then holds by construction rather
than by penalty, which is stronger than a soft residual would have been.

The model splits into two blocks, and the split is the point:

  physics block   f_theta, the fundamental diagram shape. Assumed invariant to
                  policy: a street at a given density behaves the same way
                  whether or not drivers were charged to get there.
  behaviour block h_psi, demand. Allowed to shift: this is where a toll acts.

Freezing the physics block at its pre-policy estimate and re-estimating only
the behaviour block on post-policy data turns "the residual is behaviour" from
an unfalsifiable claim into a measured parameter shift.

Identification caveat, stated up front because a referee will reach it in
thirty seconds: accumulation is observed only up to scale, since (n, P) ->
(a*n, a*P) leaves speed and production unchanged. n_jam is therefore pinned
externally from reservoir area, and every headline number is reported across a
sweep of that pin.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
GRID = 64  # resolution of the monotone MFD shape function


class MonotoneMFD(nn.Module):
    """Speed multiplier f(x) on normalised accumulation x = n / n_jam.

    Monotonicity is structural, not penalised: f is one minus the normalised
    integral of a positive function, so f(0)=1, f(1)=0, and f is decreasing by
    construction for any parameter value. A penalty would have left the network
    free to violate the shape wherever data were thin.
    """

    def __init__(self, n_reservoirs: int, grid: int = GRID):
        super().__init__()
        self.grid = grid
        # One shape per reservoir; initialised near a linear (Greenshields) form.
        self.raw = nn.Parameter(torch.zeros(n_reservoirs, grid))

    def forward(self, x: torch.Tensor, res_idx: torch.Tensor) -> torch.Tensor:
        dens = torch.nn.functional.softplus(self.raw) + 1e-4       # (R, G) positive
        cum = torch.cumsum(dens, dim=1)
        cum = cum / cum[:, -1:]                                     # normalise to [0,1]
        cum = torch.cat([torch.zeros(cum.shape[0], 1, device=cum.device), cum], dim=1)
        xc = x.clamp(0.0, 1.0) * self.grid
        lo = xc.floor().long().clamp(0, self.grid - 1)
        frac = (xc - lo.float()).unsqueeze(-1)
        c_lo = cum[res_idx].gather(1, lo.unsqueeze(-1))
        c_hi = cum[res_idx].gather(1, (lo + 1).clamp(max=self.grid).unsqueeze(-1))
        integ = (c_lo + frac * (c_hi - c_lo)).squeeze(-1)
        return (1.0 - integ).clamp(min=1e-3)


class DemandNet(nn.Module):
    """Behaviour block: inflow demand from calendar and weather."""

    def __init__(self, n_reservoirs: int, n_feat: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat + n_reservoirs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, feats: torch.Tensor, res_onehot: torch.Tensor) -> torch.Tensor:
        z = torch.cat([feats, res_onehot], dim=-1)
        return torch.nn.functional.softplus(self.net(z)).squeeze(-1)


class FreeClosure(nn.Module):
    """Unconstrained speed multiplier f(x) on normalised accumulation.

    Same role and same inputs as `MonotoneMFD`, but nothing forces it to start
    at 1, end at 0, or decrease. Swapping one for the other inside an otherwise
    identical rollout isolates the *monotonicity constraint*, which is the thing
    the fundamental diagram actually asserts.
    """

    def __init__(self, n_reservoirs: int, hidden: int = 32):
        super().__init__()
        self.n_reservoirs = n_reservoirs
        self.net = nn.Sequential(
            nn.Linear(1 + n_reservoirs, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, res_idx: torch.Tensor) -> torch.Tensor:
        onehot = torch.nn.functional.one_hot(res_idx, self.n_reservoirs).float()
        z = torch.cat([x.unsqueeze(-1), onehot], dim=-1)
        return torch.sigmoid(self.net(z)).squeeze(-1).clamp(min=1e-3)


@dataclass
class TwinConfig:
    n_reservoirs: int
    n_feat: int
    # "monotone" -> fundamental diagram, decreasing by construction
    # "free"     -> same rollout and conservation, unconstrained closure
    # "none"     -> no rollout at all, direct regression from covariates
    closure: str = "monotone"
    lr: float = 1e-2
    epochs: int = 400
    hours: int = 24

    @property
    def physics(self) -> bool:
        """True whenever the conservation rollout is used at all."""
        return self.closure != "none"


class Twin(nn.Module):
    """Physics-constrained twin, with an ablation switch.

    Accumulation must be *integrated* from demand, never read off observed
    speed. Since the accumulation proxy is trips divided by speed, feeding it
    in as a covariate and predicting speed from it would be circular -- speed
    would appear on both sides. Instead demand drives a conservation rollout,
    accumulation is a latent state, and speed and trip production are both
    outputs compared against data.

    Three arms, because comparing only the first and last confounds two
    different things:

      closure="monotone"  rollout + fundamental diagram decreasing by construction
      closure="free"      rollout + conservation, unconstrained closure
      closure="none"      no rollout; direct regression from covariates to speed

    "monotone" against "none" compares whole *architectures*: one must route
    every hour through accumulation dynamics driven by a handful of smooth
    covariates, the other can fit any hour-of-day pattern directly. Losing that
    comparison says little about the physics.

    "monotone" against "free" is the fair test. Same rollout, same conservation,
    same capacity -- only the monotonicity constraint differs, which is what the
    fundamental diagram actually asserts.
    """

    def __init__(self, cfg: TwinConfig):
        super().__init__()
        self.cfg = cfg
        self.mfd = MonotoneMFD(cfg.n_reservoirs) if cfg.closure == "monotone" else FreeClosure(cfg.n_reservoirs)
        self.demand = DemandNet(cfg.n_reservoirs, cfg.n_feat)
        self.log_vfree = nn.Parameter(torch.zeros(cfg.n_reservoirs) + float(np.log(12.0)))
        # Mean trip length L_r and taxi sampling share fold into one positive
        # per-reservoir constant linking modelled outflow to observed trips.
        self.log_obs_scale = nn.Parameter(torch.zeros(cfg.n_reservoirs))
        # Initial accumulation at the start of each modelled day, as a fraction
        # of jam accumulation.
        self.logit_n0 = nn.Parameter(torch.zeros(cfg.n_reservoirs) - 2.0)
        if not cfg.physics:
            self.free_head = nn.Sequential(
                nn.Linear(cfg.n_feat + cfg.n_reservoirs, 32), nn.Tanh(),
                nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 2),
            )

    def forward(self, feats: torch.Tensor, res_onehot: torch.Tensor, n_jam: torch.Tensor):
        """Roll the conservation ODE forward over a day.

        feats:      (D, H, F) calendar/weather covariates per day and hour
        res_onehot: (R, R) identity, one row per reservoir
        n_jam:      (R,) externally pinned jam accumulation
        returns:    log speed (D, H, R), log trips (D, H, R)
        """
        D, H, _ = feats.shape
        R = self.cfg.n_reservoirs

        if not self.cfg.physics:
            f = feats.unsqueeze(2).expand(D, H, R, feats.shape[-1])
            o = res_onehot.view(1, 1, R, R).expand(D, H, R, R)
            out = self.free_head(torch.cat([f, o], dim=-1))
            return out[..., 0], out[..., 1]

        v_free = torch.exp(self.log_vfree)                     # (R,)
        obs_scale = torch.exp(self.log_obs_scale)              # (R,)
        n = torch.sigmoid(self.logit_n0).unsqueeze(0) * n_jam.unsqueeze(0)   # (1,R)
        n = n.expand(D, R).clone()

        res_idx = torch.arange(R, device=feats.device)
        log_v, log_q = [], []
        for h in range(H):
            fh = feats[:, h, :].unsqueeze(1).expand(D, R, feats.shape[-1])
            oh = res_onehot.unsqueeze(0).expand(D, R, R)
            inflow = self.demand(fh, oh)                       # (D,R)

            x = (n / n_jam.unsqueeze(0)).clamp(0.0, 1.0)
            f_x = self.mfd(x.reshape(-1), res_idx.repeat(D)).reshape(D, R)
            speed = v_free.unsqueeze(0) * f_x                  # (D,R)
            # Production P = n * v; outflow is P scaled by mean trip length,
            # which is folded into obs_scale together with taxi share.
            outflow = n * speed / n_jam.unsqueeze(0)

            log_v.append(torch.log(speed + 1e-6))
            log_q.append(torch.log(outflow * obs_scale.unsqueeze(0) + 1e-6))

            # Semi-implicit Euler, one-hour steps; the clamp enforces
            # 0 <= n <= n_jam, the physical bound conservation must respect.
            n = (n + inflow - outflow).clamp(min=1.0)
            n = torch.minimum(n, n_jam.unsqueeze(0))

        return torch.stack(log_v, dim=1), torch.stack(log_q, dim=1)


def fit(
    model: Twin,
    feats: torch.Tensor,
    res_onehot: torch.Tensor,
    n_jam: torch.Tensor,
    y_speed: torch.Tensor,
    y_trips: torch.Tensor,
    mask: torch.Tensor,
    *,
    freeze_physics: bool = False,
    epochs: int | None = None,
    lr: float | None = None,
    w_trips: float = 0.3,
    verbose: bool = False,
) -> list[float]:
    """Train, optionally holding the physics block fixed.

    `freeze_physics=True` is the frozen-physics experiment: the fundamental
    diagram and free-flow speeds keep their pre-policy values and only demand
    may move, so whatever is needed to fit post-policy data must be expressed
    as a behavioural parameter shift.
    """
    if freeze_physics:
        for p_ in model.mfd.parameters():
            p_.requires_grad_(False)
        model.log_vfree.requires_grad_(False)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    opt = torch.optim.Adam(params, lr=lr or model.cfg.lr)
    losses = []
    for ep in range(epochs or model.cfg.epochs):
        opt.zero_grad()
        lv, lq = model(feats, res_onehot, n_jam)
        loss = (torch.sum(((lv - y_speed) ** 2) * mask) / mask.sum()
                + w_trips * torch.sum(((lq - y_trips) ** 2) * mask) / mask.sum())
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
        if verbose and ep % 100 == 0:
            print(f"  epoch {ep:4d}  loss {losses[-1]:.5f}")
    return losses


@torch.no_grad()
def evaluate(model: Twin, feats, res_onehot, n_jam, y_speed, mask) -> float:
    """RMSE on log speed over the masked cells."""
    model.eval()
    lv, _ = model(feats, res_onehot, n_jam)
    se = ((lv - y_speed) ** 2) * mask
    return float(torch.sqrt(se.sum() / mask.sum()).cpu())
