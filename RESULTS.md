# Manhattan congestion pricing: what a physics twin explains, and where it breaks

Congestion pricing began in the Manhattan CBD on 5 January 2025 ($9 peak base
toll, cordon below 60th St). This project calibrates a physics-constrained
traffic model on pre-policy data, runs it across the policy shock, and asks
which structural block fails.

**Contribution.** Effect-size estimation is not available as a contribution:
NBER w33584, *The Network-Wide Effects of Congestion Pricing* (Cook, Kreidieh,
Vasserman, Allcott, Arora, Tomkins) already measured network-wide speed effects
using Google Maps data. The question here is a model-diagnostic one — *does a
network-physics model calibrated pre-policy still hold post-policy, and if not,
which block broke?*

---

## Data

| Source | Coverage | Role |
|---|---|---|
| MTA bus route segment speeds (`58t6-89vi`, `kufs-yh3x`) | 2023-01 → 2026-07, 4 boroughs | Primary surface observable |
| TLC yellow taxi trip records | 2023-01 → 2026-06, citywide cells | OD structure, dwell-free speed |
| MTA CRZ entries (`t6yz-b64h`) | 2025-01 → 2026-07, 10-min blocks | Treatment intensity, bunching |
| MTA B&T hourly crossings (`ebfx-2m7v`) | 2023 → 2026 | Pre-period entry volumes |
| NYC DOT link speeds (`i4gi-tjb9`) | 2023 → 2026, Manhattan | Diversion channel |
| MTA subway hourly ridership | 2023 → 2026, Manhattan | Mode shift |
| NOAA Central Park | daily | Weather control |

**Observable choice departs from the obvious one, for a verified reason.** The
DOT speed feed has only 25 Manhattan links, all limited-access, and the
in-cordon ones sit on FDR Drive and the West Side Highway — *excluded roadways*
that are not tolled. A twin built on it would mostly model roads the policy does
not price. Bus segment speeds replace it: link-level, on the tolled avenues,
from a probe that is itself exempt and runs a fixed route, so it carries no
route-selection bias. Taxi records supply OD structure and a dwell-free speed
measure. The DOT links are retained for the one job they do well — measuring
diversion onto untolled roads.

**The pre-period is 2023-01 to 2024-05, not all of 2024.** Congestion pricing was
paused indefinitely in June 2024 and revived at $9 that November. Both are
anticipatory behavioural shocks; training on them would contaminate the
"pre-policy" baseline.

---

## Result 1 — Drivers respond to the price (assumption-light, robust)

The toll steps down $6.75 at 21:00 every day of the week, and entries are
published in 10-minute blocks. A local-linear regression discontinuity on entry
counts gives a jump of **+28.6%** for cars, stable across bandwidths of 60–120
minutes. The raw series shows the textbook pattern: a dip immediately before the
threshold as drivers wait, then a spike at 21:00.

The falsification test is what makes it causal:

| Vehicle class | Jump at 21:00 | Toll structure faced |
|---|---|---|
| **Cars, pickups, vans** | **+28.6%** | Time-varying: $9 peak → $2.25 overnight |
| **TLC taxi / FHV** | **+3.8%** | Flat per-trip fee, no time variation |

Taxis have no incentive to retime and do not. A recording artefact at the hour
boundary would have moved both equally. This result uses only post-period data
and needs no control group, no model, and no parallel-trends assumption.

*Figure: `outputs/figures/fig1_bunching.png`*

## Result 2 — The physics block does not break

The fundamental diagram — speed as a function of accumulation — is the physics
block, and is assumed policy-invariant. Comparing pre- and post-policy speed *at
matched accumulation* across six reservoirs:

| Reservoir | Policy-period shift | Pre-policy placebo shift |
|---|---|---|
| above 96th | +0.2% | +0.5% |
| lower Manhattan | −2.6% | −4.2% |
| midtown north | −7.6% | −1.3% |
| midtown south | −3.7% | −3.9% |
| uptown 60–96 | −4.2% | +0.3% |
| village/chelsea | −4.8% | −4.2% |

The policy-period shift is the same order as the shift between two *pre-policy*
periods. **The fundamental diagram is invariant across the shock**: a street at a
given density behaves the same way whether or not drivers were charged to get
there. Whatever the policy did, it did not change the physics.

*Figure: `outputs/figures/fig2_mfd_invariance.png`*

**Measurement caveat, stated because it matters.** Accumulation is proxied by
taxi trips divided by speed, so it tracks true accumulation only up to taxi
share of traffic — and that share is not constant across the policy date. Under a
common scale the proxy implies accumulation *rose* 11–52%, which is an artefact
of rising taxi share and yellow-cab recovery, not congestion. The invariance
test above therefore normalises within each period, which removes any level
factor constant inside the period and compares curve *shape*. Shape is the right
object anyway: the accumulation scale was never identified from speed data.

## Result 3 — The speed effect is small and not cleanly separable

Two-way fixed-effects exposure design, in-cordon bus segments against Brooklyn
never-takers (Manhattan above 60th is contaminated by diverted traffic and is
estimated separately rather than used as a control):

| Term | Coefficient | SE | t |
|---|---|---|---|
| in-cordon × post | **+0.0178** | 0.0066 | 2.71 |
| above-60th × post | +0.0098 | 0.0086 | 1.13 |

That is a **+1.8%** speed effect. But a placebo that pretends the policy began in
January 2024, using pre-period data only, returns **−0.0160 (t = −2.69)** — a
spurious effect of comparable magnitude and opposite sign. The event study shows
a visible jump at policy start, but pre-period coefficients wander from +0.003 to
+0.070, so parallel trends does not hold.

*Figure: `outputs/figures/fig3_event_study.png`*

## Result 4 — No growth in diversion onto untolled roads

The untolled (excluded-roadway) share of all zone entries is flat at **11–12%**
across the entire post-policy period, with no upward trend. Whatever rerouting
happened, it happened at once and did not accumulate.

Speeds on those untolled corridors tell the same story. DOT link speeds on FDR
Drive and the West Side Highway show no break at the policy date: the gap
against other Manhattan links stays inside its ordinary seasonal range (+1 to
+7 mph) through 2024, 2025 and 2026. If traffic had shifted onto the free
highways at scale they would have slowed, and they did not.

Two independent instruments — entry counts and link speeds — therefore agree:
**diversion onto untolled roads is not where the response went.** Combined with
Result 1, the visible margin of adjustment is *when* people drive, not which
road they take.

*Figure: `outputs/figures/fig4_excluded_share.png`*

## Result 5 — The physics constraint does not earn its place

The mandatory ablation, identical architecture with the fundamental-diagram
closure replaced by an unconstrained network, across a sweep of the externally
pinned jam-accumulation parameter:

| k_jam | Physics twin, post RMSE | Ablation, post RMSE | Frozen-physics |
|---|---|---|---|
| 30 | 0.263 | **0.138** | 0.212 |
| 60 | 0.273 | **0.138** | 0.235 |

**The unconstrained model fits about twice as well, at every value of k_jam.**
Reported rather than buried: the physics constraint costs accuracy here and does
not improve extrapolation across the policy shock.

The frozen-physics experiment — freeze the fundamental diagram at its pre-policy
values, re-estimate only the behaviour block on post-policy data, and measure how
far the behaviour parameters move — gives a relative shift of **1.78 under the
policy against 1.75 under a no-policy placebo** (2.71 vs 2.98 at k_jam = 60). The
behavioural parameter shift does not distinguish the policy from nothing
happening.

## Result 6 — Decomposition, with the bounds left visible

An accounting identity, not a regression of the residual on mode-shift
covariates: subway ridership and the traffic residual are outcomes of the same
shock, so regressing one on the other yields a coefficient with no causal
reading.

| Component | Estimate | Basis |
|---|---|---|
| **Retiming** | ~2,380 cars/day shifted into the hour after 21:00 | Bunching RD, assumption-light |
| **Vehicle volume** | Hugh L. Carey −2.6%, Queens Midtown −1.1% | The only CRZ entries with an open pre-period |
| **Mode shift** | **upper bound** 77,600–97,000 veh/day | Subway +116,390 riders/day (+6.7%) ÷ occupancy 1.2–1.5 |
| **Suppression** | sign only, not a point estimate | Residual plug |

The mode-shift bound is worth reading carefully, because it shows why it has to
be a bound. Manhattan subway ridership rose by 116,000 weekday riders, which
would mechanically "displace" up to 97,000 vehicle trips — but measured vehicle
volumes at the two tunnels fell by only about 2,500/day combined. The bound
exceeds the measured decline by more than an order of magnitude, because most of
the ridership growth is post-pandemic recovery that never had anything to do
with cars. Anyone reporting a point estimate for mode shift from this data is
reporting the recovery trend.

---

## What this adds up to

The assumption-light half works and the model-dependent half does not.

1. **Drivers demonstrably respond to the price**, and the car/taxi contrast at the
   toll threshold pins that down without any modelling.
2. **The physics is invariant** — the fundamental diagram survives the shock
   intact, which answers the project's question in its first form: the policy
   content is entirely demand-side, not in how streets behave.
3. **The twin adds nothing.** The physics constraint loses to its own ablation,
   and the frozen-physics behavioural shift cannot separate policy from placebo.
   The residual-as-behaviour framing does not survive contact with its own
   placebo tests.
4. **The speed effect is small (+1.8%) and confounded by pre-trends** that are as
   large as the effect.

The honest paper is therefore the measurement one: *what it takes to detect a
known policy effect, and why the obvious designs fail.* The failure modes are
specific and reusable — a control group contaminated by the treatment's own
diversion, a probe fleet that is itself treated, an accumulation proxy that moves
with market share, and a placebo year that moves as much as the policy year.

## Known limitations

- **No pre-period for zone entries.** Of the twelve CRZ gantry groups, only Hugh
  L. Carey and Queens Midtown tunnels appear in MTA B&T data. Holland and
  Lincoln are Port Authority; the East River bridges are free crossings that were
  never metered. Quantitative pre/post volume work is restricted accordingly.
- **Yellow taxis cannot support the exposure design.** Outer-borough coverage is
  9–27 OD pairs per month from a churning set — too thin and too unbalanced for
  never-takers. The bus panel carries that design instead.
- **Bus speed blends running time with dwell time**, so it is structurally
  attenuated: 5.3 mph in-cordon against 6.5 mph for taxis on the same streets.
- **The accumulation scale is not identified** from speed data, since (n, P) →
  (αn, αP) leaves all observables unchanged. It is pinned externally from
  reservoir area and swept over k_jam ∈ [30, 150].
- **Bus lane and busway rollout** over 2023–2025 contaminates the bus observable
  in an unknown direction and is not controlled for.
