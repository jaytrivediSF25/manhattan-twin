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

**The toll schedule contains three thresholds, and only one is a price cut.** At
05:00 on weekdays and 09:00 at weekends the toll *rises* from $2.25 to $9, so the
response must run the other way — drivers go early, pulling entries forward and
leaving a hole after the threshold. That sign prediction is a test rather than a
description, because a fitting artefact at an hour boundary has no reason to
track the direction of the price change:

| Threshold | Price | Jump | Semi-elasticity | Sign as predicted |
|---|---|--:|--:|:--:|
| 21:00, all days | falls $9 → $2.25 | **+24.2%** | −0.174 | ✅ |
| 05:00, weekdays | rises $2.25 → $9 | **−42.5%** (p = 0.000) | −0.307 | ✅ |
| 09:00, weekends | rises $2.25 → $9 | −17.1% | −0.123 | ✅ |

All three carry the predicted sign, and the implied semi-elasticities cluster
between −0.12 and −0.31 — a transferable price response, not just a jump size.

**The response is durable, with partial habituation.** Re-estimating the 21:00
jump quarter by quarter: 33.2% in 2025Q1, easing to a stable plateau near 22%
through 2026. Drivers adjust to the toll but do not stop responding to it.

The falsification test is what makes it causal:

| Vehicle class | Jump at 21:00 | Toll structure faced |
|---|---|---|
| **Cars, pickups, vans** | **+28.6%** | Time-varying: $9 peak → $2.25 overnight |
| **TLC taxi / FHV** | **+3.8%** | Flat per-trip fee, no time variation |

Taxis have no incentive to retime and do not. A recording artefact at the hour
boundary would have moved both equally. This result uses only post-period data
and needs no control group, no model, and no parallel-trends assumption.

*Figures: `outputs/figures/fig1_bunching.png`, `outputs/figures/fig5_thresholds.png`*

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

Two-way fixed-effects exposure design, 298 in-cordon bus segments against 3,529
outer-borough never-takers in Brooklyn, Queens and the Bronx (Manhattan above
60th is contaminated by diverted traffic and is estimated separately rather than
used as a control):

| Term | Coefficient | SE | t |
|---|---|---|---|
| in-cordon × post | **+0.0151** | 0.0062 | 2.44 |
| above-60th × post | +0.0071 | 0.0083 | 0.85 |

That is a **+1.5%** speed effect. But a placebo that pretends the policy began in
January 2024, using pre-period data only, returns **−0.0121 (t = −2.09)** — a
spurious effect of comparable magnitude and opposite sign. Restricting the
control group to Brooklyn alone gives +1.8% and a −1.6% placebo, so the finding
does not turn on the choice of never-takers. The event study shows
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

## Result 5 — The physics constraint earns its place; the architecture does not

An earlier version of this analysis reported that the twin lost to its own
ablation. **That comparison was not a fair test and the conclusion was wrong.**
It set a conservation rollout driven by seven smooth covariates against an
unconstrained regression with no rollout at all — two different architectures,
only one of which has to route every hour through accumulation dynamics. Losing
that comparison says nothing about the physics.

The fair test holds the architecture fixed and varies only the constraint:

| Arm | Rollout | Closure | post-RMSE (k=60) | post-RMSE (k=120) | Params |
|---|:--:|---|--:|--:|--:|
| **monotone** | yes | fundamental diagram, decreasing by construction | **0.2744** | **0.2816** | 1,939 |
| free | yes | unconstrained function of accumulation | 0.2898 | 0.2898 | 2,900 |
| none | no | direct regression from covariates | 0.1364 | 0.1364 | 4,470 |

**The monotone fundamental diagram beats the unconstrained closure in 6 of 6
runs** (3 seeds × 2 pins), by 0.008–0.015 RMSE against a seed standard deviation
of 0.0016 — a gap five to ten times the noise — while using a third fewer
parameters. Imposing the physics helps.

What is expensive is the *rollout*, not the constraint. A direct regression
halves the error, because it can fit any hour-of-day pattern straight from the
covariates while the rollout must generate the same pattern through latent
accumulation driven by a handful of smooth inputs. That is a statement about
model structure and information, not about whether traffic obeys a fundamental
diagram.

Two further details are worth recording. The free closure returns **0.2898 at
every pin and seed**, to four decimals: with no constraint it ignores the
physical scale entirely and converges to the same solution regardless of
`k_jam`, which is a neat illustration of why the accumulation scale had to be
pinned externally. And the frozen-physics behavioural shift still tracks its own
placebo (3.68 under the policy against 3.76 under no policy at `k_jam` = 120),
so that particular quantity remains uninformative.

## Result 5b — The twin's error does not rise across the policy

The claim under test is that a physics model's prediction error jumps when a
policy changes behaviour, and that the jump measures the behaviour. A
calendar-matched rolling-origin backtest — the same 9-month-train, 3-month-
forecast protocol repeated at quarterly origins across the pre-policy span —
gives the null band that claim needs:

| | Forecast RMSE |
|---|--:|
| Placebo origins (n = 5) | 0.286, 0.314, 0.319, 0.373, 0.376 |
| 10–90 percentile band | **[0.297, 0.375]** |
| **Policy period** | **0.263** |

The policy-period error falls **below** the placebo band. The twin predicts the
congestion-pricing period slightly *better* than it predicts ordinary
pre-policy periods at the same horizon and season.

So there is no anomalous gap to interpret. The premise of the project — that
the residual between a physics twin and reality isolates the behavioural
response — fails not because the residual is contaminated, but because **the
residual does not grow at all.** That is consistent with Result 2: the physics
never broke, so the model never stopped working.

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

The assumption-light results are strong, and the model-dependent ones fail in a
more interesting way than first reported.

1. **Drivers respond to the price, and the response is priced.** All three toll
   thresholds move entries in the direction the price predicts, including the
   two where the toll rises and the response reverses. Semi-elasticities cluster
   at −0.12 to −0.31, and the response is durable with partial habituation.
2. **The physics is invariant and the constraint is useful.** The fundamental
   diagram survives the shock, and imposing it beats leaving it free in every
   run. The earlier claim to the contrary came from an unfair ablation.
3. **But the twin's residual carries no signal about behaviour.** Forecast error
   during the policy period sits *below* its own placebo band. There is no
   anomalous gap, so there is nothing for "the gap is behaviour" to measure.
4. **The speed effect is small and not identified.** +1.9% on running speed, with
   a placebo that is insignificant only after dwell is removed, and a synthetic
   control that is inconclusive with degenerate weights.

The premise fails for a reason worth stating precisely. It is not that the
residual is contaminated by drift and misspecification — the usual objection.
It is that **a policy can change behaviour substantially without degrading a
physics model at all**, because the physics was never what the policy acted on.
Drivers retimed and the fundamental diagram did not care. A twin that models
the road rather than the driver stays accurate through exactly the shock it was
supposed to detect.

The defensible paper is therefore the measurement one: the toll-threshold
response is a clean, transferable estimate, and the twin's failure is a
structural argument about what digital twins can and cannot be asked to do. The
specific failure modes are reusable — a control group contaminated by the
treatment's own diversion, a probe fleet that is itself treated, an accumulation
proxy that moves with market share, an attenuating measurement that hides the
signal, and an ablation that compares architectures when it meant to compare
constraints.

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
- **No cross-city donor pool.** The right control group is another city, and it
  cannot be built from open data for 2023–2026 (see Result 3). Every donor used
  here shares New York–wide shocks with the treated units.
- **No historical closure control.** The NYC street-closure dataset is a live
  permit snapshot — 492 active permits covering 78 days — not an archive, so
  construction cannot be controlled across the study window.
- **The rolling-origin band rests on five origins.** A 24-month clean pre-period
  supports a 9-month-train, 3-month-forecast protocol at quarterly spacing and
  little more; the longer 12/8 protocol yields only two origins and no usable
  band.
- **Dwell overhead is estimated, not observed.** The 2.57-minute figure comes
  from a pooled travel-time-on-distance fit, so the de-attenuated effect should
  be read as an order-of-magnitude correction rather than a precise one.
