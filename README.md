# eeg-state-estimator

A multitaper spectral front end and a recursive state estimator for single-channel EEG —
built primarily to demonstrate **how such a component is verified**.

> **Scope.** This is a demonstration core built from open literature and exercised on
> synthetic signals with analytically known answers. It is **not a medical device**, it has
> **not been clinically validated**, and it is **not fit for any patient use**. See
> [What is deliberately out of scope](#what-is-deliberately-out-of-scope).

---

## Start here: the verification approach

The algorithms in this repository are standard. What is worth reading is how they are checked.

**1. Tests assert against analytic oracles, not against a reference implementation.**
Comparing one implementation to another proves the two agree. Comparing against a closed-form
answer proves the implementation is *right*. So a sinusoid of amplitude `A` must return power
`A²/2`; a PSD must satisfy Parseval against the time-domain variance; `1/f^a` noise must return
slope `a`; a Kalman filter's posterior variance must converge to the Riccati fixed point, which
is a quadratic with a closed-form root. Every tolerance is stated in
[`requirements.md`](requirements.md) rather than buried in a test, so the acceptance criterion
and the code that checks it cannot drift apart.

**2. A recursive estimator is checked for statistical consistency, not just accuracy.**
An estimator that is accurate on average but reports the wrong uncertainty is not usable for
anything that has to act on its own confidence. So the filter is checked four ways — all of
which it can fail while still producing a plausible-looking trajectory:

| Check | What it would catch |
|---|---|
| Self-simulation recovery | the filter does not actually track |
| Credible-interval coverage over ≥200 runs | the reported uncertainty is wrong even though the estimate is fine |
| Innovation whiteness (Ljung–Box) | the model is mis-specified — structure is left in the residual |
| NEES against its chi-square bound | the filter is over- or under-confident |

**3. Requirements are traced to tests mechanically.**
Every test carries `@pytest.mark.req("REQ-0XX")`. [`docs/traceability-matrix.md`](docs/traceability-matrix.md)
is generated from those markers, and the suite **fails** if any requirement in
[`requirements.md`](requirements.md) has no test, or if a test cites a requirement that does not
exist. CI additionally fails if the committed matrix is stale. The document cannot quietly drift
out of agreement with the code.

**4. Test-first, visibly.** The git history shows the failing test committed before the
implementation that satisfies it. Some intermediate commits therefore have red CI. That is what
test-first looks like when it is not reconstructed afterwards.

---

## Status

Built in tiers, each one finished before the next begins.

- [x] **Tier 0** — scaffold, requirements, traceability generator, CI
- [x] **Tier 1** — synthetic generators, multitaper spectral core, scalar Kalman filter with the consistency battery, documentation

**132 tests, 29 of 29 requirements verified**, ruff and mypy clean.
- [x] **Tier 2** — streaming interface, online/offline equivalence, strict causality
- [ ] **Tier 3** — one of: artifact robustness, burst suppression, public-data demonstration

---

## Install

```bash
git clone <repository-url>
cd eeg-state-estimator
pip install -e ".[dev]"
```

Requires Python 3.11 or newer. Runtime dependencies are numpy and scipy only.

```bash
pytest -q          # full suite; regenerates the traceability matrix
ruff check .       # lint
mypy               # type-check
```

## Usage

```python
import numpy as np
from eeg_state_estimator.spectral import multitaper_psd, find_peak, band_powers
from eeg_state_estimator.statespace import RandomWalkModel, filter_series

# A spectral feature per epoch, from the raw trace.
spectrum = multitaper_psd(epoch, sample_rate=128.0)
peak = find_peak(spectrum, search_low=8.0, search_high=12.0)   # None if there is no peak
powers = band_powers(spectrum)                                  # slow_delta, theta, alpha, beta

# Track one of those features over time, with an uncertainty on every estimate.
model = RandomWalkModel(process_variance=0.05, observation_variance=1.0)
track = filter_series(np.log(alpha_power_per_epoch), model)
lower, upper = track.credible_interval(0.95)
```

The second half is the point of the package. `track.variance` is a posterior variance, so
the estimator reports how confident it is — and `track.innovation` with
`track.innovation_variance` lets a caller check, on data with no ground truth, whether that
confidence is justified.

---

## Layout

```
src/eeg_state_estimator/
    types.py            shared array type aliases
    spectral.py         multitaper PSD on DPSS tapers, band powers, spectral edge
    statespace/
        kalman.py       scalar Kalman filter, random-walk process model
        diagnostics.py  NEES, NIS, Ljung-Box, interval coverage
tests/
    traceability.py     requirements <-> tests, as three pure functions
    synthetic/          generators with analytically known answers
    spectral/           spectral estimation against those answers
    statespace/         filter consistency
requirements.md         numbered, testable, with explicit tolerances
docs/
    design-decisions.md every significant choice, its rationale, the rejected alternative
    ai-assisted-development.md
    traceability-matrix.md   generated — do not edit
```

---

## Design notes worth calling out

Full rationale in [`docs/design-decisions.md`](docs/design-decisions.md). Three that a reader
might otherwise flag as mistakes:

- **The peak PSD bin is not the signal power.** Multitaper deliberately spreads a spectral line
  over the analysis bandwidth `W = NW/T`. Recovering a sinusoid's power means *integrating* over
  that bandwidth; reading the peak bin overestimates it. The test integrates.
- **Peak frequency is a power-weighted centroid, not the `argmax` bin.** Averaging `K`
  concentrated tapers turns a spectral line into a flat-topped plateau of width `2W`, so the
  argmax is pinned by taper ripple rather than by the true frequency — its error is bounded by
  `W`, not by the bin width, and zero-padding does not help. Measured argmax error at a 4-second
  record is 0.32 Hz, which fails the ±0.25 Hz requirement outright. The plateau is symmetric
  about the true frequency, so its first moment is correct where its maximum is not: the centroid
  gives 0.0096 Hz.
- **NEES is a simulation-only diagnostic.** It needs the true state. On real data the analogue is
  NIS, which is built only from the innovation and its variance. The two are separate functions so
  that the distinction is impossible to miss.

---

## What is deliberately out of scope

Named explicitly, because a small finished component is more useful than a large unfinished one:

- Clinical interpretation of any kind, and any claim of validation
- Machine-learning state classification
- Closed-loop control or drug-delivery modelling
- Multi-channel montages, source localisation, connectivity
- Any GUI, dashboard, or deployment tooling
- Redistribution of any dataset — no data is committed to this repository

---

## License

MIT. See [LICENSE](LICENSE).
