# Requirements

Each requirement has an ID, a statement, an acceptance criterion with an explicit numerical
tolerance, and the **measured margin** that tolerance was chosen against. Every ID must be
verified by at least one test marked `@pytest.mark.req("REQ-0XX")`;
`tests/test_traceability.py` fails the suite if any ID here has no test, and
`docs/traceability-matrix.md` is generated from the markers.

Tolerances live here rather than in the tests so that the acceptance criterion and the code
that checks it cannot drift apart silently.

**The margins are measured, not assumed.** Several tolerances in the first draft of this
document were unsatisfiable or statistically fragile; they were corrected after measurement
rather than after a test failed. `docs/design-decisions.md` records each one.

**Scope.** This is a demonstration core, not a clinical device. See README for what is
deliberately out of scope.

---

## Contracts

| ID | Requirement | Acceptance criterion | Measured margin |
|---|---|---|---|
| REQ-001 | Inputs shall be validated and rejected loudly rather than produce a silently wrong result. | A non-finite sample, an empty or non-1-D array, a non-positive sample rate, or a taper count exceeding `2·NW − 1` each raise `ValueError` naming the offending argument. | n/a — behavioural |
| REQ-002 | Results shall be deterministic. | The same input produces bitwise-identical output across repeated calls. Generators take an explicit seed; the legacy `numpy.random` global API is banned by lint. | n/a — exact |

## Spectral estimation

| ID | Requirement | Acceptance criterion | Measured margin |
|---|---|---|---|
| REQ-010 | The PSD shall be estimated by the multitaper method on DPSS (Slepian) tapers, with time-bandwidth `NW` and taper count `K` configurable. | `K` defaults to `2·NW − 1`; a larger `K` is rejected. Tapers carry unit energy (`Σ wₖ[n]² = 1`), asserted directly. One-sided axis from DC to Nyquist. | Taper energy exact to 1e-15 |
| REQ-011 | For a noiseless sinusoid, the peak frequency shall be recovered. | Within **±0.25 Hz**, for off-bin frequencies, record length ≥ 4 s. | **0.0096 Hz** worst case over 60 off-bin frequencies at T = 4 s — a 26× margin |
| REQ-012 | For a noiseless sinusoid of amplitude `A`, the signal power shall be recovered. | Power integrated over the analysis bandwidth about the peak within **±5%** of `A²/2`. | **0.8%** worst case over NW ∈ {2,3,4} × T ∈ {2,4,8} s — a 6× margin |
| REQ-013 | The PSD normalization shall satisfy Parseval's relation exactly. | `Σ S₁[m]·Δf` equals the taper-weighted mean square of the record to `rtol = 1e-12`. | **2.2e-16** — machine precision. Deterministic; catches any regression in the taper normalization or the one-sided doubling rule |
| REQ-014 | The PSD shall report physically correct absolute power. | For a deterministic sum of sinusoids, `Σ S₁[m]·Δf` is within **±2%** of `Σ Aᵢ²/2`. | **1e-5** for a single sinusoid |
| REQ-015 | For white noise, the PSD shall be flat at the correct absolute level. | Over an ensemble of **50 fixed-seed records**, the mean PSD in each of the bands 1–10, 10–20, 20–30, 30–40 Hz is within **±10%** of the analytic one-sided level `2σ²/fs`. | **1.6%** at T = 8 s, M = 50 — a 6× margin |
| REQ-016 | For `1/f^a` noise, the spectral slope shall be recovered. | Slope from log-log least squares over **2–40 Hz** within **±0.15** of `a`, for `a` ∈ [0.5, 3.0]. | **0.083** worst case at T = 4 s — a 1.8× margin |
| REQ-017 | Power shall be reported in named frequency bands. | Slow-delta (0.5–4), theta (4–8), alpha (8–12), beta (12–30) Hz. Band powers over a partition sum to the total over that range within **±1%**. | Exact up to trapezoid-rule error |
| REQ-018 | The spectral edge frequency shall be reported. | SEF at a configurable fraction (default 0.95), non-decreasing in the fraction. Verified against hand-constructed PSDs with closed-form answers. | Exact on analytic PSDs |
| REQ-019 | Spectral estimation shall be scale-equivariant. | Scaling the input by `c` scales the PSD by `c²` within `rtol = 1e-10`. | Exact to floating point |
| REQ-020 | Spectral estimation shall be independent of the sample rate used to represent the same signal. | The same continuous-time sinusoid sampled at different rates satisfies REQ-011 and REQ-012. | Inherits those margins |

> **Why REQ-013 and REQ-014 are separate.** The first draft had a single requirement comparing
> the PSD integral to the time-domain variance of white noise within ±2%. That measures the
> *sampling error of a variance estimate*, not the normalization: measured worst case 2.76% at
> T = 4 s and 2.21% at T = 8 s, so the requirement failed on short records while the code was
> correct. Splitting it gives one deterministic requirement that actually pins the normalization
> (REQ-013, exact) and one that states the physically meaningful claim (REQ-014).

> **Why REQ-015 is an ensemble requirement.** The first draft asked for the PSD of white noise
> to be flat within ±10% bin-to-bin. That is unsatisfiable by any correct multitaper
> implementation — the per-bin relative standard deviation is `1/√K` = 38% for `K = 7`, and
> measurement gives 200–236% worst-case per-bin deviation regardless of record length. Flatness
> is a property of the *expected* spectrum, so it is tested over an ensemble, against the
> analytic level rather than against the record's own mean. The result is a stricter test: it
> checks flatness and absolute normalization together.

## Recursive state estimation

| ID | Requirement | Acceptance criterion | Measured margin |
|---|---|---|---|
| REQ-030 | A scalar Kalman filter with a random-walk process model shall report a posterior variance with every state estimate. | Every update returns the posterior mean and its variance; the variance is strictly positive and cannot go negative under any rounding. | Exact — the update form contains no subtraction |
| REQ-031 | The filter shall expose the innovation and the innovation variance at every step. | Both returned by every update, without access to internal state. They are the only observables on which correctness can be assessed without ground truth. | n/a — API contract |
| REQ-032 | The filter shall recover a known latent trajectory from data generated by its own model. | On self-simulated data, RMSE against the true state is at least 2× lower than the RMSE of the raw observations. | ~3× at Q = 0.02, R = 1.0 |
| REQ-033 | Reported credible intervals shall be calibrated. | Across **500** independent fixed-seed runs, nominal 95% intervals contain the true state **95% ± 3%**. | Binomial SE at M = 500 is 0.0097, so ±3% is a 3.1σ bound |
| REQ-034 | Filter innovations shall be white under a correct model. | Ljung–Box on the normalized innovations does not reject at **α = 0.05**, with `p = 0` fitted parameters (Q and R are given, not estimated). | Rejection rate 0.056 against a nominal 0.05 |
| REQ-035 | The filter shall be consistent in the normalized-estimation-error sense. | Mean NEES **across ≥200 independent runs at a fixed time index** falls within the two-sided 95% chi-square bound, dof = number of runs. | Measured 0.971 against bounds [0.866, 1.143] at M = 400 |
| REQ-036 | The filter shall converge from a deliberately poor initialization. | From a state estimate far from the truth, the estimate reaches and stays within 3 posterior standard deviations within a stated number of steps, and the posterior variance converges to the Riccati fixed point independently of its initial value. | Riccati fixed point matched to 1e-6 |
| REQ-037 | Each consistency diagnostic shall be shown to **reject** a knowably wrong filter. | With Q 10× too small, Q 10× too large, or R mis-scaled, the corresponding diagnostic reports failure. | Ljung–Box rejection rate 1.000 at Q 10× too small |

> **Why REQ-035 specifies "across runs at a fixed time index".** NEES averaged *over time within
> one run* is mis-calibrated, because the estimation-error sequence is strongly autocorrelated
> (measured lag-1 ≈ 0.68) while the innovation sequence is white by construction. A time-averaged
> NEES test against a chi-square bound passes only about 65% of the time on a correct filter — it
> would pass on the seed it was written with and fail intermittently afterwards. The innovation-based
> analogue (NIS) *can* be averaged over time, and is the diagnostic that works on real recordings
> where the true state is unavailable.

> **Why REQ-037 exists.** A diagnostic suite that only ever reports PASS is indistinguishable from
> a suite of `assert True`. Each diagnostic is therefore pointed at a deliberately broken filter and
> asserted to fail.

## Streaming

| ID | Requirement | Acceptance criterion | Measured margin |
|---|---|---|---|
| REQ-050 | The library shall provide a streaming estimator interface. | An abstract base class declaring `update(chunk)`, `reset()` and `state_nbytes`. The Kalman filter and the composed pipeline both satisfy it without adaptation. | n/a — API contract |
| REQ-051 | Chunked streaming output shall equal whole-record batch output. | Feeding a record in arbitrary chunk sizes produces output identical to feeding it in one call, within `rtol = 1e-9`. | **Bitwise identical** — there is one implementation, not two |
| REQ-052 | The causal path shall contain no look-ahead. | The output at epoch `t` is unchanged when arbitrary future samples are appended to the record. | **Bitwise identical** |
| REQ-053 | Outputs produced before the estimator has converged shall be flagged, never emitted as valid values. | Each output carries an explicit validity flag and a status. Warm-up is defined by the posterior variance still exceeding the Riccati fixed point by more than a stated tolerance — a self-calibrating criterion rather than a fixed epoch count. | n/a — behavioural |
| REQ-054 | Retained internal state shall be constant in the length of the record. | `state_nbytes` after processing 10⁶ samples equals its value after 10³. | Exact |
| REQ-055 | The pipeline shall compose preprocessing, spectral estimation and recursive filtering into one streaming estimator. | A single `update(chunk)` call takes raw samples and returns per-epoch state estimates with credible intervals. | n/a — behavioural |

> **Why REQ-051 and REQ-052 come out bitwise identical rather than merely within tolerance.**
> `filter_series` and the pipeline's batch path contain no arithmetic of their own — both
> construct the estimator and call `update()`. With a single implementation there is no
> second code path to drift, so the requirement's `1e-9` tolerance is never approached. The
> tolerance is retained in the requirement because it is the honest bound to promise for a
> floating-point pipeline; meeting it exactly is a property of this design, not a guarantee
> of the specification.

## Verification infrastructure

| ID | Requirement | Acceptance criterion | Measured margin |
|---|---|---|---|
| REQ-040 | Every requirement in this document shall be verified by at least one test, and no test shall cite a requirement that does not exist. | `tests/test_traceability.py` parses the IDs from this file and fails, naming them, if either holds. | n/a — exact |
| REQ-041 | The traceability matrix shall be generated from the test suite, not maintained by hand. | `docs/traceability-matrix.md` is written from `@pytest.mark.req` markers parsed out of the test sources; CI fails if the committed copy is stale. | n/a — exact |
