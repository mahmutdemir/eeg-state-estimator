# Design decisions

Every significant choice, why it was made, and what was rejected. Maintained as the work
proceeds rather than reconstructed afterwards.

---

## Tier 0 — scaffold and verification infrastructure

### Tolerances live in `requirements.md`, not in the tests

**Decision.** Each requirement states its numerical acceptance criterion in the requirements
document; the test cites the requirement ID and checks that bound.

**Why.** If the tolerance exists only inside the test, then loosening a failing test silently
weakens the specification and nothing records that it happened. With the bound written down
separately, a test that no longer meets it visibly contradicts a document under version control.

**Rejected.** Tolerances as constants at the top of each test module. Convenient, but it makes the
specification a derivative of the code rather than the other way round.

### Traceability is scanned from the test sources, not from pytest's collected items

**Decision.** `tests/traceability.py` reads the test tree directly. `docs/traceability-matrix.md`
is regenerated at the end of every session.

**Why.** The obvious implementation is a `pytest_collection_modifyitems` hook reading markers off
collected items, and it has a serious flaw: the collected set depends on how the suite was invoked.
Running `pytest tests/spectral` would regenerate the matrix from a fraction of the tests and report
every other requirement as untested. A traceability document that changes meaning depending on the
command line is worse than none, because it looks authoritative.

A source scan is invocation-independent and reproducible without executing anything.

**Rejected.** The collection hook, for the reason above. Also rejected: a hand-maintained matrix,
which is guaranteed to drift.

**Known limitation.** Markers applied dynamically at runtime are invisible to a static scan. This
suite applies every marker as a literal decorator; `--strict-markers` and the undeclared-ID test
catch the mistakes this could otherwise hide.

### The marker scan parses the AST rather than matching a regular expression

**Decision.** Walk the parsed module and match `pytest.mark.req(...)` call nodes.

**Why.** A regex over the file text was written first, and this suite's own test caught the
problem: several tests hold marker-shaped text inside *string literals* — fixture files written to
a temporary directory to exercise the scanner. A text scan counted those as genuine coverage, which
means a requirement could appear verified by a test that only ever quoted its ID. Matching call
nodes requires a marker to be code before it counts.

**Rejected.** The regex. It is shorter and it is wrong.
`test_scan_req_markers_ignores_string_literals` is the regression test.

### CI fails if the committed traceability matrix is stale

**Decision.** After the test job, `git diff --exit-code -- docs/traceability-matrix.md`.

**Why.** The matrix is generated, so it can fall behind the markers it summarises. Since the
generated output is deterministic, a difference means someone changed a marker without
regenerating — and the document would then misreport coverage, which is the one thing it must
never do.

**Rejected.** Generating the matrix in CI and committing it from the pipeline. That puts the
build in the business of writing to the repository, which is a much larger mechanism than a
one-line check.

### mypy is strict on this package, but not `strict = true`

**Decision.** `disallow_untyped_defs`, `disallow_incomplete_defs`, `no_implicit_optional`,
`warn_return_any`, `warn_unused_ignores`, `warn_redundant_casts`, `strict_equality`.

**Why.** Full strict mode enables `disallow_any_generics`, which in numpy-heavy code produces a
continuous stream of complaints about array generics that are resolved by writing `Any` or adding
ignores — noise that makes the real errors harder to see. The flags chosen catch the mistakes that
actually occur here: an untyped function, an implicit `Optional`, a function quietly returning
`Any`.

**Rejected.** `strict = true` (noise), and no mypy at all (gives up a real class of error).

### ruff ignores `N803` and `N806` only

**Decision.** Allow uppercase argument and variable names.

**Why.** State-space code is written everywhere — textbooks, papers, every reference
implementation — with the matrices named `A`, `Q`, `C`, `R` and the Kalman gain `K`. Renaming them
to `a`, `q`, `c`, `r` to satisfy a lowercase convention would make the implementation harder to
check line-by-line against a published derivation, which is the main way anyone will verify it is
correct.

**Rejected.** Renaming to satisfy the rule; a blanket `N` disable (too broad — the other naming
rules are useful).

### Runtime dependencies are numpy and scipy only

**Decision.** Test, lint and type-check tools live in a `dev` extra. Nothing else is a runtime
dependency.

**Why.** The spectral estimator is built directly on `scipy.signal.windows.dpss` and numpy's FFT,
and the filter needs nothing beyond numpy. Keeping the runtime surface this small means the package
is easy to audit and easy to vendor — relevant for anything that would eventually ship.

**Consequence.** Ljung–Box is implemented directly rather than taking a `statsmodels` dependency
for one statistic. See the Tier 1 notes when that lands.

### Line endings are normalised to LF via `.gitattributes`

**Decision.** `* text=auto eol=lf`.

**Why.** Developed on Windows, tested on Linux in CI. Without normalisation the checkouts differ
and `ruff format --check` can legitimately disagree between the two, producing a CI failure that
reproduces nowhere locally.

---

## Tier 0 — requirements corrected by measurement

Three tolerances in the first draft of `requirements.md` were wrong. They were found by
measuring them before writing any test, not by watching a test fail. Recorded here because the
corrections are more informative than the original numbers.

### Peak frequency is a power-weighted centroid, not the `argmax` bin

**Decision.** Report the power-weighted first moment of the PSD over `f_argmax ± 1.5·W`, where
`W = NW/T` is the analysis half-bandwidth.

**Why.** The first draft specified plain `argmax`, justified by "the worst-case error is half a
bin". That reasoning is wrong. Averaging `K` spectrally concentrated tapers does not produce a
peak over a spectral line — it produces a **flat-topped plateau of width ≈ 2W**. The argmax
within a plateau is set by taper ripple, so its error is bounded by `W`, not by the bin width.
Measured, noiseless sinusoid, 60 random off-bin frequencies, NW = 4:

| record | bin width | argmax error | centroid error |
|---|---|---|---|
| 2 s | 0.500 Hz | 0.621 Hz | 0.018 Hz |
| 4 s | 0.250 Hz | **0.316 Hz** | 0.0096 Hz |
| 8 s | 0.125 Hz | 0.160 Hz | 0.0049 Hz |
| 16 s | 0.062 Hz | 0.080 Hz | 0.0025 Hz |

At 4 s the argmax error *exceeds a full bin*, which disproves the half-a-bin argument directly.
Note the error halves as `W` halves, tracking the bandwidth rather than the resolution.

The plateau is symmetric about the true frequency, so its first moment is correct where its
maximum is not.

**Rejected.** Plain `argmax` (fails at ≤ 4 s). Zero-padding (does not help — it samples the
plateau more finely, but the plateau is still flat). Three-point parabolic interpolation on the
log-PSD (assumes a locally quadratic main lobe; fits a parabola to a tabletop).

**Bonus.** The centroid shares its integration band with the peak-power integral, so one private
helper returns the band and the two results are guaranteed consistent — the zeroth moment is the
power, the normalized first moment is the frequency.

### Parseval split into an exact normalization check and a physical-power check

**Decision.** REQ-013 asserts `Σ S₁·Δf` equals the taper-weighted mean square to `rtol = 1e-12`.
REQ-014 separately asserts ±2% physical power on a deterministic sum of sinusoids.

**Why.** The first draft compared the PSD integral to the time-domain variance of white noise
within ±2%. That is a statistical test wearing a normalization test's clothes: it measures the
sampling error of a variance estimate. Measured worst case over 40 seeds — 2.76% at T = 4 s,
2.21% at T = 8 s, 1.51% at 20 s. The requirement failed on short records while the code was
correct, which is the worst possible property for an acceptance criterion.

The identity against the *taper-weighted* mean square is exact: measured 2.2e-16. It is
deterministic, it runs in milliseconds, and it is what actually catches a regression in the taper
normalization or the one-sided doubling rule.

**Rejected.** Keeping the single ±2% requirement with a fixed seed and a stated record length.
Workable, but it would state a tolerance that looks like an accuracy claim and is really a
Monte-Carlo bound.

### White-noise flatness is an ensemble requirement against the analytic level

**Decision.** REQ-015 averages the PSD over 50 fixed-seed records and compares each of four band
means to the analytic one-sided level `2σ²/fs`, within ±10%.

**Why.** The first draft required the PSD of a single white-noise record to be flat within ±10%
bin-to-bin. **No correct multitaper implementation can satisfy that.** The per-bin relative
standard deviation is `1/√K` = 38% for `K = 7` — that is the estimator's design, not a defect.
Measured worst-case per-bin deviation: 200% at T = 8 s, 236% at 30 s, 229% at 60 s. It does not
improve with record length, because a longer record buys more bins at the same per-bin scatter.

Flatness is a property of the expected spectrum, so it has to be tested over an ensemble. Testing
against the analytic level rather than the record's own band mean makes the requirement strictly
stronger: it checks flatness and absolute normalization in one assertion. Measured margin 1.6%
against a ±10% bound.

### NEES is averaged across runs, NIS over time

**Decision.** REQ-035 specifies mean NEES across independent runs at a fixed time index.

**Why.** Averaging NEES over time within a single run is mis-calibrated. The innovation sequence
of a correctly specified filter is white by construction, so NIS values along one run are
independent and may be time-averaged. The *estimation error* sequence is not — the filter carries
error forward through the state, giving a measured lag-1 autocorrelation of ≈ 0.68 — so the
effective sample size is far below the sample count and the chi-square bound is too tight. A
time-averaged NEES test on a correct filter sits inside its nominal 95% bounds only about 65% of
the time: it would pass on the seed it was developed with and fail intermittently thereafter.

This is also the practical reason the two diagnostics are not interchangeable. NEES needs the true
state and is simulation-only; NIS is built from the innovation and its variance alone and is the
one that works on a real recording.

### DPSS tapers are requested with explicit unit energy

**Decision.** Call `scipy.signal.windows.dpss(..., norm=2)` explicitly, and assert
`Σ wₖ[n]² = 1` in a test.

**Why.** The PSD normalization derivation assumes unit-energy tapers. SciPy's default happens to
give unit energy for the multi-taper call, but the normalization is a parameter of that API and a
different choice (`'approximate'`, which normalizes to unit *peak*) would scale every PSD by
roughly `N/4` with no error and no warning. Making the assumption explicit at the call site, and
asserting it in a test, converts a silent dependency into a checked one.

---

## Tier 1 — implementation

### The posterior variance is `P · R / S`, not `(1 − K·C) · P`

**Decision.** `ScalarKalmanFilter.step` computes `posterior_variance = prior_variance * R / S`.

**Why.** The two are algebraically identical: substituting `K = P·C/S` into the Joseph form
`(1 − K·C)²·P + K²·R` and simplifying gives exactly `P·R/S`. But the conventional form
constructs `1 − K·C` as a *difference*, and `K → 1` is precisely what a diffuse prior
produces. Measured against the exact value with `R = 2`:

| `P₀` | naive `(1 − K)·P` | `P·R/S` | exact |
|---|---|---|---|
| 1e6 | 1.9999960000 | 1.9999960000 | 1.9999960000 |
| 1e12 | 1.9999557566 | 2.0000000000 | 2.0000000000 |
| 1e16 | **2.2204460493** | 2.0000000000 | 2.0000000000 |
| 1e20 | **0.0000000000** | 2.0000000000 | 2.0000000000 |

At `P₀ = 1e16` the naive form is 11% high. At 1e20 it returns **exactly zero** — a filter
asserting infinite certainty, after which the Kalman gain is zero forever and the estimator
never listens to data again. That is not a precision nuisance; it is a silent, permanent
failure mode reachable from an ordinary diffuse initialisation.

The chosen form is a quotient of strictly positive quantities. It cannot cancel, and it
cannot return a negative variance under any rounding.

**Rejected.** The naive form (above). The explicit Joseph form — bit-identical here, but in
the scalar case it is three operations where one will do, and `P·R/S` *is* the Joseph form
collapsed. The vector generalisation is then the standard Joseph form for the same reason,
so nothing about this implementation has to be unlearned.

### Analysis functions take a `Spectrum`, not a signal

**Decision.** Only `multitaper_psd` consumes a signal. `band_power`,
`spectral_edge_frequency`, `find_peak` and `fit_power_law` all take a `Spectrum`.

**Why.** It makes them testable against exact answers. The spectral edge of a flat PSD is
`low + fraction·(high − low)` exactly; the spectral edge of an *estimated* white-noise
spectrum has about 1.5 Hz of seed-to-seed scatter, which is roughly 15× the difference
between the three plausible SEF conventions. Coupling the two would mean a test that
measures the seed. Separated, the algorithm gets a precise test and the estimator gets a
loose integration test, which is two useful tests instead of one vague one.

**Rejected.** A convenience API taking `(signal, sample_rate)` throughout. It reads better
in a single call and makes every downstream function untestable except through the estimator.

### `find_peak` returns `None` rather than raising or guessing

**Decision.** No interior local maximum in the search range returns `None`.

**Why.** The absence of a spectral peak is a real state, not an error and not a number.
Deep anesthetic suppression has no alpha peak. Raising would make a normal physiological
condition an exception; returning the argmax anyway would return a confident meaningless
frequency. This is the same principle as making "no valid signal" a distinct output state
rather than a value on the measurement scale.

### Ljung–Box is implemented directly rather than adding a statistics dependency

**Decision.** Eleven lines in `diagnostics.py`, instead of `statsmodels`.

**Why.** It is one statistic, the formula is short, and the runtime dependency surface stays
at numpy and scipy — which matters for anything that would eventually ship. It also forces
the `n_fitted_parameters` question to be answered explicitly: it is **0** here, because `Q`
and `R` are given rather than estimated from the data, so no degrees of freedom are consumed.
Passing a non-zero `p` when nothing was fitted is the most common error in applying this
test, and a library call makes it easy to get wrong without noticing.

### The per-sample kernel is `step()` and the chunk method is `update()`

**Decision.** Both exist from the start, with `update()` looping `step()`.

**Why.** A streaming interface's method is `update(chunk)`. Had `update` meant "one scalar"
here, adding streaming later would either break this API and its tests or leave `update`
meaning two different things in the same hierarchy. Five lines now avoids both.

`filter_series` constructs a filter and calls `update()` once, containing **no filtering
arithmetic of its own**. That single-implementation property is what makes a later
online/offline equivalence check trivially true rather than something to chase, and it means
the batch path cannot drift from the streaming path.

### `state_nbytes` is computed from a declared tuple, not by introspection

**Decision.** `len(_STATE_FIELDS) * 8`.

**Why.** `sys.getsizeof` reports CPython object overhead — 24 bytes for a bare float — which
is both misleading and unstable across interpreter versions. The property exists to express
the bounded-memory property, so it reports the numerical state carried between updates and
says so.

### The simulation-only diagnostics are 2-D with a required `time_index`

**Decision.** `nees_across_runs` and `credible_interval_coverage` take
`(n_runs, n_samples)` arrays and a keyword-only `time_index` with no default; `truth` is
their first positional parameter.

**Why.** The aggregation rule is the thing that is easy to get wrong, so the type signature
enforces it rather than a docstring requesting it. It is not possible to pass a single run,
and not possible to average over time by accident. Putting `truth` first with no default
means neither function can be reached without ground truth in hand — which is the property
that makes "simulation only" structural rather than advisory.

**Rejected.** Splitting into `diagnostics.py` and `simulation_diagnostics.py` so the
distinction appears in the import line. Arguably stronger, and worth revisiting, but at this
scope the naming convention plus the module-docstring table carries it without doubling the
module count.
