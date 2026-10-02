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
