"""Regenerate every figure in docs/verification-report.md.

    python workbooks/make_figures.py

Everything here runs from synthetic signals with seeded generators, so the figures are
reproducible by anyone who clones the repository — no download, no cached data, no
network. That is also why this script can be re-run in CI if the figures are ever
suspected of having drifted from the code.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Run standalone: `python workbooks/make_figures.py` from the repository root. The
# package lives under src/ and the synthetic generators under tests/, neither of
# which is importable by default from this directory.
_ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(_ROOT / "src"), str(_ROOT / "tests"), str(_ROOT / "workbooks")]

import numpy as np
from matplotlib import pyplot as plt
from style import (
    BAD,
    BAND,
    ESTIMATE,
    GOOD,
    GRID,
    INK_SECONDARY,
    OBSERVED,
    SEQUENTIAL,
    THIRD,
    TRUTH,
    annotate,
    save,
    use_report_style,
)

from eeg_state_estimator.pipeline import EpochPipeline
from eeg_state_estimator.spectral import (
    MultitaperConfig,
    find_peak,
    fit_power_law,
    multitaper_psd,
)
from eeg_state_estimator.statespace import RandomWalkModel, filter_series
from eeg_state_estimator.statespace.diagnostics import (
    chi_square_consistency_bounds,
    credible_interval_coverage,
    nees_across_runs,
)
from synthetic.generators import power_law_noise, sinusoid, white_noise

FS = 128.0
Q_TRUE, R_TRUE = 0.05, 1.0


# ----------------------------------------------------------------------- figure 1
def figure_peak_estimation() -> None:
    """Why the peak frequency is a centroid and not the argmax bin."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), constrained_layout=True)

    f0, duration = 10.37, 4.0
    n = int(duration * FS)
    spectrum = multitaper_psd(
        sinusoid(n_samples=n, sample_rate=FS, frequency=f0, amplitude=2.0), sample_rate=FS
    )
    peak = find_peak(spectrum, search_low=2.0, search_high=45.0)
    assert peak is not None
    near = (spectrum.frequencies > f0 - 4.0) & (spectrum.frequencies < f0 + 4.0)
    argmax_f = float(spectrum.frequencies[np.argmax(spectrum.psd)])

    ax = axes[0]
    ax.plot(spectrum.frequencies[near], spectrum.psd[near], color=ESTIMATE, marker="o", ms=3)
    ax.axvline(f0, color=TRUTH, lw=1.6, label=f"true  {f0:.2f} Hz")
    ax.axvline(argmax_f, color=INK_SECONDARY, lw=1.2, ls=":", label=f"argmax  {argmax_f:.2f} Hz")
    ax.axvline(
        peak.frequency, color=GOOD, lw=1.6, ls="--", label=f"centroid  {peak.frequency:.2f} Hz"
    )
    ax.axvspan(peak.band_low, peak.band_high, color=BAND, alpha=0.08)
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD  (µV²/Hz)")
    ax.set_title("A multitaper line is a plateau, not a peak", loc="left")
    ax.legend(loc="upper left")
    annotate(ax, f"shaded: ±1.5·W\nW = NW/T = {spectrum.half_bandwidth:.2f} Hz", loc="upper right")

    ax = axes[1]
    durations = np.array([2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0])
    rng = np.random.default_rng(0)
    probe = rng.uniform(6.0, 26.0, 60)
    argmax_err, centroid_err = [], []
    for duration in durations:
        n = int(duration * FS)
        a_err, c_err = [], []
        for frequency in probe:
            spec = multitaper_psd(
                sinusoid(n_samples=n, sample_rate=FS, frequency=frequency, amplitude=2.0),
                sample_rate=FS,
            )
            a_err.append(abs(float(spec.frequencies[np.argmax(spec.psd)]) - frequency))
            found = find_peak(spec, search_low=2.0, search_high=45.0)
            c_err.append(abs(found.frequency - frequency) if found else np.nan)
        argmax_err.append(max(a_err))
        centroid_err.append(np.nanmax(c_err))

    ax.semilogy(durations, argmax_err, color=TRUTH, marker="o", ms=5, label="argmax bin")
    ax.semilogy(durations, centroid_err, color=ESTIMATE, marker="s", ms=5, label="power centroid")
    ax.axhline(0.25, color=BAD, lw=1.4, ls="--")
    ax.text(
        durations[-1],
        0.27,
        "REQ-011 limit  0.25 Hz",
        color=BAD,
        fontsize=8,
        ha="right",
        va="bottom",
    )
    ax.set_xlabel("record length (s)")
    ax.set_ylabel("worst |error| over 60 frequencies (Hz)")
    ax.set_title("argmax fails the requirement; the centroid clears it 25×", loc="left")
    ax.legend(loc="lower left")
    save(fig, "fig01_peak_estimation.png")


# ----------------------------------------------------------------------- figure 2
def figure_spectral_oracles() -> None:
    """The four spectral requirements checked against closed-form answers."""
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)

    # (a) white-noise ensemble against the analytic level
    ax = axes[0]
    variance, n_records = 2.0, 50
    n = int(8 * FS)
    total = None
    for seed in range(n_records):
        spec = multitaper_psd(
            white_noise(n_samples=n, sample_rate=FS, variance=variance, seed=seed), sample_rate=FS
        )
        total = spec.psd if total is None else total + spec.psd
    assert total is not None
    mean_psd = total / n_records
    single = multitaper_psd(
        white_noise(n_samples=n, sample_rate=FS, variance=variance, seed=0), sample_rate=FS
    )
    band = (spec.frequencies >= 1.0) & (spec.frequencies <= 45.0)
    analytic = 2.0 * variance / FS

    ax.plot(spec.frequencies[band], single.psd[band], color=OBSERVED, lw=0.8, label="one record")
    ax.plot(spec.frequencies[band], mean_psd[band], color=ESTIMATE, lw=1.6, label="mean of 50")
    ax.axhline(analytic, color=TRUTH, lw=1.6, ls="--", label="analytic  2σ²/fs")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("PSD  (µV²/Hz)")
    ax.set_title("Flatness is an ensemble property", loc="left")
    ax.legend(loc="upper right")
    annotate(ax, "single record: ±200%\nmean of 50: ±1.6%", loc="lower right")

    # (b) 1/f^a slope recovery
    ax = axes[1]
    exponents = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    recovered = []
    for exponent in exponents:
        spec = multitaper_psd(
            power_law_noise(
                n_samples=int(8 * FS), sample_rate=FS, exponent=float(exponent), knee=1.0, seed=0
            ),
            sample_rate=FS,
        )
        recovered.append(fit_power_law(spec, low=2.0, high=40.0).exponent)
    recovered = np.array(recovered)

    ax.plot([0.3, 3.2], [0.3, 3.2], color=TRUTH, lw=1.4, ls="--", label="exact")
    ax.plot(exponents, recovered, color=ESTIMATE, marker="o", ms=5, ls="none", label="recovered")
    ax.fill_between([0.3, 3.2], [0.15, 3.05], [0.45, 3.35], color=BAND, alpha=0.10)
    ax.set_xlabel("true exponent  a")
    ax.set_ylabel("recovered exponent")
    ax.set_title("Spectral slope, 2–40 Hz fit", loc="left")
    ax.legend(loc="upper left")
    annotate(ax, f"shaded: ±0.15 limit\nworst: {np.max(np.abs(recovered - exponents)):.3f}")

    # (c) sinusoid power by band integration
    ax = axes[2]
    amplitudes = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    integrated, peak_bin = [], []
    for amplitude in amplitudes:
        spec = multitaper_psd(
            sinusoid(
                n_samples=int(8 * FS), sample_rate=FS, frequency=11.0, amplitude=float(amplitude)
            ),
            sample_rate=FS,
        )
        found = find_peak(spec, search_low=2.0, search_high=45.0)
        assert found is not None
        integrated.append(found.power)
        peak_bin.append(found.peak_psd)

    truth_power = amplitudes**2 / 2.0
    ax.loglog(truth_power, truth_power, color=TRUTH, lw=1.4, ls="--", label="exact  A²/2")
    ax.loglog(
        truth_power, integrated, color=ESTIMATE, marker="o", ms=5, ls="none", label="band integral"
    )
    ax.loglog(
        truth_power, peak_bin, color=INK_SECONDARY, marker="x", ms=6, ls="none", label="peak bin"
    )
    ax.set_xlabel("true power  A²/2  (µV²)")
    ax.set_ylabel("recovered  (µV²)")
    ax.set_title("The peak bin is not the power", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig02_spectral_oracles.png")


# ----------------------------------------------------------------------- figure 3
def figure_filter_tracking() -> None:
    """Tracking from a deliberately wrong start, and the data-free variance schedule."""
    fig, axes = plt.subplots(
        2, 1, figsize=(10.0, 5.4), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )
    rng = np.random.default_rng(2)
    n = 400
    truth = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), n))
    observed = truth + rng.normal(0.0, np.sqrt(R_TRUE), n)
    model = RandomWalkModel(Q_TRUE, R_TRUE)
    track = filter_series(observed, model, initial_mean=8.0, initial_variance=1.0)
    lower, upper = track.credible_interval(0.95)
    steps = np.arange(n)

    ax = axes[0]
    ax.plot(steps, observed, ".", color=OBSERVED, ms=3, label="observations")
    ax.plot(steps, truth, color=TRUTH, lw=1.6, label="true state (unobservable)")
    ax.plot(steps, track.mean, color=ESTIMATE, lw=1.6, label="filtered estimate")
    ax.fill_between(steps, lower, upper, color=BAND, alpha=0.18, label="95% credible interval")
    ax.set_ylim(-4.5, 9.5)
    ax.set_ylabel("state")
    ax.set_title("Convergence from a deliberately wrong start, $\\hat{x}_0 = 8$", loc="left")
    ax.legend(loc="upper right", ncol=2)

    ax = axes[1]
    ax.plot(steps, np.sqrt(track.variance), color=ESTIMATE, lw=1.6)
    ax.axhline(
        np.sqrt(model.steady_state_posterior_variance()),
        color=TRUTH,
        lw=1.4,
        ls="--",
        label="Riccati fixed point",
    )
    ax.set_xlabel("time step")
    ax.set_ylabel("posterior sd")
    ax.legend(loc="upper right")
    annotate(ax, "this curve never touches the data", loc="lower right")
    save(fig, "fig03_filter_tracking.png")


# ----------------------------------------------------------------------- figure 4
def _run_bank(model: RandomWalkModel, truth: np.ndarray, observed: np.ndarray):
    mean = np.empty_like(observed)
    variance = np.empty_like(observed)
    for index, row in enumerate(observed):
        track = filter_series(row, model)
        mean[index] = track.mean
        variance[index] = track.variance
    return mean, variance


def figure_consistency() -> None:
    """NEES and coverage for a correct filter and three knowably wrong ones."""
    n_runs, n_samples, at = 400, 320, 300
    rng = np.random.default_rng(0)
    truth = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), (n_runs, n_samples)), axis=1)
    observed = truth + rng.normal(0.0, np.sqrt(R_TRUE), (n_runs, n_samples))

    cases = [
        ("correct", RandomWalkModel(Q_TRUE, R_TRUE)),
        ("Q 10× small", RandomWalkModel(Q_TRUE * 0.1, R_TRUE)),
        ("Q 10× large", RandomWalkModel(Q_TRUE * 10.0, R_TRUE)),
        ("R 10× large", RandomWalkModel(Q_TRUE, R_TRUE * 10.0)),
    ]
    nees_values, coverage_values, rmse_values = [], [], []
    for _, model in cases:
        mean, variance = _run_bank(model, truth, observed)
        nees_values.append(nees_across_runs(truth, mean, variance, time_index=at).statistic)
        coverage_values.append(
            credible_interval_coverage(truth, mean, variance, time_index=at).fraction
        )
        rmse_values.append(float(np.sqrt(np.mean((truth[:, at] - mean[:, at]) ** 2))))

    lower, upper = chi_square_consistency_bounds(n_runs)
    labels = [name for name, _ in cases]
    positions = np.arange(len(cases))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.5), constrained_layout=True)

    ax = axes[0]
    ax.axhspan(lower, upper, color=GOOD, alpha=0.12)
    ax.axhline(1.0, color=INK_SECONDARY, lw=1.0, ls=":")
    colours = [GOOD if lower <= v <= upper else BAD for v in nees_values]
    ax.scatter(positions, nees_values, color=colours, s=70, zorder=3)
    for x, value in zip(positions, nees_values, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(0.35, 14.0)
    ax.set_ylabel("mean NEES across 400 runs")
    ax.set_title("NEES: is the claimed uncertainty honest?", loc="left")
    annotate(ax, "shaded: 95% χ² acceptance", loc="lower left")

    ax = axes[1]
    ax.axhline(0.95, color=TRUTH, lw=1.4, ls="--", label="nominal 95%")
    colours = [GOOD if abs(v - 0.95) <= 0.03 else BAD for v in coverage_values]
    ax.scatter(positions, coverage_values, color=colours, s=70, zorder=3)
    for x, value in zip(positions, coverage_values, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("fraction of runs inside the interval")
    ax.set_title("Credible-interval coverage", loc="left")
    ax.legend(loc="lower left")

    ax = axes[2]
    ax.scatter(positions, rmse_values, color=ESTIMATE, s=70, zorder=3)
    for x, value in zip(positions, rmse_values, strict=True):
        ax.annotate(
            f"{value:.2f}",
            (x, value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    ax.set_xticks(positions, labels, rotation=20, ha="right")
    ax.set_ylim(0.0, max(rmse_values) * 1.5)
    ax.set_ylabel("RMSE at the same time index")
    ax.set_title("RMSE barely notices", loc="left")
    annotate(ax, "accuracy moves 34%\nwhile NEES moves 5×", loc="lower right")
    save(fig, "fig04_consistency.png")


# ----------------------------------------------------------------------- figure 5
def figure_nees_aggregation() -> None:
    """Why NEES is averaged across runs and NIS over time."""
    n_runs, n_samples, burn = 400, 520, 200
    rng = np.random.default_rng(1)
    truth = np.cumsum(rng.normal(0.0, np.sqrt(Q_TRUE), (n_runs, n_samples)), axis=1)
    observed = truth + rng.normal(0.0, np.sqrt(R_TRUE), (n_runs, n_samples))
    model = RandomWalkModel(Q_TRUE, R_TRUE)

    mean = np.empty_like(observed)
    variance = np.empty_like(observed)
    innovation = np.empty_like(observed)
    innovation_variance = np.empty_like(observed)
    for index, row in enumerate(observed):
        track = filter_series(row, model)
        mean[index] = track.mean
        variance[index] = track.variance
        innovation[index] = track.innovation
        innovation_variance[index] = track.innovation_variance

    error = (truth - mean)[:, burn:]
    nees_time = (error**2 / variance[:, burn:]).mean(axis=1)
    nis_time = (innovation**2 / innovation_variance)[:, burn:].mean(axis=1)
    lower, upper = chi_square_consistency_bounds(error.shape[1])
    nees_pass = float(np.mean((nees_time >= lower) & (nees_time <= upper)))
    nis_pass = float(np.mean((nis_time >= lower) & (nis_time <= upper)))

    lags = np.arange(1, 16)
    err_acf = [np.mean([np.corrcoef(e[:-k], e[k:])[0, 1] for e in error]) for k in lags]
    innovation_tail = (innovation / np.sqrt(innovation_variance))[:, burn:]
    nu_acf = [np.mean([np.corrcoef(v[:-k], v[k:])[0, 1] for v in innovation_tail]) for k in lags]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.axhline(0.0, color=INK_SECONDARY, lw=1.0)
    ax.plot(lags, err_acf, color=TRUTH, marker="o", ms=5, label="estimation error")
    ax.plot(lags, nu_acf, color=ESTIMATE, marker="s", ms=5, label="innovation")
    ax.set_xlabel("lag (time steps)")
    ax.set_ylabel("autocorrelation")
    ax.set_title("The error remembers; the innovation does not", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1]
    bins = np.linspace(0.6, 1.6, 48)
    ax.hist(
        nees_time,
        bins=bins,
        color=TRUTH,
        alpha=0.65,
        label=f"NEES over time — {nees_pass:.0%} pass",
    )
    ax.hist(
        nis_time,
        bins=bins,
        color=ESTIMATE,
        alpha=0.65,
        label=f"NIS over time — {nis_pass:.0%} pass",
    )
    ax.axvline(lower, color=INK_SECONDARY, lw=1.3, ls="--")
    ax.axvline(upper, color=INK_SECONDARY, lw=1.3, ls="--")
    ax.set_xlabel("per-run statistic, averaged over that run's 320 time points")
    ax.set_ylabel("runs")
    ax.set_title("A correct filter fails a time-averaged NEES test", loc="left")
    ax.legend(loc="upper left")
    annotate(ax, "dashed: 95% χ² bounds\nboth filters are CORRECT", loc="upper right")
    save(fig, "fig05_nees_aggregation.png")


# ----------------------------------------------------------------------- figure 6
def figure_pipeline() -> None:
    """The composed estimator on a record whose alpha power rises."""
    seconds = 120.0
    n = int(seconds * FS)
    background = power_law_noise(n_samples=n, sample_rate=FS, exponent=1.5, variance=4.0, seed=0)
    envelope = np.concatenate(
        [
            np.linspace(0.15, 0.15, n // 4),
            np.linspace(0.15, 2.0, n // 2),
            np.linspace(2.0, 0.4, n - n // 4 - n // 2),
        ]
    )
    record = background + envelope * sinusoid(
        n_samples=n, sample_rate=FS, frequency=10.0, amplitude=6.0
    )

    pipeline = EpochPipeline(
        sample_rate=FS,
        band=(8.0, 12.0),
        epoch_seconds=2.0,
        model=RandomWalkModel(0.02, 0.05),
    )
    estimates = pipeline.update(record)

    config = MultitaperConfig()
    epoch_samples = pipeline.epoch_samples
    columns, times = [], []
    for index in range(len(estimates)):
        epoch = record[index * epoch_samples : (index + 1) * epoch_samples]
        spec = multitaper_psd(epoch, sample_rate=FS, config=config)
        keep = spec.frequencies <= 30.0
        columns.append(10.0 * np.log10(spec.psd[keep] + 1e-12))
        times.append(index * epoch_samples / FS)
    spectrogram = np.array(columns).T
    freqs = spec.frequencies[keep]

    fig, axes = plt.subplots(
        2, 1, figsize=(10.5, 5.6), sharex=True, height_ratios=[1.15, 1], constrained_layout=True
    )

    ax = axes[0]
    mesh = ax.pcolormesh(np.array(times), freqs, spectrogram, cmap=SEQUENTIAL, shading="auto")
    ax.axhspan(8.0, 12.0, facecolor="none", edgecolor=TRUTH, lw=1.2, ls="--")
    ax.text(2.0, 12.6, "tracked band 8–12 Hz", color=TRUTH, fontsize=8, va="bottom")
    ax.set_ylabel("frequency (Hz)")
    ax.set_title("Multitaper spectrogram, 2-second epochs", loc="left")
    ax.grid(visible=False)
    fig.colorbar(mesh, ax=ax, pad=0.01, label="dB")

    ax = axes[1]
    start = np.array([e.start_time for e in estimates])
    state = np.array([e.state for e in estimates])
    lower = np.array([e.lower for e in estimates])
    upper = np.array([e.upper for e in estimates])
    observed = np.array([e.observed for e in estimates])
    valid = np.array([e.valid for e in estimates])

    ax.plot(start, observed, ".", color=OBSERVED, ms=4, label="per-epoch log band power")
    ax.plot(start, state, color=ESTIMATE, lw=1.8, label="filtered state")
    ax.fill_between(start, lower, upper, color=BAND, alpha=0.18, label="95% credible interval")
    if (~valid).any():
        warm_end = float(start[~valid].max()) + 2.0
        ax.axvspan(start[0], warm_end, color=THIRD, alpha=0.16)
        ax.annotate(
            "warm-up\nflagged invalid",
            xy=(warm_end, 3.6),
            xytext=(warm_end + 9.0, 3.9),
            color=INK_SECONDARY,
            fontsize=8,
            arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "lw": 0.9},
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("log alpha power")
    ax.set_title("Tracked state with its uncertainty", loc="left")
    ax.set_ylim(-1.6, 5.2)
    ax.legend(
        loc="lower right", ncol=3, framealpha=0.95, frameon=True, facecolor="white", edgecolor=GRID
    )
    save(fig, "fig06_pipeline.png")


def main() -> None:
    use_report_style()
    for name, build in [
        ("fig01 peak estimation", figure_peak_estimation),
        ("fig02 spectral oracles", figure_spectral_oracles),
        ("fig03 filter tracking", figure_filter_tracking),
        ("fig04 consistency", figure_consistency),
        ("fig05 NEES aggregation", figure_nees_aggregation),
        ("fig06 pipeline", figure_pipeline),
    ]:
        print(f"building {name} ...", flush=True)
        build()
    print(f"figures written to {save.__module__ and 'docs/figures'}")


if __name__ == "__main__":
    main()
