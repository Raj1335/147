from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal

from .estimators import ModulationEstimate, classify_modulation
from .ingest import Capture


@dataclass(frozen=True)
class SignalAnalysis:
    source_name: str
    source_format: str
    sample_rate_hz: float
    sample_count: int
    total_sample_count: int | None
    first_sample: int
    nominal_center_frequency_hz: float | None
    peak_frequency_hz: float | None
    peak_offset_hz: float
    occupied_bandwidth_hz: float
    snr_estimate_db: float | None
    modulation_hint: str
    hint_score: float
    automatic_modulation: str
    automatic_modulation_score: float
    estimated_symbol_rate_hz: float | None
    estimated_frequency_offset_hz: float | None
    constellation_evm: float | None
    modulation_candidates: dict[str, float]
    modulation_estimate: ModulationEstimate
    truncated: bool
    frequency_hz: np.ndarray
    psd_db: np.ndarray
    waterfall_db: np.ndarray
    waterfall_time_s: np.ndarray
    waterfall_frequency_hz: np.ndarray

    def summary(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_format": self.source_format,
            "sample_rate_hz": self.sample_rate_hz,
            "sample_count": self.sample_count,
            "total_sample_count": self.total_sample_count,
            "first_sample": self.first_sample,
            "nominal_center_frequency_hz": self.nominal_center_frequency_hz,
            "peak_frequency_hz": self.peak_frequency_hz,
            "peak_offset_hz": self.peak_offset_hz,
            "occupied_bandwidth_hz": self.occupied_bandwidth_hz,
            "snr_estimate_db": self.snr_estimate_db,
            "modulation_hint": self.modulation_hint,
            "hint_score": self.hint_score,
            "automatic_modulation": self.automatic_modulation,
            "automatic_modulation_score": self.automatic_modulation_score,
            "estimated_symbol_rate_hz": self.estimated_symbol_rate_hz,
            "estimated_frequency_offset_hz": self.estimated_frequency_offset_hz,
            "constellation_evm": self.constellation_evm,
            "modulation_candidates": self.modulation_candidates,
            "automatic_timing_offset_samples": self.modulation_estimate.timing_offset_samples,
            "automatic_phase_rotation_rad": self.modulation_estimate.phase_rotation_rad,
            "automatic_warnings": list(self.modulation_estimate.warnings),
            "truncated": self.truncated,
        }


def extract_features(samples: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Return a compact normalized feature vector for an optional labeled model."""
    x = _finite_samples(samples)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")
    x = x[: min(len(x), 65536)]
    amplitude = np.abs(x)
    power = np.maximum(amplitude**2, np.finfo(np.float64).tiny)
    phase_delta = np.angle(x[1:] * np.conj(x[:-1]))
    freq_delta = np.diff(phase_delta) if len(phase_delta) > 1 else np.zeros(1)
    normalized_power = power / max(float(np.mean(power)), np.finfo(float).tiny)
    centered = normalized_power - np.mean(normalized_power)
    cumulant4 = float(np.mean(centered**2))
    psd = np.abs(np.fft.fft(x[: min(8192, len(x))])) ** 2
    psd /= max(float(np.sum(psd)), np.finfo(float).tiny)
    spectral_entropy = -float(np.sum(psd * np.log(psd + 1e-15)))
    moments = [
        np.mean(amplitude),
        np.std(amplitude),
        np.mean(np.abs(phase_delta)),
        np.std(phase_delta),
        np.std(freq_delta),
        np.mean(np.abs(x)),
        np.max(amplitude),
        np.percentile(amplitude, 25),
        np.percentile(amplitude, 50),
        np.percentile(amplitude, 75),
        cumulant4,
        spectral_entropy,
    ]
    return np.asarray(moments, dtype=np.float64)


def analyze_capture(capture: Capture, *, max_plot_samples: int = 262_144) -> SignalAnalysis:
    x = _finite_samples(capture.samples)
    if len(x) < 16:
        raise ValueError("At least 16 samples are needed for spectral analysis.")
    fs = capture.sample_rate_hz
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sample rate must be positive.")
    x = x[:max_plot_samples]
    x = x - np.mean(x)
    nperseg = min(4096, max(256, 2 ** int(np.floor(np.log2(len(x))))))
    nperseg = min(nperseg, len(x))
    frequencies, psd = signal.welch(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend=False,
        return_onesided=False,
        scaling="density",
    )
    frequencies = np.fft.fftshift(frequencies)
    psd = np.fft.fftshift(psd)
    psd = np.maximum(np.real(psd), np.finfo(float).tiny)
    peak_index = int(np.argmax(psd))
    peak_offset = float(frequencies[peak_index])

    cumulative = np.cumsum(psd)
    cumulative /= cumulative[-1]
    low_index = int(np.searchsorted(cumulative, 0.005))
    high_index = min(int(np.searchsorted(cumulative, 0.995)), len(frequencies) - 1)
    bandwidth = max(0.0, float(frequencies[high_index] - frequencies[low_index]))
    band_mask = (frequencies >= frequencies[low_index]) & (frequencies <= frequencies[high_index])
    noise_mask = ~band_mask
    snr_db: float | None = None
    if np.any(band_mask) and np.any(noise_mask):
        noise_per_bin = float(np.median(psd[noise_mask]))
        noise_power = noise_per_bin * int(np.count_nonzero(band_mask))
        signal_power = max(float(np.sum(psd[band_mask]) - noise_power), 0.0)
        if noise_power > 0 and signal_power > 0:
            snr_db = float(10.0 * np.log10(signal_power / noise_power))

    spec_nperseg = min(512, max(64, 2 ** int(np.floor(np.log2(len(x) // 8)))))
    spec_nperseg = min(spec_nperseg, len(x))
    spec_freq, times, spec = signal.spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=spec_nperseg,
        noverlap=spec_nperseg * 3 // 4,
        detrend=False,
        return_onesided=False,
        mode="psd",
        scaling="density",
    )
    spec = np.fft.fftshift(spec, axes=0)
    spec_freq = np.fft.fftshift(spec_freq)
    waterfall = 10.0 * np.log10(np.maximum(spec, np.finfo(float).tiny))
    hint, score = modulation_hint(x)
    estimate = classify_modulation(x, fs)
    center = capture.center_frequency_hz
    absolute_peak = center + peak_offset if center is not None else None
    return SignalAnalysis(
        source_name=capture.source_name,
        source_format=capture.source_format,
        sample_rate_hz=fs,
        sample_count=len(capture.samples),
        total_sample_count=capture.total_samples,
        first_sample=capture.first_sample,
        nominal_center_frequency_hz=center,
        peak_frequency_hz=absolute_peak,
        peak_offset_hz=peak_offset,
        occupied_bandwidth_hz=bandwidth,
        snr_estimate_db=snr_db,
        modulation_hint=hint,
        hint_score=score,
        automatic_modulation=estimate.modulation,
        automatic_modulation_score=estimate.score,
        estimated_symbol_rate_hz=estimate.symbol_rate_hz,
        estimated_frequency_offset_hz=estimate.frequency_offset_hz,
        constellation_evm=estimate.evm,
        modulation_candidates=estimate.candidates,
        modulation_estimate=estimate,
        truncated=capture.truncated,
        frequency_hz=frequencies + (center or 0.0),
        psd_db=10.0 * np.log10(psd),
        waterfall_db=waterfall,
        waterfall_time_s=times + capture.first_sample / fs,
        waterfall_frequency_hz=spec_freq + (center or 0.0),
    )


def modulation_hint(samples: np.ndarray) -> tuple[str, float]:
    """Offer a non-calibrated diagnostic family hint, never a modulation label."""
    x = _finite_samples(samples)[:65536]
    magnitude = np.abs(x)
    mean_magnitude = float(np.mean(magnitude))
    if mean_magnitude <= np.finfo(float).eps:
        return "No usable signal", 0.0
    envelope_cv = float(np.std(magnitude) / mean_magnitude)
    phase_step = np.angle(x[1:] * np.conj(x[:-1]))
    phase_spread = float(np.std(phase_step)) if phase_step.size else 0.0
    if envelope_cv < 0.12 and phase_spread > 0.45:
        return "Constant-envelope / PSK-like (unverified)", 0.35
    if envelope_cv > 0.35:
        return "Amplitude-varying / ASK-QAM-like (unverified)", 0.3
    return "Unknown; manual or trained-model review required", 0.0


def _finite_samples(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite samples.")
    return x
