from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class ModulationEstimate:
    modulation: str
    score: float
    symbol_rate_hz: float | None
    samples_per_symbol: float | None
    timing_offset_samples: float | None
    frequency_offset_hz: float | None
    fsk_deviation_hz: float | None
    phase_rotation_rad: float | None
    evm: float | None
    candidates: dict[str, float]
    warnings: tuple[str, ...]


def estimate_symbol_rate(
    samples: np.ndarray,
    sample_rate_hz: float,
    *,
    min_rate_hz: float = 100.0,
    max_rate_hz: float | None = None,
    max_samples: int = 16_384,
) -> tuple[float | None, float]:
    """Estimate symbol clock from nonzero cyclic features in power/envelope."""
    x = _samples(samples)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")
    if x.size < 128:
        return None, 0.0
    if max_samples < 128:
        raise ValueError("max_samples must be at least 128.")
    limit = min(x.size, max_samples)
    phase_differences = np.angle(x[1:limit] * np.conj(x[: limit - 1]))
    clock_features = [
        np.abs(x[:limit]) ** 2,
        np.abs(np.diff(phase_differences)),
    ]
    max_lag = min(limit // 8, int(sample_rate_hz / min_rate_hz))
    if max_lag < 3:
        return None, 0.0
    min_lag = max(4, int(np.ceil(sample_rate_hz / (max_rate_hz or sample_rate_hz / 2))))
    best_lag: int | None = None
    best_score = 0.0
    for feature in clock_features:
        feature = feature - np.mean(feature)
        if np.std(feature) <= np.finfo(float).eps * max(1.0, float(np.mean(np.abs(x)) ** 2)):
            continue
        correlation = signal.correlate(feature, feature, mode="full", method="fft")[feature.size - 1 :]
        correlation = correlation[: max_lag + 1].real / np.arange(
            feature.size, feature.size - max_lag - 1, -1
        )
        if correlation.size < 4 or correlation[0] <= np.finfo(float).tiny:
            continue
        correlation /= correlation[0]
        candidate_lags = np.arange(len(correlation))
        eligible = (candidate_lags >= min_lag) & (candidate_lags <= max_lag)
        if not np.any(eligible):
            continue
        peaks, _ = signal.find_peaks(correlation)
        positive = peaks[eligible[peaks] & (correlation[peaks] > 0)]
        if positive.size == 0:
            continue
        strongest = float(np.max(correlation[positive]))
        supported = positive[correlation[positive] >= max(0.15, 0.35 * strongest)]
        if supported.size == 0:
            continue
        peak_index = int(supported[0])
        score = float(np.clip(correlation[peak_index], 0.0, 1.0))
        if score > best_score:
            best_score = score
            best_lag = peak_index
    if best_lag is None or best_score < 0.08:
        return None, best_score
    return sample_rate_hz / best_lag, best_score


def classify_modulation(
    samples: np.ndarray,
    sample_rate_hz: float,
    *,
    symbol_rate_hz: float | None = None,
) -> ModulationEstimate:
    """Blind constellation-family estimate; rejects high-residual/low-evidence cases."""
    x = _samples(samples)
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("Sample rate must be positive.")
    if x.size < 128:
        return _unknown(None, None, None, "At least 128 samples are needed.")
    x = x[:32_768]
    if float(np.std(np.abs(x))) < 0.03 * max(float(np.mean(np.abs(x))), 1e-12):
        _, phase_clock = estimate_symbol_rate(
            np.angle(x[1:] * np.conj(x[:-1])), sample_rate_hz
        )
        if phase_clock < 0.1:
            return _unknown(None, None, None, "A constant-envelope single tone has no symbol evidence.")
    estimated_rate, clock_score = estimate_symbol_rate(x, sample_rate_hz)
    if symbol_rate_hz is not None:
        if not np.isfinite(symbol_rate_hz) or symbol_rate_hz <= 0:
            raise ValueError("Symbol rate must be positive.")
        estimated_rate = float(symbol_rate_hz)
        clock_score = 1.0
    if estimated_rate is None:
        return _unknown(None, None, None, "No stable symbol-clock feature was found.")
    sps = sample_rate_hz / estimated_rate
    if sps < 2:
        return _unknown(estimated_rate, sps, None, "At least two samples per symbol are required.")

    sample_count = min(256, int((x.size - sps) / sps))
    if sample_count < 32:
        return _unknown(estimated_rate, sps, None, "Too few symbols for classification.")

    residuals: dict[str, float] = {}
    offsets: dict[str, float] = {}
    timings: dict[str, float | None] = {}
    phase_rotations: dict[str, float | None] = {}
    sample_indices = np.arange(x.size)
    symbol_times = [
        phase + np.arange(sample_count) * sps
        for phase in np.linspace(0.0, sps, 4, endpoint=False)
    ]
    for name, points in _constellations().items():
        order = {"BPSK": 2, "QPSK": 4, "8PSK": 8, "16QAM": 4}[name]
        frequency_offset = _psk_frequency(x, sample_rate_hz, order)
        corrected = x * np.exp(-2j * np.pi * frequency_offset * sample_indices / sample_rate_hz)
        best_raw = 1.0
        best_used = points.size
        best_timing: float | None = None
        best_rotation: float | None = None
        for times_at_symbols in symbol_times:
            symbols = np.interp(times_at_symbols, sample_indices, corrected.real) + 1j * np.interp(
                times_at_symbols, sample_indices, corrected.imag
            )
            symbols = symbols / max(float(np.sqrt(np.mean(np.abs(symbols) ** 2))), 1e-12)
            rotations = _rotations(name)
            if name in {"BPSK", "QPSK", "8PSK", "16QAM"}:
                power_order = order
                powers = symbols**power_order
                period = 2.0 * np.pi / order
                reference_phase = np.angle(np.mean(points**power_order)) / power_order
                observed_phase = np.angle(np.mean(powers)) / power_order
                candidate_rotation = float(
                    (reference_phase - observed_phase + period / 2) % period - period / 2
                )
                rotations = np.asarray([candidate_rotation, *rotations])
            for rotation in rotations:
                rotated_points = points * np.exp(-1j * rotation)
                distances = np.abs(symbols[:, None] - rotated_points[None, :])
                nearest_index = np.argmin(distances, axis=1)
                nearest = rotated_points[nearest_index]
                evm = float(np.sqrt(np.mean(np.abs(symbols - nearest) ** 2)))
                counts = np.bincount(nearest_index, minlength=rotated_points.size)
                used_points = int(np.count_nonzero(counts >= max(5, int(sample_count * 0.01))))
                if evm < best_raw:
                    best_raw = evm
                    best_used = used_points
                    best_timing = float(times_at_symbols[0] % sps)
                    best_rotation = float(rotation)
                    offsets[name] = frequency_offset
        residuals[name] = best_raw + 0.25 * (1.0 - best_used / points.size)
        timings[name] = best_timing
        phase_rotations[name] = best_rotation

    fsk_fit = _fsk_fit(x, sample_rate_hz, estimated_rate)
    if fsk_fit is not None:
        residuals["2FSK"] = fsk_fit[0]
    constellation_ranked = sorted(
        ((name, value) for name, value in residuals.items() if name != "2FSK"),
        key=lambda item: item[1],
    )
    if (
        fsk_fit is not None
        and fsk_fit[0] + 1e-6 < constellation_ranked[0][1]
    ):
        ranked = sorted(residuals.items(), key=lambda item: item[1])
    else:
        ranked = constellation_ranked
    winner, evm = ranked[0]
    fsk_parameters = fsk_fit if winner == "2FSK" else None
    frequency_offset = fsk_parameters[1] if fsk_parameters else offsets.get(winner, 0.0)
    timing_offset = fsk_parameters[3] if fsk_parameters else timings.get(winner)
    phase_rotation = phase_rotations.get(winner)
    fsk_deviation = fsk_parameters[2] if fsk_parameters else None
    margin = ranked[1][1] - evm if len(ranked) > 1 else 0.0
    score = float(np.clip((1.0 - evm / 0.5) * min(1.0, margin / 0.15) * clock_score, 0.0, 1.0))
    if evm > 0.38 or margin < 0.025 or clock_score < 0.1 or score < 0.25:
        return ModulationEstimate(
            modulation="Unknown",
            score=score,
            symbol_rate_hz=estimated_rate,
            samples_per_symbol=sps,
            timing_offset_samples=timing_offset,
            frequency_offset_hz=frequency_offset,
            fsk_deviation_hz=fsk_deviation,
            phase_rotation_rad=phase_rotation,
            evm=evm,
            candidates=residuals,
            warnings=("Blind estimate is ambiguous or has high constellation residual.",),
        )
    return ModulationEstimate(
        modulation=winner,
        score=score,
        symbol_rate_hz=estimated_rate,
        samples_per_symbol=sps,
        timing_offset_samples=timing_offset,
        frequency_offset_hz=frequency_offset,
        fsk_deviation_hz=fsk_deviation,
        phase_rotation_rad=phase_rotation,
        evm=evm,
        candidates=residuals,
        warnings=("Blind estimates require validation against a known sync word or CRC.",),
    )


def _constellations() -> dict[str, np.ndarray]:
    return {
        "BPSK": np.exp(1j * np.pi * np.arange(2)),
        "QPSK": np.exp(1j * (np.pi / 4 + np.pi / 2 * np.arange(4))),
        "8PSK": np.exp(1j * np.pi / 8 + 1j * np.pi / 4 * np.arange(8)),
        "16QAM": (
            np.asarray([-3, -1, 1, 3])[:, None] + 1j * np.asarray([-3, -1, 1, 3])[None, :]
        ).reshape(-1)
        / np.sqrt(10),
    }


def _rotations(name: str) -> np.ndarray:
    period = np.pi if name == "BPSK" else np.pi / 2 if name in {"QPSK", "16QAM"} else 2 * np.pi
    return np.linspace(0.0, period, 12, endpoint=False)


def _dominant_offset(samples: np.ndarray, sample_rate_hz: float) -> float:
    nfft = min(16_384, 2 ** int(np.floor(np.log2(len(samples)))))
    frequencies, density = signal.welch(
        samples,
        fs=sample_rate_hz,
        nperseg=nfft,
        return_onesided=False,
        detrend="constant",
    )
    frequencies = np.fft.fftshift(frequencies)
    density = np.fft.fftshift(density)
    if density.size < 3:
        return 0.0
    return float(frequencies[int(np.argmax(density))])


def _psk_frequency(samples: np.ndarray, sample_rate_hz: float, order: int) -> float:
    powered = samples**order
    if np.std(powered) <= 1e-6 * max(float(np.mean(np.abs(powered))), 1.0):
        return 0.0
    nfft = min(16_384, 2 ** int(np.floor(np.log2(len(powered)))))
    frequencies, density = signal.welch(
        powered,
        fs=sample_rate_hz,
        nperseg=nfft,
        return_onesided=False,
        detrend="constant",
    )
    frequencies = np.fft.fftshift(frequencies)
    density = np.fft.fftshift(density)
    if density.size < 3 or np.max(density) <= np.finfo(float).tiny:
        return 0.0
    offset = float(frequencies[int(np.argmax(density))]) / order
    alias_period = sample_rate_hz / order
    return float((offset + alias_period / 2) % alias_period - alias_period / 2)


def _fsk_fit(
    samples: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
) -> tuple[float, float, float, float] | None:
    phase = np.angle(samples[1:] * np.conj(samples[:-1]))
    frequency = phase * sample_rate_hz / (2 * np.pi)
    samples_per_symbol = sample_rate_hz / symbol_rate_hz
    symbol_count = min(1024, int((len(samples) - samples_per_symbol) / samples_per_symbol))
    if symbol_count < 32:
        return None
    best: tuple[float, float, float, float] | None = None
    for timing in np.linspace(0.0, samples_per_symbol, 8, endpoint=False):
        symbols: list[float] = []
        for index in range(symbol_count):
            start = int(round(timing + index * samples_per_symbol))
            end = min(len(frequency), int(round(timing + (index + 1) * samples_per_symbol)))
            if end <= start:
                break
            symbols.append(float(np.mean(frequency[start:end])))
        values = np.asarray(symbols)
        if values.size < 32:
            continue
        low, high = np.percentile(values, [20, 80])
        midpoint = (low + high) / 2
        labels = values > midpoint
        if min(np.mean(labels), np.mean(~labels)) < 0.05:
            continue
        centers = np.asarray([np.mean(values[~labels]), np.mean(values[labels])])
        if centers[1] - centers[0] < sample_rate_hz / len(samples):
            continue
        errors = values - centers[labels.astype(int)]
        scale = max(float(np.std(values)), 1e-12)
        residual = float(np.sqrt(np.mean(errors**2)) / scale)
        candidate = (
            residual,
            float(np.mean(centers)),
            float((centers[1] - centers[0]) / 2.0),
            float(timing % samples_per_symbol),
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    return best


def _unknown(
    symbol_rate: float | None,
    samples_per_symbol: float | None,
    frequency_offset: float | None,
    warning: str,
) -> ModulationEstimate:
    return ModulationEstimate(
        modulation="Unknown",
        score=0.0,
        symbol_rate_hz=symbol_rate,
        samples_per_symbol=samples_per_symbol,
        timing_offset_samples=None,
        frequency_offset_hz=frequency_offset,
        fsk_deviation_hz=None,
        phase_rotation_rad=None,
        evm=None,
        candidates={},
        warnings=(warning,),
    )


def _samples(samples: np.ndarray) -> np.ndarray:
    x = np.asarray(samples, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains NaN or infinite samples.")
    return x
