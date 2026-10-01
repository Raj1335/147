from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sih26147.analysis import SignalAnalysis, analyze_capture
from sih26147.decode import deinterleave, ldpc_decode, rs_decode, viterbi_decode
from sih26147.demod import (
    bits_to_bytes,
    correlate_bitstreams,
    demodulate,
    demodulate_from_estimate,
)
from sih26147.ingest import Capture, CaptureError, DEFAULT_SAMPLE_LIMIT, read_capture
from sih26147.training import predict_with_trained_model


SAMPLE_CAPTURE_PATH = (
    Path("data")
    / "real"
    / "pslv-436500kHz-2018-01-13-095446-prefix-16MiB.wav"
)

st.set_page_config(
    page_title="SIH26147 | Signal Intelligence Workbench",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2.1rem; max-width: 1440px;}
        .hero {padding: 1.2rem 1.4rem; border: 1px solid #25404f; border-radius: 14px;
               background: linear-gradient(120deg, #102532 0%, #101c27 65%, #102a2b 100%);}
        .hero h1 {margin: 0; font-size: 2rem; letter-spacing: .02em;}
        .hero p {margin: .45rem 0 0; color: #aac1cf;}
        .muted {color: #9cb1bf; font-size: .9rem;}
        </style>
        <div class="hero">
          <h1>Signal Intelligence Workbench</h1>
          <p>SIH26147 · File-first IQ/WAV analysis · local processing by default</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Research prototype: outputs are signal-analysis aids, not operational intelligence. "
        "Do not upload classified, sensitive, or personally identifying recordings to a public host."
    )

    with st.sidebar:
        st.subheader("Capture input")
        hosted = os.environ.get("SIH_HOSTED", "").lower() == "true"
        if hosted:
            input_mode = "Upload file"
            uploaded = st.file_uploader("WAV or raw IQ file", type=["wav", "iq", "bin", "dat"])
            local_path = ""
        else:
            input_mode = st.radio("Input method", ("Upload file", "Local file path"))
            uploaded = (
                st.file_uploader("WAV or raw IQ file", type=["wav", "iq", "bin", "dat"])
                if input_mode == "Upload file"
                else None
            )
            local_path = (
                st.text_input(
                    "Path to local WAV/IQ file",
                    value=str(SAMPLE_CAPTURE_PATH) if SAMPLE_CAPTURE_PATH.is_file() else "",
                    placeholder=r"C:\captures\recording.wav",
                )
                if input_mode == "Local file path"
                else ""
            )
        raw_format = st.selectbox("Raw IQ storage", ("cf32", "ci16", "cu8"), help="Ignored for WAV.")
        raw_rate = st.number_input(
            "Raw IQ sample rate (samples/s)",
            min_value=1.0,
            value=1_000_000.0,
            step=100_000.0,
            format="%.0f",
            help="Raw IQ has no header; supply the rate from capture metadata.",
        )
        center_mhz = st.number_input(
            "Known center frequency (MHz; optional)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.6f",
        )
        first_sample = st.number_input("First sample index", min_value=0, value=0, step=1)
        sample_limit = st.number_input(
            "Maximum samples to analyze",
            min_value=16_384,
            max_value=1_000_000,
            value=DEFAULT_SAMPLE_LIMIT,
            step=16_384,
            help="Bounds analysis memory. Choose a different start sample to inspect another window.",
        )
        analyze_clicked = st.button("Analyze capture", type="primary", width="stretch")
        st.divider()
        st.caption("No capture is stored on a server by this application code. Hosted platform logs/storage policies still apply.")

    if analyze_clicked:
        if input_mode == "Upload file" and uploaded is None:
            st.error("Choose a WAV or raw IQ file first.")
        elif input_mode == "Local file path" and not local_path.strip():
            st.error("Enter a local WAV or raw IQ file path.")
        else:
            try:
                if input_mode == "Local file path":
                    path = Path(local_path).expanduser()
                    with path.open("rb") as stream:
                        capture = read_capture(
                            stream,
                            source_name=path.name,
                            raw_format=raw_format,
                            sample_rate_hz=raw_rate,
                            center_frequency_hz=center_mhz * 1_000_000 if center_mhz else None,
                            first_sample=int(first_sample),
                            sample_limit=int(sample_limit),
                        )
                else:
                    capture = read_capture(
                        uploaded,
                        source_name=uploaded.name,
                        raw_format=raw_format,
                        sample_rate_hz=raw_rate,
                        center_frequency_hz=center_mhz * 1_000_000 if center_mhz else None,
                        first_sample=int(first_sample),
                        sample_limit=int(sample_limit),
                    )
                result = analyze_capture(capture)
                st.session_state["capture"] = capture
                st.session_state["analysis"] = result
                st.session_state["bits"] = ""
                model_path = Path("models/modulation.joblib")
                if model_path.is_file():
                    st.session_state["trained_prediction"] = predict_with_trained_model(
                        model_path, capture.samples, capture.sample_rate_hz
                    )
                else:
                    st.session_state.pop("trained_prediction", None)
                st.success(f"Analyzed {capture.samples.size:,} samples from {capture.source_name}.")
            except (CaptureError, ValueError, OSError) as exc:
                st.error(f"Capture analysis failed: {exc}")

    capture: Capture | None = st.session_state.get("capture")
    result: SignalAnalysis | None = st.session_state.get("analysis")
    if result is None or capture is None:
        _show_welcome()
        return

    tabs = st.tabs(("Signal analysis", "Demodulate & decode", "Training workflow", "Report"))
    with tabs[0]:
        _show_analysis(capture, result)
    with tabs[1]:
        _show_processing(capture, result)
    with tabs[2]:
        _show_training_info()
    with tabs[3]:
        _show_report(result)


def _show_welcome() -> None:
    st.subheader("Start with a capture")
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            **Supported input**
            - Uncompressed PCM `.wav` (mono real or stereo I/Q)
            - Raw interleaved `.iq` / `.bin` / `.dat`: complex float32, signed int16, or unsigned uint8
            - Raw IQ rate and format must come from capture metadata; they cannot be recovered from raw bytes alone.
            """
        )
    with right:
        st.markdown(
            """
            **Analysis flow**
            1. Inspect the waveform, PSD, waterfall, and constellation.
            2. Review measured estimates and their limitations.
            3. Select known modulation / symbol timing before demodulation.
            4. Verify decoded output using sync words, CRC, or trusted reference bits.
            """
        )
    st.info("For a public real over-the-air dataset and provenance, see `datasets/README.md` in the project.")


def _show_analysis(capture: Capture, result: SignalAnalysis) -> None:
    if capture.truncated:
        st.warning(
            f"Showing samples {capture.first_sample:,}–{capture.first_sample + capture.samples.size - 1:,} "
            f"of {capture.total_samples:,}. Adjust the start sample to inspect another window."
        )
    first, second, third, fourth = st.columns(4)
    first.metric("Sample rate", f"{result.sample_rate_hz / 1e6:.6g} Msps")
    peak_label = (
        f"{result.peak_frequency_hz / 1e6:.6f} MHz"
        if result.peak_frequency_hz is not None
        else f"{result.peak_offset_hz / 1e3:.3f} kHz offset"
    )
    second.metric("Dominant spectral peak", peak_label)
    third.metric("99% occupied bandwidth", f"{result.occupied_bandwidth_hz / 1e3:.2f} kHz")
    fourth.metric(
        "Noise-floor SNR estimate",
        f"{result.snr_estimate_db:.1f} dB" if result.snr_estimate_db is not None else "Unavailable",
        help="A rough PSD/noise-floor estimate; not a calibrated receiver SNR.",
    )

    st.markdown(
        f"**Diagnostic modulation hint:** {result.modulation_hint} "
        f"(heuristic score {result.hint_score:.0%}; not a probability)"
    )
    symbol_rate = (
        f"{result.estimated_symbol_rate_hz:,.2f} symbols/s"
        if result.estimated_symbol_rate_hz is not None
        else "not estimated"
    )
    st.markdown(
        f"**Automatic estimate:** {result.automatic_modulation} · symbol rate: {symbol_rate} · "
        f"score: {result.automatic_modulation_score:.0%}"
    )
    if result.modulation_candidates:
        st.caption(
            "Candidate residuals (lower is closer; complexity-penalized): "
            + " · ".join(
                f"{name} {residual:.3f}"
                for name, residual in sorted(
                    result.modulation_candidates.items(), key=lambda item: item[1]
                )
            )
        )
    prediction = st.session_state.get("trained_prediction")
    if prediction:
        st.success(
            f"Local real-capture model: {prediction['label']} "
            f"({prediction['confidence']:.1%} uncalibrated model score). "
            f"Group-holdout accuracy: {prediction['validation_accuracy']:.1%}."
        )
    st.caption(
        "Hint is intentionally low-confidence and is not a trained or calibrated AMC result. "
        "Automatic modulation and symbol-rate estimates are blind classical estimators, not trained "
        "or calibrated results. FEC and interleaver parameters are not automatically identified."
    )
    spectrum_tab, waterfall_tab, constellation_tab = st.tabs(("Spectrum", "Waterfall", "Constellation"))
    with spectrum_tab:
        figure = go.Figure()
        figure.add_trace(go.Scattergl(x=result.frequency_hz, y=result.psd_db, mode="lines", name="Welch PSD"))
        figure.update_layout(
            height=410,
            template="plotly_dark",
            title="Power spectral density",
            xaxis_title="Frequency (Hz; relative unless center frequency is supplied)",
            yaxis_title="Power / Hz (dB)",
        )
        st.plotly_chart(figure, width="stretch")
    with waterfall_tab:
        figure = go.Figure(
            go.Heatmap(
                x=result.waterfall_time_s,
                y=result.waterfall_frequency_hz,
                z=result.waterfall_db,
                colorscale="Turbo",
                colorbar={"title": "dB"},
            )
        )
        figure.update_layout(
            height=500,
            template="plotly_dark",
            title="Short-time spectral power",
            xaxis_title="Time (s)",
            yaxis_title="Frequency (Hz)",
        )
        st.plotly_chart(figure, width="stretch")
    with constellation_tab:
        constellation = capture.samples[: min(capture.samples.size, 20_000)]
        constellation = constellation - np.mean(constellation)
        figure = go.Figure(
            go.Scattergl(
                x=constellation.real,
                y=constellation.imag,
                mode="markers",
                marker={"size": 3, "opacity": 0.55, "color": np.angle(constellation), "colorscale": "Turbo"},
            )
        )
        figure.update_layout(
            height=500,
            template="plotly_dark",
            title="Raw sample constellation (before carrier/timing recovery)",
            xaxis_title="In-phase",
            yaxis_title="Quadrature",
            yaxis={"scaleanchor": "x", "scaleratio": 1},
        )
        st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Source: {result.source_name} · {result.source_format} · "
        f"{result.sample_count:,} analyzed samples"
    )


def _show_processing(capture: Capture, analysis: SignalAnalysis) -> None:
    demod_tab, deinterleave_tab, fec_tab, correlate_tab = st.tabs(
        ("Demodulation", "De-interleaving", "FEC", "Bit correlation")
    )
    with demod_tab:
        st.caption("Automatic path estimates carrier/timing for supported PSK/QAM/2FSK; framing and FEC still require validation.")
        if st.button("Auto-estimate and demodulate PSK/QAM/2FSK", key="run_auto_demod"):
            try:
                bits = demodulate_from_estimate(
                    capture.samples,
                    analysis.modulation_estimate,
                    capture.sample_rate_hz,
                )
                st.session_state["bits"] = "".join(map(str, bits.tolist()))
                st.success(
                    f"Auto-demodulated {bits.size:,} hard-decision bits using the estimated "
                    "carrier, timing, and constellation rotation. Verify with framing/CRC."
                )
            except ValueError as exc:
                st.error(str(exc))
        scheme = st.selectbox("Known modulation", ("BPSK", "QPSK", "8PSK", "16QAM", "2FSK"))
        samples_per_symbol = st.number_input("Samples per symbol", min_value=1, value=8, step=1)
        fsk_deviation = st.number_input("2FSK tone deviation (Hz)", min_value=1.0, value=1200.0)
        if st.button("Demodulate", key="run_demod"):
            try:
                bits = demodulate(
                    capture.samples,
                    scheme=scheme,
                    samples_per_symbol=int(samples_per_symbol),
                    fsk_deviation_hz=float(fsk_deviation),
                    sample_rate_hz=capture.sample_rate_hz,
                )
                text = "".join(map(str, bits.tolist()))
                st.session_state["bits"] = text
                st.success(f"Produced {bits.size:,} hard-decision bits. Validate framing/CRC before trusting them.")
            except ValueError as exc:
                st.error(str(exc))
        st.text_area("Current bit stream", key="bits", height=140)
        st.download_button(
            "Download bit stream",
            data=st.session_state.get("bits", ""),
            file_name="decoded-bits.txt",
            mime="text/plain",
            disabled=not bool(st.session_state.get("bits")),
        )
    with deinterleave_tab:
        st.caption("These operations invert the specific parameterized interleavers implemented here; unknown framing is not inferred.")
        if st.button("Load demodulated bits", key="load_demodulated_for_deinterleave"):
            st.session_state["deinterleave_input"] = st.session_state.get("bits", "")
        bits_text = st.text_area("Input bits", key="deinterleave_input")
        method = st.selectbox("Interleaver type", ("block", "diagonal", "convolutional", "pseudorandom"))
        rows, columns, branches = st.columns(3)
        with rows:
            row_count = st.number_input("Rows", min_value=1, value=8)
        with columns:
            column_count = st.number_input("Columns", min_value=1, value=8)
        with branches:
            branch_count = st.number_input("Convolutional branches", min_value=1, value=4)
        original_length = st.number_input(
            "Original bit count (convolutional; removes padding)", min_value=1, value=64
        )
        seed = st.number_input("Pseudo-random seed", min_value=0, value=0)
        if st.button("De-interleave", key="run_deinterleave"):
            try:
                bits = _parse_bit_array(bits_text)
                decoded = deinterleave(
                    bits,
                    method=method,
                    rows=int(row_count),
                    columns=int(column_count),
                    branches=int(branch_count),
                    seed=int(seed),
                    original_length=int(original_length),
                )
                output = "".join(map(str, decoded.tolist()))
                st.session_state["deinterleaved_bits"] = output
                st.code(output[:10_000] + ("…" if len(output) > 10_000 else ""))
                st.download_button("Download de-interleaved bits", output, "deinterleaved-bits.txt")
            except ValueError as exc:
                st.error(str(exc))
    with fec_tab:
        if st.button("Load de-interleaved bits (or last demodulation)", key="load_bits_for_fec"):
            st.session_state["fec_input"] = st.session_state.get(
                "deinterleaved_bits", st.session_state.get("bits", "")
            )
        fec_kind = st.selectbox(
            "Decoder",
            ("Convolutional / Viterbi", "Reed–Solomon", "LDPC (custom parity-check matrix)"),
        )
        coded_text = st.text_area("Coded bits (or bytes as 0/1 bits)", key="fec_input")
        if fec_kind == "Convolutional / Viterbi":
            constraint = st.number_input("Constraint length", min_value=2, max_value=12, value=7)
            generator_text = st.text_input("Octal generator polynomials", value="171,133")
            terminated = st.checkbox("Codeword is terminated with zero tail bits", value=True)
            if st.button("Viterbi decode", key="run_viterbi"):
                try:
                    generators = tuple(int(value.strip(), 8) for value in generator_text.split(","))
                    decoded = viterbi_decode(
                        _parse_bit_array(coded_text),
                        constraint_length=int(constraint),
                        generators=generators,
                        terminated=terminated,
                    )
                    output = "".join(map(str, decoded.tolist()))
                    st.session_state["fec_output"] = output
                    st.code(output[:10_000])
                except (ValueError, OverflowError) as exc:
                    st.error(str(exc))
        elif fec_kind == "Reed–Solomon":
            parity_symbols = st.number_input("Parity symbols", min_value=1, max_value=254, value=32)
            if st.button("Reed–Solomon decode", key="run_rs"):
                from reedsolo import ReedSolomonError

                try:
                    decoded = rs_decode(
                        bits_to_bytes(_parse_bit_array(coded_text)),
                        parity_symbols=int(parity_symbols),
                    )
                    st.session_state["fec_output"] = "".join(f"{byte:08b}" for byte in decoded)
                    st.download_button("Download decoded payload", decoded, "rs-payload.bin")
                except (ValueError, RuntimeError, ReedSolomonError) as exc:
                    st.error(f"Reed–Solomon decoding failed: {exc}")
        else:
            matrix_text = st.text_area(
                "Parity-check matrix H (JSON list of 0/1 rows)",
                value="[[1,1,0,1],[0,1,1,1]]",
            )
            if st.button("LDPC decode", key="run_ldpc"):
                try:
                    matrix = np.asarray(json.loads(matrix_text))
                    llrs = np.where(_parse_bit_array(coded_text) == 0, 8.0, -8.0)
                    decoded, converged, iterations = ldpc_decode(llrs, matrix)
                    st.session_state["fec_output"] = "".join(map(str, decoded.tolist()))
                    st.write(f"Syndrome passed: {converged} · iterations: {iterations}")
                    st.code("".join(map(str, decoded.tolist()))[:10_000])
                except (ValueError, json.JSONDecodeError) as exc:
                    st.error(f"LDPC decoding failed: {exc}")
    with correlate_tab:
        if st.button("Load latest decoded output", key="load_bits_for_correlation"):
            st.session_state["candidate_bits"] = st.session_state.get(
                "fec_output", st.session_state.get("bits", "")
            )
        reference = st.text_area("Reference/header/payload bit stream")
        candidate = st.text_area("Observed bit stream", key="candidate_bits")
        offset = st.number_input("Maximum positive alignment shift", min_value=0, value=4096)
        if st.button("Correlate bit streams", key="run_correlation"):
            try:
                comparison = correlate_bitstreams(reference, candidate, max_offset=int(offset))
                st.metric("Best bit error rate", f"{comparison['bit_error_rate']:.6f}")
                st.write(
                    f"Alignment offset: {comparison['offset_bits']} bits · "
                    f"Compared: {comparison['compared_bits']} bits"
                )
            except ValueError as exc:
                st.error(str(exc))


def _show_training_info() -> None:
    st.subheader("Train on measured captures — not generated waveforms")
    st.write(
        "This build intentionally ships no synthetic-trained modulation model. "
        "Use independently recorded, human-annotated IQ windows and the CSV manifest + CLI trainer."
    )
    st.code(
        "python train_model.py --manifest datasets/real_capture_manifest.csv "
        "--model models/modulation.joblib",
        language="powershell",
    )
    st.markdown(
        """
        The trainer requires source URL, license, annotation reference, and capture-group fields, and
        keeps entire recording groups out of the test split to reduce train/test leakage. It refuses
        to report a model if any class lacks independent capture groups. See `datasets/README.md` for
        a public measured UHF recording and its documented labels.

        A usable real model for multiple modulation families still depends on acquiring and annotating
        enough independent real recordings per class. The app does **not** pretend that one satellite
        capture is a balanced training set or that a synthetic benchmark is over-the-air evidence.
        """
    )


def _show_report(result: SignalAnalysis) -> None:
    report = {
        "schema_version": "1.0",
        "problem_statement": "SIH26147",
        "analysis": result.summary(),
        "limitations": [
            "PSD-derived occupied bandwidth and SNR are estimates, not calibrated measurements.",
            "The blind modulation/symbol-rate estimate is untrained, uncalibrated, and not guaranteed.",
            "FEC and interleaver types/parameters are not automatically inferred.",
            "Manual demodulation requires known symbol timing and does not recover frame synchronization.",
            "Validate recovered bits with known synchronization, CRC, or independent ground truth.",
        ],
    }
    if "trained_prediction" in st.session_state:
        report["trained_model_result"] = st.session_state["trained_prediction"]
    st.download_button(
        "Download JSON analysis report",
        data=json.dumps(report, indent=2),
        file_name=Path(result.source_name).stem + "-analysis.json",
        mime="application/json",
        type="primary",
    )
    st.json(report)


def _parse_bit_array(text: str) -> np.ndarray:
    normalized = "".join(text.split())
    if not normalized or set(normalized) - {"0", "1"}:
        raise ValueError("Enter a non-empty stream containing only 0 and 1.")
    return np.fromiter((character == "1" for character in normalized), dtype=np.uint8)


if __name__ == "__main__":
    main()
