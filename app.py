from __future__ import annotations

import json
import os
from html import escape
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
from sih26147.validation import run_reference_validation


SAMPLE_CAPTURE_PATHS = (
    Path("sih26147")
    / "dsd"
    / "dsd.2021-12-02T16_44_54_046.wav",
    Path("data")
    / "real"
    / "pslv-436500kHz-2018-01-13-095446-prefix-16MiB.wav",
)
SAMPLE_CAPTURE_PATH = next(
    (path for path in SAMPLE_CAPTURE_PATHS if path.is_file()),
    SAMPLE_CAPTURE_PATHS[-1],
)

st.set_page_config(
    page_title="SIH26147 | Signal Analysis",
    page_icon="∿",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp {background:#0b1118;}
        .block-container {padding-top:1.15rem; max-width:1540px; padding-bottom:3rem;}
        [data-testid="stHeader"] {background:transparent;}
        [data-testid="stSidebar"] {background:#0e1720; border-right:1px solid #24313d;}
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {letter-spacing:.02em;}
        .masthead {display:flex; align-items:center; gap:1rem; padding:.35rem 0 1.05rem;
                   border-bottom:1px solid #26333e; margin-bottom:1rem;}
        .brand-mark {display:flex; align-items:center; justify-content:center; width:2.7rem;
                     height:2.7rem; border:1px solid #486777; border-radius:7px; color:#9ad9cf;
                     font-size:1.25rem; font-weight:650; background:#12212b;}
        .brand-copy {flex:1;}
        .brand-copy h1 {font-size:1.42rem; line-height:1.3; margin:0; letter-spacing:.015em;
                        color:#e6edf3; font-weight:620;}
        .brand-copy p {margin:.14rem 0 0; color:#91a2b1; font-size:.83rem;}
        .eyebrow, .section-kicker {color:#8dbdb8; font:600 .68rem/1.4 ui-monospace,Consolas,monospace;
                                   letter-spacing:.11em; text-transform:uppercase;}
        .system-status {display:flex; align-items:center; gap:.45rem; color:#a8b7c4;
                        border:1px solid #30404c; border-radius:4px; padding:.38rem .58rem;
                        font:600 .65rem ui-monospace,Consolas,monospace; letter-spacing:.06em;}
        .status-dot {width:.42rem; height:.42rem; border-radius:50%; background:#70cbb4;
                     box-shadow:0 0 0 3px #70cbb41c;}
        .workspace-intro {border:1px solid #283945; border-radius:7px; background:#101923;
                          padding:1.1rem 1.25rem; margin:.65rem 0 1rem;}
        .workspace-intro h2 {font-size:1.15rem; margin:.3rem 0 .4rem; font-weight:600;}
        .workspace-intro p {color:#9cabb8; margin:.15rem 0; font-size:.9rem;}
        .workflow-list {display:grid; gap:.68rem; margin-top:.45rem;}
        .workflow-item {display:flex; gap:.72rem; align-items:flex-start; color:#c1ccd5;
                         font-size:.88rem;}
        .workflow-step {color:#83bcb3; font:600 .72rem ui-monospace,Consolas,monospace;
                        border:1px solid #304a51; border-radius:4px; padding:.14rem .32rem;}
        .capture-strip {display:flex; align-items:center; justify-content:space-between;
                        padding:.7rem .9rem; border:1px solid #2c3b47; background:#101923;
                        border-radius:6px; margin:.25rem 0 1rem; color:#d4dce3;}
        .capture-strip small {display:block; color:#92a2ae; font-size:.76rem; margin-top:.18rem;}
        .capture-badge {border:1px solid #39675f; color:#9dd7c8; border-radius:4px;
                        padding:.25rem .45rem; font:600 .65rem ui-monospace,Consolas,monospace;
                        letter-spacing:.05em;}
        [data-testid="stMetric"] {background:#101923; border:1px solid #273743;
                                  border-radius:6px; padding:.7rem .8rem;}
        [data-testid="stMetricLabel"] {color:#9eafbc;}
        [data-testid="stTabs"] button {font-weight:550;}
        .stCaption {color:#92a1ad;}
        div[data-testid="stAlert"] {border-radius:5px;}
        .muted {color:#9cb1bf; font-size:.9rem;}
        </style>
        <div class="masthead">
          <div class="brand-mark">∿</div>
          <div class="brand-copy">
            <div class="eyebrow">SIH26147 / SIGNAL ANALYSIS</div>
            <h1>Signal Intelligence Workbench</h1>
            <p>Capture review · parameter estimation · decode verification</p>
          </div>
          <div class="system-status"><span class="status-dot"></span> SESSION ACTIVE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Research prototype · file-based analysis only · do not upload sensitive recordings to a public host.")

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
        _show_validation_lab()
        return

    st.markdown(
        f"""
        <div class="capture-strip">
          <div><span class="section-kicker">ACTIVE CAPTURE</span>
            <small>{escape(result.source_name)} · {escape(result.source_format)} · {result.sample_count:,} samples</small>
          </div>
          <span class="capture-badge">READY FOR REVIEW</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    tabs = st.tabs(("Signal overview", "Decode chain", "Validation lab", "Export report"))
    with tabs[0]:
        _show_analysis(capture, result)
    with tabs[1]:
        _show_processing(capture, result)
    with tabs[2]:
        _show_validation_lab()
    with tabs[3]:
        _show_report(result)


def _show_welcome() -> None:
    st.markdown('<div class="section-kicker">WORKSPACE / NEW ANALYSIS</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="workspace-intro">
              <div class="eyebrow">01 / INGEST</div>
              <h2>Load a capture</h2>
              <p>Review complex baseband recordings without changing the source file.</p>
              <p>PCM WAV · raw IQ (cf32 / ci16 / cu8) · bounded sample windows</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="workspace-intro">
              <div class="eyebrow">02 / WORKFLOW</div>
              <h2>From samples to evidence</h2>
              <div class="workflow-list">
                <div class="workflow-item"><span class="workflow-step">01</span> Inspect spectrum, waterfall and constellation.</div>
                <div class="workflow-item"><span class="workflow-step">02</span> Review estimates; keep unknowns unknown.</div>
                <div class="workflow-item"><span class="workflow-step">03</span> Demodulate and decode with stated parameters.</div>
                <div class="workflow-item"><span class="workflow-step">04</span> Validate against reference bits, sync or CRC.</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(
        "Raw IQ does not contain its own sample rate or center frequency. Supply these from capture metadata; "
        "the analyzer does not infer missing acquisition metadata."
    )


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


def _show_validation_lab() -> None:
    st.markdown('<div class="section-kicker">VALIDATION / KNOWN ANSWER</div>', unsafe_allow_html=True)
    st.subheader("Prove the decode chain")
    st.write(
        "Run a deterministic, locally generated reference frame through BPSK demodulation and a "
        "terminated rate-1/2 convolutional code. Because the source payload is known, the result is "
        "checked by bit-for-bit comparison rather than a confidence score."
    )
    if st.button("Run reference pipeline test", type="primary", key="run_reference_validation"):
        try:
            st.session_state["reference_validation"] = run_reference_validation()
        except ValueError as exc:
            st.error(f"Reference validation failed: {exc}")

    validation = st.session_state.get("reference_validation")
    if validation is not None:
        if validation.passed:
            st.success("PASS · recovered payload matches the known source bit-for-bit.")
        else:
            st.error("FAIL · output did not match the known source. Do not treat this pipeline as verified.")
        first, second, third = st.columns(3)
        first.metric("Known payload", f"{validation.payload_bits:,} bits")
        second.metric("Coded / demodulated", f"{validation.coded_bits:,} bits")
        third.metric("Decoded payload BER", f"{validation.decoded_ber:.6f}")
        st.caption(
            f"Demodulator BER: {validation.demodulated_ber:.6f} · "
            "Generated noiseless reference fixture · not evidence of performance on an unknown capture."
        )
    else:
        st.info("No reference test has been run in this session.")

    with st.expander("Model training and known limitations"):
        _show_training_info()


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
    validation = st.session_state.get("reference_validation")
    if validation is not None:
        report["reference_validation"] = {
            "payload_bits": validation.payload_bits,
            "coded_bits": validation.coded_bits,
            "demodulated_ber": validation.demodulated_ber,
            "decoded_ber": validation.decoded_ber,
            "passed": validation.passed,
            "fixture": "deterministic generated BPSK and convolutional code",
        }
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
