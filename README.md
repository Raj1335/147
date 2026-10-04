# SIH26147 Signal Intelligence Workbench

Local-first signal-analysis workbench for the SIH26147 workflow: ingest WAV/raw IQ, inspect spectrum/waterfall/constellation, estimate a limited set of signal parameters, automatically or manually demodulate supported schemes, de-interleave known layouts, decode supported FEC forms, correlate bit streams, and export a JSON report. Blind estimates are conservative and may return **Unknown**; they are not validated sponsor-grade recognition.

## Quick start

This project targets Python 3.12. CI and deployment use the same minor release; use Python 3.12 locally for consistent dependency and DSP behavior.

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .[dev]
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Inputs and behavior

- `.wav`: uncompressed PCM mono real or stereo I/Q; sample rate comes from the WAV header.
- Raw `.iq`, `.bin`, `.dat`: interleaved `cf32`, `ci16`, or `cu8`; the sample rate, format, and optional center frequency must be supplied by the analyst.
- The web uploader is capped at 100 MB and analysis is capped at 1,000,000 samples per selected window. The UI lets a user change the starting sample to inspect another window. This avoids claiming whole multi-gigabyte uploads are processed in a free web instance.
- **Load synthetic QPSK demo** generates a small capture in memory, runs analysis and preamble-assisted demodulation, and compares against its known bits. The UI and JSON report label it as synthetic; it is not a real capture or accuracy benchmark.
- `python datasets/download_public_capture.py --confirm-download-7-7gb` explicitly downloads a public, real UHF recording. See [`datasets/README.md`](datasets/README.md) for its source, license, labels, size, and annotation caveats.
- The development workspace may contain a 16 MiB measured-data prefix at `data/real/pslv-436500kHz-2018-01-13-095446-prefix-16MiB.wav`. Choose **Local file path** in the app; the path is prefilled when this sample exists. Its adjacent sidecar identifies it as an unverified, incomplete prefix, not a full capture or training corpus.
- Train a real-data model only after labeling multiple independent captured sessions. First copy the manifest template, then populate and review every row:

  ```sh
  cp datasets/real_capture_manifest.example.csv datasets/real_capture_manifest.csv
  python train_model.py --manifest datasets/real_capture_manifest.csv --model models/modulation.joblib
  ```

  The training workflow is not presented as complete until the data and group-wise validation exist.
- Follow [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) for the end-to-end real-capture labeling, manifest audit, grouped evaluation, local model use, and copy-ready vibe-coding prompts.

## Important DSP limitations

The tool reports WAV/raw IQ sample rate, dominant spectral peak, an approximate 99%-power bandwidth, and a rough PSD/noise-floor SNR. These estimates need calibration against known captures. Raw samples cannot reveal an unknown sample rate or absolute RF center frequency without metadata.

The automatic classical estimator attempts symbol-clock and BPSK/QPSK/8PSK/16QAM/2FSK family estimates, and the automatic hard-decision demodulator uses its carrier/timing estimates for supported schemes. These are untrained and uncalibrated, operate on bounded selected windows, and may correctly return **Unknown** when evidence is weak. The independent family hint is also not a trained classifier. FEC and interleaver types/parameters are not inferred automatically. Viterbi is a hard-decision rate-1/n decoder; RS requires byte-aligned codewords and user-supplied parity length; LDPC requires a supplied parity-check matrix. Interleavers require the exact dimensions/branch count/seed, and their conventions may not match a particular transmitter. Validate output with sync words, CRC, re-encoding/BER, and authoritative protocol parameters.

The experimental `sih26147.dsp.demodulate_chain` helper currently supports hard-decision QPSK with integer samples per symbol, an RRC matched filter, and optional preamble-bit phase resolution. It is not yet wired into the app's decode UI, does not implement differential QPSK or soft decisions, and cannot resolve the 90-degree phase ambiguity without known reference bits.

No encryption decoding, live SDR capture, or server-side capture retention is implemented. Do not upload sensitive captures to a public deployment. A free hosted instance has ephemeral storage, resource limits, and no guarantee of availability; use local mode for large or sensitive captures.

## Official problem-statement traceability

| SIH26147 requirement | Current implementation | Status / gap |
|---|---|---|
| `.wav` / `.IQ` GUI input | PCM WAV, float32 WAV, and raw `cf32` / `ci16` / `cu8` input | Implemented within window/upload bounds |
| Sampling frequency | WAV header or operator entry for raw IQ | Input metadata surfaced; unknown raw-IQ rate cannot be inferred from bytes alone |
| Modulation type / symbol rate | Blind classical estimates for BPSK/QPSK/8PSK/16QAM/2FSK; optional model trained from annotated captures | Automatic estimator is tested on deterministic fixtures, not validated/calibrated on labeled real captures; can return unknown |
| Spectrum, constellation, waterfall | Welch PSD, raw-sample constellation, STFT waterfall | Implemented for selected window |
| FSK/QAM/PSK demodulation | Hard-decision manual path and automatic carrier/timing-assisted path for 2FSK/BPSK/QPSK/8PSK/16QAM | No protocol/frame synchronization; validate bits against sync words, CRC, or known truth |
| Block/convolutional/diagonal/pseudo-random de-interleaving | Parameterized transforms | Requires exact layout/seed; convolutional mode also requires original bit count to remove padding; not blind inference |
| Viterbi, RS, concatenated, LDPC | Viterbi, RS, generic H-matrix LDPC; sequential stages can be wired via UI | Concatenated chain is manual; code family, polynomials, framing, and soft input are not auto-discovered |
| Bitstream correlation | Sliding offset BER comparison | Implemented for operator-provided reference |
| Automated FEC/interleaver extraction | None | Not yet implemented; needs protocol metadata, labeled captures, and ground-truth validation |
| Real-data training | Provenance-requiring trainer and public capture acquisition instructions | No trained real model shipped: currently located dataset is a single large, incompletely time-annotated recording |

This status table is intentional: the system integrates analysis and decoding workflows, but it is not yet a fully validated blind signal-intelligence solution. Do not pitch incomplete rows as completed features.

## Free hosting

`render.yaml` defines a free Docker-based Render web service. Sign in to Render, create a Blueprint from this repository, and deploy the `render.yaml` service. It runs as a non-root user, uses a health check, and excludes local captures and models from the image. The free tier can sleep and has limited CPU/RAM/disk; 100 MB upload limits and ephemeral files make it a demo host, not an operational data platform. Keep large capture data out of Git and out of hosted sessions. The app processes uploads in memory for bounded sample windows; use local mode for the multi-gigabyte public capture.

To run the same hosted-mode container locally under the demo memory/CPU limits:

```sh
docker build -t sih26147:local .
docker run --rm --publish 8501:8501 --cpus=0.1 --memory=512m --read-only --tmpfs /tmp:rw,noexec,nosuid,size=64m sih26147:local
```

Open `http://localhost:8501`. Hosted mode intentionally disables local-path access; use the native local quick start for private files.

## Validation

```sh
python -m pytest --cov=sih26147 --cov-report=term-missing --cov-fail-under=85
```

Tests use small deterministic signals only to verify algorithms. They are not training data and are not evidence of real-world modulation-classification accuracy.
