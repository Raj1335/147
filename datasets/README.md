# Real measured data and model training

## Public over-the-air UHF capture

The first candidate for SIH26147 validation is Daniel Estévez's **RF recording of PSLV 2018-004 launch cubesats**:

- Dataset page and download: <https://zenodo.org/records/6403199>
- DOI: <https://doi.org/10.5281/zenodo.6403199>
- License shown by Zenodo: **Creative Commons Attribution 4.0 International (CC BY 4.0)**; retain attribution and verify the current record terms before redistribution.
- Capture: 4 Msps complex IQ, WAV, nominal center 436.5 MHz, recorded over the air near Madrid with a LimeSDR and antenna on 2018-01-13.
- File size in the Zenodo record checked for this project: 7,736,807,424 bytes (about 7.2 GiB). It is deliberately **not** copied into the source tree or downloaded automatically.
- The first 128 bytes were checked with an HTTP range request: this real file has zeroed RIFF and `data` length fields despite the multi-gigabyte body. The local PCM reader has a bounded fallback that determines the available frame count from the file size when those fields are zero.
- Author's capture notes / documented signal labels: <https://destevez.net/2018/01/decoding-satellites-from-the-pslv-2018-004-launch/>

The author reports signals including PicSat at 435.525 MHz (1.2 kbaud BPSK, G3RUH/AX.25), CNUSail-1 at 437.100 MHz (9.6 kbaud BPSK), VZLUSAT-I at 437.240 MHz (4.8 kbaud GMSK), and INDUST-10 at 435.080 MHz (about 1.2 kbaud FSK/AX.25). These are **reported frequencies and modes**, not sample-accurate time-window labels. Review the waterfall and annotate clean windows before training; do not copy invented sample offsets into the manifest.

## Supplementary real WAV decoder fixtures

The [satellite-recordings collection](https://github.com/daniestevez/satellite-recordings) describes short amateur-satellite packet recordings as mono WAV at 48 kHz; its repository README says they are for decoder testing and its license is Unlicense/public domain. They can exercise WAV ingest and packet-workflow plumbing, but are already mono audio (not complex raw RF I/Q), do not form a verified modulation-labeled training set, and should not be used to claim over-the-air IQ AMC performance.

### Download

The capture is multi-gigabyte. Check available disk space and bandwidth first (the downloader requires the full file plus 512 MiB free). Download explicitly with:

```powershell
python datasets\download_public_capture.py --confirm-download-7-7gb
```

Files go under ignored `data\real\`; the application does not bundle them or upload them to a deployment. The downloader checks the published MD5 checksum after completion and writes an adjacent `.metadata.json` provenance sidecar. Interrupted partial downloads are retained and are not resumable by this script.

For a small, bounded **real-data prefix** suitable for checking ingestion without downloading 7.2 GiB, use:

```powershell
python datasets\download_public_capture.py --sample-bytes 16777216
```

This requests bytes 0–16,777,215 using HTTP Range and stores an adjacent provenance sidecar with a SHA-256 of that prefix. The sample file currently present in this workspace is `data\real\pslv-436500kHz-2018-01-13-095446-prefix-16MiB.wav`; SHA-256: `ad24578b1612cc82932bc21589c5a10caa9b6d483a637c18409ec8d870a5432a`. The source's full-file MD5 is **not** verified by a prefix download. In the GUI, choose local-path mode, inspect this file, and enter 436.5 MHz as the known center frequency if absolute frequency labels are desired.

## Training requirements and limits

For the full step-by-step annotation, manifest, audit, training, evaluation, and vibe-coding workflow, see [`../TRAINING_GUIDE.md`](../TRAINING_GUIDE.md).

`real_capture_manifest.example.csv` is a schema-only template. Populate it only after manually selecting windows and verifying their labels. `capture_group` must identify the independent recording/session, not each window cut from one recording. Include exact provenance, license, and annotation source for every row.

The trainer (`train_model.py`) deliberately:

1. rejects rows not identified as measured/recorded and rows without provenance;
2. requires at least two independent recording groups per label;
3. keeps recording groups disjoint between training and test data; and
4. refuses to report an accuracy if a valid group-wise split cannot be made.

That means the single PSLV recording is valuable **real held-out signal-analysis evidence**, but by itself it cannot establish a generalizable, multi-class AMC model. Collect or locate multiple independently captured examples per class (with permission and ground-truth annotation) before claiming real-data training performance. RadioML and other generated channel-model datasets must be separately identified as synthetic/simulated and must not be presented as real captures.

## Why no trained model is shipped

The available public capture is one large recording with some modes documented in a field report, not a balanced, fully time-annotated modulation/FEC/interleaver corpus. Training a model on synthetic examples and calling it real-world trained would be misleading. This project ships a provenance-enforcing training path and an uncalibrated DSP hint instead. A model will only be credible after collecting independently recorded, annotated data and reporting its capture-group holdout results.

This public capture is a satellite/UHF example; it does not prove performance over every HF/VHF/UHF sensor, waveform, or receiver chain. For operational use, obtain sponsor-approved captures and ground truth.
