# Real-Capture Training Guide (Vibe-Coding Edition)

This guide is a practical, prompt-driven workflow for building and evaluating a modulation classifier from **real, labeled IQ captures** in this repository. It is written so you can work through it yourself or give each numbered prompt to a coding assistant in your editor.

> **Do not train on the current 16 MiB prefix.** It is one incomplete capture prefix, has no sample-accurate labels, and is not a training corpus. The project intentionally ships no trained classifier. The deterministic signals in `tests/` are algorithm fixtures, not training data.

## What you are building

The current trainer extracts a fixed feature vector from each selected IQ window and fits a Random Forest. It requires data provenance and evaluates on recording groups withheld from training. A generated model is only a research baseline: passing the script does not establish useful performance on new receivers, locations, signal conditions, or modulation classes.

The complete workflow is:

1. Collect or locate lawfully usable, real captures with defensible labels.
2. Keep each original recording intact and record its provenance.
3. Annotate exact sample windows and review labels against trusted evidence.
4. Populate the CSV manifest with one row per labeled window.
5. Run group-disjoint evaluation and inspect every class result.
6. Save and test the model locally in the app.
7. Report limitations and only claim performance supported by held-out real captures.

## 1. Prepare the environment and inspect the project

From the project root in Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py --help
```

Before changing code, ask your coding assistant to inspect the current implementation:

> Read `TRAINING_GUIDE.md`, `sih26147/training.py`, `sih26147/analysis.py`, `sih26147/ingest.py`, `train_model.py`, and `tests/test_training.py`. Summarize the exact current manifest schema, accepted IQ formats, feature vector, group-split rules, model artifact path, and app loading behavior. Do not edit files or claim that existing data are trainable.

Check the dataset's usage terms before downloading, processing, or sharing it. Keep large recordings and derived data under ignored `data/` directories, not in source control or a public hosted app.

## 2. Build a trustworthy capture collection

For every source recording, preserve the original file and write down:

- where it came from and its license/permission;
- the receiver, antenna, location (if safe and permitted), date, and capture session;
- sample rate, I/Q representation, center frequency, and any known preprocessing;
- how modulation labels were established (protocol documentation, transmitter configuration, expert review, or other ground truth);
- known impairments, competing signals, clipping, filtering, and uncertain intervals.

Prefer several **independent recordings per class**, ideally across different sessions and capture conditions. A recording cut into many windows is still one recording group. More windows from the same file do not substitute for independent recordings.

Do not infer exact time-window labels from a source article that only lists signal frequencies or modes. Establish and review the actual sample intervals first. If a source does not provide enough evidence to identify a window's modulation, mark it as uncertain and leave it out until verified. Do not label noise, transitions, mixtures, or undecodable captures as a modulation class merely to make the dataset larger.

The public PSLV recording is a useful real-data ingestion and analysis example, but its published notes are not sample-accurate annotations. The local prefix is incomplete and is not sufficient for training. See [`datasets/README.md`](datasets/README.md) for its source and limits.

### Suggested project data layout

Keep files local and ignored by Git:

```text
data/
  real/
    source-a/
      recording-001.wav
      recording-001-notes.md
    source-b/
      recording-002.iq
      recording-002-notes.md
  annotations/
    real_capture_manifest.csv
```

Never put credentials or private access links in the manifest. Use a stable public provenance URL where possible, otherwise a non-secret reference that collaborators with permission can resolve.

## 3. Annotate windows and populate the manifest

Start with the repository's schema:

```powershell
Copy-Item datasets\real_capture_manifest.example.csv data\annotations\real_capture_manifest.csv
```

There must be **one CSV row per labeled window**. Required columns are:

| Column | What to enter |
|---|---|
| `capture_path` | Path to the recording, relative to the manifest's directory. Use a Windows-compatible path such as `..\real\source-a\recording-001.wav`. |
| `label` | The verified modulation name, consistently spelled across all rows (for example `BPSK`, `QPSK`, `2FSK`, or `GMSK`). Do not mix aliases for one class. |
| `sample_rate_hz` | Positive sample rate in samples/second. For WAV, it must agree with the file header; raw IQ requires the known capture rate. |
| `raw_format` | `cf32`, `ci16`, or `cu8` for raw interleaved IQ. WAV rows are read from the WAV header; keep this field populated for schema consistency. |
| `first_sample` | Zero-based sample index of the first sample in this window. |
| `sample_count` | Window length from 128 to 1,000,000 samples. Use the same order of magnitude across examples where practical. |
| `capture_group` | Stable ID for the **original independent recording/session**. All windows from the same recording use exactly the same group ID. |
| `data_origin` | One of `measured`, `over_the_air`, or `recorded`. The trainer rejects values such as `synthetic`. |
| `source_url` | HTTPS URL identifying the recording/source or authoritative dataset record. |
| `license` | Actual license or permission basis, including attribution requirements where applicable. |
| `annotation_reference` | Verifiable reference for the label and interval, such as a notes file plus page/section, annotation record, protocol config, or expert review record. |
| `center_frequency_hz` | Optional center frequency in Hz; this is not required by the trainer. |

Example **schema illustration only** (the path, interval, source, and annotation below are placeholders and must not be copied as real labels):

```csv
capture_path,label,sample_rate_hz,raw_format,first_sample,sample_count,capture_group,data_origin,source_url,license,annotation_reference,center_frequency_hz
..\real\session-a\capture.wav,BPSK,4000000,cf32,12000000,262144,session-a,measured,https://example.org/dataset,CC BY 4.0,annotations/session-a.md#verified-window,436500000
```

The project only checks that the referenced HTTPS URL is syntactically valid; it does not verify the source, license, label, interval, or group identity for you. Those require human review.

### Make a window manifest with an assistant

Use a prompt like this after you have a recording and actual verified annotations:

> I have a real IQ recording at `[path]`, sample rate `[Hz]`, format `[WAV stereo I/Q | cf32 | ci16 | cu8]`, and center frequency `[Hz or unknown]`. The independently verified annotations are: `[list exact start sample, sample count, label, and evidence for each window]`. Create CSV rows for `data/annotations/real_capture_manifest.csv` using the existing example header. Keep the same `capture_group` for every window from this original recording. Do not invent offsets, labels, licenses, URLs, or missing metadata. First validate the row values against the file and report any unresolved fields; only then write the rows.

For an assistant to inspect or measure files, use approved local tools and provide a small bounded range. Do not ask it to guess annotations from a filename, plot, or article that lacks exact timing evidence.

## 4. Audit the dataset before fitting

Manually check:

- every path exists and resolves relative to the manifest;
- every window is in bounds and contains the stated number of samples;
- WAV header rate and I/Q channel convention match the manifest;
- raw IQ byte format, endianness, scaling, and sample rate are known;
- labels have consistent spelling and were independently verified;
- each group corresponds to one original recording/session;
- no recording (or near-duplicate copy) appears under multiple group IDs;
- each class has examples from multiple independent groups;
- class balance and capture conditions are documented;
- no synthetic, augmented, or simulated sample is labeled as measured.

Ask your coding assistant to help with an **audit only** before training:

> Inspect `data/annotations/real_capture_manifest.csv` and the referenced local files. Do not train or edit the manifest. Check schema, paths, sample-rate/format consistency where verifiable, window bounds, per-class window counts, and per-class distinct `capture_group` counts. Flag any identical recording assigned multiple group IDs if detectable. Do not validate labels by guessing. Print a concise audit report and stop if a required fact cannot be verified.

The current trainer requires at least four manifest rows, at least two distinct recording groups for every label, a group-wise split that contains every class on both sides, and `test_size` between 0.05 and 0.5. Two groups per class is only a code minimum, **not** a good evidence threshold. Collect several more independent groups per class before interpreting performance.

## 5. Train and read the evaluation

Run from the project root:

```powershell
python train_model.py `
  --manifest data\annotations\real_capture_manifest.csv `
  --model models\modulation.joblib `
  --test-size 0.25 `
  --seed 26147
```

The output includes the model path, row count, class labels, group-holdout accuracy, per-class precision/recall/F1, confusion matrix, and provenance. The saved model is also written to `models\modulation.joblib` (ignored by Git). Training raises an explicit error if provenance, data files, class groups, or a valid grouped split are missing.

Interpret results in this order:

1. Confirm every class appears in both training and held-out groups.
2. Inspect the confusion matrix, per-class recall, and class support—not just overall accuracy.
3. Identify whether the held-out groups cover genuinely independent recordings and conditions.
4. Inspect errors against the original IQ and annotations. Correct data issues only from evidence, then rerun the evaluation.
5. Record the manifest version/hash, code version, command, seed, metrics, and known limitations.

The current script does a single group-disjoint holdout. Results can vary with the groups and seed. A high score on a small or narrow dataset is not a generalization guarantee. Do not use windows from the same original recording on both sides of evaluation, and do not tune labels/features repeatedly against the held-out set and then present it as an untouched final test.

### If training refuses the manifest

Do not weaken the checks just to make training complete. Use the error to identify the missing evidence:

- **Too few groups:** acquire more independent recordings per class.
- **A class missing from a split:** add independent recordings; repeated windows from one recording do not solve group leakage.
- **Capture not found:** fix the relative path from the manifest directory.
- **Wrong number of samples:** check sample indexing, rate, end-of-file, and WAV frame structure.
- **Invalid origin/provenance:** correct it from the source documentation; never relabel synthetic data as real.
- **No defensible annotation:** do not include that row yet.

## 6. Check the trained artifact in the app

The GUI checks for `models/modulation.joblib` after analyzing a capture and, if it exists, applies that local model to the selected window. Start locally:

```powershell
streamlit run app.py
```

Load a separate, known real capture window not used in training. Compare the prediction with authoritative ground truth, and also inspect the classical estimator. The model result's score is not guaranteed to be calibrated confidence. The artifact stores the manifest path and evaluation metadata; keep the manifest and provenance records with the model so results can be reproduced.

Only load joblib files you created or otherwise trust. Like pickle-based formats, joblib artifacts can execute code when loaded. Do not accept a model upload from an untrusted source.

Do not copy a locally trained artifact into a public Render deployment as a substitute for validation. The web service is a constrained demo host; use local processing for large/private recordings, and follow dataset license/privacy terms.

## 7. A safe vibe-coding sequence for improving the training system

Give one prompt at a time. Review the proposed diff and run tests after each step.

### Prompt A — manifest audit utility

> Add a read-only manifest audit command for the existing schema. Report file/path errors, invalid numeric fields, out-of-range windows when determinable, label counts, group counts per label, and possible recording-group mistakes. Do not train, change labels, relax current trainer validation, or silently skip bad rows. Reuse existing ingestion code. Add focused tests for both valid and invalid manifests, update the CLI help and README, then run the targeted tests.

### Prompt B — reproducible evaluation report

> Extend training output so the grouped train/test recording IDs, per-class support, seed, test fraction, feature version, and dependency/model configuration are recorded in a machine-readable report beside the model. Keep all train/test groups disjoint and fail explicitly if not. Do not report accuracy as real-world performance. Add tests proving that no group overlaps and that report metadata is reproducible.

### Prompt C — improve evaluation, not just the score

> Review the current feature extraction and grouped holdout design. Propose a minimal evaluation improvement that tests generalization across independent real recording groups. Before implementing, explain the split strategy and how class representation is preserved. Do not use synthetic test signals as training data, do not leak groups, and do not optimize against a final untouched test set. Implement only after the split is testable and add tests for leakage and missing-class edge cases.

### Prompt D — app-facing model safety

> Inspect how the app loads and reports a locally trained model. Improve user-facing reporting so the model label, uncalibrated score, held-out validation metadata, and data limitations are clear. Keep model loading local and trusted-only; do not add arbitrary model upload or server-side persistence. Add a regression test or app smoke check and update documentation.

### Prompt E — dataset readiness review

> Review the current real-capture dataset candidates and annotations. Produce a readiness checklist of verified facts, missing labels, license/permission concerns, and group leakage risks. Use only evidence in the supplied source records and local annotation files. Do not fabricate capture offsets or call generated waveforms real. Stop without training if sample-level labels or independent groups are insufficient.

Avoid prompts such as “make accuracy 99%,” “fill missing labels,” or “use the public capture to create lots of training rows.” Those requests reward leakage and fabricated certainty rather than a reliable model.

## 8. What a credible training release should contain

Before describing a model as validated, retain and review:

- source recordings and license/provenance records (stored privately if required);
- reviewed sample-accurate annotations and the exact manifest used;
- independent capture-group definitions;
- repeatable train/test groups, command, seed, package versions, and code revision;
- per-class metrics, confusion matrix, and held-out support;
- error analysis, receiver/channel limitations, and known unknowns;
- a truly independent final evaluation set if model development used the group-holdout results;
- evidence that artifact predictions work on intended hardware/data without leaking test recordings.

**Current status:** the trainer and workflow are implemented, but no qualifying real-data classifier is shipped. Automatic FEC/interleaver identification and protocol/frame synchronization are also outside this classifier-training workflow.

## Reference decode validation in the app

The app's **Validation lab** runs a deterministic known-answer vertical slice: a generated payload is convolutionally encoded, BPSK-modulated, demodulated, and Viterbi-decoded. A passing result means the controlled software chain exactly recovered that generated payload. It does not establish blind modulation-classification performance or validate any unknown/real capture. Use independent ground-truth recordings and report their held-out results before claiming real-signal performance.
