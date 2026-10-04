# SIH26147 Signal Intelligence Workbench: Completion Plan

Scope: engineering completion only (no pitch, no submission material).
Audited: `Raj1335/147` @ `f500e80` (shallow clone, 30 tracked files, ~3,160 lines of Python).
Method: read every source file, ran the test suite, and wrote extra probe scripts against realistic signals.

---

## 1. Where the project actually stands

### What is solid (keep it)

- Clean module split: `ingest`, `analysis`, `estimators`, `demod`, `decode`, `analog`, `training`, `validation`.
- Input handling is careful: bounded windows, malformed-file errors, and a fallback for the PSLV capture's zeroed WAV length fields.
- The training path enforces provenance and recording-level (group) holdout, which is rare and correct.
- The docs are honest about limits. Keep that discipline; it is what makes the later claims believable.
- 36 tests pass in ~3 s. The known-answer chain (BPSK → conv. encode → Viterbi) is exact.

### What is not production-grade (measured, not guessed)

| # | Finding | Evidence |
|---|---|---|
| 1 | **The blind estimator fails on realistically shaped signals.** | RRC-shaped (β=0.35, 8 sps) BPSK/QPSK/8PSK/16QAM with 150 Hz carrier offset: **0/5 correct at 25, 15 and 8 dB SNR for every class.** Even clean rectangular-pulse QPSK at 40 dB returned `Unknown`. |
| 2 | **Symbol-rate estimate is quantized to integer lags.** | True 6000 Bd at fs=48 kHz came back as 5333 Bd (= fs/9) in 3 of 5 runs at 25 dB; at 10 dB it returned `None` or 2.0–2.8 kBd. |
| 3 | **Manual demod only works if the symbol centre lands on sample `sps//2`.** | No matched filter, no timing recovery. On an RRC-shaped QPSK stream with symbol peaks at sample offset 0, midpoint sampling gave ≈0.48 BER (coin flip). |
| 4 | **There is no channel extraction.** | No mixer/decimator/channel filter anywhere. The repo's own real capture is 4 Msps wide with several narrowband signals; the whole band is fed to a single-signal estimator. Its documented result on that capture is `Unknown`. |
| 5 | **There is no integrity oracle.** | No sync-word search, CRC, HDLC/AX.25 deframer, or descrambler. The words "sync" and "CRC" appear only in warning strings. Nothing can tell the user whether a decode is right. |
| 6 | **FEC is a demo-grade subset.** | Viterbi is hard-decision only, no puncturing, ~0.3 ms/bit (5,000 bits = 1.5 s; guard caps it near 156k coded symbols). `rs_decode` exposes only the parity count: `reedsolo` defaults (prim 0x11d, fcr 0) are not CCSDS (0x187, fcr 112, dual basis). LDPC uses a dense float matrix and Python loops with `np.delete`; a DVB-S2-size code (64,800 columns) would not fit in memory. |
| 7 | **Mapping conventions are non-standard.** | QPSK/8PSK/16QAM labels are natural binary by angle/level, not Gray. No differential decoding, no NRZI. Real systems use Gray. |
| 8 | **No automated FEC/interleaver detection**, which the problem statement asks for ("extraction of … FEC and interleaving parameters"). | The README says so itself. |
| 9 | **Repo hygiene.** | `README.md` still ends with 7 NUL bytes (`# 1 6 0` in UTF-16); `SESSION_LOG.md` says they were removed. Two PNGs sit in the repo root (one named `power-spectral-density (1).png`). No LICENSE, no CI, no `pyproject.toml`, no Dockerfile, no lint/type checks. `pytest` is in runtime `requirements.txt`. Dev is Python 3.14.7 but deploy is pinned to 3.12.8, so you are testing on a different interpreter than you ship. |
| 10 | **Single 727-line `app.py`**, no caching (`st.cache_*` appears nowhere), blocking compute on the UI thread. | |

Bottom line: the scaffolding and honesty are good, but the signal chain between "bytes on disk" and "trusted bits" is mostly missing. Phases 1 to 4 below are the heart of the completion; everything else is hardening.

---

## 2. Definition of done (production-grade, measurable)

Treat these as release gates. Numbers are proposed targets; tune them after the first benchmark run, but write them down before you tune.

| Area | Gate |
|---|---|
| Demod correctness | BPSK/QPSK/8PSK/16QAM BER within **1 dB** of theory (AWGN, coded or uncoded) at BER 1e-3, with timing offset, carrier offset ≤ 2% of symbol rate, and RRC shaping. |
| Blind estimation | Symbol rate within **0.5%**; modulation accuracy **≥ 90%** at ≥ 10 dB SNR over the supported class list; confusion matrix committed to the repo. |
| Integrity | Every decode path ends in a pass/fail oracle (sync word, CRC, or re-encode BER) shown in the UI and the report. |
| Real data | At least **one real over-the-air signal decoded end-to-end with CRC-valid frames** (see Phase 7). |
| FEC | Viterbi ≥ 200 kbit/s; RS/LDPC match published test vectors; auto-detection finds the right code from a catalog in ≥ 95% of synthetic trials ≥ 6 dB above threshold. |
| Scale | Whole-file processing by streaming chunks; memory stays flat on a 7 GB file; no UI freeze. |
| Quality | `ruff`, `mypy` (strict on `sih26147/`), `pytest` ≥ 85% line coverage on the core, all in CI on every push. |
| Security | Threat model written; no unsafe deserialization; local-path mode sandboxed; `pip-audit` clean. |
| Reproducibility | One command (`make verify`) rebuilds the env, runs tests, and regenerates the benchmark tables. Every report embeds code version, parameters, and input hash. |

---

## 3. Phased plan

Order matters: each phase unlocks the next. Effort is a rough single-developer estimate in focused days.

### Phase 0: Foundation and hygiene (1 day)

1. Strip the NUL bytes from `README.md`; add a CI check that fails on NULs in any `*.md`.
2. Delete both root PNGs (or move one to `docs/img/`), and add a `.gitignore` rule so `*.png` in the root cannot come back.
3. Add `LICENSE` (pick one deliberately; if you reuse CC BY 4.0 data keep the attribution in `datasets/README.md`).
4. Add `pyproject.toml`: package metadata, `requires-python = ">=3.12,<3.13"`, runtime deps, and a `[dev]` extra (`pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`, `pip-audit`). Remove `pytest` from `requirements.txt`.
5. **Pick one Python version for dev, CI and deploy** (3.12 recommended; numba support on 3.14 is not something to bet on). Say so in the README.
6. Lock dependencies (`uv lock` or `pip-compile`); commit the lockfile.
7. `ruff` + `mypy --strict sih26147/` + `pre-commit`. Fix what they find.
8. GitHub Actions: lint → type-check → tests with coverage → `pip-audit`. Cache pip. Fail under 85% core coverage.
9. Replace Windows-only paths in docs (`datasets\...`, PowerShell only) with cross-platform commands, or show both.
10. Fix section numbering in `SESSION_LOG.md` (7, 8 appear before 6) or retire it into `CHANGELOG.md`.

Done when: CI is green on a fresh clone and `git status` is clean after `make verify`.

### Phase 1: The signal chain (the biggest gap) (6–8 days)

Create a `sih26147/dsp/` package of small, pure, individually tested blocks. Each block takes arrays and returns arrays plus a small metadata dataclass. No Streamlit imports.

| Block | File | What it must do |
|---|---|---|
| Channelizer / DDC | `dsp/channelizer.py` | Pick a center offset and bandwidth from the waterfall/PSD, mix to baseband, low-pass (polyphase FIR via `scipy.signal.firwin` + `resample_poly`), decimate to ~4–8 sps. Also: auto-detect candidate channels from the PSD (peak picking + occupied bandwidth) and return a ranked list. |
| Cleanup | `dsp/preprocess.py` | DC removal, I/Q imbalance estimate, AGC, burst detection (energy detector) so continuous-wave noise is not decoded. |
| Pulse shaping | `dsp/filters.py` | RRC/RC taps generator (β, span, sps), Gaussian taps (for GMSK/GFSK), matched-filter application. |
| Timing recovery | `dsp/timing.py` | Gardner (PSK/QAM) and Mueller-Müller; Oerder-Meyr feed-forward for bursts. Fractional-delay interpolator (cubic or polyphase). Output: one symbol per output sample with timing-error trace. |
| Carrier recovery | `dsp/carrier.py` | Coarse: FFT of x^M (already prototyped in `_psk_frequency`). Fine: Costas loop (BPSK/QPSK), decision-directed PLL for 8PSK/QAM. Return lock indicator and phase-ambiguity set. |
| Equalizer (optional, last) | `dsp/equalizer.py` | CMA / LMS for multipath. |
| Symbol mapping | `dsp/mapping.py` | Gray-coded constellations as the default (BPSK, QPSK, 8PSK, 16QAM, 64QAM), natural-binary as an option, differential encode/decode, NRZI, soft LLR output (max-log). |
| FSK family | `dsp/fsk.py` | 2FSK/4FSK, MSK, GFSK, GMSK: FM discriminator + symbol-synchronous integrate-and-dump, deviation estimate, modulation index estimate. (Your target satellites include GMSK and FSK.) |

Integrate them in one object:

```python
@dataclass
class DemodConfig: scheme, symbol_rate_hz, rolloff, gray, differential, ...
@dataclass
class DemodResult: symbols, bits, llrs, timing_trace, freq_trace, lock, evm, warnings
def demodulate_chain(iq, fs, cfg) -> DemodResult: ...
```

Keep the old `demodulate()` as a legacy "assumes aligned" function, and label it that way in the UI.

**First failing test to write (it encodes finding #1 and #3):** generate RRC-shaped QPSK at 6 kBd, fs = 48 kHz, 150 Hz carrier offset, 0.37-symbol timing offset, 15 dB SNR; `demodulate_chain` must recover the payload with BER < 1e-2 after resolving phase ambiguity. It fails today.

Done when: BER curves for all supported schemes sit within 1 dB of theory (Phase 2 generates the plots), and the old `Unknown`-on-clean-QPSK case passes.

### Phase 2: A trustworthy test and benchmark harness (3 days; start alongside Phase 1)

You need a transmitter to test a receiver.

1. `sih26147/sim/` (**test-only, clearly labeled synthetic**, consistent with your existing "never present synthetic as real" rule):
   - modulators for every supported scheme with RRC/Gaussian shaping;
   - impairments: AWGN at a given Eb/N0 or SNR, carrier offset + phase noise, timing offset and clock drift, simple multipath, DC offset, I/Q imbalance;
   - encoders matching your decoders (conv. encode already exists in `validation.py`; add RS, interleavers, LDPC).
2. `benchmarks/ber_curves.py`: sweeps Eb/N0 for each scheme, compares with closed-form theory (`scipy.special.erfc`), writes a PNG and a CSV. Commit the CSV; CI fails if any point drifts > 1 dB.
3. `benchmarks/amc_confusion.py`: Monte-Carlo confusion matrix over SNR × class. Commit the table.
4. Property-based tests with `hypothesis` for the pure functions (de-interleave ∘ interleave = identity for random shapes, bit packing round trips, mapping/demapping round trip, parser fuzzing for `read_capture`).
5. Fuzz `ingest.py` with truncated, oversize-header, and odd-length files (cybersecurity-style negative tests; also good portfolio material).
6. Mutation spot-check (`mutmut`) on `decode.py` only, to see whether your tests would notice a flipped bit order.

Done when: `make bench` regenerates every table and figure, and README embeds them.

### Phase 3: Blind estimation rewrite (4–5 days)

Replace "try each constellation and compare residuals" with the standard pipeline; keep the old estimator as a cross-check.

1. **Symbol rate**: cyclic autocorrelation / spectral-line method on |x|² and on x² (BPSK/QPSK show a line at 1/T). Refine the peak with parabolic interpolation to sub-bin accuracy (fixes finding #2). Also report a confidence value from peak-to-median ratio.
2. **Bandwidth/rolloff**: from the channelizer's occupied bandwidth, `rolloff ≈ BW·T − 1`.
3. **Modulation class** from higher-order cumulants (C20, C21, C40, C42, C63) after normalization, plus FM-discriminator statistics for the FSK family. Start with decision rules from the literature; then train a small classifier on top (see Phase 7 caveat). Candidate classes: OOK/ASK, 2FSK, 4FSK, MSK/GMSK, BPSK, QPSK/OQPSK, 8PSK, 16QAM, 64QAM, plus **Unknown** as a first-class output.
4. **Parameter refinement** after class selection: carrier offset, timing, deviation, index, via the Phase 1 loops. Report the lock state.
5. Output a ranked list with a calibrated score (fit on the Phase 2 Monte-Carlo, not on guesses), and say when the evidence is too low.

Done when: symbol-rate error < 0.5% and classification ≥ 90% at ≥ 10 dB on the synthetic suite; AM/FM/CW analog branch gets an equivalent detector so it is not purely manual.

### Phase 4: Framing, descrambling, and the integrity oracle (4 days)

This is what turns "some bits" into "verified data". Do it before more FEC work.

1. `sih26147/framing/crc.py`: table-driven CRC engine with a catalog (CRC-16/CCITT-FALSE, CRC-16/X-25, CRC-16/XMODEM, CRC-32, CRC-8). Test against the standard check values for the string `"123456789"`.
2. `framing/sync.py`: attached-sync-marker search (e.g. CCSDS `1ACFFC1D`) with bit-error tolerance, all four phase ambiguities and polarity inversion tried automatically, returning offsets and quality.
3. `framing/hdlc.py` + `ax25.py`: flag detect, bit-unstuffing, FCS check, AX.25 header parse (callsigns, control, PID).
4. `framing/scramblers.py`: G3RUH (x¹⁷+x¹²+1) self-synchronizing, CCSDS additive pseudo-randomizer (x⁸+x⁷+x⁵+x³+1), NRZI.
5. Wire a "Frame finder" tab into the decode chain: given hard/soft bits, try all (polarity × differential/NRZI × scrambler × frame type) combinations and list which yields valid CRCs. This is the correct place for "automation" without guessing.

Done when: synthetic AX.25/G3RUH BPSK at 1.2 kBd round-trips with CRC-valid frames, including with a flipped polarity and a 1-bit slip.

### Phase 5: FEC and interleaving, completed (7–9 days)

**Convolutional / Viterbi** (`decode/viterbi.py`)
- Vectorize add-compare-select across states with a precomputed trellis (or `numba`), keep the pure-Python version as the reference for tests.
- Soft-decision (LLR/quantized) metrics, **puncturing** patterns (2/3, 3/4, 5/6, 7/8), tail-biting, constraint lengths up to 9, rate 1/2, 1/3.
- Traceback depth ≥ 5K for streaming (continuous) decoding, not only terminated blocks.
- Target ≥ 200 kbit/s; document the speed-up with a benchmark.

**Reed-Solomon** (`decode/rs.py`)
- Expose `nsize`, `nsym`, `fcr`, `prim`, `generator`, `c_exp`, shortening, virtual fill, erasures.
- Add CCSDS RS(255,223) with the dual-basis (Berlekamp) conversion and interleave depth I = 1…8; verify against a CCSDS test vector.
- Report corrected-symbol counts, not just output bytes.

**LDPC** (`decode/ldpc.py`)
- Replace the dense `q`/`r` matrices with `scipy.sparse` (CSR/CSC) and vectorized check-node updates; use normalized min-sum (faster, less numerically fragile than `arctanh`) with an optional sum-product mode.
- Import H from `.alist` files; ship CCSDS C2 (8176,7156) and one AR4JA code as built-in catalog entries, with generator support so you can encode in tests.
- Syndrome check every iteration, early stop (already done), iteration statistics.

**Interleavers** (`decode/interleave.py`)
- Keep block, diagonal, convolutional, pseudo-random; add the matching *interleave* functions so every pair round-trips in tests.
- Convolutional interleaver: support the Forney (B, M) form and CCSDS.
- Document each convention (row/column fill order, direction) in the docstring; wrong convention is the main cause of "decoder doesn't work".

**Concatenated chain** (`decode/chain.py`)
- A declarative pipeline: `[deinterleave → viterbi → deinterleave → RS → descramble → CRC]`, serializable to JSON so reports are reproducible, and runnable from CLI and UI.

**Automated parameter detection** (this closes the README's "not yet implemented" row)
- *Convolutional code*: for each (rate, K, polynomial set) in a catalog, decode and score by the re-encode mismatch rate and the survivor-path metric; accept only if the best candidate beats the second by a margin and beats a noise-only baseline. Include inverted-G2 variants and bit-order variants.
- *Block interleaver size*: search rows × columns candidates and score by the downstream syndrome/CRC pass rate, or by rank of the bit matrix when a linear block code is present.
- *RS parameters*: try common (n, k, fcr, prim) tuples and score by number of codewords that decode with ≤ t errors.
- *Pseudo-random interleavers* cannot be inferred from the bits alone without knowing the generator; support catalog PRNGs / seeds only and say so explicitly in the UI.
- Every detector must return `Unknown` with the reason when evidence is weak.

Done when: each decoder passes published or independently generated test vectors, the detectors hit the ≥ 95% synthetic target, and the chain decodes a concatenated (conv + RS + interleaver) synthetic stream from IQ to CRC-valid bytes.

### Phase 6: Performance, streaming, and the CLI (3–4 days)

1. `ingest.py`: add an `np.memmap`-based reader so multi-GB files are never fully loaded; expose `iter_windows(path, window, hop)` with overlap for filters.
2. Chunked processing with state carried across chunks (filter memory, loop state), so a 7 GB capture can be channelized end-to-end in constant RAM.
3. Hot loops: `numba.njit` for Viterbi ACS, LDPC check update, and the per-symbol FSK loops (`demodulate_from_estimate` has a Python loop over up to 1e6 symbols). If you avoid numba, vectorize and document the speed.
4. Streamlit: wrap pure computations in `st.cache_data` keyed by (file hash, window, params); run long jobs off the main thread with progress reporting, or move to a small worker (`concurrent.futures`) with a cancel button.
5. Headless CLI: `sih26147 analyze <file> --fs ... --json out.json` and `sih26147 decode <file> --chain chain.json`. A tool that only works through a browser is hard to test and automate.
6. Benchmarks recorded in `benchmarks/perf.md`: samples/s for channelizer, demod, Viterbi bit/s, LDPC frames/s.

Done when: a 1 GB file processes without exceeding a fixed memory ceiling and the UI never blocks for more than ~1 s.

### Phase 7: Real-data validation (3–5 days, depends on Phases 1 and 4)

Your current blocker is "no sample-accurate labels". The fix is to **let the CRC label the data** instead of hand-annotating it.

1. With the channelizer, extract each known PSLV signal from the 4 Msps capture at the frequencies the repo already cites (PicSat 435.525 MHz, CNUSail-1 437.100 MHz, VZLUSAT-I 437.240 MHz, INDUST-10 435.080 MHz). Note the capture is centered at 436.5 MHz, so offsets are −975 kHz, +600 kHz, +740 kHz and −1.42 MHz; check the last against the 4 MHz span before you rely on it.
2. Run the Phase 4 frame finder on each. Windows that produce **CRC-valid frames are verified ground truth** for the modulation and framing at those sample indices. Record start/length in the manifest; `annotation_reference` becomes "CRC-valid AX.25 frame at sample N". That is honest, sample-accurate, and independent of the blog post.
3. Fetch only the portions of the 7.7 GB file you need using HTTP range requests (the downloader already does this for the prefix), and keep SHA-256 of each slice.
4. For more independent recording groups per class, evaluate other real sources (the `satellite-recordings` repo is already noted; look for others and verify each license yourself before use). Do not count windows of the same file as separate groups.
5. Retrain with the existing group-holdout trainer once ≥ 3 independent groups per class exist. If they do not, keep shipping the DSP-based classifier from Phase 3 and say the learned model is not yet trained.

Done when: at least one real signal decodes to CRC-valid frames from the raw capture through the app/CLI, and the committed report shows frame count, BER proxy, and SHA-256 of the input slice.

### Phase 8: Application and UX restructuring (3–4 days)

1. Split `app.py` into `app/` modules: `sidebar.py`, `overview.py`, `channel_select.py`, `decode_chain.py`, `validation.py`, `report.py`. Keep state in one typed `SessionState` dataclass.
2. **Channel selection on the waterfall**: click-drag a band → runs the channelizer → feeds the rest of the pipeline. This replaces "analyze the whole band".
3. Show the oracle result (sync/CRC status) next to every bit output; red/green, with counts.
4. Make the report JSON versioned (`schema_version`), include code version/commit, library versions, input SHA-256, all parameters, and each stage's status. Add a "replay from report" command to reproduce a run.
5. Error boundaries: user-facing messages for expected failures (bad file, wrong rate), full traceback only in logs.
6. Basic accessibility/contrast pass on the custom CSS; keep `unsafe_allow_html` limited to static strings.

### Phase 9: Security and operations (2–3 days; good portfolio signal for your cybersecurity focus)

1. Write `docs/THREAT_MODEL.md` (assets, entry points: uploads, local paths, model files, report export; STRIDE table).
2. **Models**: `joblib.load` executes code from the file. Replace with `skops` (with its trusted-types allowlist) or ONNX, and verify a SHA-256 recorded at training time before loading.
3. **Local-path mode**: resolve and restrict reads to a configured root (`SIH_DATA_ROOT`), reject symlinks that escape it, and keep it disabled when `SIH_HOSTED=true` (already done; add a test).
4. Uploads: enforce size, extension *and* header sanity (WAV `RIFF`/`WAVE`, raw size alignment), cap decode time, and write nothing to disk unless asked.
5. Audit every `unsafe_allow_html=True` call to confirm no user-controlled string (file names, labels) reaches it; use `html.escape` where needed.
6. `pip-audit` + Dependabot/Renovate in CI; generate an SBOM (`cyclonedx-py`).
7. Logging: structured logs, no capture contents, no absolute user paths in hosted mode.
8. Container: non-root user, read-only root filesystem, no secrets in the image, healthcheck on `/_stcore/health`.

### Phase 10: Packaging, deployment, and docs (2–3 days)

1. `Dockerfile` (multi-stage, pinned base image) + `docker-compose.yml` for local runs; keep Render for the free demo; document the memory ceiling.
2. Version with SemVer, tag releases, keep a `CHANGELOG.md`.
3. Docs: `docs/ARCHITECTURE.md` (block diagram of the chain), `docs/ALGORITHMS.md` (what each block assumes and its limits), `docs/OPERATIONS.md` (config env vars, resource limits, troubleshooting), `docs/BENCHMARKS.md` (generated).
4. README: rewrite top-down as install → run → verify → capabilities (with the new status table) → limits. Remove "this session" language; write it for a stranger.
5. Keep the honest status table, but update every row to *measured* status with links to the benchmark outputs.

---

## 4. Build order, effort, and the cut line

| Order | Phase | Days (est.) | Cut line |
|---|---|---|---|
| 1 | 0 Foundation | 1 | **Must** |
| 2 | 2 Harness (start) | 1.5 | **Must** |
| 3 | 1 Signal chain | 6–8 | **Must** |
| 4 | 4 Framing/oracle | 4 | **Must** |
| 5 | 3 Blind estimation | 4–5 | **Must** |
| 6 | 5 FEC + detection | 7–9 | **Must** for Viterbi/RS/interleavers; LDPC catalog can follow |
| 7 | 7 Real-data validation | 3–5 | **Must** (one real CRC-valid decode) |
| 8 | 6 Performance/CLI | 3–4 | Should |
| 9 | 8 App restructuring | 3–4 | Should |
| 10 | 9 Security | 2–3 | Should (cheap and high value) |
| 11 | 10 Packaging/docs | 2–3 | Should |

Roughly 35–45 focused days in total; the "Must" rows alone are ~25–30. These are my estimates and depend on your pace; the order matters more than the numbers. If time is tight, the shortest path to something genuinely demonstrable is Phases 0 → 2 → 1 → 4 → 7 (a real satellite packet decoded from raw IQ with CRC proof), then fill in 3 and 5.

---

## 5. Working method (how to avoid rebuilding things twice)

1. Write the failing test first; paste its output into the PR/commit message.
2. One block per commit. Run `make verify` before pushing.
3. Never change a threshold to make a test pass. If you must change it, record why and the measured numbers.
4. Keep the "synthetic vs real" labeling rule already in your docs: simulated data is for algorithm checks only; any accuracy claim must come from real, group-held-out data.
5. When using an AI coding assistant, give it one block at a time with this template:

> Implement `<block>` in `sih26147/<file>.py` per `COMPLETION_PLAN.md` Phase N. First add a test in `tests/` that fails on the current code, using the simulator in `sih26147/sim/`. Then implement until it passes. Do not change existing thresholds, do not mark synthetic data as real, add type hints, run `ruff`, `mypy`, and `pytest`, and report the numbers. Stop and ask if a convention (bit order, polynomial form, mapping) is ambiguous.

---


The completion plan is already in `SIH26147_COMPLETION_GUIDE.md` (P0 to P10). This is the hosting plan to go with it. The main point is to submit a link that never sleeps, so the evaluator never sees a loading screen.

## What free hosting looks like right now

- **Render free:** spins down after 15 minutes idle and takes about a minute to wake. Your repo already has a `render.yaml`, so this is the host you're currently set up for, and it's the one that can lose you points. Its free bandwidth was also cut to 5 GB in April 2026.
- **Streamlit Community Cloud:** sleeps after 12 hours, so it's good for a mirror but not for the main link.
- **Hugging Face Spaces:** Docker and Gradio Spaces now need a paid plan, and the Streamlit SDK option is gone, so skip it.
- **Oracle Cloud Always Free:** a real VM that doesn't sleep. It needs a card, you run the server yourself, and the allowance was halved in June 2026, to 2 ARM cores and 12 GB, which is plenty for this app.
- **Others:** Koyeb's free tier has reportedly closed. One site says Northflank's free tier is always-on, but I haven't verified that, so don't build the plan on it.

Free tiers change quietly, so recheck the terms the day you deploy.

## The plan: always-on host plus an instant front page

**1. Primary host: Oracle Always Free VM (about half a day).** If you have a card that Oracle accepts (it's only for identity checks), this is the only option with no cold start.
- Create an Ubuntu 24.04 ARM VM (A1 Flex, 2 cores, 12 GB) in your nearest region. "Out of capacity" errors are common on ARM, so retry later or try another availability domain.
- Open ports 80 and 443 in the VCN security list **and** in the VM's own iptables, since Oracle's images block them by default.
- Install Docker and run your app with `--restart unless-stopped`.
- Put Caddy in front for automatic HTTPS (it handles Streamlit's websockets), with a free DuckDNS subdomain.
- Add a free UptimeRobot monitor on the URL so you get an email if it goes down.
- Check Oracle's current idle-reclaim policy. As far as I know, instances with very low usage can be reclaimed, but verify this.
- Build the image on ARM, and confirm `numpy`, `scipy`, `scikit-learn` and `reedsolo` all install (aarch64 wheels exist for the first three).

**2. Fallback if there's no card: Render with warm-keeping.**
- Add a free external pinger (UptimeRobot or cron-job.org) hitting the app every 10 minutes. Render's free tier covers 750 hours a month, enough for one always-on service.
- This is a workaround, and Render's terms or limits could change. Don't rely on it without testing it for a few days.
- Don't use Streamlit Cloud as the main link. Plain HTTP pings don't reliably keep it awake, so I'd use it only as a backup mirror.

**3. Make the submitted link an instant static page.** Host a one-file page on GitHub Pages (always instant). It should show:
- the project title and a pipeline diagram
- 3 or 4 screenshots
- a 90-second demo video (unlisted YouTube)
- a prominent **Launch live app** button
- a status dot

On page load, have it silently wake the backend so the app is already up by the time they click:
```html
<script>
const APP="https://YOUR-APP-URL";
fetch(APP+"/_stcore/health",{mode:"no-cors"}).then(()=>dot("ready")).catch(()=>dot("waking"));
</script>
```
Even on the worst case (the host asleep), the evaluator sees a working page with a video and a "starting, up to 60 s" message. A blank spinner is the thing to avoid. If you're forced to submit the app URL directly, use the Oracle host.

**4. Make the app start fast and be usable without files.** Evaluators won't have IQ captures.
- Bundle 2 or 3 small synthetic captures (from the P1 generator, under 2 MB each). Add a **Load demo capture** button that runs the full pipeline in one click.
- Test the real constraints locally: `docker run --cpus=0.1 --memory=512m` for Render. Measure time to first page and peak memory.
- Lazy-import heavy modules (`sklearn`, `plotly` charts) inside the functions that use them. Keep `pytest` out of runtime requirements.
- Cap hosted analysis at about 1 M samples, as you already do, and keep hosted mode on so the server never reads arbitrary paths.


## Order of work

1. Add `Dockerfile` and the demo-capture button (P9 and P6 in the guide).
2. Deploy to Render first, since it's the quickest, to get a working link.
3. Set up Oracle and Caddy and switch the main link when it's stable.
4. Add the GitHub Pages front page last, once the app URL is final.



## 6. Final release checklist

- [ ] CI green on a fresh clone (lint, types, tests, coverage ≥ 85% core, `pip-audit`).
- [ ] `make verify` regenerates BER curves, AMC confusion matrix, perf tables with no drift.
- [ ] BER within 1 dB of theory for every supported scheme.
- [ ] Symbol-rate error < 0.5%; AMC ≥ 90% at ≥ 10 dB on the synthetic suite.
- [ ] At least one real over-the-air signal decoded to CRC-valid frames; report committed.
- [ ] Viterbi, RS (incl. CCSDS), LDPC pass known test vectors; detectors meet the 95% target.
- [ ] A multi-GB file processed in constant memory, headless via CLI and interactively via UI.
- [ ] Threat model written; no unsafe deserialization; local-path sandboxed.
- [ ] Docker image builds, runs as non-root, passes the healthcheck.
- [ ] README, ARCHITECTURE, ALGORITHMS, OPERATIONS, CHANGELOG are current; every capability claim links to a measurement.
