from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


URL = "https://zenodo.org/api/records/6403199/files/pslv-436500kHz-2018-01-13-095446.wav/content"
EXPECTED_SIZE = 7_736_807_424
EXPECTED_MD5 = "d0f5b667dae0a7b40c519799ca928eb5"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "real" / "pslv-436500kHz-2018-01-13-095446.wav"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a public 7.2 GiB real over-the-air UHF capture.")
    parser.add_argument(
        "--sample-bytes",
        type=int,
        help="Download a bounded prefix sample (minimum 4 MiB, maximum 256 MiB) instead of the full file.",
    )
    parser.add_argument(
        "--confirm-download-7-7gb",
        action="store_true",
        help="Required acknowledgement before starting the multi-gigabyte download.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_bytes is not None:
        return _download_prefix(args.sample_bytes, args.output)
    if not args.confirm_download_7_7gb:
        parser.error("The capture is about 7.2 GiB; inspect disk space and pass --confirm-download-7-7gb.")
    output = args.output or OUTPUT
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error(f"Refusing to overwrite existing file: {output}")
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if metadata_path.exists():
        parser.error(f"Refusing to overwrite existing provenance sidecar: {metadata_path}")
    free_bytes = shutil.disk_usage(output.parent).free
    if free_bytes < EXPECTED_SIZE + 512 * 1024 * 1024:
        parser.error(
            f"Need at least {EXPECTED_SIZE + 512 * 1024 * 1024:,} free bytes; "
            f"only {free_bytes:,} are available."
        )

    digest = hashlib.md5()
    downloaded = 0
    request = urllib.request.Request(URL, headers={"User-Agent": "SIH26147-research-prototype/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, output.open("xb") as destination:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) != EXPECTED_SIZE:
                raise RuntimeError(f"Unexpected dataset size: {content_length} bytes.")
            while chunk := response.read(4 * 1024 * 1024):
                destination.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                print(f"\rDownloaded {downloaded:,} / {EXPECTED_SIZE:,} bytes", end="", flush=True)
    except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        print(f"\nDownload failed after {downloaded:,} bytes: {exc}", file=sys.stderr)
        print("The partial file is retained for inspection; remove it manually before retrying.", file=sys.stderr)
        return 1
    print()
    if downloaded != EXPECTED_SIZE or digest.hexdigest() != EXPECTED_MD5:
        print(
            f"Dataset integrity check failed (size={downloaded}, md5={digest.hexdigest()}).",
            file=sys.stderr,
        )
        return 1
    provenance = {
        "title": "RF recording of PSLV 2018-004 launch cubesats",
        "creator": "Daniel Estévez",
        "doi": "10.5281/zenodo.6403199",
        "source_url": "https://zenodo.org/records/6403199",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "capture": {
            "sample_rate_hz": 4_000_000,
            "center_frequency_hz": 436_500_000,
            "format": "stereo I/Q WAV; check the header before use",
        },
        "file_size_bytes": downloaded,
        "md5": digest.hexdigest(),
        "note": "Use the author's write-up for reported signal labels; it is not a sample-accurate annotation.",
    }
    metadata_path.write_text(
        json.dumps(provenance, indent=2),
        encoding="utf-8",
    )
    print(f"Verified CC-BY-4.0 capture saved to {output}")
    return 0


def _download_prefix(sample_bytes: int, output: Path | None) -> int:
    if sample_bytes < 4 * 1024 * 1024 or sample_bytes > 256 * 1024 * 1024:
        raise SystemExit("--sample-bytes must be between 4 MiB and 256 MiB.")
    if sample_bytes >= EXPECTED_SIZE:
        raise SystemExit("Sample range must be smaller than the full capture.")
    suffix = f"{sample_bytes // (1024 * 1024)}MiB"
    output = output or OUTPUT.with_name(f"{OUTPUT.stem}-prefix-{suffix}.wav")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = output.with_suffix(output.suffix + ".metadata.json")
    if output.exists() or metadata_path.exists():
        raise SystemExit(f"Refusing to overwrite the sample or its metadata: {output}")
    request = urllib.request.Request(
        URL,
        headers={
            "Range": f"bytes=0-{sample_bytes - 1}",
            "User-Agent": "SIH26147-research-prototype/0.1",
        },
    )
    digest = hashlib.sha256()
    received = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            expected_range = f"bytes 0-{sample_bytes - 1}/{EXPECTED_SIZE}"
            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                raise RuntimeError("Dataset server did not return the exact requested byte range.")
            with output.open("xb") as destination:
                while received < sample_bytes:
                    chunk = response.read(min(4 * 1024 * 1024, sample_bytes - received))
                    if not chunk:
                        break
                    destination.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    print(f"\rDownloaded prefix {received:,} / {sample_bytes:,} bytes", end="", flush=True)
    except (OSError, RuntimeError, TimeoutError, urllib.error.URLError) as exc:
        print(f"\nPrefix download failed after {received:,} bytes: {exc}", file=sys.stderr)
        print("Any partial output is retained; remove it manually before retrying.", file=sys.stderr)
        return 1
    print()
    if received != sample_bytes:
        print(f"Incomplete prefix: expected {sample_bytes:,}, received {received:,} bytes.", file=sys.stderr)
        return 1
    metadata = {
        "title": "RF recording of PSLV 2018-004 launch cubesats (partial prefix)",
        "creator": "Daniel Estévez",
        "doi": "10.5281/zenodo.6403199",
        "source_url": "https://zenodo.org/records/6403199",
        "license": "CC BY 4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "capture": {"sample_rate_hz": 4_000_000, "center_frequency_hz": 436_500_000},
        "source_byte_range": f"bytes 0-{sample_bytes - 1} of {EXPECTED_SIZE}",
        "full_file_checksum_verified": False,
        "prefix_sha256": digest.hexdigest(),
        "note": "This is an unverified prefix sample, not the full recording or an annotated training corpus.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved measured prefix sample to {output}; this is not the full dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
