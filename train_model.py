from __future__ import annotations

import argparse

from sih26147.training import result_json, train_real_capture_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an auditable modulation model from labeled real IQ captures."
    )
    parser.add_argument("--manifest", required=True, help="CSV with capture windows and source provenance.")
    parser.add_argument("--model", default="models/modulation.joblib", help="Output model path.")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=26147)
    args = parser.parse_args()
    result = train_real_capture_model(
        args.manifest,
        model_path=args.model,
        test_size=args.test_size,
        random_state=args.seed,
    )
    print(result_json(result))


if __name__ == "__main__":
    main()
