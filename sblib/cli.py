"""Command-line interface for single-file FFT processing."""

import argparse
from pathlib import Path

try:
    from sblib.SignalBurner import SignalBurner
except ImportError:  # pragma: no cover - allows direct script execution
    from SignalBurner import SignalBurner


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SignalBurner on a single H5 file")
    parser.add_argument("input", help="Path to the input .h5 file")
    parser.add_argument(
        "--dataset", default="rf_data", help="Dataset name inside the H5 file"
    )
    parser.add_argument("--output", default=None, help="Optional output path")
    args = parser.parse_args(argv)

    burner = SignalBurner(dataset_name=args.dataset)
    out_mag = burner.process_file(Path(args.input))
    print(f"Processed {args.input}: {out_mag.shape[0]} magnitude samples")


if __name__ == "__main__":
    main()
