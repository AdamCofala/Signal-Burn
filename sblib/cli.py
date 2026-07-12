from pathlib import Path
import argparse

from SignalBurner import SignalBurner


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run SignalBurner on a single H5 file")
    parser.add_argument("input", help="Path to the input .h5 file")
    parser.add_argument("--dataset", default="rf_data", help="Dataset name inside the H5 file")
    parser.add_argument("--output", default=None, help="Optional output .bin path")
    args = parser.parse_args(argv)

    burner = SignalBurner()
    output_path = Path(args.output) if args.output is not None else None
    out_mag = burner.process_file(args.input, output_path=output_path, dataset_name=args.dataset)
    print(f"Processed {args.input}: {out_mag.shape[0]} magnitude samples")


if __name__ == "__main__":
    main()
