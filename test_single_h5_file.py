from pathlib import Path

import h5py

from sblib.core import SignalBurner, load_iq_samples


DEFAULT_H5 = Path(__file__).resolve().parent / "cha1" / "2026-07-10T12-00-00" / "rf@1783686190.000.h5"
DEFAULT_DATASET = "rf_data"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "bin" / f"{DEFAULT_H5.stem}_spectrum.bin"


def _preview_pairs(data, count=5):
    pairs = data.reshape(-1, 2)
    preview = pairs[:count]
    return [(int(pair[0]), int(pair[1])) for pair in preview]


def _resolve_dataset_name(h5_path, preferred="rf_data"):
    with h5py.File(h5_path, "r") as handle:
        if preferred in handle:
            return preferred
        keys = list(handle.keys())
        if len(keys) != 1:
            raise KeyError(f"Dataset {preferred!r} not found in {h5_path}; available keys: {keys}")
        return keys[0]


def main():
    h5_path = DEFAULT_H5
    dataset_name = _resolve_dataset_name(h5_path, preferred=DEFAULT_DATASET)
    burner = SignalBurner()
    iq_data = load_iq_samples(h5_path, dataset_name=dataset_name)

    print(f"Input file: {h5_path}")
    print(f"Dataset: {dataset_name}")
    print("First 5 I/Q samples before processing:")
    for index, (real, imag) in enumerate(_preview_pairs(iq_data, count=5), start=1):
        print(f"  {index}: I={real}, Q={imag}")

    output_path = DEFAULT_OUTPUT
    out_mag = burner.process_file(h5_path, output_path=output_path, dataset_name=dataset_name)

    print(f"Output file: {output_path}")
    print("First 5 magnitude samples after FFT:")
    for index, magnitude in enumerate(out_mag[:5], start=1):
        print(f"  {index}: {float(magnitude):.6f}")


if __name__ == "__main__":
    main()

