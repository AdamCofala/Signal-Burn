"""Single-channel spectrogram from all H5 files in a folder."""

import argparse
import time
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sblib.SignalBurner import SignalBurner

# ---------- defaults ----------
INPUT_ROOT = Path("/dev/shm/signal-burn/hf25/cha1")
CACHE_DIR = Path("/pool/signal_storage/cache")
DATASET_NAME = "rf_data"
FFT_SIZE = 262144
FS = 25_000_000
OUT_PNG = "spectrogram_final.png"
# ------------------------------


def latest_input_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return base_dir
    return max(subdirs, key=lambda p: (p.stat().st_mtime, p.name))


def main():
    parser = argparse.ArgumentParser(
        description="Single‑channel spectrogram from folder"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_ROOT,
        help="Root folder (latest subdir is used)",
    )
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--fs", type=float, default=FS, help="Sampling frequency Hz")
    parser.add_argument("--output", type=Path, default=OUT_PNG, help="Output PNG")
    args = parser.parse_args()

    input_dir = latest_input_dir(args.input)
    print(f"Using input directory: {input_dir}")

    print(f"Init SignalBurner (FFT_SIZE={args.fft_size})...")
    sb = SignalBurner(
        cache_path=args.cache,
        dataset_name=args.dataset,
        use_cache=True,
        fft_size=args.fft_size,
        show_logs=True,
    )

    print("Starting processing (GPU accelerated)...")
    t_start = time.perf_counter()
    results = sb.process_fft_files(input_dir)
    print("Shutting down SignalBurner...")
    sb.shutdown()
    sb.clean_cache(10)
    t_total = time.perf_counter() - t_start

    if not results:
        print("No results to show.")
        return

    print(
        f"Processed {len(results)} files in {t_total:.2f}s "
        f"({len(results) / t_total:.2f} file/s)"
    )
    columns = [res[1] for res in results]
    spectrogram = np.stack(columns, axis=1)
    spectrogram_db = 10 * np.log10(spectrogram + 1e-6)

    print("Rendering plot...")
    plt.figure(figsize=(16, 9))
    plt.imshow(
        spectrogram_db,
        aspect="auto",
        origin="lower",
        cmap="jet",
        extent=[0, len(results), 0, args.fs / 1e6],
        vmin=np.percentile(spectrogram_db, 5),
        vmax=np.percentile(spectrogram_db, 95),
    )
    plt.xlabel("Time [Index/Files]")
    plt.ylabel("Frequency [MHz]")
    plt.colorbar(label="Power [dB]")
    plt.title(f"{input_dir.name} | {len(results)} files | FFT Window {args.fft_size}")
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
