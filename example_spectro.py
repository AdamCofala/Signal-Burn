import time
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")
import numpy as np
from pathlib import Path
from sblib.SignalBurner import SignalBurner

INPUT_ROOT = Path("/dev/shm/signal-burn/hf25/cha1")

FFT_SIZE = 262144
FS = 25_000_000
OUT_PNG = "spectrogram_final.png"


def latest_input_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    subdirs = [path for path in base_dir.iterdir() if path.is_dir()]
    if not subdirs:
        return base_dir
    return max(subdirs, key=lambda path: (path.stat().st_mtime, path.name))


def main():
    input_dir = latest_input_dir(INPUT_ROOT)
    print(f"Using input directory: {input_dir}")

    print(f"Init SignalBurner (FFT_SIZE={FFT_SIZE})...")
    sb = SignalBurner(
        cache_path=Path("/pool/signal_storage/cache"),
        dataset_name="rf_data",
        use_cache=True,
        fft_size=FFT_SIZE,
        show_logs=True,
    )

    print("Starting processing (GPU Accelerated)...")
    t_start = time.perf_counter()

    # Nowe API: przetwarzamy wszystkie pliki w folderze
    results = sb.process_fft_files(input_dir)

    print("Shutting down SignalBurner...")
    sb.shutdown()
    sb.clean_cache(10)

    t_total = time.perf_counter() - t_start

    if not results:
        print("No results to show.")
        return

    print(
        f"Processed {len(results)} files in {t_total:.2f}s. ({len(results) / t_total:.2f} file/s)"
    )
    print("Building spectogram matrix...")

    # results to lista krotek (Path, widmo)
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
        extent=[
            0,
            len(results),
            0,
            FS / 1e6,
        ],
        vmin=np.percentile(spectrogram_db, 5),
        vmax=np.percentile(spectrogram_db, 95),
    )

    plt.xlabel("Time [Index/Files]")
    plt.ylabel("Frequency [MHz]")
    plt.colorbar(label="Power [dB]")
    plt.title(f"{input_dir.name} | {len(results)} files | FFT Window {FFT_SIZE}")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
