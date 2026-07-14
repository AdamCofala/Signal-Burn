import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sblib.SignalBurner import SignalBurner

INPUT_DIR = Path("/pool/signal_storage/hf25/cha1/2026-07-14T09-00-00")

FFT_SIZE = 262144
FS = 25_000_000
OUT_PNG = "spectrogram_final.png"


def main():
    print(f"Init SignalBurner (FFT_SIZE={FFT_SIZE})...")
    sb = SignalBurner(input_path=INPUT_DIR, fft_size=FFT_SIZE)

    print("Starting processing (GPU Accelerated)...")
    t_start = time.perf_counter()

    results = sb.run()

    print("Shutting down SignalBurner...")
    sb.shutdown()

    t_total = time.perf_counter() - t_start

    if not results:
        print("No results to show.")
        return

    print(
        f"Processed {len(results)} files in {t_total:.2f}s. ({len(results) / t_total:.2f} file/s)"
    )
    print("Building spectogram matrix...")

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
    plt.title(f"{INPUT_DIR.name} | {len(results)} files | FFT Window {FFT_SIZE}")

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.show()

    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
