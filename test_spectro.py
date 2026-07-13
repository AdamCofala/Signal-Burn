import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sblib.SignalBurner import SignalBurner


INPUT_DIR = Path("/pool/signal_storage/hf25/cha1/2026-07-13T08-00-00")
MAX_FREQ_BINS = 500_000
FS = 25_000_000
OUT_PNG = "spectrogram_final.png"


def main():
    print(f"Inicjalizacja SignalBurner (MAX_BINS={MAX_FREQ_BINS})...")

    sb = SignalBurner(input_path=INPUT_DIR, max_bins=MAX_FREQ_BINS)

    print("Rozpoczynam przetwarzanie (GPU Accelerated)...")
    t_start = time.perf_counter()

    results = sb.run()

    t_total = time.perf_counter() - t_start

    if not results:
        print("Brak wyników do wyświetlenia.")
        return

    print(
        f"Przetworzono {len(results)} plików w {t_total:.2f}s. ({len(results) / t_total:.2f} plików/s)"
    )
    print("Budowanie macierzy spektrogramu...")

    columns = [res[1] for res in results]
    spectrogram = np.stack(columns, axis=1)

    spectrogram_db = 10 * np.log10(spectrogram + 1e-6)

    print("Renderowanie wykresu...")
    plt.figure(figsize=(16, 9))

    freqs_mhz = np.linspace(0, FS, MAX_FREQ_BINS) / 1e6
    plt.pcolormesh(
        range(len(results)),
        freqs_mhz,
        spectrogram_db,
        shading="auto",
        cmap="jet",
        vmin=np.percentile(spectrogram_db, 5),
        vmax=np.percentile(spectrogram_db, 95),
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [MHz]")
    plt.colorbar(label="Power [dB]")
    plt.title(f"{INPUT_DIR.name} | {len(results)} plików ")

    plt.savefig(OUT_PNG, dpi=300)
    plt.show()
    print(f"Zapisano wykres: {OUT_PNG}")
    plt.close()

    sb.shutdown()
    print("Zasoby GPU zwolnione.")


if __name__ == "__main__":
    main()
