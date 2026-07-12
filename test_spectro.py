"""
Testowy skrypt: generuje spektrogram (czas x czestotliwosc) na podstawie
folderu plikow .h5 przy uzyciu SignalBurner.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sblib.SignalBurner import (
    SignalBurner,
)  # dopasuj import do faktycznej lokalizacji klasy

INPUT_DIR = Path("/pool/signal_storage/hf25/cha1/2026-07-12T10-00-00/")
CACHE_PATH = Path("/pool/signal_storage/cache")
OUTPUT_PATH = INPUT_DIR / "processed"

DATASET_NAME = "rf_data"
OUT_BINS = 2048
FS = 25_000_000
SAVE_FILES = False
OUT_PNG = "spectrogram.png"
LIB_PATH = None  # None -> domyslna sciezka z SignalBurner (obok modulu)


def main():
    if not INPUT_DIR.exists():
        raise SystemExit(f"Folder nie istnieje: {INPUT_DIR}")

    sb = SignalBurner(
        input_path=INPUT_DIR,
        save_files=SAVE_FILES,
        output_path=OUTPUT_PATH,
        cache_path=CACHE_PATH,
        dataset_name=DATASET_NAME,
        out_bins=OUT_BINS,
        lib_path=LIB_PATH,
    )

    print(f"Przetwarzanie folderu: {INPUT_DIR}")
    print(f"out_bins={OUT_BINS}  fs={FS:.0f} Hz  save_files={SAVE_FILES}")

    columns = []
    file_names = []

    t_start = time.perf_counter()

    for i, (h5_path, mag) in enumerate(sb.process_files()):
        columns.append(mag)
        file_names.append(h5_path.name)
        elapsed = time.perf_counter() - t_start
        print(f"  [{i + 1}] {h5_path.name}  ({elapsed:.2f}s od startu)")

    t_total = time.perf_counter() - t_start

    if not columns:
        raise SystemExit("Brak plikow .h5 do przetworzenia w podanym folderze.")

    print(
        f"\nPrzetworzono {len(columns)} plikow w {t_total:.2f}s "
        f"({t_total / len(columns):.3f}s/plik)"
    )

    # (out_bins, num_files) - kazda kolumna to jeden plik/sekunda
    spectrogram = np.stack(columns, axis=1)

    # magnitude -> dB, zabezpieczone przed log(0)
    spectrogram_db = 20 * np.log10(spectrogram + 1e-6)

    freqs_mhz = np.linspace(-FS / 2, FS / 2, OUT_BINS) / 1e6
    time_axis_s = np.arange(len(columns))  # 1 plik = 1 sekunda

    plt.figure(figsize=(14, 6))
    plt.pcolormesh(
        time_axis_s, freqs_mhz, spectrogram_db, shading="auto", cmap="viridis"
    )
    plt.xlabel("Czas [s] (1 plik = 1s)")
    plt.ylabel("Czestotliwosc [MHz]")
    plt.colorbar(label="Magnitude [dB]")
    plt.title(f"Spektrogram: {INPUT_DIR.name} ({len(columns)} plikow)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    print(f"Zapisano spektrogram: {OUT_PNG}")


if __name__ == "__main__":
    main()
