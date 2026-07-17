#!/usr/bin/env python3
"""Coherence spectrogram with nearest-timestamp pairing, diagnostics, and per-file timers."""

# Wyłączamy blokowanie plików HDF5 na samym początku
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import re
from pathlib import Path
import time
import gc
import tempfile
import shutil

import numpy as np
import matplotlib.pyplot as plt
from sblib.SignalBurner import SignalBurner

# ---------- config ----------
CHA1_ROOT = Path("/dev/shm/signal-burn/hf25/cha1")
CHA2_ROOT = Path("/dev/shm/signal-burn/hf25/cha2")
CACHE_DIR = Path("/pool/signal_storage/cache")
DATASET_NAME = "rf_data"
FFT_SIZE = 262144
FS = 25_000_000  # Hz
OUT_PNG = "coherence_spectrogram.png"
MAX_RETRIES = 3
RETRY_DELAY = 0.2
MAX_TIME_DIFF = 0.0  # seconds – max allowed mismatch for pairing
# ----------------------------


def get_ram_usage_mb() -> float:
    """Return Resident Set Size (RSS) memory in MB for the current Linux process."""
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    return int(parts[1]) / 1024.0  # kB to MB
    except Exception:
        pass
    return 0.0


def parse_timestamp(filename: Path) -> float:
    """Extract UNIX timestamp from 'rf@123456789.123.h5'."""
    m = re.match(r"rf@(\d+)\.(\d+)", filename.stem)
    if not m:
        raise ValueError(f"Invalid filename: {filename.name}")
    return int(m.group(1)) + int(m.group(2)) / 1000.0


def latest_input_dir(base: Path) -> Path:
    """Return the subdirectory with the most recent timestamp in its name."""
    if not base.exists():
        return base
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}$")
    subdirs = [p for p in base.iterdir() if p.is_dir() and pattern.match(p.name)]
    if not subdirs:
        subdirs = [p for p in base.iterdir() if p.is_dir()]
        if not subdirs:
            return base
    subdirs.sort(key=lambda p: p.name)
    return subdirs[-1]


def collect_files_with_timestamps(folder: Path):
    """Return sorted list of (timestamp, Path) for all .h5 files."""
    entries = []
    for p in folder.glob("*.h5"):
        if not p.exists():
            continue
        try:
            ts = parse_timestamp(p)
            entries.append((ts, p))
        except ValueError:
            continue
    entries.sort(key=lambda x: x[0])
    return entries


def pair_nearest(list1, list2, max_diff=MAX_TIME_DIFF):
    """Greedy pairing of two sorted timestamp lists."""
    pairs = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        ts1, p1 = list1[i]
        ts2, p2 = list2[j]
        diff = abs(ts1 - ts2)
        if diff <= max_diff:
            pairs.append((p1, p2, ts1, ts2))
            i += 1
            j += 1
        elif ts1 < ts2:
            i += 1
        else:
            j += 1
    return pairs


def wait_for_file(
    path: Path, retries: int = MAX_RETRIES, delay: float = RETRY_DELAY
) -> bool:
    """Wait until file appears (or give up)."""
    for _ in range(retries):
        if path.exists():
            return True
        time.sleep(delay)
    return False


def main() -> None:
    cha1_dir = latest_input_dir(CHA1_ROOT)
    cha2_dir = latest_input_dir(CHA2_ROOT)
    print(f"cha1 dir : {cha1_dir}")
    print(f"cha2 dir : {cha2_dir}")

    # Collect files with timestamps
    list1 = collect_files_with_timestamps(cha1_dir)
    list2 = collect_files_with_timestamps(cha2_dir)
    print(f"Found {len(list1)} files in cha1, {len(list2)} in cha2")

    if not list1 or not list2:
        print("One of the directories is empty.")
        return

    # Create nearest-time pairs
    raw_pairs = pair_nearest(list1, list2, MAX_TIME_DIFF)
    print(f"Formed {len(raw_pairs)} pairs within {MAX_TIME_DIFF}s tolerance")

    if not raw_pairs:
        print("No matching pairs found.")
        return

    # Filter out pairs where files are temporarily missing
    valid_pairs = []
    for p1, p2, ts1, ts2 in raw_pairs:
        if not wait_for_file(p1) or not wait_for_file(p2):
            print(f"  skipping {p1.name} / {p2.name}: file(s) missing")
            continue
        valid_pairs.append((p1, p2, ts1, ts2))

    print(f"Processing {len(valid_pairs)} stable pairs …")

    burner = SignalBurner(
        fft_size=FFT_SIZE,
        dataset_name=DATASET_NAME,
        cache_path=CACHE_DIR,
        use_cache=True,
        show_logs=False,
    )
    burner.clean_cache(0)

    # Przygotowanie katalogu tymczasowego dla np.memmap
    temp_dir = tempfile.mkdtemp()
    mmap_path = Path(temp_dir) / "spec_matrix.dat"
    spec_matrix_mmapped = None

    t_start = time.perf_counter()
    timestamps = []
    kept = 0

    for i, (p1, p2, ts1, ts2) in enumerate(valid_pairs, 1):
        try:
            # Sprawdzamy stan RAM-u przed operacją
            ram_usage = get_ram_usage_mb()

            # Wypisujemy początek statusu bez znaku nowej linii
            print(
                f"  [{i}/{len(valid_pairs)}] [RAM: {ram_usage:.1f} MB] "
                f"Processing {p1.name} / {p2.name} ... ",
                end="",
                flush=True,
            )

            # Mierzymy czas procesowania konkretnego pliku
            t_pair_start = time.perf_counter()
            # Używamy process_coherence zamiast process_cross
            coherence = burner.process_coherence(p1, p2)
            t_pair_elapsed = time.perf_counter() - t_pair_start

            # Inicjalizacja pliku na dysku podczas pierwszej udanej iteracji
            if spec_matrix_mmapped is None:
                shape = (len(coherence), len(valid_pairs))
                spec_matrix_mmapped = np.memmap(
                    mmap_path, dtype=coherence.dtype, mode="w+", shape=shape
                )

            # Zapisz wynik bezpośrednio na dysk
            spec_matrix_mmapped[:, kept] = coherence
            timestamps.append((ts1 + ts2) / 2)  # mean timestamp
            kept += 1

            # Wyświetlamy status wraz ze zmierzonym czasem
            print(f"OK ({t_pair_elapsed:.3f} s)")

            # Ręczne czyszczenie pamięci po każdej iteracji
            del coherence
            gc.collect()

        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

    t_total = time.perf_counter() - t_start

    if kept == 0:
        print("\nNo valid pairs processed.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    print(f"\nProcessed {kept} pairs in {t_total:.2f} s: {t_total / kept:.2f} s/pair")

    # Odcinamy ewentualne puste kolumny i odwracamy chronologię (najstarsze z lewej)
    spec_matrix = spec_matrix_mmapped[:, :kept][:, ::-1]
    timestamps = timestamps[:kept][::-1]

    # Coherence jest w zakresie [0, 1], nie logarytmujemy
    spec_data = spec_matrix

    # Time axis: assume ~1 s per file, show seconds from first timestamp
    timestamps = np.array(timestamps)
    t0 = int(round(timestamps[0]))
    t_end = int(round(timestamps[-1] - t0)) + 1

    # Frequency axis (fftshifted: -FS/2 ... FS/2)
    freq_axis = np.fft.fftshift(np.fft.fftfreq(FFT_SIZE, d=1 / FS)) / 1e6  # MHz

    plt.figure(figsize=(16, 9))
    plt.imshow(
        spec_data,
        aspect="auto",
        origin="lower",
        extent=[0, t_end, freq_axis[0], freq_axis[-1]],
        cmap="jet",
        vmin=0.0,
        vmax=1.0,  # Coherence jest w zakresie [0, 1]
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [MHz]")
    plt.colorbar(label="Magnitude Squared Coherence")
    plt.title(
        f"Coherence Spectrogram – {cha1_dir.name} / {cha2_dir.name}  ({kept} pairs)"
    )
    plt.tight_layout()

    plt.savefig(OUT_PNG, dpi=300)
    print(f"Saved: {OUT_PNG}")

    # Usuwamy pliki tymczasowe mapowane w pamięci
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
