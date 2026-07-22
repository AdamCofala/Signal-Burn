"""Cross-spectrogram with nearest-timestamp pairing."""

import argparse
import os, re, time, gc, tempfile, shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sblib.SignalBurner import SignalBurner

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# ---------- defaults ----------
CHA1_ROOT = Path("/dev/shm/signal-burn/hf25/cha1")
CHA2_ROOT = Path("/dev/shm/signal-burn/hf25/cha2")
CACHE_DIR = Path("/pool/signal_storage/cache")
DATASET_NAME = "rf_data"
FFT_SIZE = 262144
FS = 25_000_000
OUT_PNG = "images/cross_spectrogram.png"
MAX_RETRIES = 3
RETRY_DELAY = 0.2
MAX_TIME_DIFF = 0.0
# ------------------------------


def get_ram_usage_mb() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return 0.0


def parse_timestamp(filename: Path) -> float:
    m = re.match(r"rf@(\d+)\.(\d+)", filename.stem)
    if not m:
        raise ValueError(f"Invalid filename: {filename.name}")
    return int(m.group(1)) + int(m.group(2)) / 1000.0


def latest_input_dir(base: Path) -> Path:
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
    entries = []
    for p in folder.glob("*.h5"):
        try:
            ts = parse_timestamp(p)
            entries.append((ts, p))
        except ValueError:
            continue
    entries.sort(key=lambda x: x[0])
    return entries


def pair_nearest(list1, list2, max_diff):
    pairs = []
    i = j = 0
    while i < len(list1) and j < len(list2):
        ts1, p1 = list1[i]
        ts2, p2 = list2[j]
        if abs(ts1 - ts2) <= max_diff:
            pairs.append((p1, p2, ts1, ts2))
            i += 1
            j += 1
        elif ts1 < ts2:
            i += 1
        else:
            j += 1
    return pairs


def wait_for_file(path, retries=MAX_RETRIES, delay=RETRY_DELAY):
    for _ in range(retries):
        if path.exists():
            return True
        time.sleep(delay)
    return False


def main():
    parser = argparse.ArgumentParser(description="Cross‑spectrogram")
    parser.add_argument("--cha1", type=Path, default=CHA1_ROOT)
    parser.add_argument("--cha2", type=Path, default=CHA2_ROOT)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--out", type=Path, default=OUT_PNG)
    parser.add_argument(
        "--max-diff",
        type=float,
        default=MAX_TIME_DIFF,
        help="Max allowed timestamp mismatch (s)",
    )
    args = parser.parse_args()

    cha1_dir = latest_input_dir(args.cha1)
    cha2_dir = latest_input_dir(args.cha2)
    print(f"cha1 dir: {cha1_dir}\ncha2 dir: {cha2_dir}")

    list1 = collect_files_with_timestamps(cha1_dir)
    list2 = collect_files_with_timestamps(cha2_dir)
    print(f"Found {len(list1)} files in cha1, {len(list2)} in cha2")
    if not list1 or not list2:
        print("One directory is empty.")
        return

    raw_pairs = pair_nearest(list1, list2, args.max_diff)
    print(f"Formed {len(raw_pairs)} pairs within {args.max_diff}s tolerance")
    if not raw_pairs:
        print("No matching pairs.")
        return

    valid_pairs = []
    for p1, p2, ts1, ts2 in raw_pairs:
        if not wait_for_file(p1) or not wait_for_file(p2):
            print(f"  skipping {p1.name}/{p2.name}: file missing")
            continue
        valid_pairs.append((p1, p2, ts1, ts2))
    print(f"Processing {len(valid_pairs)} stable pairs …")

    burner = SignalBurner(
        fft_size=args.fft_size,
        dataset_name=args.dataset,
        cache_path=args.cache,
        use_cache=True,
        show_logs=False,
    )
    burner.clean_cache(0)

    temp_dir = tempfile.mkdtemp()
    mmap_path = Path(temp_dir) / "spec_matrix.dat"
    spec_matrix_mmapped = None
    t_start = time.perf_counter()
    timestamps = []
    kept = 0

    for i, (p1, p2, ts1, ts2) in enumerate(valid_pairs, 1):
        try:
            ram_usage = get_ram_usage_mb()
            print(
                f"  [{i}/{len(valid_pairs)}] [RAM: {ram_usage:.1f} MB] "
                f"Processing {p1.name} / {p2.name} ... ",
                end="",
                flush=True,
            )
            t_pair = time.perf_counter()
            mag = burner.process_cross(p1, p2)
            t_pair = time.perf_counter() - t_pair
            if spec_matrix_mmapped is None:
                shape = (len(mag), len(valid_pairs))
                spec_matrix_mmapped = np.memmap(
                    mmap_path, dtype=mag.dtype, mode="w+", shape=shape
                )
            spec_matrix_mmapped[:, kept] = mag
            timestamps.append((ts1 + ts2) / 2)
            kept += 1
            print(f"OK ({t_pair:.3f} s)")
            del mag
            gc.collect()
        except Exception as exc:
            print(f"FAILED: {exc}")
            continue

    t_total = time.perf_counter() - t_start
    if kept == 0:
        print("No valid pairs processed.")
        shutil.rmtree(temp_dir, ignore_errors=True)
        return

    print(f"\nProcessed {kept} pairs in {t_total:.2f}s ({t_total / kept:.2f}s/pair)")
    spec_matrix = spec_matrix_mmapped[:, :kept][:, ::-1]
    timestamps = timestamps[:kept][::-1]
    spec_db = 10 * np.log10(spec_matrix + 1e-12)

    timestamps = np.array(timestamps)
    t0 = int(round(timestamps[0]))
    t_end = int(round(timestamps[-1] - t0)) + 1
    freq_axis = np.fft.fftshift(np.fft.fftfreq(args.fft_size, d=1 / args.fs)) / 1e6

    plt.figure(figsize=(16, 9))
    plt.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        extent=[0, t_end, freq_axis[0], freq_axis[-1]],
        cmap="jet",
        vmin=np.percentile(spec_db, 5),
        vmax=np.percentile(spec_db, 95),
    )
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [MHz]")
    plt.colorbar(label="Cross-spectrum magnitude [dB]")
    plt.title(f"Cross-spectrogram - {cha1_dir.name}/{cha2_dir.name} ({kept} pairs)")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved: {args.out}")
    shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
