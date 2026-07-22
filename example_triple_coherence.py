from sblib import SignalBurner
from pathlib import Path
import time
import re


CHA1_ROOT = Path("/dev/shm/signal-burn/hf25/cha1")
CHA2_ROOT = Path("/dev/shm/signal-burn/hf25/cha2")
CHA3_ROOT = Path("/dev/shm/signal-burn/hf25/cha3")
COH_OUT = Path("/dev/shm/signal-burn/hf25/coherence")
CACHE_DIR = Path("/pool/signal_storage/cache")
DATASET_NAME = "rf_data"
FFT_SIZE = 262144
FS = 25_000_000


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


def match_files(dir1, dir2, max_time_diff=0.0):
    files1 = collect_files_with_timestamps(dir1)
    files2 = collect_files_with_timestamps(dir2)

    matched_pairs = []
    for ts1, file1 in files1:
        closest_file2 = None
        closest_time_diff = float("inf")
        for ts2, file2 in files2:
            time_diff = abs(ts1 - ts2)
            if time_diff < closest_time_diff and time_diff <= max_time_diff:
                closest_time_diff = time_diff
                closest_file2 = file2
        if closest_file2 is not None:
            matched_pairs.append((file1, closest_file2))
    return matched_pairs


def process_coherence(sb: SignalBurner, file1: Path, file2: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"coherence_{file1.name}.h5"

    if output_file.exists():
        print(f"Skipping {output_file}, already exists.")
        return

    print(f"Processing coherence for {file1.name} and {file2.name}...")

    coherence_result = sb.process_coherence(file1, file2)

    sb.save_to_h5(
        h5_path=output_file,
        datasets={"coherence": coherence_result},
        metadata={
            "file1": str(file1),
            "file2": str(file2),
            "fs": FS,
            "fft_size": FFT_SIZE,
        },
    )


def main():

    sb = SignalBurner(
        cache_path=CACHE_DIR,
        dataset_name=DATASET_NAME,
        fft_size=FFT_SIZE,
        use_cache=False,
        show_logs=False,
    )

    sb.clean_cache(0)

    latest_cha1_dir = latest_input_dir(CHA1_ROOT)
    latest_cha2_dir = latest_input_dir(CHA2_ROOT)
    latest_cha3_dir = latest_input_dir(CHA3_ROOT)

    pairs12 = match_files(latest_cha1_dir, latest_cha2_dir)
    pairs13 = match_files(latest_cha1_dir, latest_cha3_dir)
    pairs23 = match_files(latest_cha2_dir, latest_cha3_dir)

    t_start = time.perf_counter()
    for i in range(len(min(pairs12, pairs13, pairs23))):
        t_pair = time.perf_counter()

        file1, file2 = pairs12[i]
        process_coherence(sb, file1, file2, COH_OUT / "cha1_cha2")

        file1, file3 = pairs13[i]
        process_coherence(sb, file1, file3, COH_OUT / "cha1_cha3")

        file2, file3 = pairs23[i]
        process_coherence(sb, file2, file3, COH_OUT / "cha2_cha3")

        t_pair = time.perf_counter() - t_pair
        print(
            f"Processed coherence for 3 pairs {i + 1}/{len(pairs12)} in {t_pair:.2f}s"
        )

    t_end = time.perf_counter() - t_start
    print(f"Total processing time: {t_end:.2f}s")
    print("All coherence processing completed.")
    print(f"Results saved in {COH_OUT}")
    print(
        f"STATS: 'num_pairs': {len(pairs12)} 'total_time': {t_end:.2f}s == {t_end / len(pairs12):.2f}s per 3 pairs"
    )


if __name__ == "__main__":
    main()
