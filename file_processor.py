#!/usr/bin/env python3
"""
Real-time I/Q processor for channels /hf25/cha*/data

- Ignores files present at startup, processes only **new** ones.
- Every **second**: saves FFT amplitudes at selected frequencies to CSV.
- Every **minute** (when timestamp % 60 == 0): saves the **single packet** that
  arrived at that second - full FFT (all bins) and coherence for 3 channel pairs
  - into an HDF5 file, and prints a timing summary.
- On exit: shuts down SignalBurner and optionally kills leftover Python
  processes on the GPU to ensure a clean state for future runs.
"""

import argparse
import csv
import re
import signal
import subprocess
import time
from collections import defaultdict
from pathlib import Path
import os

import numpy as np

from sblib.SignalBurner import SignalBurner

# ---------- defaults (can be overridden via CLI) ----------
CHA_ROOTS = [
    Path("/dev/shm/hf25/cha1/data"),
    Path("/dev/shm/hf25/cha2/data"),
    Path("/dev/shm/hf25/cha3/data"),
]
FFT_SIZE = 262144
FS = 25_000_000  # Hz
CACHE_DIR = Path("/pool/signal_storage/cache")
DATASET_NAME = "rf_data"
OUTPUT_DIR = Path("/pool/signal_storage/output")
SELECTED_FREQUENCIES_HZ = [1e6, 5e6, 10e6]  # example: 1, 5, 10 MHz
POLL_INTERVAL = 0.5  # seconds between directory scans
MAX_TIME_DIFF = 0.0  # required timestamp accuracy for pairing
# -----------------------------------------------------------


def latest_input_dir(base: Path) -> Path:
    """Return the most recent timestamp-named subdirectory."""
    if not base.exists():
        return base
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return base
    return max(subdirs, key=lambda p: (p.stat().st_mtime, p.name))


def parse_timestamp(filename: Path) -> float:
    """Extract UNIX timestamp from a name like 'rf@123456789.123.h5'."""
    m = re.match(r"rf@(\d+)\.(\d+)", filename.stem)
    if not m:
        raise ValueError(f"Invalid filename: {filename.name}")
    return int(m.group(1)) + int(m.group(2)) / 1000.0


class LiveProcessor:
    def __init__(
        self,
        cha_roots,
        fft_size,
        fs,
        cache_dir,
        dataset_name,
        output_dir,
        selected_freqs_hz,
        poll_interval,
        max_time_diff,
    ):
        self.cha_roots = [Path(p) for p in cha_roots]
        self.num_channels = len(self.cha_roots)
        self.fft_size = fft_size
        self.fs = fs
        self.output_dir = Path(output_dir)
        self.selected_freqs_hz = selected_freqs_hz
        self.poll_interval = poll_interval
        self.max_time_diff = max_time_diff

        # Create output directories
        self.second_log_path = self.output_dir / "second_fft.csv"
        self.minute_h5_dir = self.output_dir / "minute_h5"
        self.minute_h5_dir.mkdir(parents=True, exist_ok=True)

        # Frequency axis for bin selection (after fftshift as in CUDA code)
        freq_axis = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / fs))
        self.selected_bins = []
        for f_hz in selected_freqs_hz:
            idx = np.argmin(np.abs(freq_axis - f_hz))
            self.selected_bins.append(idx)

        # SignalBurner engine - cache włączony, by przy ponownym przetwarzaniu
        # tego samego zestawu plików (np. restart) odczytywać gotowe wyniki z .npz
        t0 = time.perf_counter()
        self.sb = SignalBurner(
            fft_size=fft_size,
            dataset_name=dataset_name,
            cache_path=cache_dir,
            use_cache=True,
            show_logs=False,
        )
        print(f"[init] SignalBurner ready ({time.perf_counter() - t0:.4f}s)")

        # Incremental discovery - "seen" set per channel
        self.seen = [dict() for _ in range(self.num_channels)]
        for ch in range(self.num_channels):
            self._ignore_existing_files(ch)

        # Pending seconds: ts_int -> {channel_idx: (ts, path)}
        self.pending = defaultdict(dict)

    def _ignore_existing_files(self, channel_idx: int):
        """Add all existing .h5 files to 'seen' to exclude from processing."""
        root = self.cha_roots[channel_idx]
        cur_dir = latest_input_dir(root)
        cnt = 0
        for fp in cur_dir.glob("*.h5"):
            stem = fp.stem
            try:
                ts = parse_timestamp(fp)
            except ValueError:
                continue
            self.seen[channel_idx][stem] = (ts, fp)
            cnt += 1
        print(f"[init] Channel {channel_idx + 1}: ignored {cnt} existing file(s)")

    def _add_new_files(self, channel_idx: int):
        """Detect new files since last check."""
        root = self.cha_roots[channel_idx]
        cur_dir = latest_input_dir(root)
        new_count = 0
        for fp in cur_dir.glob("*.h5"):
            stem = fp.stem
            if stem in self.seen[channel_idx]:
                continue
            try:
                ts = parse_timestamp(fp)
            except ValueError:
                continue
            self.seen[channel_idx][stem] = (ts, fp)
            ts_int = int(ts)
            self.pending[ts_int][channel_idx] = (ts, fp)
            new_count += 1
        if new_count:
            print(f"[discover] cha{channel_idx + 1}: {new_count} new file(s)")

    def _process_second(self, ts_int: int):
        """Process one second using the new combined pair processing."""
        if len(self.pending[ts_int]) < self.num_channels:
            return False

        # Collect files for all channels (order zgodny z oryginałem)
        files = []
        timestamps = []
        for ch in range(self.num_channels):
            ts, fp = self.pending[ts_int][ch]
            files.append(fp)
            timestamps.append(ts)

        files.reverse()
        timestamps.reverse()

        # Timestamp matching (dla MAX_TIME_DIFF == 0 muszą być identyczne)
        if self.max_time_diff == 0.0 and not all(
            abs(t - timestamps[0]) < 1e-9 for t in timestamps[1:]
        ):
            print(f"[warn] second {ts_int}: timestamps not identical, skipping")
            del self.pending[ts_int]
            return False

        total_start = time.perf_counter()

        try:
            # Miejsca na wyniki
            ffts = {idx: None for idx in range(self.num_channels)}
            fft_times = {idx: 0.0 for idx in range(self.num_channels)}
            coh_pairs = []
            pair_labels = []
            coh_times = []

            need_coh = ts_int % 60 == 0  # tylko na granicy minuty zapisujemy koherencję

            # Pary: (0,1), (0,2), (1,2)
            pair_indices = [(0, 1), (0, 2), (1, 2)]

            for i, j in pair_indices:
                t0 = time.perf_counter()
                res = self.sb.process_pair_all(files[i], files[j])
                dt = time.perf_counter() - t0

                # Zapamiętaj widma mocy (jeśli jeszcze nie mamy)
                if ffts[i] is None:
                    ffts[i] = res["power1"]
                    fft_times[i] = dt  # przybliżenie, ale wystarczające do podsumowania
                if ffts[j] is None:
                    ffts[j] = res["power2"]
                    fft_times[j] = dt

                # Dla podsumowania minutowego zbieramy koherencję
                if need_coh:
                    coh_pairs.append(res["coherence"])
                    coh_times.append(dt)
                    pair_labels.append((i, j))

            # Mamy już wszystkie trzy widma mocy -> piszemy CSV
            self._write_second_log(ts_int, [ffts[i] for i in range(self.num_channels)])

            total_time = time.perf_counter() - total_start

            # Snapshot minutowy
            if need_coh:
                self._save_minute_snapshot(
                    ts_int,
                    [ffts[i] for i in range(self.num_channels)],
                    coh_pairs,
                    pair_labels,
                    [fft_times[i] for i in range(self.num_channels)],
                    coh_times,
                    total_time,
                )

            del self.pending[ts_int]
            return True

        except FileNotFoundError as e:
            print(f"[error] second {ts_int}: file not found - {e}. Skipping.")
            if ts_int in self.pending:
                del self.pending[ts_int]
            return False
        except Exception as e:
            print(f"[error] second {ts_int}: unexpected - {e}. Skipping.")
            if ts_int in self.pending:
                del self.pending[ts_int]
            return False

    def _write_second_log(self, ts_int, ffts):
        """Append selected frequency magnitudes to CSV."""
        row = [ts_int]
        for ch_fft in ffts:
            for bin_idx in self.selected_bins:
                row.append(ch_fft[bin_idx])

        write_header = not self.second_log_path.exists()
        with open(self.second_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                header = ["second"]
                for ch in range(1, self.num_channels + 1):
                    for f_hz in self.selected_freqs_hz:
                        header.append(f"cha{ch}_{f_hz / 1e6:.2f}MHz")
                writer.writerow(header)
            writer.writerow(row)

    def _save_minute_snapshot(
        self, ts_int, ffts, coh_pairs, pair_labels, fft_times, coh_times, total_time
    ):
        """Save a single-second snapshot to HDF5 and print a summary."""
        minute_start = (ts_int // 60) * 60
        h5_path = self.minute_h5_dir / f"minute_{minute_start}.h5"

        datasets = {}
        for ch in range(self.num_channels):
            datasets[f"cha{ch + 1}/fft"] = ffts[ch].reshape(1, -1)

        for idx, (i, j) in enumerate(pair_labels):
            datasets[f"pairs/{i + 1}{j + 1}/coherence"] = coh_pairs[idx].reshape(1, -1)

        datasets["timestamps"] = np.array([ts_int], dtype=np.float64)

        t0 = time.perf_counter()
        try:
            self.sb.save_to_h5(
                h5_path=h5_path,
                datasets=datasets,
                metadata={
                    "fs": self.fs,
                    "fft_size": self.fft_size,
                    "minute_start": minute_start,
                    "timestamp": ts_int,
                },
            )
            dt = time.perf_counter() - t0
            print(f"[minute] HDF5 saved to {h5_path} in {dt:.4f}s")
        except Exception as e:
            print(f"[minute] FAILED to save HDF5: {e}")
            return

        # Print summary
        fft_total = sum(fft_times)
        coh_total = sum(coh_times)
        print(f"[minute] ====== Minute {minute_start} summary ======")
        for idx, d in enumerate(fft_times):
            print(f"         FFT cha{idx + 1}: {d:.4f}s")
        print(f"         FFT total     : {fft_total:.4f}s")
        for idx, d in enumerate(coh_times):
            i, j = pair_labels[idx]
            print(f"         Coherence {i + 1}-{j + 1}: {d:.4f}s")
        print(f"         Coherence total: {coh_total:.4f}s")
        print(f"         Second total   : {total_time:.4f}s")
        print(f"[minute] ===================================")

    def run(self):
        print("[main] Processor started. Press Ctrl+C to stop.")
        try:
            while True:
                loop_start = time.perf_counter()

                # 1. Discover new files
                for ch in range(self.num_channels):
                    self._add_new_files(ch)

                # 2. Process complete seconds
                ready_seconds = sorted(
                    [
                        ts
                        for ts in self.pending
                        if len(self.pending[ts]) == self.num_channels
                    ]
                )
                for ts_int in ready_seconds:
                    self._process_second(ts_int)

                # 3. Clean up stale incomplete seconds (older than 10 s)
                now = time.time()
                for ts_int in list(self.pending.keys()):
                    if now - ts_int > 10:
                        del self.pending[ts_int]

                # 4. Clean sb cache (optional, can be commented out if not needed)
                self.sb.clean_cache(5)

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0, self.poll_interval - elapsed))

        except KeyboardInterrupt:
            print("[main] Interrupted. Shutting down...")
        finally:
            self.sb.shutdown()
            print("[main] SignalBurner GPU resources released.")
            # Cleanup leftover GPU processes (just in case)
            self._cleanup_gpu_processes()

    def _cleanup_gpu_processes(self):
        """Kill any remaining Python processes that are still using the GPU.
        This helps to ensure a clean state after crashes or forced exits."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,process_name",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode != 0:
                print(
                    "[cleanup] nvidia-smi not available or returned error, skipping GPU cleanup."
                )
                return

            lines = result.stdout.strip().splitlines()
            for line in lines:
                if not line.strip():
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    continue
                pid_str, name = parts[0].strip(), parts[1].strip()
                if "python" in name.lower():
                    try:
                        pid = int(pid_str)
                        print(
                            f"[cleanup] Killing leftover Python process on GPU: PID {pid} ({name})"
                        )
                        os.kill(pid, signal.SIGKILL)
                    except (ValueError, ProcessLookupError) as e:
                        print(f"[cleanup] Could not kill PID {pid_str}: {e}")
        except Exception as e:
            print(f"[cleanup] GPU process cleanup failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Real-time I/Q processor")
    parser.add_argument("--cha-roots", nargs="+", type=Path, default=CHA_ROOTS)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--freqs", nargs="+", type=float, default=SELECTED_FREQUENCIES_HZ
    )
    parser.add_argument("--poll", type=float, default=POLL_INTERVAL)
    parser.add_argument("--max-diff", type=float, default=MAX_TIME_DIFF)
    args = parser.parse_args()

    processor = LiveProcessor(
        cha_roots=args.cha_roots,
        fft_size=args.fft_size,
        fs=args.fs,
        cache_dir=args.cache,
        dataset_name=args.dataset,
        output_dir=args.output,
        selected_freqs_hz=args.freqs,
        poll_interval=args.poll,
        max_time_diff=args.max_diff,
    )
    processor.run()


if __name__ == "__main__":
    main()
