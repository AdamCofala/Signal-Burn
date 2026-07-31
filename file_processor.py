#!/usr/bin/env python3
"""
Real-time I/Q processor for channels /hf25/cha*/data

- Ignores files present at startup, processes only **new** ones.
- Every **second**: saves FFT amplitudes at selected frequencies to CSV.
- Every **minute** (when timestamp % 60 == 0): saves the **single packet** that
  arrived at that second - full FFT (all bins) and coherence for 3 channel pairs
  - into an HDF5 file, and logs a timing summary.
- On exit: shuts down SignalBurner and optionally kills leftover Python
  processes on the GPU to ensure a clean state for future runs.

Logging
-------
- Console: concise, INFO level by default (use --verbose for DEBUG).
- File: full DEBUG-level log, rotated daily, kept for --log-retention-days
  (default 14), stored under <output>/logs/live_processor.log.
- A periodic heartbeat line is emitted every --heartbeat-interval seconds
  (default 30s) so it's obvious the process is alive even when idle.
"""

import argparse
import csv
import logging
import logging.handlers
import re
import signal
import subprocess
import sys
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
POLL_INTERVAL = 0.4  # seconds between directory scans
MAX_TIME_DIFF = 0.0  # required timestamp accuracy for pairing
HEARTBEAT_INTERVAL = 30.0  # seconds between "still alive" status lines
LOG_RETENTION_DAYS = 14
# -----------------------------------------------------------

logger = logging.getLogger("sdr_live")


def setup_logging(output_dir: Path, verbose: bool, retention_days: int) -> None:
    """Configure console + rotating file logging.

    Console gets INFO (or DEBUG with --verbose); the log file always gets
    DEBUG so nothing is lost for later diagnostics, even if the console is
    kept quiet during normal operation.
    """
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    console_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S"
    )
    file_fmt = logging.Formatter(
        fmt="%(asctime)s %(levelname)-7s [%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "live_processor.log"

    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_path),
        when="midnight",
        backupCount=retention_days,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    logger.info(
        "Logging initialised -> console=%s, file=%s (retention=%dd)",
        "DEBUG" if verbose else "INFO",
        log_path,
        retention_days,
    )


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


def is_temp_file(fp: Path) -> bool:
    """True for in-progress writer artifacts like 'tmp.rf@....h5' that should
    never be picked up - they get renamed to their final name once the
    writer finishes, and will be discovered normally at that point."""
    return fp.name.startswith("tmp.")


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
        heartbeat_interval=HEARTBEAT_INTERVAL,
    ):
        self.cha_roots = [Path(p) for p in cha_roots]
        self.num_channels = len(self.cha_roots)
        self.fft_size = fft_size
        self.fs = fs
        self.output_dir = Path(output_dir)
        self.selected_freqs_hz = selected_freqs_hz
        self.poll_interval = poll_interval
        self.max_time_diff = max_time_diff
        self.heartbeat_interval = heartbeat_interval

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
        logger.info(
            "SignalBurner engine ready in %.3fs (fft_size=%d, fs=%.3f MHz)",
            time.perf_counter() - t0,
            fft_size,
            fs / 1e6,
        )

        # Incremental discovery - "seen" set per channel
        self.seen = [dict() for _ in range(self.num_channels)]
        ignored_counts = []
        for ch in range(self.num_channels):
            ignored_counts.append(self._ignore_existing_files(ch))
        logger.info(
            "Startup scan complete - ignored existing files per channel: %s (total=%d)",
            ", ".join(f"cha{ch + 1}={n}" for ch, n in enumerate(ignored_counts)),
            sum(ignored_counts),
        )

        # Pending seconds: ts_int -> {channel_idx: (ts, path)}
        self.pending = defaultdict(dict)

        # Runtime counters used for the heartbeat line
        self._start_time = time.time()
        self._last_heartbeat = self._start_time
        self._seconds_processed = 0
        self._minutes_saved = 0
        self._last_minute_saved_at = None
        self._discovered_since_heartbeat = 0
        self._errors_since_heartbeat = 0

    def _ignore_existing_files(self, channel_idx: int) -> int:
        """Add all existing .h5 files to 'seen' to exclude from processing."""
        root = self.cha_roots[channel_idx]
        cur_dir = latest_input_dir(root)
        cnt = 0
        for fp in cur_dir.glob("*.h5"):
            if is_temp_file(fp):
                continue
            stem = fp.stem
            try:
                ts = parse_timestamp(fp)
            except ValueError:
                logger.debug(
                    "Skipping unparseable filename during startup scan: %s", fp.name
                )
                continue
            self.seen[channel_idx][stem] = (ts, fp)
            cnt += 1
        return cnt

    def _add_new_files(self, channel_idx: int):
        """Detect new files since last check."""
        root = self.cha_roots[channel_idx]
        cur_dir = latest_input_dir(root)
        new_count = 0
        for fp in cur_dir.glob("*.h5"):
            if is_temp_file(fp):
                continue
            stem = fp.stem
            if stem in self.seen[channel_idx]:
                continue
            try:
                ts = parse_timestamp(fp)
            except ValueError:
                logger.warning("Ignoring file with unparseable timestamp: %s", fp.name)
                continue
            self.seen[channel_idx][stem] = (ts, fp)
            ts_int = int(ts)
            self.pending[ts_int][channel_idx] = (ts, fp)
            new_count += 1
        if new_count:
            logger.debug(
                "Discovered %d new file(s) on cha%d", new_count, channel_idx + 1
            )
            self._discovered_since_heartbeat += new_count

    def _process_second(self, ts_int: int) -> bool:
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

        # Timestamp matching (dla MAX_TIME_DIFF == 0 muszą być identyczne)
        if self.max_time_diff == 0.0 and not all(
            abs(t - timestamps[0]) < 1e-9 for t in timestamps[1:]
        ):
            logger.warning(
                "second %d: timestamps not identical across channels (%s) - skipping",
                ts_int,
                timestamps,
            )
            del self.pending[ts_int]
            self._errors_since_heartbeat += 1
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
            logger.debug("second %d processed in %.4fs", ts_int, total_time)

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
            self._seconds_processed += 1
            return True

        except FileNotFoundError as e:
            logger.error("second %d: file not found - %s. Skipping.", ts_int, e)
            if ts_int in self.pending:
                del self.pending[ts_int]
            self._errors_since_heartbeat += 1
            return False
        except Exception:
            logger.exception(
                "second %d: unexpected error while processing. Skipping.", ts_int
            )
            if ts_int in self.pending:
                del self.pending[ts_int]
            self._errors_since_heartbeat += 1
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
                logger.debug("Created new CSV log at %s", self.second_log_path)
            writer.writerow(row)

    def _save_minute_snapshot(
        self, ts_int, ffts, coh_pairs, pair_labels, fft_times, coh_times, total_time
    ):
        """Save a single-second snapshot to HDF5 and log a timing summary."""
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
        except Exception:
            logger.exception("Failed to save minute HDF5 snapshot to %s", h5_path)
            self._errors_since_heartbeat += 1
            return

        self._minutes_saved += 1
        self._last_minute_saved_at = minute_start

        fft_total = sum(fft_times)
        coh_total = sum(coh_times)
        summary_lines = [
            f"Minute {minute_start} snapshot saved -> {h5_path} ({dt:.3f}s)"
        ]
        for idx, d in enumerate(fft_times):
            summary_lines.append(f"    FFT cha{idx + 1}:        {d:.4f}s")
        summary_lines.append(f"    FFT total:        {fft_total:.4f}s")
        for idx, d in enumerate(coh_times):
            i, j = pair_labels[idx]
            summary_lines.append(f"    Coherence {i + 1}-{j + 1}:   {d:.4f}s")
        summary_lines.append(f"    Coherence total:  {coh_total:.4f}s")
        summary_lines.append(f"    Second total:     {total_time:.4f}s")
        logger.info("\n".join(summary_lines))

    def _maybe_heartbeat(self):
        """Emit a periodic 'still alive' status line, regardless of activity."""
        now = time.time()
        if now - self._last_heartbeat < self.heartbeat_interval:
            return
        uptime = now - self._start_time
        last_minute = (
            f"minute_{self._last_minute_saved_at}"
            if self._last_minute_saved_at is not None
            else "none yet"
        )
        logger.info(
            "status: alive | uptime=%.0fs | seconds_processed=%d | minutes_saved=%d "
            "| pending=%d | new_files=%d | errors=%d | last_snapshot=%s",
            uptime,
            self._seconds_processed,
            self._minutes_saved,
            len(self.pending),
            self._discovered_since_heartbeat,
            self._errors_since_heartbeat,
            last_minute,
        )
        self._last_heartbeat = now
        self._discovered_since_heartbeat = 0
        self._errors_since_heartbeat = 0

    def run(self):
        logger.info("Processor started. Press Ctrl+C to stop.")
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
                stale = [ts for ts in list(self.pending.keys()) if now - ts > 10]
                for ts_int in stale:
                    logger.warning(
                        "Dropping stale incomplete second %d (only %d/%d channels arrived)",
                        ts_int,
                        len(self.pending[ts_int]),
                        self.num_channels,
                    )
                    del self.pending[ts_int]

                # 4. Clean sb cache (optional, can be commented out if not needed)
                removed = self.sb.clean_cache(5)
                if removed:
                    logger.debug("Cache cleanup removed %d stale file(s)", removed)

                # 5. Heartbeat
                self._maybe_heartbeat()

                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0, self.poll_interval - elapsed))

        except KeyboardInterrupt:
            logger.info("Interrupted by user (Ctrl+C). Shutting down...")
        finally:
            self.sb.shutdown()
            logger.info("SignalBurner GPU resources released.")
            self._cleanup_gpu_processes()
            logger.info(
                "Final stats: uptime=%.0fs, seconds_processed=%d, minutes_saved=%d",
                time.time() - self._start_time,
                self._seconds_processed,
                self._minutes_saved,
            )

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
                logger.debug(
                    "nvidia-smi not available or returned an error, skipping GPU cleanup."
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
                        logger.warning(
                            "Killing leftover Python process on GPU: PID %d (%s)",
                            pid,
                            name,
                        )
                        os.kill(pid, signal.SIGKILL)
                    except (ValueError, ProcessLookupError) as e:
                        logger.error("Could not kill PID %s: %s", pid_str, e)
        except Exception:
            logger.exception("GPU process cleanup failed")


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
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=HEARTBEAT_INTERVAL,
        help="Seconds between 'still alive' status log lines (default: %(default)s)",
    )
    parser.add_argument(
        "--log-retention-days",
        type=int,
        default=LOG_RETENTION_DAYS,
        help="How many rotated daily log files to keep (default: %(default)s)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show DEBUG-level messages on the console as well as in the log file",
    )
    args = parser.parse_args()

    setup_logging(
        args.output, verbose=args.verbose, retention_days=args.log_retention_days
    )

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
        heartbeat_interval=args.heartbeat_interval,
    )
    processor.run()


if __name__ == "__main__":
    main()
