#!/usr/bin/env python3
"""
Real-time I/Q processor - unified high-rate phase sampling with online
oscillator-drift (linear trend) subtraction.

phase.csv now contains phase already corrected for the local-oscillator /
reference-carrier frequency offset (the linear drift you'd otherwise see
in a raw unwrapped-phase plot). The correction is estimated online from a
rolling window of recent samples and updated once per second; what's left
in phase.csv is the residual (e.g. the slower ionospheric/diurnal wave),
not the raw drifting phase.
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
from collections import defaultdict, deque
from pathlib import Path
import os


import numpy as np

from sblib.SignalBurner import SignalBurner

# ---------- defaults ----------
CHA_ROOTS = [
    Path("/dev/shm/hf25/cha1/data"),
    Path("/dev/shm/hf25/cha2/data"),
    Path("/dev/shm/hf25/cha3/data"),
]
FFT_SIZE = 262144
FS = 25_000_000
CACHE_DIR = Path("/pool/signal_storage/cache")
DATASET_NAME = "rf_data"
OUTPUT_DIR = Path("/pool/signal_storage/output")
SELECTED_FREQUENCIES_HZ = [1e6, 5e6, 10e6]
POLL_INTERVAL = 0.4
MAX_TIME_DIFF = 0.0
HEARTBEAT_INTERVAL = 30.0
LOG_RETENTION_DAYS = 14
TARGET_FREQ_HZ = 12_500_000 - 225_000  # = 12_275_000 Hz
PHASE_OFFSET_DEG = 36.0  # e-czas phase shift
PHASE_SAMPLING_HZ = 500.0  # unified phase.csv sampling rate
DETREND_WINDOW_SEC = 300.0  # rolling window used to estimate oscillator drift
DETREND_REFIT_INTERVAL_SEC = 1.0  # how often the linear fit is recomputed
# ----------------------------------


logger = logging.getLogger("sdr_live")


def setup_logging(output_dir, verbose, retention_days):
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
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=str(log_dir / "live_processor.log"),
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
        log_dir / "live_processor.log",
        retention_days,
    )


def latest_input_dir(base: Path) -> Path:
    if not base.exists():
        return base
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return base
    return max(subdirs, key=lambda p: (p.stat().st_mtime, p.name))


def parse_timestamp(filename: Path) -> float:
    m = re.match(r"rf@(\d+)\.(\d+)", filename.stem)
    if not m:
        raise ValueError(f"Invalid filename: {filename.name}")
    return int(m.group(1)) + int(m.group(2)) / 1000.0


def is_temp_file(fp: Path) -> bool:
    return fp.name.startswith("tmp.")


class PhaseDetrender:
    """Per-channel online phase unwrap + rolling-window linear-drift removal.

    Tracks a continuously unwrapped phase (so wraparounds at +-180 deg don't
    show up as jumps), keeps a rolling window of (time, unwrapped_deg)
    samples, and periodically refits a linear trend (deg/s) to that window.
    The trend is evaluated (extrapolated) for every incoming sample between
    refits, and the residual (unwrapped - trend) is what gets written out -
    this is the oscillator/frequency-offset component removed, leaving the
    slower structure (e.g. diurnal ionospheric wave) visible on its own.
    """

    def __init__(self, sampling_hz, window_sec, refit_interval_sec):
        buffer_len = max(2, int(sampling_hz * window_sec))
        self._buffer_t = deque(maxlen=buffer_len)
        self._buffer_phase = deque(maxlen=buffer_len)
        self._last_unwrapped_deg = None
        self._refit_interval_sec = refit_interval_sec
        self._last_refit_t = None
        self._slope_deg_per_sec = 0.0
        self._intercept_deg = 0.0
        self._fit_t0 = None

    def update(self, t_sec, wrapped_deg):
        # --- incremental unwrap relative to the last sample ---
        if self._last_unwrapped_deg is None:
            unwrapped = wrapped_deg
        else:
            last_wrapped = ((self._last_unwrapped_deg + 180) % 360) - 180
            delta = wrapped_deg - last_wrapped
            if delta > 180:
                delta -= 360
            elif delta < -180:
                delta += 360
            unwrapped = self._last_unwrapped_deg + delta
        self._last_unwrapped_deg = unwrapped

        self._buffer_t.append(t_sec)
        self._buffer_phase.append(unwrapped)

        # --- periodically refit the linear (oscillator-drift) trend ---
        if self._last_refit_t is None or (t_sec - self._last_refit_t) >= self._refit_interval_sec:
            if len(self._buffer_t) >= 2:
                t_arr = np.asarray(self._buffer_t, dtype=np.float64)
                p_arr = np.asarray(self._buffer_phase, dtype=np.float64)
                t0 = t_arr[0]
                slope, intercept = np.polyfit(t_arr - t0, p_arr, 1)
                self._slope_deg_per_sec = slope
                self._intercept_deg = intercept
                self._fit_t0 = t0
            self._last_refit_t = t_sec

        if self._fit_t0 is None:
            trend = 0.0
        else:
            trend = self._slope_deg_per_sec * (t_sec - self._fit_t0) + self._intercept_deg

        residual = unwrapped - trend
        return residual, self._slope_deg_per_sec


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
        target_freq_hz,
        phase_offset_deg=0.0,
        phase_sampling_hz=0.0,
        detrend_window_sec=DETREND_WINDOW_SEC,
        detrend_refit_interval_sec=DETREND_REFIT_INTERVAL_SEC,
        heartbeat_interval=30,
    ):
        self.cha_roots = [Path(p) for p in cha_roots]
        self.num_channels = len(self.cha_roots)
        self.fft_size = fft_size
        self.fs = fs
        self.output_dir = Path(output_dir)
        self.selected_freqs_hz = selected_freqs_hz
        self.poll_interval = poll_interval
        self.max_time_diff = max_time_diff
        self.target_freq_hz = target_freq_hz
        self.phase_offset_rad = np.radians(phase_offset_deg)
        self.phase_sampling_hz = phase_sampling_hz
        self.heartbeat_interval = heartbeat_interval

        self.second_log_path = self.output_dir / "second_fft.csv"
        self.phase_log_path = self.output_dir / "phase.csv"
        self.minute_h5_dir = self.output_dir / "minute_h5"
        self.minute_h5_dir.mkdir(parents=True, exist_ok=True)

        freq_axis = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / fs))
        self.selected_bins = [
            np.argmin(np.abs(freq_axis - f)) for f in selected_freqs_hz
        ]

        t0 = time.perf_counter()
        self.sb = SignalBurner(
            fft_size=fft_size,
            dataset_name=dataset_name,
            cache_path=cache_dir,
            use_cache=True,
            show_logs=True,
        )
        logger.info(
            "SignalBurner ready in %.3fs (fft_size=%d, fs=%.3f MHz)",
            time.perf_counter() - t0,
            fft_size,
            fs / 1e6,
        )

        self.seen = [dict() for _ in range(self.num_channels)]
        ignored_counts = []
        for ch in range(self.num_channels):
            ignored_counts.append(self._ignore_existing_files(ch))
        logger.info(
            "Startup scan complete - ignored existing files per channel: %s (total=%d)",
            ", ".join(f"cha{ch + 1}={n}" for ch, n in enumerate(ignored_counts)),
            sum(ignored_counts),
        )

        self.pending = defaultdict(dict)
        self._start_time = time.time()
        self._last_heartbeat = self._start_time
        self._seconds_processed = 0
        self._minutes_saved = 0
        self._last_minute_saved_at = None
        self._discovered_since_heartbeat = 0
        self._errors_since_heartbeat = 0

        self._fft_times_rolling = deque(maxlen=59)
        self._phase_times_rolling = deque(maxlen=59)

        # For high-rate phase sampling + online oscillator-drift removal
        if self.phase_sampling_hz > 0:
            self._phase_period = 1.0 / self.phase_sampling_hz
            self._last_phase_sample_time = 0.0
            self._detrenders = [
                PhaseDetrender(
                    sampling_hz=self.phase_sampling_hz,
                    window_sec=detrend_window_sec,
                    refit_interval_sec=detrend_refit_interval_sec,
                )
                for _ in range(self.num_channels)
            ]

    def _ignore_existing_files(self, channel_idx: int) -> int:
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
                continue
            self.seen[channel_idx][stem] = (ts, fp)
            cnt += 1
        return cnt

    def _add_new_files(self, channel_idx: int):
        root = self.cha_roots[channel_idx]
        cur_dir = latest_input_dir(root)
        new_count = 0
        for fp in cur_dir.glob("*.h5"):
            if is_temp_file(fp) or fp.stem in self.seen[channel_idx]:
                continue
            stem = fp.stem
            try:
                ts = parse_timestamp(fp)
            except ValueError:
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
        if len(self.pending[ts_int]) < self.num_channels:
            return False

        files, timestamps = [], []
        for ch in range(self.num_channels):
            ts, fp = self.pending[ts_int][ch]
            files.append(fp)
            timestamps.append(ts)

        if self.max_time_diff == 0.0 and not all(
            abs(t - timestamps[0]) < 1e-9 for t in timestamps[1:]
        ):
            logger.warning(
                "second %d: timestamps not identical across channels - skipping", ts_int
            )
            del self.pending[ts_int]
            self._errors_since_heartbeat += 1
            return False

        total_start = time.perf_counter()
        need_coh = ts_int % 60 == 0

        try:
            # 1. Power spectra
            if not need_coh:
                ffts = {}
                fft_time_sum = 0.0
                for ch in range(self.num_channels):
                    t0 = time.perf_counter()
                    ffts[ch] = self.sb.process_file(files[ch])
                    fft_time_sum += time.perf_counter() - t0
                self._fft_times_rolling.append(fft_time_sum)
            else:
                t0 = time.perf_counter()
                res = self.sb.process_triple_all(files[0], files[1], files[2])
                triple_time = time.perf_counter() - t0
                ffts = {0: res["power1"], 1: res["power2"], 2: res["power3"]}
                coh_data = {
                    (0, 1): res["coherence12"],
                    (0, 2): res["coherence13"],
                    (1, 2): res["coherence23"],
                }

            self._write_second_log(ts_int, [ffts[i] for i in range(self.num_channels)])

            # 2. High-rate phase sampling (unified phase.csv, oscillator drift
            # already subtracted online before writing)
            phase_time = 0.0
            if self.phase_sampling_hz > 0:
                phase_start = time.perf_counter()
                self._phase_sample(files, timestamps)
                phase_time = time.perf_counter() - phase_start
                if not need_coh:
                    self._phase_times_rolling.append(phase_time)

            # 3. Minute snapshot
            if need_coh:
                total_time = time.perf_counter() - total_start
                self._save_minute_snapshot(
                    ts_int,
                    ffts,
                    coh_data,
                    triple_time,
                    total_time,
                    phase_time=phase_time,
                )

            del self.pending[ts_int]
            self._seconds_processed += 1
            return True

        except Exception:
            logger.exception("Error processing second %d", ts_int)
            self._errors_since_heartbeat += 1
            return False

    def _phase_sample(self, files, timestamps):
        """Take a high-rate phase sample, subtract the online-estimated
        oscillator drift, and append it to the unified phase.csv."""
        now = time.time()
        if now - self._last_phase_sample_time < self._phase_period:
            return
        self._last_phase_sample_time = now

        t_sec = timestamps[0]
        row = [int(t_sec * 1000)]  # timestamp in ms
        for ch, fp in enumerate(files):
            z = self.sb.process_baseband_iq(
                fp, target_freq_hz=self.target_freq_hz, fs=self.fs
            )
            z_shifted = z * np.exp(1j * self.phase_offset_rad)
            wrapped_deg = np.degrees(np.angle(z_shifted))
            residual_deg, _slope = self._detrenders[ch].update(t_sec, wrapped_deg)
            row.append(residual_deg)
            row.append(abs(z_shifted))

        write_header = not self.phase_log_path.exists()
        with open(self.phase_log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                header = ["timestamp_ms"]
                for ch in range(1, self.num_channels + 1):
                    header.append(f"cha{ch}_phase_deg")
                    header.append(f"cha{ch}_amplitude")
                writer.writerow(header)
            writer.writerow(row)

    def _write_second_log(self, ts_int, ffts):
        row = [ts_int]
        for ch_fft in ffts:
            for bin_idx in self.selected_bins:
                row.append(ch_fft[bin_idx])
        write_header = not self.second_log_path.exists()
        with open(self.second_log_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                h = ["second"]
                for ch in range(1, self.num_channels + 1):
                    for f_hz in SELECTED_FREQUENCIES_HZ:
                        h.append(f"cha{ch}_{f_hz / 1e6:.2f}MHz")
                w.writerow(h)
            w.writerow(row)

    def _save_minute_snapshot(
        self, ts_int, ffts, coh_data, triple_time, total_time, phase_time=0.0
    ):
        minute_start = (ts_int // 60) * 60
        h5_path = self.minute_h5_dir / f"minute_{minute_start}.h5"
        datasets = {}
        for ch in range(self.num_channels):
            datasets[f"cha{ch + 1}/fft"] = ffts[ch].reshape(1, -1)
        for (i, j), coh in coh_data.items():
            datasets[f"pairs/{i + 1}{j + 1}/coherence"] = coh.reshape(1, -1)
        datasets["timestamps"] = np.array([ts_int], dtype=np.float64)
        t0 = time.perf_counter()
        self.sb.save_to_h5(
            h5_path=h5_path,
            datasets=datasets,
            metadata={"fs": self.fs, "fft_size": self.fft_size},
        )
        save_time = time.perf_counter() - t0
        self._minutes_saved += 1
        self._last_minute_saved_at = minute_start
        avg_fft = (
            sum(self._fft_times_rolling) / len(self._fft_times_rolling)
            if self._fft_times_rolling
            else 0
        )
        avg_phase = (
            sum(self._phase_times_rolling) / len(self._phase_times_rolling)
            if self._phase_times_rolling
            else 0
        )
        logger.info(
            f"Minute {minute_start} snapshot -> {h5_path} ({save_time:.4f}s save)\n"
            f"    Triple-all: {triple_time:.4f}s\n"
            f"    Phase sample (detrend): {phase_time:.6f}s\n"
            f"    Avg FFT/sec: {avg_fft:.4f}s\n"
            f"    Avg phase/sec: {avg_phase:.6f}s\n"
            f"    Total second: {total_time:.4f}s"
        )

    def _maybe_heartbeat(self):
        now = time.time()
        if now - self._last_heartbeat < self.heartbeat_interval:
            return
        uptime = now - self._start_time
        last_minute = (
            f"minute_{self._last_minute_saved_at}"
            if self._last_minute_saved_at
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
                for ch in range(self.num_channels):
                    self._add_new_files(ch)
                ready_seconds = sorted(
                    [
                        ts
                        for ts in self.pending
                        if len(self.pending[ts]) == self.num_channels
                    ]
                )
                for ts_int in ready_seconds:
                    self._process_second(ts_int)
                now = time.time()
                stale = [ts for ts in list(self.pending.keys()) if now - ts > 10]
                for ts_int in stale:
                    del self.pending[ts_int]
                self.sb.clean_cache(2)
                self._maybe_heartbeat()
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0, self.poll_interval - elapsed))
        except KeyboardInterrupt:
            logger.info("Stopped.")
        finally:
            self.sb.shutdown()


def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--target-freq-hz", type=float, default=TARGET_FREQ_HZ)
    parser.add_argument(
        "--phase-offset-deg",
        type=float,
        default=PHASE_OFFSET_DEG,
        help="Phase offset for e-czas (default: 36°)",
    )
    parser.add_argument(
        "--phase-sampling-hz",
        type=float,
        default=PHASE_SAMPLING_HZ,
        help="Unified phase.csv sampling rate in Hz, 0=off (default: %(default)s)",
    )
    parser.add_argument(
        "--detrend-window-sec",
        type=float,
        default=DETREND_WINDOW_SEC,
        help="Rolling window (seconds) used to estimate and subtract the "
        "oscillator/frequency-offset drift before writing phase.csv "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--detrend-refit-interval-sec",
        type=float,
        default=DETREND_REFIT_INTERVAL_SEC,
        help="How often (seconds) the linear drift fit is recomputed "
        "(default: %(default)s)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    setup_logging(args.output, args.verbose, LOG_RETENTION_DAYS)
    proc = LiveProcessor(
        cha_roots=args.cha_roots,
        fft_size=args.fft_size,
        fs=args.fs,
        cache_dir=args.cache,
        dataset_name=args.dataset,
        output_dir=args.output,
        selected_freqs_hz=args.freqs,
        poll_interval=args.poll,
        max_time_diff=args.max_diff,
        target_freq_hz=args.target_freq_hz,
        phase_offset_deg=args.phase_offset_deg,
        phase_sampling_hz=args.phase_sampling_hz,
        detrend_window_sec=args.detrend_window_sec,
        detrend_refit_interval_sec=args.detrend_refit_interval_sec,
    )
    proc.run()


if __name__ == "__main__":
    main()