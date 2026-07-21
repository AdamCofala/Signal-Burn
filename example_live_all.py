#!/usr/bin/env python3
"""Live dual-channel spectrograms + coherence (PyQt5 + pyqtgraph).

Displays three rolling spectrograms side-by-side:
  - Channel 1 power [dB]
  - Channel 2 power [dB]
  - Magnitude squared coherence [0-1]
Files from two folders are paired by nearest timestamp and processed on the GPU.
Updates immediately when a new pair is processed - no artificial timer delay.
"""

import argparse
import sys
import time
import re

from pathlib import Path
from collections import deque
import bisect

import numpy as np
import pyqtgraph as pg
import matplotlib.cm as cm
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QRectF
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from sblib.SignalBurner import SignalBurner

# ----------------------------- defaults ---------------------------------
CHA1_ROOT = Path("/dev/shm/signal-burn/hf25/cha1")
CHA2_ROOT = Path("/dev/shm/signal-burn/hf25/cha2")
CACHE_DIR = Path("/pool/signal_storage/cache")
FFT_SIZE = 262144
FS = 25_000_000  # Hz
WINDOW_MINUTES = 1.0  # rolling window length
DATASET_NAME = "rf_data"
MAX_TIME_DIFF = 0.0  # max timestamp mismatch for pairing
CLEAR_INPUT_FOLDER = True  # delete existing .h5 files in input folders on start
DOWNSAMPLE_DISPLAY = 1  # keep every N-th file in display
FREQ_DOWNSAMPLE = 64  # downsampling factor on frequency axis (1 = no reduction)
X_FUTURE = 5.0  # seconds to extend x-axis into the future
# ------------------------------------------------------------------------


def latest_input_dir(base: Path) -> Path:
    """Return the most recent timestamp-named subdirectory."""
    if not base.exists():
        return base
    subdirs = [p for p in base.iterdir() if p.is_dir()]
    if not subdirs:
        return base
    return max(subdirs, key=lambda p: (p.stat().st_mtime, p.name))


def parse_timestamp(filename: Path) -> float:
    """Extract UNIX timestamp from 'rf@123456789.123.h5'."""
    m = re.match(r"rf@(\d+)\.(\d+)", filename.stem)
    if not m:
        raise ValueError(f"Invalid filename: {filename.name}")
    return int(m.group(1)) + int(m.group(2)) / 1000.0


class Worker(QThread):
    """Background thread – scans two folders incrementally, pairs files,
    computes FFTs + coherence, and emits dB-converted spectra."""

    data_ready = pyqtSignal(object)  # (times, spec1_dB, spec2_dB, coh_spec)

    def __init__(
        self,
        cha1_dir,
        cha2_dir,
        cache_dir,
        fft_size,
        dataset_name,
        window_seconds,
        max_time_diff,
        clear_input,
    ):
        super().__init__()
        self.cha1_dir = Path(cha1_dir)
        self.cha2_dir = Path(cha2_dir)
        self.cache_dir = Path(cache_dir)
        self.fft_size = fft_size
        self.dataset_name = dataset_name
        self.window_seconds = window_seconds
        self.max_time_diff = max_time_diff
        self.clear_input = clear_input

        self.sb = None
        self.timer = None

        # Incremental discovery
        self.seen1 = {}  # stem -> (ts, path)
        self.seen2 = {}
        self.unpaired1 = []  # (ts, path)
        self.unpaired2 = []

        # History buffers
        self.hist1 = deque()
        self.hist2 = deque()
        self.hist_coh = deque()

    def run(self):
        self.sb = SignalBurner(
            fft_size=self.fft_size,
            dataset_name=self.dataset_name,
            cache_path=self.cache_dir,
            use_cache=False,
            show_logs=True,
        )
        print("[Worker] SignalBurner ready.")

        if self.clear_input:
            for folder in (self.cha1_dir, self.cha2_dir):
                if folder.exists():
                    for f in folder.glob("*.h5"):
                        try:
                            f.unlink()
                        except Exception:
                            pass

        self.timer = QTimer()
        self.timer.timeout.connect(self._poll)
        self.timer.start(500)  # ms
        self.exec_()
        self.sb.shutdown()
        print("[Worker] Stopped.")

    def _add_new_files(self, folder: Path, seen: dict, unpaired: list):
        """Add new .h5 files and return how many were added."""
        new_count = 0
        for fp in folder.glob("*.h5"):
            stem = fp.stem
            if stem in seen:
                continue
            try:
                ts = parse_timestamp(fp)
            except ValueError:
                continue
            seen[stem] = (ts, fp)
            bisect.insort(unpaired, (ts, fp), key=lambda x: x[0])
            new_count += 1
        return new_count

    def _pair_nearest(self, list1, list2):
        """Yield (p1, p2, ts1, ts2) for matching pairs within max_time_diff."""
        i = j = 0
        while i < len(list1) and j < len(list2):
            ts1, p1 = list1[i]
            ts2, p2 = list2[j]
            if abs(ts1 - ts2) <= self.max_time_diff:
                yield p1, p2, ts1, ts2
                i += 1
                j += 1
            elif ts1 < ts2:
                i += 1
            else:
                j += 1

    def _poll(self):
        poll_start = time.time()

        dir1 = latest_input_dir(self.cha1_dir)
        dir2 = latest_input_dir(self.cha2_dir)

        # Discover new files and log counts
        n_new1 = self._add_new_files(dir1, self.seen1, self.unpaired1)
        n_new2 = self._add_new_files(dir2, self.seen2, self.unpaired2)
        if n_new1 > 0 or n_new2 > 0:
            print(
                f"[Worker] Discovered {n_new1} new file(s) in cha1, {n_new2} in cha2 "
                f"(total unpaired: {len(self.unpaired1)}, {len(self.unpaired2)})"
            )

        if not self.unpaired1 or not self.unpaired2:
            return

        # Trim old unpaired
        cutoff = max(self.unpaired1[-1][0], self.unpaired2[-1][0]) - self.window_seconds
        while self.unpaired1 and self.unpaired1[0][0] < cutoff:
            self.unpaired1.pop(0)
        while self.unpaired2 and self.unpaired2[0][0] < cutoff:
            self.unpaired2.pop(0)

        pairs = list(self._pair_nearest(self.unpaired1, self.unpaired2))
        if pairs:
            print(f"[Worker] Paired {len(pairs)} file(s) for processing")

        new_data = False
        processing_times = []

        for p1, p2, ts1, ts2 in pairs:
            # Remove from unpaired
            try:
                self.unpaired1.remove((ts1, p1))
            except ValueError:
                pass
            try:
                self.unpaired2.remove((ts2, p2))
            except ValueError:
                pass

            t_start = time.time()
            try:
                spec1 = self.sb.process_file(p1)
                spec2 = self.sb.process_file(p2)
                coh = self.sb.process_coherence(p1, p2)
                t_end = time.time()
                dt = t_end - t_start
                processing_times.append(dt)

                spec1_db = 10 * np.log10(spec1 + 1e-12)
                spec2_db = 10 * np.log10(spec2 + 1e-12)
                avg_ts = (ts1 + ts2) / 2

                self.hist1.append((avg_ts, spec1_db))
                self.hist2.append((avg_ts, spec2_db))
                self.hist_coh.append((avg_ts, coh))
                new_data = True

                print(
                    f"[Worker] Processed pair: {p1.name} / {p2.name} "
                    f"(ts diff: {abs(ts1 - ts2):.6f}s) in {dt:.4f}s"
                )
            except Exception as e:
                t_end = time.time()
                print(
                    f"[Worker] Error processing {p1.name}/{p2.name}: {e} "
                    f"(failed after {t_end - t_start:.4f}s)"
                )

        # Trim history deques
        if self.hist1:
            cutoff_hist = self.hist1[-1][0] - self.window_seconds
            while self.hist1 and self.hist1[0][0] < cutoff_hist:
                self.hist1.popleft()
            while self.hist2 and self.hist2[0][0] < cutoff_hist:
                self.hist2.popleft()
            while self.hist_coh and self.hist_coh[0][0] < cutoff_hist:
                self.hist_coh.popleft()

        if new_data and self.hist1:
            times = [t for t, _ in self.hist1]
            mat1 = np.stack([s for _, s in self.hist1], axis=0)
            mat2 = np.stack([s for _, s in self.hist2], axis=0)
            matc = np.stack([s for _, s in self.hist_coh], axis=0)
            self.data_ready.emit((times, mat1, mat2, matc))

        poll_duration = time.time() - poll_start
        if processing_times:
            total_proc = sum(processing_times)
            avg_proc = total_proc / len(processing_times)
            print(
                f"[Worker] Poll cycle completed in {poll_duration:.4f}s, "
                f"GPU total: {total_proc:.4f}s, avg/pair: {avg_proc:.4f}s"
            )
        else:
            print(
                f"[Worker] Poll cycle completed in {poll_duration:.4f}s (no new pairs)"
            )

    def stop(self):
        if self.timer:
            self.timer.stop()
        self.quit()


class MainWindow(QMainWindow):
    def __init__(
        self,
        cha1_root,
        cha2_root,
        cache_dir,
        fft_size,
        dataset_name,
        window_minutes,
        fs,
        downsample,
        freq_downsample,
        x_future,
        max_time_diff,
        clear_input,
    ):
        super().__init__()
        self.setWindowTitle("Live Dual Channel + Coherence")
        self.fs = fs
        self.window_seconds = window_minutes * 60
        self.downsample = downsample
        self.freq_downsample = freq_downsample
        self.x_future = x_future
        self.fft_size = fft_size

        self.cha1_dir = latest_input_dir(Path(cha1_root))
        self.cha2_dir = latest_input_dir(Path(cha2_root))
        print(f"cha1 dir: {self.cha1_dir}\ncha2 dir: {self.cha2_dir}")

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        titles = ["Channel 1 Power [dB]", "Channel 2 Power [dB]", "Coherence"]
        self.plots = []
        self.img_items = []
        for idx, title in enumerate(titles):
            pw = pg.PlotWidget()
            pw.setLabel("left", "Freq", units="MHz", color="w")
            pw.setLabel("bottom", "Time", units="s", color="w")
            pw.setBackground("k")
            for ax in ("left", "bottom"):
                pw.getAxis(ax).setPen("w")
                pw.getAxis(ax).setTextPen("w")
            pw.showGrid(x=True, y=True, alpha=0.3)
            pw.setAspectLocked(False)
            pw.setXRange(-self.window_seconds, self.x_future, padding=0)
            pw.setYRange(0, fs / 1e6, padding=0)
            pw.setTitle(title, color="w", size="12pt")

            img = pg.ImageItem()
            pw.addItem(img)
            if idx < 2:
                lut = cm.jet(np.arange(256))[:, :3] * 255
            else:
                lut = cm.plasma(np.arange(256))[:, :3] * 255
            img.setLookupTable(lut.astype(np.uint8))
            img.setAutoDownsample(True)
            img.setImage(np.zeros((1, fft_size)))

            layout.addWidget(pw)
            self.plots.append(pw)
            self.img_items.append(img)

        self.setGeometry(100, 100, 1800, 600)

        self.latest_data = None

        self.worker = Worker(
            cha1_dir=self.cha1_dir,
            cha2_dir=self.cha2_dir,
            cache_dir=cache_dir,
            fft_size=fft_size,
            dataset_name=dataset_name,
            window_seconds=self.window_seconds,
            max_time_diff=max_time_diff,
            clear_input=clear_input,
        )
        self.worker.data_ready.connect(self._on_data_ready)
        self.worker.start()

        self.last_log_time = 0

    def _on_data_ready(self, data):
        self.latest_data = data
        self._update_display()

    def _update_display(self):
        data = self.latest_data
        if data is None:
            return
        times, mat1_db, mat2_db, matc = data
        if mat1_db.size == 0:
            return

        if self.downsample > 1:
            times = times[:: self.downsample]
            mat1_db = mat1_db[:: self.downsample, :]
            mat2_db = mat2_db[:: self.downsample, :]
            matc = matc[:: self.downsample, :]

        if self.freq_downsample > 1:

            def downsample_freq(arr, factor):
                n_freq = arr.shape[-1]
                new_len = n_freq // factor
                if new_len == 0:
                    return arr
                arr_trim = arr[..., : new_len * factor]
                return arr_trim.reshape(arr.shape[:-1] + (new_len, factor)).mean(
                    axis=-1
                )

            mat1_db = downsample_freq(mat1_db, self.freq_downsample)
            mat2_db = downsample_freq(mat2_db, self.freq_downsample)
            matc = downsample_freq(matc, self.freq_downsample)

        latest_time = times[-1]
        times_rel = [t - latest_time for t in times]
        x_min, x_max = times_rel[0], 0.0
        y_min, y_max = 0, self.fs / 1e6

        vmin1, vmax1 = np.percentile(mat1_db, 5), np.percentile(mat1_db, 95)
        vmin2, vmax2 = np.percentile(mat2_db, 5), np.percentile(mat2_db, 95)
        if vmin1 == vmax1:
            vmin1 -= 1
            vmax1 += 1
        if vmin2 == vmax2:
            vmin2 -= 1
            vmax2 += 1

        self.img_items[0].setImage(mat1_db, autoLevels=False, levels=[vmin1, vmax1])
        self.img_items[1].setImage(mat2_db, autoLevels=False, levels=[vmin2, vmax2])
        self.img_items[2].setImage(matc, autoLevels=False, levels=[0.0, 1.0])

        for i in range(3):
            self.img_items[i].setRect(
                QRectF(x_min, y_min, x_max - x_min, y_max - y_min)
            )

        for pw in self.plots:
            pw.setXRange(-self.window_seconds, self.x_future, padding=0)
            pw.setYRange(y_min, y_max, padding=0)

        now_log = time.time()
        if now_log - self.last_log_time >= 5.0:
            self.last_log_time = now_log
            print(f"[GUI] {len(times)} pairs in window, time {x_min:.1f}–{x_max:.1f} s")

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()


def main():
    parser = argparse.ArgumentParser(
        description="Live dual-channel spectrograms + coherence"
    )
    parser.add_argument("--cha1", type=Path, default=CHA1_ROOT)
    parser.add_argument("--cha2", type=Path, default=CHA2_ROOT)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument("--window", type=float, default=WINDOW_MINUTES)
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--max-diff", type=float, default=MAX_TIME_DIFF)
    parser.add_argument(
        "--clear-input", action="store_true", default=CLEAR_INPUT_FOLDER
    )
    parser.add_argument("--downsample", type=int, default=DOWNSAMPLE_DISPLAY)
    parser.add_argument("--freq-downsample", type=int, default=FREQ_DOWNSAMPLE)
    parser.add_argument("--x-future", type=float, default=X_FUTURE)
    args = parser.parse_args()

    pg.setConfigOptions(useOpenGL=False, antialias=False)
    app = QApplication(sys.argv)
    window = MainWindow(
        cha1_root=args.cha1,
        cha2_root=args.cha2,
        cache_dir=args.cache,
        fft_size=args.fft_size,
        dataset_name=args.dataset,
        window_minutes=args.window,
        fs=args.fs,
        downsample=args.downsample,
        freq_downsample=args.freq_downsample,
        x_future=args.x_future,
        max_time_diff=args.max_diff,
        clear_input=args.clear_input,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
