"""Live rolling spectrogram (PyQt5). Updates every 0.5 s."""

import argparse
import sys, time, re, threading
from pathlib import Path
import numpy as np
import pyqtgraph as pg
import matplotlib.cm as cm
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QRectF
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from sblib.SignalBurner import SignalBurner

# ---------- defaults ----------
INPUT_ROOT = Path("/pool/signal_storage/hf25/cha1")
CACHE_DIR = Path("/pool/signal_storage/cache")
FFT_SIZE = 262144
FS = 25_000_000
WINDOW_MINUTES = 5.0
UPDATE_INTERVAL = 0.5
DATASET_NAME = "rf_data"
CLEAR_INPUT_FOLDER = True
SAVE_PNG = False
OUTPUT_FILE = Path("./live_spectrogram.png")
DOWNSAMPLE_DISPLAY = 1
# ------------------------------


def latest_input_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if not subdirs:
        return base_dir
    return max(subdirs, key=lambda p: (p.stat().st_mtime, p.name))


def parse_timestamp(filename: Path) -> float:
    m = re.match(r"rf@(\d+)\.(\d+)", filename.stem)
    if not m:
        raise ValueError(f"Invalid filename: {filename.name}")
    return int(m.group(1)) + int(m.group(2)) / 1000.0


class Worker(QThread):
    data_ready = pyqtSignal(object)

    def __init__(
        self, input_dir, cache_dir, fft_size, dataset_name, window_seconds, clear_input
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.cache_dir = Path(cache_dir)
        self.fft_size = fft_size
        self.dataset_name = dataset_name
        self.window_seconds = window_seconds
        self.clear_input = clear_input
        self.sb = None
        self.timer = None
        self.processed = set()
        self.history = []

    def run(self):
        if not self.input_dir.exists():
            print(f"[Worker] WARNING: input directory missing: {self.input_dir}")
        self.sb = SignalBurner(
            fft_size=self.fft_size,
            dataset_name=self.dataset_name,
            cache_path=self.cache_dir,
            use_cache=False,
            show_logs=True,
        )
        print("[Worker] SignalBurner ready.")
        if self.clear_input and self.input_dir.exists():
            print(f"[Worker] Clearing input directory: {self.input_dir}")
            for f in self.input_dir.glob("*.h5"):
                try:
                    f.unlink()
                    print(f"  removed {f.name}")
                except Exception as e:
                    print(f"  error removing {f.name}: {e}")
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_for_new_files)
        self.timer.start(500)
        self.exec_()
        self.sb.shutdown()
        print("[Worker] Stopped.")

    def check_for_new_files(self):
        all_files = sorted(self.input_dir.glob("*.h5"), key=lambda p: p.name)
        file_entries = []
        for fp in all_files:
            try:
                ts = parse_timestamp(fp)
                file_entries.append((fp, ts))
            except ValueError:
                continue
        if not file_entries:
            return
        latest_ts = max(ts for _, ts in file_entries)
        cutoff = latest_ts - self.window_seconds
        window_files = [(fp, ts) for fp, ts in file_entries if ts >= cutoff]
        new_count = 0
        for fp, ts in window_files:
            if fp.stem not in self.processed:
                try:
                    mag = self.sb.process_file(fp)
                    self.processed.add(fp.stem)
                    self.history.append((ts, mag))
                    new_count += 1
                except Exception as e:
                    print(f"[Worker] Error processing {fp.name}: {e}")
        self.history = [(t, s) for t, s in self.history if t >= cutoff]
        self.history.sort(key=lambda x: x[0])
        if new_count > 0 and self.history:
            times = [t for t, _ in self.history]
            spectra = [s for _, s in self.history]
            if len(set(s.shape for s in spectra)) != 1:
                print("[Worker] Inconsistent spectrum shapes")
            else:
                spec_matrix = np.stack(spectra, axis=0)
                self.data_ready.emit((times, spec_matrix))
                print(
                    f"[Worker] {new_count} new file(s), total in window: {len(self.history)}"
                )

    def stop(self):
        if self.timer:
            self.timer.stop()
        self.quit()


class MainWindow(QMainWindow):
    def __init__(
        self,
        input_dir,
        cache_dir,
        fft_size,
        dataset_name,
        window_seconds,
        fs,
        downsample,
    ):
        super().__init__()
        self.setWindowTitle("Live Spectrogram - pyqtgraph")
        self.setGeometry(100, 100, 1200, 600)
        self.fs = fs
        self.window_seconds = window_seconds
        self.downsample = downsample

        self.input_dir = latest_input_dir(input_dir)
        print(f"[GUI] Using input directory: {self.input_dir}")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_widget.setLabel("left", "Frequency", units="MHz", color="w")
        self.plot_widget.setLabel("bottom", "Time", units="s", color="w")
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.setBackground("k")
        for axis in ("left", "bottom"):
            self.plot_widget.getAxis(axis).setPen("w")
            self.plot_widget.getAxis(axis).setTextPen("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)
        self.img_item.setImage(np.zeros((1, FFT_SIZE)))
        jet_cmap = cm.jet(np.arange(256))[:, :3] * 255
        self.img_item.setLookupTable(jet_cmap.astype(np.uint8))
        self.img_item.setAutoDownsample(True)

        self.plot_widget.setXRange(-window_seconds, 0, padding=0)
        self.plot_widget.setYRange(0, fs / 1e6, padding=0)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(int(UPDATE_INTERVAL * 1000))

        self.latest_data = None
        self.lock = threading.Lock()

        self.worker = Worker(
            input_dir=self.input_dir,
            cache_dir=cache_dir,
            fft_size=fft_size,
            dataset_name=dataset_name,
            window_seconds=window_seconds,
            clear_input=CLEAR_INPUT_FOLDER,
        )
        self.worker.data_ready.connect(self.on_data_ready)
        self.worker.start()

        self.last_log_time = 0

    def on_data_ready(self, data):
        with self.lock:
            self.latest_data = data

    def update_display(self):
        with self.lock:
            data = self.latest_data
            if data is None:
                return
            times, spec_matrix = data
            self.latest_data = None

        if spec_matrix.size == 0:
            return
        if self.downsample > 1:
            spec_matrix = spec_matrix[:: self.downsample, :]
            times = times[:: self.downsample]

        spec_db = 10 * np.log10(spec_matrix + 1e-12)
        y_min, y_max = 0, self.fs / 1e6
        latest_time = times[-1]
        times_rel = [t - latest_time for t in times]
        x_min, x_max = times_rel[0], 0.0

        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 95)
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        self.img_item.setImage(spec_db, autoLevels=False)
        self.img_item.setLevels([vmin, vmax])
        self.img_item.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))
        self.plot_widget.setXRange(-self.window_seconds, 0, padding=0)
        self.plot_widget.setYRange(y_min, y_max, padding=0)

        if SAVE_PNG:
            exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
            exporter.export(str(OUTPUT_FILE))

        now_log = time.time()
        if now_log - self.last_log_time >= 5.0:
            self.last_log_time = now_log
            print(
                f"[GUI] window: {len(times)} files, "
                f"rel time {x_min:.1f}-{x_max:.1f} s, "
                f"power {vmin:.1f}-{vmax:.1f} dB"
            )

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="Live rolling spectrogram")
    parser.add_argument("--input", type=Path, default=INPUT_ROOT)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--fs", type=float, default=FS)
    parser.add_argument(
        "--window", type=float, default=WINDOW_MINUTES, help="Window length in minutes"
    )
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument(
        "--downsample",
        type=int,
        default=DOWNSAMPLE_DISPLAY,
        help="Keep every N-th file in display",
    )
    args = parser.parse_args()

    pg.setConfigOptions(useOpenGL=False, antialias=True)
    app = QApplication(sys.argv)
    window = MainWindow(
        input_dir=args.input,
        cache_dir=args.cache,
        fft_size=args.fft_size,
        dataset_name=args.dataset,
        window_seconds=args.window * 60,
        fs=args.fs,
        downsample=args.downsample,
    )
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
