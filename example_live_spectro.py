#!/usr/bin/env python3
"""
Live spectrogram – 5-minutowe okno, aktualizacja co 1 s.
GUI: PyQt5 + pyqtgraph (software rendering – bez OpenGL).
Przetwarzanie plików w tle za pomocą QTimer (brak time.sleep).
Automatyczne usuwanie starych danych na bazie czasu z plików.
"""

import sys
import time
import re
import threading
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import matplotlib.cm as cm
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, QRectF
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from sblib.SignalBurner import SignalBurner

# ========== KONFIGURACJA ==========
INPUT_DIR = Path("/pool/signal_storage/hf25/cha1/2026-07-14T18-00-00")
CACHE_DIR = Path("/pool/signal_storage/cache")
FFT_SIZE = 262144
FS = 25_000_000  # Hz
WINDOW_MINUTES = 5.0
UPDATE_INTERVAL = 0.5  # sekundy (odświeżanie GUI)
DATASET_NAME = "rf_data"
CLEAR_INPUT_FOLDER = True
SAVE_PNG = False
OUTPUT_FILE = Path("./live_spectrogram.png")
DOWNSAMPLE_DISPLAY = 1  # wyświetlaj co N-ty plik (zmniejsza liczbę kolumn)
# ==================================


def parse_timestamp(filename: Path) -> float:
    m = re.match(r"rf@(\d+)\.(\d+)", filename.stem)
    if not m:
        raise ValueError(f"Nieprawidłowa nazwa: {filename.name}")
    return int(m.group(1)) + int(m.group(2)) / 1000.0


class Worker(QThread):
    """Wątek przetwarzający nowe pliki sterowany timerem (non-blocking)."""

    data_ready = pyqtSignal(object)  # (times, spectra_matrix)

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
        self.history = []  # lista (timestamp, spectrum)

    def run(self):
        if not self.input_dir.exists():
            print(f"[Worker] UWAGA: Katalog {self.input_dir} nie istnieje!")

        self.sb = SignalBurner(
            fft_size=self.fft_size,
            dataset_name=self.dataset_name,
            cache_path=self.cache_dir,
            use_cache=True,
            show_logs=True,
        )
        print("[Worker] SignalBurner gotowy.")

        if self.clear_input and self.input_dir.exists():
            print(f"[Worker] Czyszczę folder wejściowy: {self.input_dir}")
            for f in self.input_dir.glob("*.h5"):
                try:
                    f.unlink()
                    print(f"  usunięto {f.name}")
                except Exception as e:
                    print(f"  błąd usuwania {f.name}: {e}")

        # Zamiast pętli while z sleep, odpalamy QTimer wewnątrz event loopa wątku
        self.timer = QTimer()
        self.timer.timeout.connect(self.check_for_new_files)
        self.timer.start(500)  # Sprawdzaj katalog co 500ms

        self.exec_()  # Uruchamia pętlę zdarzeń i blokuje do momentu quit()

        self.sb.shutdown()
        print("[Worker] Zatrzymano.")

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

        # NAJWAŻNIEJSZE: Czas referencyjny to czas najnowszego pliku w katalogu!
        latest_ts = max(ts for _, ts in file_entries)
        cutoff = latest_ts - self.window_seconds

        # Wybieramy pliki, które mieszczą się w naszym 5-minutowym oknie wstecz
        window_files = [(fp, ts) for fp, ts in file_entries if ts >= cutoff]

        new_count = 0
        for fp, ts in window_files:
            stem = fp.stem
            if stem not in self.processed:
                try:
                    mag = self.sb.process_file(fp)
                    self.processed.add(stem)
                    self.history.append((ts, mag))
                    new_count += 1
                except Exception as e:
                    print(f"[Worker] Błąd {fp.name}: {e}")

        # Przycina historię w pamięci podręcznej – usuwa to, co wypadło poza okno 5 min
        self.history = [(t, s) for t, s in self.history if t >= cutoff]
        self.history.sort(key=lambda x: x[0])

        # Jeśli pojawiły się nowe pliki, wypychamy pełne, zaktualizowane okno do GUI
        if new_count > 0 and self.history:
            times = [t for t, _ in self.history]
            spectra = [s for _, s in self.history]
            shapes = [s.shape for s in spectra]

            if len(set(shapes)) != 1:
                print(f"[Worker] Błąd: różne kształty widm: {shapes}")
            else:
                spec_matrix = np.stack(spectra, axis=0)
                self.data_ready.emit((times, spec_matrix))
                print(
                    f"[Worker] Nowe dane: {new_count} plików, w oknie: {len(self.history)}"
                )

    def stop(self):
        if self.timer:
            self.timer.stop()
        self.quit()  # Kończy exec_()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Live Spectrogram – pyqtgraph (soft render)")
        self.setGeometry(100, 100, 1200, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)

        # Główny wykres
        self.plot_widget = pg.PlotWidget()
        layout.addWidget(self.plot_widget)
        self.plot_widget.setLabel("left", "Częstotliwość", units="MHz", color="w")
        self.plot_widget.setLabel("bottom", "Czas", units="s", color="w")
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.setBackground("k")

        self.plot_widget.getAxis("left").setPen("w")
        self.plot_widget.getAxis("left").setTextPen("w")
        self.plot_widget.getAxis("bottom").setPen("w")
        self.plot_widget.getAxis("bottom").setTextPen("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)

        # ImageItem – wyświetlanie spektrogramu
        self.img_item = pg.ImageItem()
        self.plot_widget.addItem(self.img_item)

        # Domyślny obraz (pusty)
        self.img_item.setImage(np.zeros((1, FFT_SIZE)))
        jet_cmap = cm.jet(np.arange(256))[:, :3] * 255
        self.img_item.setLookupTable(jet_cmap.astype(np.uint8))
        self.img_item.setAutoDownsample(True)

        # Stały, sztywny podgląd na osie: X od -300s do 0s, Y od 0 do 25 MHz
        self.plot_widget.setXRange(-WINDOW_MINUTES * 60, 0, padding=0)
        self.plot_widget.setYRange(0, FS / 1e6, padding=0)

        # Timer do odświeżania GUI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_display)
        self.timer.start(int(UPDATE_INTERVAL * 1000))

        # Bufor na dane z wątku
        self.latest_data = None
        self.lock = threading.Lock()

        # Uruchom wątek
        self.worker = Worker(
            input_dir=INPUT_DIR,
            cache_dir=CACHE_DIR,
            fft_size=FFT_SIZE,
            dataset_name=DATASET_NAME,
            window_seconds=WINDOW_MINUTES * 60,
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

        # Próbkowanie w czasie (wymiar 0 to Czas)
        if DOWNSAMPLE_DISPLAY > 1:
            spec_matrix = spec_matrix[::DOWNSAMPLE_DISPLAY, :]
            times = times[::DOWNSAMPLE_DISPLAY]

        # Konwersja na dB
        spec_db = 10 * np.log10(spec_matrix + 1e-12)

        y_min = 0
        y_max = FS / 1e6  # 25.0 MHz

        # Liczymy czas relatywny względem NAJNOWSZEJ odebranej próbki
        latest_data_time = times[-1]
        times_rel = [t - latest_data_time for t in times]
        x_min = times_rel[0]
        x_max = 0.0  # Najnowsza próbka zawsze ląduje na 0.0s (prawa strona)

        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 95)
        if vmin == vmax:
            vmin -= 1
            vmax += 1

        # Podmiana obrazu w pyqtgraph
        self.img_item.setImage(spec_db, autoLevels=False)
        self.img_item.setLevels([vmin, vmax])

        # setRect mapuje macierz na fizyczne osie wykresu
        self.img_item.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))

        # Trzymamy stabilne, stałe granice wyświetlania
        self.plot_widget.setXRange(-WINDOW_MINUTES * 60, 0, padding=0)
        self.plot_widget.setYRange(y_min, y_max, padding=0)

        if SAVE_PNG:
            self.save_png()

        now_log = time.time()
        if now_log - self.last_log_time >= 5.0:
            self.last_log_time = now_log
            print(
                f"[GUI] Okno: {len(times)} plików, relatywny czas: {x_min:.1f}s do {x_max:.1f}s, "
                f"moc: {vmin:.1f}–{vmax:.1f} dB"
            )

    def save_png(self):
        exporter = pg.exporters.ImageExporter(self.plot_widget.scene())
        exporter.export(str(OUTPUT_FILE))

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait()
        event.accept()


def main():
    pg.setConfigOptions(useOpenGL=False, antialias=True)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
