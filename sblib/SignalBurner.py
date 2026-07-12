import ctypes
import json
import queue
import threading
from pathlib import Path

import h5py
import numpy as np


class SignalBurner:
    def __init__(
        self,
        input_path=None,
        save_files=False,
        output_path=None,
        cache_path=None,
        dataset_name="rf_data",
        out_bins=2048,
        lib_path=None,
    ):
        self.input_path = input_path
        self.output_path = output_path
        self.cache_path = cache_path
        self.dataset_name = dataset_name
        self.save_files = save_files
        self.out_bins = out_bins

        self._lib_path = (
            Path(lib_path)
            if lib_path is not None
            else Path(__file__).parent.parent / "bin" / "libsb_core.so"
        )
        self._lib = None

    # --- properties ---------------------------------------------------

    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, value):
        self._input_path = Path(value) if value is not None else None

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, value):
        self._output_path = Path(value) if value is not None else None

    @property
    def cache_path(self):
        return self._cache_path

    @cache_path.setter
    def cache_path(self, value):
        self._cache_path = Path(value) if value is not None else None

    @property
    def dataset_name(self):
        return self._dataset_name

    @dataset_name.setter
    def dataset_name(self, value):
        self._dataset_name = value

    @property
    def out_bins(self):
        return self._out_bins

    @out_bins.setter
    def out_bins(self, value):
        if value <= 0:
            raise ValueError("out_bins musi byc > 0")
        self._out_bins = value

    # --- biblioteka (cache'owana, lazy) --------------------------------

    def load_library(self):
        if self._lib is None:
            if not self._lib_path.exists():
                raise FileNotFoundError(f"Nie znaleziono biblioteki: {self._lib_path}")
            lib = ctypes.CDLL(str(self._lib_path))
            lib.sb_process_iq.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.sb_process_iq.restype = ctypes.c_int
            self._lib = lib
        return self._lib

    # --- cache index (persystentny, JSON) ------------------------------

    def _cache_index_path(self):
        return self._cache_path / "index.json"

    def _load_cache_index(self):
        if self._cache_path is None:
            return {}
        self._cache_path.mkdir(parents=True, exist_ok=True)
        idx_path = self._cache_index_path()
        if not idx_path.exists():
            return {}
        try:
            with open(idx_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_cache_index(self, index):
        if self._cache_path is None:
            return
        idx_path = self._cache_index_path()
        tmp_path = idx_path.with_suffix(".json.tmp")
        with open(tmp_path, "w") as f:
            json.dump(index, f)
        tmp_path.replace(idx_path)

    def _fingerprint(self, h5_path):
        stat = h5_path.stat()
        return {"size": stat.st_size, "mtime": stat.st_mtime}

    def _is_cached(self, h5_path, index):
        entry = index.get(h5_path.name)
        if entry is None:
            return False
        fp = self._fingerprint(h5_path)
        if entry.get("size") != fp["size"] or entry.get("mtime") != fp["mtime"]:
            return False
        out_bin_path = Path(entry.get("output", ""))
        return out_bin_path.exists()

    def _prune_stale_cache(self, index, current_names):
        stale = [name for name in index if name not in current_names]
        for name in stale:
            del index[name]
        return index

    # --- wczytywanie IQ z h5 -------------------------------------------

    def _load_iq_data(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            if self._dataset_name not in f:
                raise KeyError(
                    f"Dataset '{self._dataset_name}' nie istnieje w {h5_path}"
                )
            raw = f[self._dataset_name][:]

        raw = raw.ravel()

        if raw.dtype.fields and {"r", "i"}.issubset(raw.dtype.fields):
            num_samples = raw.shape[0]
            data = np.empty(num_samples * 2, dtype=np.int16)
            data[0::2] = raw["r"].astype(np.int16, copy=False)
            data[1::2] = raw["i"].astype(np.int16, copy=False)
        else:
            data = np.ascontiguousarray(raw, dtype=np.int16)
            num_samples = data.shape[0] // 2

        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        return data, num_samples

    # --- przetwarzanie jednego pliku ------------------------------------

    def _run_gpu(self, data, num_samples):
        lib = self.load_library()
        out_mag = np.empty(self._out_bins, dtype=np.float32)

        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        out_ptr = out_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = lib.sb_process_iq(
            data_ptr,
            ctypes.c_size_t(num_samples),
            ctypes.c_int(self._out_bins),
            out_ptr,
        )
        if ret != 0:
            raise RuntimeError(f"sb_process_iq failed (code {ret})")
        return out_mag

    def process_file(self, h5_path):
        data, num_samples = self._load_iq_data(h5_path)
        if num_samples == 0:
            raise ValueError(f"Brak probek IQ w {h5_path}")
        return self._run_gpu(data, num_samples)

    # --- generator z overlapem I/O <-> GPU ------------------------------

    def process_files(self):
        if self._input_path is None:
            raise ValueError("input_path nie jest ustawiony")

        index = self._load_cache_index()
        h5_files = sorted(self._input_path.glob("*.h5"))
        current_names = {p.name for p in h5_files}
        index = self._prune_stale_cache(index, current_names)

        to_process = []
        for h5_path in h5_files:
            if self._is_cached(h5_path, index):
                # szybka sciezka: wynik juz na dysku, maly plik, czytamy inline
                out_bin_path = Path(index[h5_path.name]["output"])
                yield h5_path, np.fromfile(out_bin_path, dtype=np.float32)
            else:
                to_process.append(h5_path)

        if not to_process:
            self._save_cache_index(index)
            return

        # producer: czyta+konwertuje kolejne pliki w tle, podczas gdy GPU liczy poprzedni
        read_queue = queue.Queue(maxsize=2)
        SENTINEL = object()

        def producer():
            for h5_path in to_process:
                try:
                    data, num_samples = self._load_iq_data(h5_path)
                    read_queue.put((h5_path, data, num_samples, None))
                except Exception as e:
                    read_queue.put((h5_path, None, None, e))
            read_queue.put(SENTINEL)

        thread = threading.Thread(target=producer, daemon=True)
        thread.start()

        try:
            while True:
                item = read_queue.get()
                if item is SENTINEL:
                    break
                h5_path, data, num_samples, err = item
                if err is not None:
                    raise err

                out_mag = self._run_gpu(data, num_samples)

                if self.save_files:
                    if self._output_path is None:
                        raise ValueError(
                            "save_files=True wymaga ustawionego output_path"
                        )
                    self._output_path.mkdir(parents=True, exist_ok=True)
                    out_file = self._output_path / f"{h5_path.stem}.bin"
                    out_mag.tofile(out_file)

                    index[h5_path.name] = {
                        **self._fingerprint(h5_path),
                        "output": str(out_file),
                    }

                yield h5_path, out_mag
        finally:
            self._save_cache_index(index)
            thread.join(timeout=1.0)

    def run(self):
        return list(self.process_files())
