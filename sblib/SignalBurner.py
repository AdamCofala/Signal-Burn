import ctypes
from pathlib import Path
import h5py
import numpy as np
import os


class SignalBurner:
    """
    Class SignarBurner is responsible for processing I/Q data from HDF5 files
    using a CUDA library. It loads the data, processes it on the GPU, and optionally
    saves the results. The class handles loading the CUDA library, managing input and
    output paths, and ensuring that the data is in the correct format for processing.
    """

    def __init__(
        self,
        input_path=None,  # Input path to the directory containing HDF5 files
        save_files=False,  # Whether to save the processed files or not
        output_path=None,  # Output path to save the processed files (if save_files is True)
        cache_path=None,  # Path to cache intermediate results
        dataset_name=None,  # Name of the dataset in the HDF5 files to process
        lib_path=None,  # Path to the CUDA library (if None, defaults to 'bin/libsb_core.so')
        fft_size=8192,  # FFT window size for processing (number of bins)
        use_cache=True,  # Whether to use caching for intermediate results
    ) -> None:

        self.input_path = input_path
        self.output_path = output_path
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else Path(__file__).parent.parent / "cache"
        )
        self.dataset_name = (
            dataset_name
            if dataset_name is not None
            else print("Dataset name not provided.")
        )
        self.save_files = save_files
        self.fft_size = fft_size
        self.use_cache = use_cache

        self._lib_path = (
            Path(lib_path)
            if lib_path is not None
            else Path(__file__).parent.parent / "bin" / "libsb_core.so"
        )

        self._lib = None

    @property
    def input_path(self):
        return self._input_path

    @input_path.setter
    def input_path(self, value):
        self._input_path = Path(value) if value is not None else None

    def load_library(self):
        """
        Load the CUDA library for processing I/Q data.
        """

        if self._lib is None:
            if not self._lib_path.exists():
                raise FileNotFoundError(f"Library not found: {self._lib_path}")

            lib = ctypes.CDLL(str(self._lib_path))

            # sb_process_fft: (int16_t* in, size_t num_samples, float* out, int fft_size)
            lib.sb_process_fft.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_fft.restype = ctypes.c_int

            # sb_shutdown
            if hasattr(lib, "sb_shutdown"):
                lib.sb_shutdown.argtypes = []
                lib.sb_shutdown.restype = None

            self._lib = lib
        return self._lib

    def load_iq_data(self, h5_path):
        with h5py.File(h5_path, "r") as f:
            raw = f[self.dataset_name][:]

        if raw.dtype.fields and {"r", "i"}.issubset(raw.dtype.fields):
            num_samples = raw.shape[0]
            data = np.empty(num_samples * 2, dtype=np.int16)
            data[0::2] = raw["r"].ravel()
            data[1::2] = raw["i"].ravel()
        else:
            data = np.asarray(raw, dtype=np.int16).ravel()
            if data.size % 2 != 0:
                raise ValueError(
                    "I/Q data size is not even, cannot reshape into complex pairs."
                )

        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)

        return data, data.size // 2

    def run_gpu(self, data, num_samples) -> np.ndarray:
        lib = self.load_library()

        out_mag = np.empty(self.fft_size, dtype=np.float32)

        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        out_ptr = out_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = lib.sb_process_fft(
            data_ptr, ctypes.c_size_t(num_samples), out_ptr, ctypes.c_int(self.fft_size)
        )

        if ret != 0:
            raise RuntimeError(f"sb_process_fft failed ( {ret})")

        return out_mag

    def get_cache_file(self, h5_path):
        if self.cache_path is None:
            print("Cache path is not set. Caching is disabled.")
            return None

        self.cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_path / f"{h5_path.stem}_fft{self.fft_size}.npy"
        return cache_file

    def is_cache_valid(self, h5_path: Path, cache_file: Path) -> bool:
        if not cache_file.exists():
            return False
        src_mtime = os.path.getmtime(h5_path)
        cache_mtime = os.path.getmtime(cache_file)
        return cache_mtime >= src_mtime

    def process_file(self, h5_path: Path) -> np.ndarray:

        if self.use_cache:
            cache_file = self.get_cache_file(h5_path)
            if cache_file and self.is_cache_valid(h5_path, cache_file):
                print(f"Loading cached result for {h5_path.name}...")
                return np.load(cache_file)

        data, num_samples = self.load_iq_data(h5_path)

        if num_samples == 0:
            raise ValueError(f"File {h5_path} contains no samples.")

        out = self.run_gpu(data, num_samples)

        if self.use_cache:
            cache_file = self.get_cache_file(h5_path)
            if cache_file:
                np.save(cache_file, out)

        return out

    def shutdown(self):
        """
        Shutdown the CUDA library and free GPU resources.
        Really important to call this after processing ALL files,
        the CUDA library was designed to allocate GPU resources once and
        reuse them for all files for faster processing. If you call this after each file,
        it will drasticly decrease performance because of reallocation gpu resources.
        """
        lib = self.load_library()
        if hasattr(lib, "sb_shutdown"):
            lib.sb_shutdown()

    def run(self) -> list[tuple[Path, np.ndarray]]:
        if self._input_path is None:
            raise ValueError("input_path is not set.")

        h5_files = list(self._input_path.glob("*.h5"))
        # If u want u can sort them by name or date here
        h5_files = sorted(h5_files, key=lambda x: x.name)
        results = []

        for i, h5_path in enumerate(h5_files):
            try:
                mag = self.process_file(h5_path)
                print(f"Processed {h5_path.name}: {i + 1}/{len(h5_files)}")
                results.append((h5_path, mag))
            except Exception as e:
                print(f"Error {h5_path.name}: {e}")

        return results

    def clean_cache(self, max_age_days: int = 30) -> int:
        if self.cache_path is None or not self.cache_path.exists():
            return 0

        import time

        now = time.time()
        deleted = 0
        for f in self.cache_path.glob("*.npy"):
            if now - f.stat().st_mtime > max_age_days * 86400:
                f.unlink()
                deleted += 1
        return deleted
