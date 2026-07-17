"""GPU‑accelerated I/Q HDF5 processing library (SignalBurner)."""

import ctypes
from pathlib import Path
import h5py
import numpy as np
import os
import time
from typing import Optional, List, Tuple


class SignalBurner:
    """Process I/Q HDF5 files with CUDA‑accelerated FFT, cross‑spectrum,
    and coherence computations.

    Parameters
    -
    fft_size : int
        Number of FFT bins (default 8192).
    dataset_name : str
        HDF5 dataset name (default ``"rf_data"``).
    lib_path : Path or None
        Path to ``libsb_core.so``.  If None, uses ``bin/libsb_core.so``
        relative to the repository root.
    use_cache : bool
        Whether to cache results on disk.
    cache_path : Path or None
        Directory for cached ``.npy`` files.
    save_files : bool
        (Unused reserved flag).
    output_path : Path or None
        (Unused reserved flag).
    show_logs : bool
        Print progress messages.
    """

    def __init__(
        self,
        save_files: bool = False,
        output_path: Optional[Path] = None,
        cache_path: Optional[Path] = None,
        dataset_name: Optional[str] = None,
        lib_path: Optional[Path] = None,
        fft_size: int = 8192,
        use_cache: bool = True,
        show_logs: bool = False,
    ) -> None:
        self.output_path = output_path
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else Path(__file__).parent.parent / "cache"
        )
        self.dataset_name = dataset_name
        self.save_files = save_files
        self.fft_size = fft_size
        self.use_cache = use_cache
        self.show_logs = show_logs

        self._lib_path = (
            Path(lib_path)
            if lib_path is not None
            else Path(__file__).parent.parent / "bin" / "libsb_core.so"
        )
        self._lib = None

    # -- GPU library loading
    def load_library(self) -> ctypes.CDLL:
        """Load and configure the CUDA shared library."""
        if self._lib is None:
            if not self._lib_path.exists():
                raise FileNotFoundError(f"Library not found: {self._lib_path}")
            lib = ctypes.CDLL(str(self._lib_path))

            # sb_process_fft
            lib.sb_process_fft.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_fft.restype = ctypes.c_int

            # sb_process_cross_fft
            lib.sb_process_cross_fft.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_cross_fft.restype = ctypes.c_int

            # sb_process_coherence
            lib.sb_process_coherence.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_coherence.restype = ctypes.c_int

            if hasattr(lib, "sb_shutdown"):
                lib.sb_shutdown.argtypes = []
                lib.sb_shutdown.restype = None

            self._lib = lib
        return self._lib

    # -- I/O helpers
    def load_iq_data(self, h5_path: Path) -> Tuple[np.ndarray, int]:
        """Read interleaved I/Q int16 data from an HDF5 file.

        Returns (data, num_samples).
        """
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

    # -- GPU runners
    def _run_single(self, data: np.ndarray, num_samples: int) -> np.ndarray:
        """Run single‑channel FFT on GPU."""
        self.load_library()
        out = np.empty(self.fft_size, dtype=np.float32)
        ret = self._lib.sb_process_fft(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_size_t(num_samples),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.fft_size),
        )
        if ret != 0:
            raise RuntimeError(f"sb_process_fft failed (code {ret})")
        return out

    def _run_pair(
        self, data1: np.ndarray, data2: np.ndarray, num_samples: int, lib_func
    ) -> np.ndarray:
        """Run a two-channel GPU operation (cross or coherence)."""
        self.load_library()
        out = np.empty(self.fft_size, dtype=np.float32)
        ret = lib_func(
            data1.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            data2.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_size_t(num_samples),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.fft_size),
        )
        if ret != 0:
            raise RuntimeError(f"GPU pair operation failed (code {ret})")
        return out

    # -- Cache helpers -------
    def get_cache_file(self, h5_path: Path) -> Optional[Path]:
        """Return cache path for a single‑file FFT."""
        if self.cache_path is None:
            return None
        self.cache_path.mkdir(parents=True, exist_ok=True)
        return self.cache_path / f"{h5_path.stem}_fft{self.fft_size}.npy"

    def is_cache_valid(self, h5_path: Path, cache_file: Path) -> bool:
        """Check whether a cache file is newer than its source HDF5."""
        return cache_file.exists() and (
            os.path.getmtime(cache_file) >= os.path.getmtime(h5_path)
        )

    def _pair_cache_file(self, p1: Path, p2: Path, tag: str) -> Optional[Path]:
        """Return cache path for a pair operation."""
        if not self.use_cache or self.cache_path is None:
            return None
        stems = sorted([p1.stem, p2.stem])
        return self.cache_path / f"{stems[0]}_{stems[1]}_{tag}_fft{self.fft_size}.npy"

    # -- Public processing methods
    def process_file(self, h5_path: Path) -> np.ndarray:
        """Compute averaged power spectrum for a single HDF5 file."""
        if self.use_cache:
            cache_file = self.get_cache_file(h5_path)
            if cache_file and self.is_cache_valid(h5_path, cache_file):
                if self.show_logs:
                    print(f"Loading cached result for {h5_path.name}...")
                return np.load(cache_file)

        data, num_samples = self.load_iq_data(h5_path)
        if num_samples == 0:
            raise ValueError(f"File {h5_path} contains no samples.")
        out = self._run_single(data, num_samples)

        if self.use_cache:
            cache_file = self.get_cache_file(h5_path)
            if cache_file:
                np.save(cache_file, out)
        return out

    def process_cross(self, h5_path1: Path, h5_path2: Path) -> np.ndarray:
        """Cross‑spectrum magnitude between two HDF5 files."""
        cache_file = self._pair_cache_file(h5_path1, h5_path2, "cross")
        if cache_file and cache_file.exists():
            mtime1 = os.path.getmtime(h5_path1)
            mtime2 = os.path.getmtime(h5_path2)
            if cache_file.stat().st_mtime >= max(mtime1, mtime2):
                if self.show_logs:
                    print(
                        f"Loading cached cross‑spectrum for {h5_path1.name} & {h5_path2.name}..."
                    )
                return np.load(cache_file)

        data1, nsamp1 = self.load_iq_data(h5_path1)
        data2, nsamp2 = self.load_iq_data(h5_path2)
        if nsamp1 != nsamp2:
            raise ValueError(f"Sample count mismatch: {nsamp1} vs {nsamp2}")
        if nsamp1 == 0:
            raise ValueError("No samples in input files.")

        out = self._run_pair(data1, data2, nsamp1, self._lib.sb_process_cross_fft)

        if cache_file:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, out)
        return out

    def process_coherence(self, h5_path1: Path, h5_path2: Path) -> np.ndarray:
        """Magnitude‑squared coherence between two HDF5 files."""
        cache_file = self._pair_cache_file(h5_path1, h5_path2, "coherence")
        if cache_file and cache_file.exists():
            mtime1 = os.path.getmtime(h5_path1)
            mtime2 = os.path.getmtime(h5_path2)
            if cache_file.stat().st_mtime >= max(mtime1, mtime2):
                if self.show_logs:
                    print(
                        f"Loading cached coherence for {h5_path1.name} & {h5_path2.name}..."
                    )
                return np.load(cache_file)

        data1, nsamp1 = self.load_iq_data(h5_path1)
        data2, nsamp2 = self.load_iq_data(h5_path2)
        if nsamp1 != nsamp2:
            raise ValueError(f"Sample count mismatch: {nsamp1} vs {nsamp2}")
        if nsamp1 == 0:
            raise ValueError("No samples in input files.")

        out = self._run_pair(data1, data2, nsamp1, self._lib.sb_process_coherence)

        if cache_file:
            self.cache_path.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, out)
        return out

    # -- Batch processing ---
    def process_fft_files(self, folder: Path) -> List[Tuple[Path, np.ndarray]]:
        """Process all ``.h5`` files in *folder* and return
        ``[(file, spectrum), ...]``."""
        folder = Path(folder)
        if not folder.is_dir():
            raise NotADirectoryError(f"{folder} is not a directory")
        h5_files = sorted(folder.glob("*.h5"), key=lambda p: p.name)
        results = []
        for i, fp in enumerate(h5_files):
            try:
                spectrum = self.process_file(fp)
                results.append((fp, spectrum))
                if self.show_logs:
                    print(f"[{i + 1}/{len(h5_files)}] {fp.name}")
            except Exception as e:
                print(f"Error processing {fp.name}: {e}")
        return results

    def process_cross_files(
        self, folder1: Path, folder2: Path
    ) -> List[Tuple[Path, Path, np.ndarray]]:
        """Compute cross‑spectra for paired files in two folders.
        Files are paired by sorted name."""
        folder1 = Path(folder1)
        folder2 = Path(folder2)
        if not folder1.is_dir() or not folder2.is_dir():
            raise NotADirectoryError("Both arguments must be directories")
        files1 = sorted(folder1.glob("*.h5"), key=lambda p: p.name)
        files2 = sorted(folder2.glob("*.h5"), key=lambda p: p.name)

        if len(files1) != len(files2):
            print(
                f"Warning: folder1 has {len(files1)} files, folder2 has {len(files2)}. "
                f"Processing only the first {min(len(files1), len(files2))} pairs."
            )
        min_len = min(len(files1), len(files2))
        results = []
        for i in range(min_len):
            fp1, fp2 = files1[i], files2[i]
            try:
                cross = self.process_cross(fp1, fp2)
                results.append((fp1, fp2, cross))
                if self.show_logs:
                    print(f"[{i + 1}/{min_len}] {fp1.name} & {fp2.name}")
            except Exception as e:
                print(f"Error processing pair ({fp1.name}, {fp2.name}): {e}")
        return results

    # -- Cleanup ----
    def shutdown(self) -> None:
        """Release GPU resources."""
        lib = self.load_library()
        if hasattr(lib, "sb_shutdown"):
            lib.sb_shutdown()

    def clean_cache(self, max_age_minutes: int = 30) -> int:
        """Remove cached ``.npy`` files older than *max_age_minutes*."""
        if self.cache_path is None or not self.cache_path.exists():
            return 0
        now = time.time()
        deleted = 0
        for f in self.cache_path.glob("*.npy"):
            if now - f.stat().st_mtime > max_age_minutes * 60:
                f.unlink()
                deleted += 1
        return deleted
