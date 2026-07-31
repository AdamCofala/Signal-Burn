"""GPU-accelerated I/Q HDF5 processing library (SignalBurner)."""

import ctypes
import hashlib
from pathlib import Path
import h5py
import numpy as np
import os
import time
from typing import Optional, List, Tuple


class SignalBurner:
    """Process I/Q HDF5 files with CUDA-accelerated FFT, cross-spectrum,
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
        Directory for cached ``.npy`` / ``.npz`` files.
    show_logs : bool
        Print progress messages.
    """

    def __init__(
        self,
        cache_path: Path | None = None,
        dataset_name: Path | None = None,
        lib_path: Path | None = None,
        fft_size: int = 8192,
        use_cache: bool = True,
        show_logs: bool = False,
    ) -> None:

        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else Path(__file__).parent.parent / "cache"
        )
        self.dataset_name = dataset_name
        self.fft_size = fft_size
        self.use_cache = use_cache
        self.show_logs = show_logs

        self._lib_path = (
            Path(lib_path)
            if lib_path is not None
            else Path(__file__).parent.parent / "bin" / "libsb_core.so"
        )
        self._lib = None
        self.load_library()

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

            # sb_process_pair_full
            lib.sb_process_pair_full.argtypes = [
                ctypes.POINTER(ctypes.c_int16),  # in1
                ctypes.POINTER(ctypes.c_int16),  # in2
                ctypes.c_size_t,  # num_samples
                ctypes.POINTER(ctypes.c_float),  # out_pow1
                ctypes.POINTER(ctypes.c_float),  # out_pow2
                ctypes.POINTER(ctypes.c_float),  # out_cross_mag
                ctypes.POINTER(ctypes.c_float),  # out_coherence
                ctypes.c_int,  # fft_size
            ]
            lib.sb_process_pair_full.restype = ctypes.c_int

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
            ds = f[self.dataset_name]
            raw = np.empty(ds.shape, dtype=ds.dtype)
            ds.read_direct(raw)

        if raw.dtype.fields and {"r", "i"}.issubset(raw.dtype.fields):
            if raw.dtype.itemsize == 4 and raw.flags["C_CONTIGUOUS"]:
                data = raw.view(np.int16).ravel()
            else:
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
        """Run single-channel FFT on GPU."""
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

    def _run_pair_full(
        self, data1: np.ndarray, data2: np.ndarray, num_samples: int
    ) -> dict:
        """Execute the combined GPU operation returning all spectra."""
        pow1 = np.empty(self.fft_size, dtype=np.float32)
        pow2 = np.empty(self.fft_size, dtype=np.float32)
        cross_mag = np.empty(self.fft_size, dtype=np.float32)
        coherence = np.empty(self.fft_size, dtype=np.float32)

        ret = self._lib.sb_process_pair_full(
            data1.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            data2.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_size_t(num_samples),
            pow1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pow2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cross_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            coherence.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.fft_size),
        )
        if ret != 0:
            raise RuntimeError(f"sb_process_pair_full failed (code {ret})")

        return {
            "power1": pow1,
            "power2": pow2,
            "cross_magnitude": cross_mag,
            "coherence": coherence,
        }

    # -- Cache helpers -------
    def _file_cache_key(self, p: Path) -> str:
        """Unique-per-file cache key.

        Using only ``Path.stem`` is NOT safe here: files coming from
        different source directories (e.g. separate channels cha1/cha2/cha3)
        can legitimately share the exact same basename/timestamp (that's
        even required for pairing them by second). Using the bare stem as a
        cache key then makes two *different* files collide onto the same
        cache path, silently returning one channel's cached result for
        another channel's request. We disambiguate by folding in a short
        hash of the resolved parent directory.
        """
        parent_hash = hashlib.sha1(str(p.resolve().parent).encode()).hexdigest()[:8]
        return f"{p.stem}_{parent_hash}"

    def get_cache_file(self, h5_path: Path) -> Optional[Path]:
        """Return cache path for a single-file FFT (`.npy`)."""
        if self.cache_path is None:
            return None
        self.cache_path.mkdir(parents=True, exist_ok=True)
        key = self._file_cache_key(h5_path)
        return self.cache_path / f"{key}_fft{self.fft_size}.npy"

    def is_cache_valid(self, h5_path: Path, cache_file: Path) -> bool:
        """Check whether a cache file is newer than its source HDF5."""
        return cache_file.exists() and (
            os.path.getmtime(cache_file) >= os.path.getmtime(h5_path)
        )

    def _pair_base_path(self, p1: Path, p2: Path) -> Optional[Path]:
        """Return base cache path (without extension) for a pair.
        The actual files will be <base>.npz (full) or <base>_<product>.npy."""
        if not self.use_cache or self.cache_path is None:
            return None
        keys = sorted([self._file_cache_key(p1), self._file_cache_key(p2)])
        self.cache_path.mkdir(parents=True, exist_ok=True)
        return self.cache_path / f"{keys[0]}_{keys[1]}_fft{self.fft_size}"

    def _full_cache_path(self, p1: Path, p2: Path) -> Optional[Path]:
        """Path to the combined `.npz` cache file."""
        base = self._pair_base_path(p1, p2)
        return base.with_suffix(".npz") if base else None

    def _is_pair_cache_fresh(self, p1: Path, p2: Path, cache_file: Path) -> bool:
        """Check if a pair cache file is newer than both source HDF5 files."""
        if not cache_file.exists():
            return False
        mtime_c = cache_file.stat().st_mtime
        return mtime_c >= os.path.getmtime(p1) and mtime_c >= os.path.getmtime(p2)

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
        """Cross-spectrum magnitude between two HDF5 files."""
        # 1. Try the combined cache first
        full_npz = self._full_cache_path(h5_path1, h5_path2)
        if full_npz and self._is_pair_cache_fresh(h5_path1, h5_path2, full_npz):
            if self.show_logs:
                print(f"Using full cache for {h5_path1.name} & {h5_path2.name}")
            data = np.load(full_npz)
            return data["cross_mag"]

        # 2. Fallback to old single-product cache
        old_cache = self._pair_base_path(h5_path1, h5_path2)
        if old_cache:
            old_cache = old_cache.with_name(old_cache.name + "_cross.npy")
            if self._is_pair_cache_fresh(h5_path1, h5_path2, old_cache):
                if self.show_logs:
                    print(
                        f"Loading cached cross-spectrum for {h5_path1.name} & {h5_path2.name}..."
                    )
                return np.load(old_cache)

        # 3. Compute full pair and cache it
        all_res = self.process_pair_all(h5_path1, h5_path2)
        return all_res["cross_magnitude"]

    def process_coherence(self, h5_path1: Path, h5_path2: Path) -> np.ndarray:
        """Magnitude-squared coherence between two HDF5 files."""
        # 1. Try the combined cache first
        full_npz = self._full_cache_path(h5_path1, h5_path2)
        if full_npz and self._is_pair_cache_fresh(h5_path1, h5_path2, full_npz):
            if self.show_logs:
                print(f"Using full cache for {h5_path1.name} & {h5_path2.name}")
            data = np.load(full_npz)
            return data["coherence"]

        # 2. Fallback to old single-product cache
        old_cache = self._pair_base_path(h5_path1, h5_path2)
        if old_cache:
            old_cache = old_cache.with_name(old_cache.name + "_coherence.npy")
            if self._is_pair_cache_fresh(h5_path1, h5_path2, old_cache):
                if self.show_logs:
                    print(
                        f"Loading cached coherence for {h5_path1.name} & {h5_path2.name}..."
                    )
                return np.load(old_cache)

        # 3. Compute full pair and cache it
        all_res = self.process_pair_all(h5_path1, h5_path2)
        return all_res["coherence"]

    def process_pair_all(self, h5_path1: Path, h5_path2: Path) -> dict:
        """Compute all pair products in a single GPU pass.

        Returns a dictionary with keys:
            'power1'         - power spectrum of channel 1
            'power2'         - power spectrum of channel 2
            'cross_magnitude'- cross-spectrum magnitude
            'coherence'      - magnitude-squared coherence
        """
        full_npz = self._full_cache_path(h5_path1, h5_path2)
        if full_npz and self._is_pair_cache_fresh(h5_path1, h5_path2, full_npz):
            if self.show_logs:
                print(
                    f"Loading cached full pair for {h5_path1.name} & {h5_path2.name}..."
                )
            data = np.load(full_npz)
            return {
                "power1": data["power1"],
                "power2": data["power2"],
                "cross_magnitude": data["cross_mag"],
                "coherence": data["coherence"],
            }

        data1, nsamp1 = self.load_iq_data(h5_path1)
        data2, nsamp2 = self.load_iq_data(h5_path2)
        if nsamp1 != nsamp2:
            raise ValueError(f"Sample count mismatch: {nsamp1} vs {nsamp2}")
        if nsamp1 == 0:
            raise ValueError("No samples in input files.")

        results = self._run_pair_full(data1, data2, nsamp1)

        if self.use_cache and full_npz:
            np.savez(
                full_npz,
                power1=results["power1"],
                power2=results["power2"],
                cross_mag=results["cross_magnitude"],
                coherence=results["coherence"],
            )
        return results

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
        """Compute cross-spectra for paired files in two folders.
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

    def save_to_h5(
        self,
        h5_path: Path,
        datasets: dict,
        metadata: Optional[dict] = None,
        mode: str = "a",
    ) -> Path:
        """Universal save — works for FFT spectra, cross-spectra, coherence,
        or any other array result.

        Parameters
        ----------
        h5_path : Path
            Target .h5 file.
        datasets : dict
            Maps a path -> array, e.g.:
                {
                    "fft/file1": spectrum,
                    "cross/file1_file2": cross_spec,
                    "coherence/file1_file2": coh,
                }
            "/" in the key auto-creates nested groups (h5py handles this).
            A value can also be (array, attrs_dict) to attach per-dataset
            metadata, e.g. {"fft/file1": (spectrum, {"fft_size": 8192})}.
        metadata : dict, optional
            File-level attrs (sample_rate, fft_size, timestamp, source files...).
        mode : str
            "a" (default) = append/create, "w" = overwrite whole file.
        """
        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_path, mode) as f:
            for key, value in datasets.items():
                if isinstance(value, tuple):
                    arr, ds_attrs = value
                else:
                    arr, ds_attrs = value, None

                arr = np.asarray(arr)
                if key in f:
                    del f[key]  # overwrite if it already exists
                ds = f.create_dataset(key, data=arr)

                if ds_attrs:
                    for k, v in ds_attrs.items():
                        ds.attrs[k] = v

            if metadata:
                f.attrs["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                for k, v in metadata.items():
                    f.attrs[k] = v

        return h5_path

    # -- Cleanup ----
    def shutdown(self) -> None:
        """Release GPU resources."""
        if hasattr(self._lib, "sb_shutdown"):
            self._lib.sb_shutdown()

    def clean_cache(self, max_age_minutes: int = 30) -> int:
        """Remove cached ``.npy`` / ``.npz`` files older than *max_age_minutes*."""
        if self.cache_path is None or not self.cache_path.exists():
            return 0
        now = time.time()
        deleted = 0
        for pattern in ("*.npy", "*.npz"):
            for f in self.cache_path.glob(pattern):
                if now - f.stat().st_mtime > max_age_minutes * 60:
                    f.unlink()
                    deleted += 1
        return deleted
