"""GPU-accelerated I/Q HDF5 processing library for SignalBurner."""

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

    def load_library(self) -> ctypes.CDLL:
        """Load and configure the CUDA shared library."""
        if self._lib is None:
            if not self._lib_path.exists():
                raise FileNotFoundError(f"Library not found: {self._lib_path}")
            lib = ctypes.CDLL(str(self._lib_path))

            lib.sb_process_fft.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_fft.restype = ctypes.c_int

            lib.sb_process_cross_fft.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_cross_fft.restype = ctypes.c_int

            lib.sb_process_coherence.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_coherence.restype = ctypes.c_int

            lib.sb_process_pair_full.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_pair_full.restype = ctypes.c_int

            lib.sb_process_triple_all.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
                ctypes.c_int,
            ]
            lib.sb_process_triple_all.restype = ctypes.c_int

            lib.sb_process_baseband_iq_full.argtypes = [
                ctypes.POINTER(ctypes.c_int16),
                ctypes.c_size_t,
                ctypes.c_float,
                ctypes.POINTER(ctypes.c_float),
                ctypes.POINTER(ctypes.c_float),
            ]
            lib.sb_process_baseband_iq_full.restype = ctypes.c_int

            if hasattr(lib, "sb_shutdown"):
                lib.sb_shutdown.argtypes = []
                lib.sb_shutdown.restype = None

            # New function
            if hasattr(lib, "sb_process_triple_cross_spectra"):
                lib.sb_process_triple_cross_spectra.argtypes = [
                    ctypes.POINTER(ctypes.c_int16),  # in1
                    ctypes.POINTER(ctypes.c_int16),  # in2
                    ctypes.POINTER(ctypes.c_int16),  # in3
                    ctypes.c_size_t,  # num_samples
                    ctypes.POINTER(ctypes.c_float),  # out_pow1
                    ctypes.POINTER(ctypes.c_float),  # out_pow2
                    ctypes.POINTER(ctypes.c_float),  # out_pow3
                    ctypes.POINTER(ctypes.c_float),  # out_cross12_real
                    ctypes.POINTER(ctypes.c_float),  # out_cross12_imag
                    ctypes.POINTER(ctypes.c_float),  # out_cross13_real
                    ctypes.POINTER(ctypes.c_float),  # out_cross13_imag
                    ctypes.POINTER(ctypes.c_float),  # out_cross23_real
                    ctypes.POINTER(ctypes.c_float),  # out_cross23_imag
                    ctypes.POINTER(ctypes.c_float),  # out_cross12_phase
                    ctypes.POINTER(ctypes.c_float),  # out_cross13_phase
                    ctypes.POINTER(ctypes.c_float),  # out_cross23_phase
                    ctypes.c_int,  # fft_size
                    ctypes.c_uint,  # flags
                ]
                lib.sb_process_triple_cross_spectra.restype = ctypes.c_int

            self._lib = lib
        return self._lib

    def load_iq_data(self, h5_path: Path) -> Tuple[np.ndarray, int]:
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
                raise ValueError("I/Q data size is not even.")

        if not data.flags["C_CONTIGUOUS"]:
            data = np.ascontiguousarray(data)
        return data, data.size // 2

    def _run_single(self, data: np.ndarray, num_samples: int) -> np.ndarray:
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

    def _run_pair(self, data1, data2, num_samples, lib_func):
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

    def _run_pair_full(self, data1, data2, num_samples):
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

    def _file_cache_key(self, p: Path) -> str:
        return hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:16]

    def get_cache_file(self, h5_path: Path) -> Optional[Path]:
        if self.cache_path is None:
            return None
        self.cache_path.mkdir(parents=True, exist_ok=True)
        key = self._file_cache_key(h5_path)
        return self.cache_path / f"{key}_fft{self.fft_size}.npy"

    def is_cache_valid(self, h5_path: Path, cache_file: Path) -> bool:
        return cache_file.exists() and (
            os.path.getmtime(cache_file) >= os.path.getmtime(h5_path)
        )

    def _pair_base_path(self, p1, p2):
        if not self.use_cache or self.cache_path is None:
            return None
        key1 = self._file_cache_key(p1)
        key2 = self._file_cache_key(p2)
        keys = sorted([key1, key2])
        self.cache_path.mkdir(parents=True, exist_ok=True)
        return self.cache_path / f"{keys[0]}_{keys[1]}_fft{self.fft_size}"

    def _full_cache_path(self, p1, p2):
        base = self._pair_base_path(p1, p2)
        return base.with_suffix(".npz") if base else None

    def _is_pair_cache_fresh(self, p1, p2, cache_file):
        if not cache_file.exists():
            return False
        mtime_c = cache_file.stat().st_mtime
        return mtime_c >= os.path.getmtime(p1) and mtime_c >= os.path.getmtime(p2)

    # Public methods
    def process_file(self, h5_path: Path) -> np.ndarray:
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
        full_npz = self._full_cache_path(h5_path1, h5_path2)
        if full_npz and self._is_pair_cache_fresh(h5_path1, h5_path2, full_npz):
            if self.show_logs:
                print(f"Using full cache for {h5_path1.name} & {h5_path2.name}")
            return np.load(full_npz)["cross_mag"]

        old_cache = self._pair_base_path(h5_path1, h5_path2)
        if old_cache:
            old_cache = old_cache.with_name(old_cache.name + "_cross.npy")
            if self._is_pair_cache_fresh(h5_path1, h5_path2, old_cache):
                if self.show_logs:
                    print(f"Loading cached cross-spectrum...")
                return np.load(old_cache)

        all_res = self.process_pair_all(h5_path1, h5_path2)
        return all_res["cross_magnitude"]

    def process_coherence(self, h5_path1: Path, h5_path2: Path) -> np.ndarray:
        full_npz = self._full_cache_path(h5_path1, h5_path2)
        if full_npz and self._is_pair_cache_fresh(h5_path1, h5_path2, full_npz):
            if self.show_logs:
                print(f"Using full cache for {h5_path1.name} & {h5_path2.name}")
            return np.load(full_npz)["coherence"]

        old_cache = self._pair_base_path(h5_path1, h5_path2)
        if old_cache:
            old_cache = old_cache.with_name(old_cache.name + "_coherence.npy")
            if self._is_pair_cache_fresh(h5_path1, h5_path2, old_cache):
                if self.show_logs:
                    print(f"Loading cached coherence...")
                return np.load(old_cache)

        all_res = self.process_pair_all(h5_path1, h5_path2)
        return all_res["coherence"]

    def process_pair_all(self, h5_path1: Path, h5_path2: Path) -> dict:
        full_npz = self._full_cache_path(h5_path1, h5_path2)
        if full_npz and self._is_pair_cache_fresh(h5_path1, h5_path2, full_npz):
            if self.show_logs:
                print(f"Loading cached full pair...")
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

    def process_triple_all(
        self, h5_path1: Path, h5_path2: Path, h5_path3: Path
    ) -> dict:
        data1, nsamp1 = self.load_iq_data(h5_path1)
        data2, nsamp2 = self.load_iq_data(h5_path2)
        data3, nsamp3 = self.load_iq_data(h5_path3)
        if nsamp1 != nsamp2 or nsamp2 != nsamp3:
            raise ValueError("Sample count mismatch")
        if nsamp1 == 0:
            raise ValueError("No samples")

        n = self.fft_size
        pow1 = np.empty(n, dtype=np.float32)
        pow2 = np.empty(n, dtype=np.float32)
        pow3 = np.empty(n, dtype=np.float32)
        cross12 = np.empty(n, dtype=np.float32)
        cross13 = np.empty(n, dtype=np.float32)
        cross23 = np.empty(n, dtype=np.float32)
        coh12 = np.empty(n, dtype=np.float32)
        coh13 = np.empty(n, dtype=np.float32)
        coh23 = np.empty(n, dtype=np.float32)

        ret = self._lib.sb_process_triple_all(
            data1.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            data2.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            data3.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_size_t(nsamp1),
            pow1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pow2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            pow3.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cross12.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cross13.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            cross23.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            coh12.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            coh13.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            coh23.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(self.fft_size),
        )
        if ret != 0:
            raise RuntimeError(f"sb_process_triple_all failed (code {ret})")

        return {
            "power1": pow1,
            "power2": pow2,
            "power3": pow3,
            "cross12": cross12,
            "cross13": cross13,
            "cross23": cross23,
            "coherence12": coh12,
            "coherence13": coh13,
            "coherence23": coh23,
        }

    def process_baseband_iq(
        self,
        h5_path: Path,
        target_freq_hz: float = 0.0,
        fs: float = 25e6,
    ) -> complex:
        """Return the average complex phasor over the entire file."""
        data, num_samples = self.load_iq_data(h5_path)
        phase_inc = -2.0 * np.pi * target_freq_hz / fs

        real = ctypes.c_float(0.0)
        imag = ctypes.c_float(0.0)

        ret = self._lib.sb_process_baseband_iq_full(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_size_t(num_samples),
            ctypes.c_float(phase_inc),
            ctypes.byref(real),
            ctypes.byref(imag),
        )
        if ret != 0:
            raise RuntimeError(f"sb_process_baseband_iq_full failed (code {ret})")
        return complex(real.value, imag.value)

    def process_fft_files(self, folder: Path) -> List[Tuple[Path, np.ndarray]]:
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
                    del f[key]
                ds = f.create_dataset(key, data=arr)
                if ds_attrs:
                    for k, v in ds_attrs.items():
                        ds.attrs[k] = v
            if metadata:
                f.attrs["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                for k, v in metadata.items():
                    f.attrs[k] = v
        return h5_path

    def shutdown(self) -> None:
        if hasattr(self._lib, "sb_shutdown"):
            self._lib.sb_shutdown()

    def clean_cache(self, max_age_minutes: int = 30) -> int:
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

    # ------------------------------------------------------------------
    # NEW: process_triple_cross_spectra
    # ------------------------------------------------------------------
    def process_triple_cross_spectra(
        self,
        h5_path1: Path,
        h5_path2: Path,
        h5_path3: Path,
        compute_power: bool = True,
        compute_cross_spectrum: bool = True,
        compute_phase: bool = False,
    ) -> dict:
        """
        Process three I/Q files and return cross-spectra, power spectra,
        and optionally phases – without coherence calculation.

        Parameters
        ----------
        compute_power : if True, return 'power1', 'power2', 'power3'
        compute_cross_spectrum : if True, return 'cross12_real', 'cross12_imag', etc.
        compute_phase : if True, return 'cross12_phase', etc.

        Returns
        -------
        dict with the requested arrays.
        """
        if not hasattr(self._lib, "sb_process_triple_cross_spectra"):
            raise RuntimeError(
                "Library does not support sb_process_triple_cross_spectra"
            )

        data1, nsamp1 = self.load_iq_data(h5_path1)
        data2, nsamp2 = self.load_iq_data(h5_path2)
        data3, nsamp3 = self.load_iq_data(h5_path3)
        if nsamp1 != nsamp2 or nsamp2 != nsamp3:
            raise ValueError("Sample count mismatch")
        if nsamp1 == 0:
            raise ValueError("No samples")

        n = self.fft_size

        # Prepare arrays (only if requested, otherwise None)
        pow1 = np.empty(n, dtype=np.float32) if compute_power else None
        pow2 = np.empty(n, dtype=np.float32) if compute_power else None
        pow3 = np.empty(n, dtype=np.float32) if compute_power else None

        cr12_real = np.empty(n, dtype=np.float32) if compute_cross_spectrum else None
        cr12_imag = np.empty(n, dtype=np.float32) if compute_cross_spectrum else None
        cr13_real = np.empty(n, dtype=np.float32) if compute_cross_spectrum else None
        cr13_imag = np.empty(n, dtype=np.float32) if compute_cross_spectrum else None
        cr23_real = np.empty(n, dtype=np.float32) if compute_cross_spectrum else None
        cr23_imag = np.empty(n, dtype=np.float32) if compute_cross_spectrum else None

        ph12 = np.empty(n, dtype=np.float32) if compute_phase else None
        ph13 = np.empty(n, dtype=np.float32) if compute_phase else None
        ph23 = np.empty(n, dtype=np.float32) if compute_phase else None

        flags = 0
        if compute_power:
            flags |= 0x01
        if compute_cross_spectrum:
            flags |= 0x02
        if compute_phase:
            flags |= 0x04

        # Helper: pass NULL pointer if array is None
        def ptr(arr):
            return (
                arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
                if arr is not None
                else ctypes.POINTER(ctypes.c_float)()
            )

        ret = self._lib.sb_process_triple_cross_spectra(
            data1.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            data2.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            data3.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),
            ctypes.c_size_t(nsamp1),
            ptr(pow1),
            ptr(pow2),
            ptr(pow3),
            ptr(cr12_real),
            ptr(cr12_imag),
            ptr(cr13_real),
            ptr(cr13_imag),
            ptr(cr23_real),
            ptr(cr23_imag),
            ptr(ph12),
            ptr(ph13),
            ptr(ph23),
            ctypes.c_int(n),
            ctypes.c_uint(flags),
        )
        if ret != 0:
            raise RuntimeError(f"sb_process_triple_cross_spectra failed (code {ret})")

        result = {}
        if compute_power:
            result.update(power1=pow1, power2=pow2, power3=pow3)
        if compute_cross_spectrum:
            result.update(
                cross12_real=cr12_real,
                cross12_imag=cr12_imag,
                cross13_real=cr13_real,
                cross13_imag=cr13_imag,
                cross23_real=cr23_real,
                cross23_imag=cr23_imag,
            )
        if compute_phase:
            result.update(
                cross12_phase=ph12,
                cross13_phase=ph13,
                cross23_phase=ph23,
            )
        return result
