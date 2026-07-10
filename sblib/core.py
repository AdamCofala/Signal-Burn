import ctypes
from pathlib import Path

import h5py
import numpy as np

_lib = None

__all__ = ["SignalBurner", "load_iq_samples"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _candidate_lib_paths(lib_path=None):
    if lib_path is not None:
        yield Path(lib_path)

    bin_dir = _repo_root() / "bin"
    yield bin_dir / "libsb_core.so"


def _load_symbol(lib):
    symbol = lib.sb_process_iq
    symbol.argtypes = [
        ctypes.POINTER(ctypes.c_int16),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_float),
    ]
    symbol.restype = ctypes.c_int
    return symbol


def _get_lib(lib_path=None):
    global _lib
    if _lib is None:
        resolved_path = None
        for candidate in _candidate_lib_paths(lib_path):
            candidate_path = candidate if candidate.is_absolute() else candidate.resolve()
            if candidate_path.exists():
                resolved_path = candidate_path
                break

        if resolved_path is None:
            resolved_path = next(_candidate_lib_paths(lib_path)).resolve()

        _lib = ctypes.CDLL(str(resolved_path))
        _lib.sb_process_iq = _load_symbol(_lib)
    return _lib


def _normalize_iq(raw):
    if raw.dtype.fields and {"r", "i"}.issubset(raw.dtype.fields):
        num_samples = raw.shape[0]
        data = np.empty(num_samples * 2, dtype=np.int16)
        data[0::2] = np.asarray(raw["r"], dtype=np.int16).ravel()
        data[1::2] = np.asarray(raw["i"], dtype=np.int16).ravel()
        return data

    if np.iscomplexobj(raw):
        real = np.asarray(np.real(raw), dtype=np.int16).ravel()
        imag = np.asarray(np.imag(raw), dtype=np.int16).ravel()
        data = np.empty(real.size * 2, dtype=np.int16)
        data[0::2] = real
        data[1::2] = imag
        return data

    data = np.asarray(raw, dtype=np.int16).ravel()
    if data.size % 2 != 0:
        raise ValueError("Interleaved I/Q data must contain an even number of int16 values")
    return data


def load_iq_samples(h5_path, dataset_name="rf_data"):
    with h5py.File(h5_path, "r") as handle:
        raw = handle[dataset_name][:]
    return _normalize_iq(raw)


class SignalBurner:
    def __init__(self, lib_path=None):
        self.lib = _get_lib(lib_path)

    def process_file(self, h5_path, output_path=None, dataset_name="rf_data"):
        data = load_iq_samples(h5_path, dataset_name=dataset_name)
        num_samples = data.shape[0] // 2
        out_mag = np.empty(num_samples, dtype=np.float32)

        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        out_ptr = out_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self.lib.sb_process_iq(data_ptr, ctypes.c_size_t(num_samples), out_ptr)
        if ret != 0:
            raise RuntimeError(f"sb_process_iq failed for {h5_path} (code {ret})")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            out_mag.tofile(output_path)

        return out_mag

    def run_batch(self, input_dir, output_dir, dataset_name="rf_data"):
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for h5_file in in_path.glob("*.h5"):
            output_file = out_path / f"{h5_file.stem}_spectrum.bin"
            self.process_file(h5_file, output_path=output_file, dataset_name=dataset_name)


