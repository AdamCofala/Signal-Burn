import ctypes
from pathlib import Path

import h5py
import numpy as np

_lib = None

def _get_lib(lib_path="./bin/libdgp_core.so"):
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(str(Path(lib_path).resolve()))
        _lib.process_iq.argtypes = [
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_float),
        ]
        _lib.process_iq.restype = ctypes.c_int
    return _lib


class SignalBurnManager:
    def __init__(self, lib_path="./bin/libdgp_core.so"):
        self.lib = _get_lib(lib_path)

    def run_batch(self, input_dir, output_dir, dataset_name="rf_data"):
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        for h5_file in in_path.glob("*.h5"):
            output_file = out_path / f"{h5_file.stem}_spectrum.bin"
            self._process_single(h5_file, output_file, dataset_name)

    def _process_single(self, h5_path, output_path, dataset_name):
        with h5py.File(h5_path, "r") as f:
            raw = f[dataset_name][:]

            if raw.dtype.fields and {"r", "i"}.issubset(raw.dtype.fields):
                num_samples = raw.shape[0]
                data = np.empty(num_samples * 2, dtype=np.int16)
                data[0::2] = raw["r"].astype(np.int16, copy=False).ravel()
                data[1::2] = raw["i"].astype(np.int16, copy=False).ravel()
            else:
                data = np.asarray(raw, dtype=np.int16).ravel()
                num_samples = data.shape[0] // 2

        out_mag = np.empty(num_samples, dtype=np.float32)

        data_ptr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_int16))
        out_ptr = out_mag.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self.lib.process_iq(data_ptr, ctypes.c_size_t(num_samples), out_ptr)
        if ret != 0:
            raise RuntimeError(f"process_iq failed for {h5_path} (code {ret})")

        out_mag.tofile(output_path)