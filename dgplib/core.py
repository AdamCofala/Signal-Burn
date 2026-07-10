import subprocess
import h5py
import numpy as np
from pathlib import Path

class SignalBurnManager:
    def __init__(self, binary_path="./bin/dgp_core"):
        self.binary_path = Path(binary_path).resolve()

    def run_batch(self, input_dir, output_dir, dataset_name="rf_data", N=2048):
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Znajdujemy wszystkie pliki .h5 w folderze
        for h5_file in in_path.glob("*.h5"):
            output_file = out_path / f"{h5_file.stem}_spectrum.bin"
            self._process_single(h5_file, output_file, dataset_name, N)

    def _process_single(self, h5_path, output_path, dataset_name, N):
        with h5py.File(h5_path, "r") as f:
            raw = f[dataset_name][:]

            if raw.dtype.fields and {"r", "i"}.issubset(raw.dtype.fields):
                data = np.empty(raw.shape[0] * 2, dtype=np.int16)
                data[0::2] = raw["r"].astype(np.int16, copy=False).ravel()
                data[1::2] = raw["i"].astype(np.int16, copy=False).ravel()
            else:
                data = np.asarray(raw, dtype=np.int16).ravel()

        size = data.size
        cmd = [str(self.binary_path), "-", str(output_path), str(size), str(N)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

        try:
            proc.stdin.write(data.tobytes())
            proc.stdin.close()
            rc = proc.wait()
            if rc != 0:
                stderr = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
                raise RuntimeError(
                    f"Processing failed for {h5_file} (exit code {rc}). {stderr}".strip()
                )
        except BrokenPipeError:
            stderr = proc.stderr.read().decode(errors="ignore") if proc.stderr else ""
            rc = proc.wait()
            raise RuntimeError(
                f"Downstream process closed the pipe for {h5_file} "
                f"(exit code {rc}). {stderr}".strip()
            )
        print(f"Processed: {output_path.name}")