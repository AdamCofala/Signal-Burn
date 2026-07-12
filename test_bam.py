import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

path = "cha1/2026-07-10T12-00-00/rf@1783686190.000.h5"
dataset_name = "rf_data"

with h5py.File(path, "r") as f:
    raw = f[dataset_name][:]
    print("shape:", raw.shape, "dtype:", raw.dtype)

# spłaszcz (25_000_000, 1) -> (25_000_000,)
raw = raw.ravel()

iq = raw["r"].astype(np.float32) + 1j * raw["i"].astype(np.float32)

fs = 25_000_000

f_axis, t_axis, Zxx = signal.stft(
    iq, fs=fs, nperseg=4096, noverlap=2048, return_onesided=False
)

plt.figure(figsize=(12, 6))
plt.pcolormesh(
    t_axis,
    np.fft.fftshift(f_axis),
    np.fft.fftshift(20 * np.log10(np.abs(Zxx) + 1e-12), axes=0),
    shading="auto",
)
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [s] (w ramach jednego pliku)")
plt.colorbar(label="dB")
plt.title(f"Spektrogram wewnątrz jednego pliku: {path}")
plt.tight_layout()
plt.savefig("diagnostic_spectrogram.png", dpi=150)
print("Zapisano diagnostic_spectrogram.png")
