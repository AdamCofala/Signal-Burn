#!/usr/bin/env python3
"""
Render spectrograms and coherence plots from minute HDF5 snapshots.

Usage:
    python3 render_spectrograms.py --input /path/to/minute_h5/ --output ./images/
"""

import argparse
import re
import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np


def find_h5_files(input_dir):
    """Return sorted list of minute HDF5 files."""
    input_dir = Path(input_dir)
    files = sorted(
        input_dir.glob("minute_*.h5"),
        key=lambda p: int(re.search(r"minute_(\d+)\.h5", p.name).group(1)),
    )
    return files


def load_data(files, fs=25e6, fft_size=262144):
    """
    Read all HDF5 files and return:
        timestamps : 1D array of UNIX timestamps (float)
        fft_data   : dict channel_index -> 2D array (time, freq) [linear power]
        coh_data   : dict (ch_i, ch_j) -> 2D array (time, freq) [0-1]
    """
    num_channels = 3
    fft_data = {ch: [] for ch in range(1, num_channels + 1)}
    coh_data = {(1, 2): [], (1, 3): [], (2, 3): []}
    timestamps = []

    for fpath in files:
        with h5py.File(fpath, "r") as f:
            ts = f["timestamps"][0]
            timestamps.append(ts)

            for ch in range(1, num_channels + 1):
                arr = f[f"cha{ch}/fft"][:].squeeze()
                fft_data[ch].append(arr)

            for i, j in coh_data.keys():
                arr = f[f"pairs/{i}{j}/coherence"][:].squeeze()
                coh_data[(i, j)].append(arr)

    timestamps = np.array(timestamps)
    for ch in fft_data:
        fft_data[ch] = np.stack(fft_data[ch], axis=0)
    for pair in coh_data:
        coh_data[pair] = np.stack(coh_data[pair], axis=0)

    return timestamps, fft_data, coh_data


def build_freq_axis(fft_size, fs):
    """Return frequency axis in MHz (after fftshift) – used only for extent."""
    freq = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / fs))
    return freq / 1e6


def downsample_freq(arr, factor):
    """Downsample frequency axis by averaging bins."""
    if factor <= 1:
        return arr
    n_freq = arr.shape[-1]
    new_len = n_freq // factor
    arr = arr[..., : new_len * factor]
    return arr.reshape(arr.shape[:-1] + (new_len, factor)).mean(axis=-1)


def plot_spectrogram(
    ax,
    data_linear,
    timestamps,
    freq_axis_mhz,
    vmin_perc=5,
    vmax_perc=95,
    title="",
    cmap="jet",
):
    """Plot a 2D spectrogram (dB). Y-axis relabelled to 0–25 MHz."""
    data_db = 10 * np.log10(data_linear + 1e-12)
    vmin = np.percentile(data_db, vmin_perc)
    vmax = np.percentile(data_db, vmax_perc)

    # Use original -12.5..12.5 extent
    dt = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    extent = [
        mdates.date2num(dt[0]),
        mdates.date2num(dt[-1]),
        freq_axis_mhz[0],
        freq_axis_mhz[-1],
    ]

    im = ax.imshow(
        data_db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    # Shift Y-axis labels by +12.5 MHz so the display reads 0–25 MHz
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x + 12.5:.1f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time [HH:MM]")
    ax.set_ylabel("Frequency [MHz]")
    ax.set_title(title)
    return im


def plot_coherence(ax, coh_data, timestamps, freq_axis_mhz, title=""):
    """Plot coherence (0-1). Y-axis relabelled to 0–25 MHz."""
    dt = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]
    extent = [
        mdates.date2num(dt[0]),
        mdates.date2num(dt[-1]),
        freq_axis_mhz[0],
        freq_axis_mhz[-1],
    ]

    im = ax.imshow(
        coh_data.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="plasma",
    )
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{x + 12.5:.1f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time [HH:MM]")
    ax.set_ylabel("Frequency [MHz]")
    ax.set_title(title)
    return im


def main():
    parser = argparse.ArgumentParser(
        description="Render spectrograms from minute HDF5 files."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Directory with minute_*.h5 files"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./images"),
        help="Output directory for plots",
    )
    parser.add_argument(
        "--fs", type=float, default=25e6, help="Sampling frequency [Hz]"
    )
    parser.add_argument("--fft-size", type=int, default=262144, help="FFT size")
    parser.add_argument(
        "--freq-downsample", type=int, default=1, help="Frequency downsampling factor"
    )
    parser.add_argument(
        "--vmin-percentile", type=float, default=5, help="Lower percentile for dB scale"
    )
    parser.add_argument(
        "--vmax-percentile",
        type=float,
        default=95,
        help="Upper percentile for dB scale",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = find_h5_files(args.input)
    if not files:
        print("No minute_*.h5 files found.")
        return

    print(f"Loading {len(files)} files...")
    timestamps, fft_data, coh_data = load_data(
        files, fs=args.fs, fft_size=args.fft_size
    )

    freq_axis_full = build_freq_axis(args.fft_size, args.fs)
    if args.freq_downsample > 1:
        freq_axis = downsample_freq(
            freq_axis_full[np.newaxis, :], args.freq_downsample
        ).squeeze()
        for ch in fft_data:
            fft_data[ch] = downsample_freq(fft_data[ch], args.freq_downsample)
        for pair in coh_data:
            coh_data[pair] = downsample_freq(coh_data[pair], args.freq_downsample)
    else:
        freq_axis = freq_axis_full

    # Plot each channel spectrogram
    for ch in range(1, 4):
        if ch in fft_data and fft_data[ch].size > 0:
            fig, ax = plt.subplots(figsize=(16, 9))
            im = plot_spectrogram(
                ax,
                fft_data[ch],
                timestamps,
                freq_axis,
                vmin_perc=args.vmin_percentile,
                vmax_perc=args.vmax_percentile,
                title=f"Channel {ch} Spectrogram",
            )
            plt.colorbar(im, ax=ax, label="Power [dB]")
            fig.tight_layout()
            fig.savefig(output_dir / f"channel_{ch}_spectrogram.png", dpi=300)
            plt.close(fig)
        print(
            f"Channel {ch} spectrogram saved to {output_dir / f'channel_{ch}_spectrogram.png'}"
        )

    # Plot coherence pairs
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        pair_key = (i, j)
        if pair_key in coh_data and coh_data[pair_key].size > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            im = plot_coherence(
                ax,
                coh_data[pair_key],
                timestamps,
                freq_axis,
                title=f"Coherence: Channel {i} vs Channel {j}",
            )
            plt.colorbar(im, ax=ax, label="Coherence")
            fig.tight_layout()
            fig.savefig(output_dir / f"coherence_{i}_{j}.png", dpi=150)
            plt.close(fig)
        print(
            f"Coherence plot for channels {i} and {j} saved to {output_dir / f'coherence_{i}_{j}.png'}"
        )

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
