#!/usr/bin/env python3
"""
Render spectrograms and coherence plots from minute HDF5 snapshots.
Additionally plot phase vs time from the live processor's unified phase.csv
(timestamp_ms, cha{ch}_phase_deg, cha{ch}_amplitude - written at up to
500 Hz, with the oscillator/frequency-offset drift already subtracted
online before it's written).

Usage:
    python3 diagnosys_plot.py --input /path/to/minute_h5/ --output ./images/
    python3 diagnosys_plot.py --phase-csv /path/to/phase.csv --output ./images/
    python3 diagnosys_plot.py --aggregated-h5 /path/to/aggregated.h5 --output ./images/
"""

import argparse
import csv
import re
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import h5py

# ---------- defaults ----------
DEFAULT_INPUT_DIR = Path("/pool/signal_storage/output/minute_h5")
DEFAULT_OUTPUT_DIR = Path("./images")
DEFAULT_FS = 25_000_000
DEFAULT_FFT_SIZE = 262144
DEFAULT_FREQ_DOWNSAMPLE = 1
DEFAULT_TIME_DOWNSAMPLE = 1
DEFAULT_PHASE_DOWNSAMPLE = 1
DEFAULT_VMIN = 5
DEFAULT_VMAX = 95
DEFAULT_DPI = 300
DEFAULT_WORKERS = 4
DEFAULT_CHUNK_CACHE_MB = 100
# ------------------------------

LOCAL_TZ = ZoneInfo("Europe/Warsaw")

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": 10,
        "figure.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
    }
)


def find_h5_files(input_dir):
    input_dir = Path(input_dir)
    files = sorted(
        input_dir.glob("minute_*.h5"),
        key=lambda p: int(re.search(r"minute_(\d+)\.h5", p.name).group(1)),
    )
    return files


def load_single_file_metadata(filepath):
    with h5py.File(filepath, 'r') as f:
        return f['timestamps'][0]


def load_single_file_data(filepath, dataset_path, fft_size, chunk_cache_nbytes=None):
    kwargs = {}
    if chunk_cache_nbytes is not None:
        kwargs['rdcc_nbytes'] = chunk_cache_nbytes
        kwargs['rdcc_nslots'] = 20000
    with h5py.File(filepath, 'r', **kwargs) as f:
        data = f[dataset_path][:].squeeze().astype(np.float32)
        if data.shape[-1] != fft_size:
            if data.shape[-1] < fft_size:
                tmp = np.full(fft_size, np.nan, dtype=np.float32)
                tmp[:data.shape[-1]] = data
                data = tmp
            else:
                data = data[..., :fft_size]
        return data


def load_all_data_parallel(files, fft_size, workers=4, chunk_cache_mb=100):
    files = list(files)
    n = len(files)
    chunk_cache_nbytes = chunk_cache_mb * 1024 * 1024

    # 1. Wczytaj timestampy równolegle
    timestamps = np.empty(n, dtype=np.float64)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(load_single_file_metadata, f): idx for idx, f in enumerate(files)}
        for fut in as_completed(futures):
            idx = futures[fut]
            timestamps[idx] = fut.result()

    # Posortuj wg czasu
    sort_idx = np.argsort(timestamps)
    files_sorted = [files[i] for i in sort_idx]
    timestamps_sorted = timestamps[sort_idx]

    datasets = {
        'cha1': 'cha1/fft',
        'cha2': 'cha2/fft',
        'cha3': 'cha3/fft',
        'coh12': 'pairs/12/coherence',
        'coh13': 'pairs/13/coherence',
        'coh23': 'pairs/23/coherence',
    }

    results = {'timestamps': timestamps_sorted}
    for key, path in datasets.items():
        arr = np.empty((n, fft_size), dtype=np.float32)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for idx, fp in enumerate(files_sorted):
                fut = pool.submit(load_single_file_data, fp, path, fft_size, chunk_cache_nbytes)
                futures.append((idx, fut))
            for idx, fut in futures:
                arr[idx] = fut.result()
        results[key] = arr
    return results


def load_from_aggregated(agg_path):
    with h5py.File(agg_path, 'r') as f:
        return {
            'timestamps': f['timestamps'][:],
            'cha1': f['cha1/fft'][:],
            'cha2': f['cha2/fft'][:],
            'cha3': f['cha3/fft'][:],
            'coh12': f['pairs/12/coherence'][:],
            'coh13': f['pairs/13/coherence'][:],
            'coh23': f['pairs/23/coherence'][:],
        }


def build_freq_axis(fft_size=DEFAULT_FFT_SIZE, fs=DEFAULT_FS):
    freq = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / fs))
    return freq.astype(np.float32) / 1e6


def downsample_freq(arr, factor):
    if factor <= 1:
        return arr
    n_freq = arr.shape[-1]
    new_len = n_freq // factor
    arr = arr[..., : new_len * factor]
    return arr.reshape(arr.shape[:-1] + (new_len, factor)).mean(axis=-1)


def build_time_locator(timestamps):
    timestamps = np.asarray(timestamps)
    span_sec = timestamps[-1] - timestamps[0]
    if span_sec <= 60:
        return mdates.SecondLocator(bysecond=range(0, 60, 5), tz=LOCAL_TZ)
    elif span_sec <= 60 * 60:
        return mdates.MinuteLocator(byminute=range(0, 60, 5), tz=LOCAL_TZ)
    elif span_sec <= 3 * 60 * 60:
        return mdates.MinuteLocator(byminute=range(0, 60, 15), tz=LOCAL_TZ)
    elif span_sec <= 12 * 60 * 60:
        return mdates.MinuteLocator(byminute=[0, 30], tz=LOCAL_TZ)
    elif span_sec <= 24 * 60 * 60:
        return mdates.HourLocator(byhour=range(0, 24, 3), tz=LOCAL_TZ)
    elif span_sec <= 3 * 24 * 60 * 60:
        return mdates.HourLocator(byhour=range(0, 24, 3), tz=LOCAL_TZ)
    else:
        return mdates.HourLocator(byhour=range(0, 24, 6), tz=LOCAL_TZ)


def add_date_top_axis(ax, timestamps):
    timestamps = np.asarray(timestamps)
    dt_start = datetime.datetime.fromtimestamp(timestamps[0], tz=LOCAL_TZ)
    dt_end = datetime.datetime.fromtimestamp(timestamps[-1], tz=LOCAL_TZ)
    tick_locs = [mdates.date2num(dt_start)]
    tick_labels = [dt_start.strftime("%Y-%m-%d")]
    day = dt_start.date() + datetime.timedelta(days=1)
    while day <= dt_end.date():
        midnight = datetime.datetime.combine(day, datetime.time.min, tzinfo=LOCAL_TZ)
        if dt_start <= midnight <= dt_end:
            tick_locs.append(mdates.date2num(midnight))
            tick_labels.append(midnight.strftime("%Y-%m-%d"))
        day += datetime.timedelta(days=1)
    top_ax = ax.secondary_xaxis("top")
    top_ax.set_xticks(tick_locs)
    top_ax.set_xticklabels(tick_labels)
    top_ax.tick_params(axis="x", labelsize=10)
    return top_ax


def style_time_axis(ax, timestamps, fmt="%H:%M"):
    ax.xaxis.set_major_locator(build_time_locator(timestamps))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=LOCAL_TZ))
    ax.set_xlabel("Time [HH:MM]" if fmt == "%H:%M" else "Time")
    add_date_top_axis(ax, timestamps)


def style_freq_axis(ax, freq_mhz, tick_step_mhz=5.0):
    nyquist_mhz = -freq_mhz[0]
    y_ticks = np.arange(freq_mhz[0], nyquist_mhz + 1e-6, tick_step_mhz)
    ax.set_ylim(freq_mhz[0], nyquist_mhz)
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x + nyquist_mhz:.1f}")
    )
    ax.set_ylabel("Frequency [MHz]")


def plot_spectrogram(ax, data, timestamps, freq_mhz, vmin_p=5, vmax_p=95, title="", cmap="jet"):
    db = 10 * np.log10(data + 1e-12)
    vmin, vmax = np.percentile(db, vmin_p), np.percentile(db, vmax_p)
    dt = [datetime.datetime.fromtimestamp(t, tz=LOCAL_TZ) for t in timestamps]
    extent = [mdates.date2num(dt[0]), mdates.date2num(dt[-1]), freq_mhz[0], freq_mhz[-1]]
    im = ax.imshow(
        db.T, aspect="auto", origin="lower", extent=extent,
        vmin=vmin, vmax=vmax, cmap=cmap, interpolation="bilinear", rasterized=True
    )
    style_freq_axis(ax, freq_mhz)
    style_time_axis(ax, timestamps)
    ax.set_title(title)
    ax.grid(which="major", axis="both", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.set_box_aspect(9 / 16)
    return im


def plot_coherence(ax, coh_data, timestamps, freq_mhz, title=""):
    dt = [datetime.datetime.fromtimestamp(t, tz=LOCAL_TZ) for t in timestamps]
    extent = [mdates.date2num(dt[0]), mdates.date2num(dt[-1]), freq_mhz[0], freq_mhz[-1]]
    im = ax.imshow(
        coh_data.T, aspect="auto", origin="lower", extent=extent,
        vmin=0, vmax=1, cmap="plasma", interpolation="nearest", rasterized=True
    )
    style_freq_axis(ax, freq_mhz)
    style_time_axis(ax, timestamps)
    ax.set_title(title)
    ax.grid(which="major", axis="both", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.set_box_aspect(9 / 16)
    return im


def plot_phases(phase_csv, output_dir, time_downsample=1):
    if not phase_csv.exists():
        print(f"File {phase_csv} not found.")
        return
    times_ms = []
    data = {}
    with open(phase_csv) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        num_channels = 0
        for h in headers:
            if h.startswith("cha") and h.endswith("_phase_deg"):
                ch_num = int(h.split("_")[0][3:])
                num_channels = max(num_channels, ch_num)
        for ch in range(1, num_channels + 1):
            data[ch] = {"phase": [], "amplitude": []}
        for row in reader:
            ts_ms = float(row["timestamp_ms"])
            times_ms.append(ts_ms)
            for ch in range(1, num_channels + 1):
                data[ch]["phase"].append(float(row[f"cha{ch}_phase_deg"]))
                data[ch]["amplitude"].append(float(row[f"cha{ch}_amplitude"]))
    if not times_ms:
        print("No data in phase CSV.")
        return
    timestamps_unix = np.array(times_ms, dtype=np.float64) / 1000.0
    step = max(1, time_downsample)
    plot_idx = slice(None, None, step)
    timestamps_plot = timestamps_unix[plot_idx]
    times_dt = [datetime.datetime.fromtimestamp(ts, tz=LOCAL_TZ) for ts in timestamps_plot]
    for ch in range(1, num_channels + 1):
        phase_vals = np.array(data[ch]["phase"])[plot_idx]
        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(times_dt, phase_vals, color="tab:green", linewidth=0.4, alpha=0.85, rasterized=True)
        ax.set_ylabel("Phase residual [deg]")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_title(f"Channel {ch} - Phase (oscillator drift removed)")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)
        style_time_axis(ax, timestamps_plot, fmt="%H:%M:%S")
        fig.tight_layout()
        fig.savefig(output_dir / f"phase_cha{ch}.png", dpi=DEFAULT_DPI)
        plt.close(fig)
    print(f"Phase plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render spectrograms and phase plots.")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--fft-size", type=int, default=DEFAULT_FFT_SIZE)
    parser.add_argument("--freq-downsample", type=int, default=1)
    parser.add_argument("--time-downsample", type=int, default=1)
    parser.add_argument("--vmin-percentile", type=float, default=DEFAULT_VMIN)
    parser.add_argument("--vmax-percentile", type=float, default=DEFAULT_VMAX)
    parser.add_argument("--phase-csv", type=Path, help="Path to phase.csv")
    parser.add_argument("--phase-time-downsample", type=int, default=DEFAULT_PHASE_DOWNSAMPLE)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--aggregated-h5", type=Path, help="Path to aggregated HDF5 file")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Number of parallel I/O threads")
    parser.add_argument("--chunk-cache-mb", type=int, default=DEFAULT_CHUNK_CACHE_MB, help="Chunk cache per HDF5 file (MB)")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase_csv:
        plot_phases(args.phase_csv, output_dir, time_downsample=args.phase_time_downsample)
        if not args.input and not args.aggregated_h5:
            return

    data = None
    if args.aggregated_h5:
        print(f"Loading aggregated data from {args.aggregated_h5}...")
        data = load_from_aggregated(args.aggregated_h5)
    elif args.input:
        files = find_h5_files(args.input)
        if not files:
            print("No minute_*.h5 files found.")
            return
        print(f"Loading {len(files)} files with {args.workers} workers...")
        data = load_all_data_parallel(
            files, args.fft_size,
            workers=args.workers,
            chunk_cache_mb=args.chunk_cache_mb
        )
    else:
        print("No input data. Use --input, --aggregated-h5 or --phase-csv.")
        return

    timestamps = data['timestamps']
    if args.time_downsample > 1:
        step = args.time_downsample
        timestamps = timestamps[::step]
        for key in data:
            if key != 'timestamps':
                data[key] = data[key][::step]

    freq_mhz_full = build_freq_axis(args.fft_size, args.fs)
    if args.freq_downsample > 1:
        freq_mhz = downsample_freq(freq_mhz_full[np.newaxis, :], args.freq_downsample).squeeze()
        for key in list(data.keys()):
            if key.startswith('cha') or key.startswith('coh'):
                data[key] = downsample_freq(data[key], args.freq_downsample)
    else:
        freq_mhz = freq_mhz_full

    # Generuj wykresy bezpośrednio z danych w pamięci
    for ch in [1, 2, 3]:
        fft_ch = data[f'cha{ch}']
        fig, ax = plt.subplots(figsize=(16, 9))
        im = plot_spectrogram(
            ax, fft_ch, timestamps, freq_mhz,
            vmin_p=args.vmin_percentile, vmax_p=args.vmax_percentile,
            title=f"Channel {ch} Spectrogram"
        )
        plt.colorbar(im, ax=ax, label="Power [dB]")
        fig.savefig(output_dir / f"channel_{ch}_spectrogram.png", dpi=args.dpi)
        plt.close(fig)
        print(f"Channel {ch} spectrogram saved.")

    for i, j in [(1, 2), (1, 3), (2, 3)]:
        coh = data[f'coh{i}{j}']
        fig, ax = plt.subplots(figsize=(16, 9))
        im = plot_coherence(ax, coh, timestamps, freq_mhz, title=f"Coherence: Channel {i} vs Channel {j}")
        plt.colorbar(im, ax=ax, label="Coherence")
        fig.savefig(output_dir / f"coherence_{i}_{j}.png", dpi=args.dpi)
        plt.close(fig)
        print(f"Coherence {i}-{j} saved.")

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()