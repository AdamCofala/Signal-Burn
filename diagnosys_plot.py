#!/usr/bin/env python3
"""
Render spectrograms, cross-spectrum amplitude, and clean phase plots.

Inputs:
  --input        :  directory with minute_*.h5 files (spectrograms + cross-spectra)
  --phase-csv    :  phase.csv from file_processor (clean phase plots)


python3 diagnosys_plot.py --input /pool/signal_storage/output/minute_h5/ --output images/ --freq-downsample 5 --time-downsample 60


"""

import argparse
import csv
import re
import datetime
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import h5py
from scipy.signal import butter, filtfilt

# Default settings
DEFAULT_OUTPUT_DIR = Path("./images")
DEFAULT_FS = 25_000_000
DEFAULT_FFT_SIZE = 262144
DEFAULT_FREQ_DOWNSAMPLE = 1
DEFAULT_TIME_DOWNSAMPLE = 1
DEFAULT_PHASE_DOWNSAMPLE = 1
DEFAULT_VMIN = 5
DEFAULT_VMAX = 95
DEFAULT_DPI = 300
CLEAN_WINDOW_SEC = 60.0
CLEAN_PERCENTILE = 5.0
DEFAULT_FILTER_CUTOFF = 0.007
DEFAULT_FILTER_ORDER = 4

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


def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return b, a


def lowpass_filter(data, cutoff, fs, order=4):
    b, a = butter_lowpass(cutoff, fs, order)
    return filtfilt(b, a, data)


def find_minute_files(input_dir: Path) -> List[Path]:
    input_dir = Path(input_dir)
    files = sorted(
        input_dir.glob("minute_*.h5"),
        key=lambda p: int(re.search(r"minute_(\d+)\.h5", p.name).group(1)),
    )
    return files


def filter_minute_files(
    files: List[Path],
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
) -> List[Path]:
    if start_ts is None and end_ts is None:
        return files
    filtered = []
    for f in files:
        minute_start = int(re.search(r"minute_(\d+)\.h5", f.name).group(1))
        minute_end = minute_start + 59
        if start_ts is not None and minute_end < start_ts:
            continue
        if end_ts is not None and minute_start > end_ts:
            continue
        filtered.append(f)
    return filtered


def load_minute_data(
    files: List[Path],
    start_ts: Optional[int] = None,
    end_ts: Optional[int] = None,
    time_downsample: int = 1,
) -> Tuple[np.ndarray, dict]:
    all_timestamps = []
    data_arrays = defaultdict(list)

    for fp in files:
        with h5py.File(fp, "r") as f:
            ts = f["timestamps"][:]
            mask = np.ones(len(ts), dtype=bool)
            if start_ts is not None:
                mask &= ts >= start_ts
            if end_ts is not None:
                mask &= ts <= end_ts
            if not np.any(mask):
                continue

            ts_filtered = ts[mask]
            all_timestamps.append(ts_filtered)

            def collect_datasets(name, obj):
                if isinstance(obj, h5py.Dataset) and name != "timestamps":
                    arr = obj[:]
                    data_arrays[name].append(arr[mask])

            f.visititems(collect_datasets)

    if not all_timestamps:
        return np.array([]), {}

    timestamps = np.concatenate(all_timestamps)
    sort_idx = np.argsort(timestamps)
    timestamps = timestamps[sort_idx]

    result = {}
    for key, arrays in data_arrays.items():
        concat = np.concatenate(arrays, axis=0)
        result[key] = concat[sort_idx]

    if time_downsample > 1:
        timestamps = timestamps[::time_downsample]
        for key in result:
            result[key] = result[key][::time_downsample]

    return timestamps, result


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


def plot_spectrogram(
    ax, data, timestamps, freq_mhz, vmin_p=5, vmax_p=95, title="", cmap="jet"
):
    db = 10 * np.log10(data + 1e-12)
    vmin, vmax = np.percentile(db, vmin_p), np.percentile(db, vmax_p)
    dt = [datetime.datetime.fromtimestamp(t, tz=LOCAL_TZ) for t in timestamps]
    extent = [
        mdates.date2num(dt[0]),
        mdates.date2num(dt[-1]),
        freq_mhz[0],
        freq_mhz[-1],
    ]
    im = ax.imshow(
        db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
        rasterized=True,
    )
    style_freq_axis(ax, freq_mhz)
    style_time_axis(ax, timestamps)
    ax.set_title(title)
    ax.grid(which="major", axis="both", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.set_box_aspect(9 / 16)
    return im


def plot_cross_amplitude(ax, data, timestamps, freq_mhz, title=""):
    db = 10 * np.log10(data + 1e-12)
    vmin, vmax = np.percentile(db, 5), np.percentile(db, 95)
    dt = [datetime.datetime.fromtimestamp(t, tz=LOCAL_TZ) for t in timestamps]
    extent = [
        mdates.date2num(dt[0]),
        mdates.date2num(dt[-1]),
        freq_mhz[0],
        freq_mhz[-1],
    ]
    im = ax.imshow(
        db.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
        interpolation="bilinear",
        rasterized=True,
    )
    style_freq_axis(ax, freq_mhz)
    style_time_axis(ax, timestamps)
    ax.set_title(title)
    ax.grid(which="major", axis="both", linestyle=":", linewidth=0.4, alpha=0.4)
    ax.set_box_aspect(9 / 16)
    return im


# ----------------------------------------------------------------------
# Phase CSV plotting – always clean phase with raw residual background
# ----------------------------------------------------------------------
def plot_phases(
    phase_csv: Path,
    output_dir: Path,
    time_downsample=1,
    clean_window_sec=CLEAN_WINDOW_SEC,
    clean_percentile=CLEAN_PERCENTILE,
    filter_cutoff=DEFAULT_FILTER_CUTOFF,
    filter_order=DEFAULT_FILTER_ORDER,
):
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
    times_dt = [
        datetime.datetime.fromtimestamp(ts, tz=LOCAL_TZ) for ts in timestamps_plot
    ]

    print("Computing carrier-only clean phase...")
    clean_data = {}
    fs_clean = 1.0 / clean_window_sec

    for ch in range(1, num_channels + 1):
        raw_phase = np.array(data[ch]["phase"])[plot_idx]

        t_full = timestamps_unix
        ph_full = np.array(data[ch]["phase"])
        amp_full = np.array(data[ch]["amplitude"])
        t_clean, ph_clean = clean_phase_data(
            t_full,
            ph_full,
            amp_full,
            window_sec=clean_window_sec,
            percentile=clean_percentile,
        )

        ph_clean_filtered = None
        try:
            ph_clean_filtered = lowpass_filter(
                ph_clean, filter_cutoff, fs_clean, filter_order
            )
        except ValueError as e:
            print(
                f"Channel {ch}: {e}. Skipping low-pass filter, using unfiltered clean phase."
            )
            ph_clean_filtered = ph_clean

        clean_data[ch] = (t_clean, ph_clean, ph_clean_filtered)

        if len(t_clean) == 0:
            print(f"Warning: no clean phase computed for channel {ch}.")
            continue

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(
            times_dt,
            raw_phase,
            color="gray",
            linewidth=0.2,
            alpha=0.4,
            label="raw residual",
            rasterized=True,
        )
        dt_clean = [datetime.datetime.fromtimestamp(ts, tz=LOCAL_TZ) for ts in t_clean]
        label = (
            f"clean + LP filter (cutoff {filter_cutoff} Hz)"
            if ph_clean_filtered is not ph_clean
            else "clean (no filter)"
        )
        ax.plot(
            dt_clean,
            ph_clean_filtered,
            color="tab:blue",
            linewidth=0.8,
            alpha=0.95,
            label=label,
            rasterized=True,
        )
        ax.set_ylabel("Phase [deg]")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)
        ax.set_title(f"Channel {ch} - Clean Phase (carrier only)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)
        style_time_axis(ax, timestamps_plot, fmt="%H:%M:%S")
        fig.tight_layout()
        fig.savefig(output_dir / f"phase_cha{ch}_clean.png", dpi=DEFAULT_DPI)
        plt.close(fig)

    # Save CSVs
    clean_csv_path = output_dir / "phase_clean.csv"
    with open(clean_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp_ms"]
        for ch in range(1, num_channels + 1):
            header.append(f"cha{ch}_clean_phase_deg")
        writer.writerow(header)
        all_times = set()
        for ch in range(1, num_channels + 1):
            all_times.update(clean_data[ch][0])
        all_times = sorted(all_times)
        for t in all_times:
            row = [int(t * 1000)]
            for ch in range(1, num_channels + 1):
                t_ch, ph_ch, _ = clean_data[ch]
                if len(t_ch) == 0:
                    row.append("")
                    continue
                idx = np.argmin(np.abs(t_ch - t))
                if abs(t_ch[idx] - t) < clean_window_sec * 0.5:
                    row.append(f"{ph_ch[idx]:.6f}")
                else:
                    row.append("")
            writer.writerow(row)
    print(f"Unfiltered clean phase CSV saved to {clean_csv_path}")

    filt_csv_path = output_dir / "phase_clean_filtered.csv"
    with open(filt_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["timestamp_ms"]
        for ch in range(1, num_channels + 1):
            header.append(f"cha{ch}_clean_filtered_phase_deg")
        writer.writerow(header)
        all_times = set()
        for ch in range(1, num_channels + 1):
            all_times.update(clean_data[ch][0])
        all_times = sorted(all_times)
        for t in all_times:
            row = [int(t * 1000)]
            for ch in range(1, num_channels + 1):
                t_ch, _, ph_filt = clean_data[ch]
                if len(t_ch) == 0:
                    row.append("")
                    continue
                idx = np.argmin(np.abs(t_ch - t))
                if abs(t_ch[idx] - t) < clean_window_sec * 0.5:
                    row.append(f"{ph_filt[idx]:.6f}")
                else:
                    row.append("")
            writer.writerow(row)
    print(f"Filtered clean phase CSV saved to {filt_csv_path}")
    print(f"Clean phase plots saved to {output_dir}")


def clean_phase_data(timestamps, phase, amplitude, window_sec=60.0, percentile=10.0):
    t = np.asarray(timestamps, dtype=np.float64)
    p = np.asarray(phase, dtype=np.float64)
    a = np.asarray(amplitude, dtype=np.float64)
    if len(t) < 2:
        return np.array([]), np.array([])
    t_start = t[0]
    t_end = t[-1]
    window_starts = np.arange(t_start, t_end, window_sec)
    times_out = window_starts + window_sec / 2.0
    phase_out = np.full_like(times_out, np.nan, dtype=np.float64)
    idx_left = np.searchsorted(t, window_starts, side="left")
    idx_right = np.searchsorted(t, window_starts + window_sec, side="right")
    for i in range(len(window_starts)):
        l = idx_left[i]
        r = idx_right[i]
        if r - l < 5:
            continue
        amp_win = a[l:r]
        thresh = np.percentile(amp_win, percentile)
        mask = amp_win < thresh
        if not np.any(mask):
            continue
        phase_out[i] = np.mean(p[l:r][mask])
    valid = ~np.isnan(phase_out)
    return times_out[valid], phase_out[valid]


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Render spectrograms, cross-spectrum amplitude, and clean phase."
    )
    parser.add_argument(
        "--input", type=Path, default=None, help="Directory with minute_*.h5 files"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--fft-size", type=int, default=DEFAULT_FFT_SIZE)
    parser.add_argument("--freq-downsample", type=int, default=1)
    parser.add_argument("--time-downsample", type=int, default=1)
    parser.add_argument("--vmin-percentile", type=float, default=DEFAULT_VMIN)
    parser.add_argument("--vmax-percentile", type=float, default=DEFAULT_VMAX)
    parser.add_argument("--phase-csv", type=Path, help="Path to phase.csv")
    parser.add_argument(
        "--phase-time-downsample",
        type=int,
        default=DEFAULT_PHASE_DOWNSAMPLE,
        help="Render only every Nth sample of the phase CSV",
    )
    parser.add_argument(
        "--clean-window",
        type=float,
        default=CLEAN_WINDOW_SEC,
        help="Rolling window length in seconds for clean phase averaging",
    )
    parser.add_argument(
        "--clean-percentile",
        type=float,
        default=CLEAN_PERCENTILE,
        help="Amplitude percentile threshold for selecting carrier-only samples",
    )
    parser.add_argument(
        "--filter-cutoff",
        type=float,
        default=DEFAULT_FILTER_CUTOFF,
        help="Low-pass filter cutoff frequency in Hz",
    )
    parser.add_argument(
        "--filter-order",
        type=int,
        default=DEFAULT_FILTER_ORDER,
        help="Order of the Butterworth low-pass filter",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--start-ts",
        type=int,
        default=None,
        help="Start Unix timestamp (inclusive) for data filtering",
    )
    parser.add_argument(
        "--end-ts",
        type=int,
        default=None,
        help="End Unix timestamp (inclusive) for data filtering",
    )
    parser.add_argument(
        "--no-spectrograms", action="store_true", help="Skip spectrogram plots"
    )
    parser.add_argument(
        "--no-cross-amplitude",
        action="store_true",
        help="Skip cross-spectrum amplitude plots",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase CSV – always clean
    if args.phase_csv:
        plot_phases(
            args.phase_csv,
            output_dir,
            time_downsample=args.phase_time_downsample,
            clean_window_sec=args.clean_window,
            clean_percentile=args.clean_percentile,
            filter_cutoff=args.filter_cutoff,
            filter_order=args.filter_order,
        )

    # Minute-level H5 data
    if args.input:
        files = find_minute_files(args.input)
        files = filter_minute_files(files, args.start_ts, args.end_ts)
        if not files:
            print(f"No minute_*.h5 files found in {args.input} matching time range.")
            return

        print(f"Loading data from {len(files)} minute files...")
        timestamps, data = load_minute_data(
            files,
            start_ts=args.start_ts,
            end_ts=args.end_ts,
            time_downsample=args.time_downsample,
        )
        if len(timestamps) == 0:
            print("No data in selected time range.")
            return

        freq_mhz_full = build_freq_axis(args.fft_size, args.fs)
        if args.freq_downsample > 1:
            freq_mhz = downsample_freq(
                freq_mhz_full[np.newaxis, :], args.freq_downsample
            ).squeeze()
        else:
            freq_mhz = freq_mhz_full

        # Spectrograms
        if not args.no_spectrograms:
            for ch in range(1, 4):
                key = f"cha{ch}/fft"
                if key not in data:
                    print(f"Warning: {key} missing in data, skipping.")
                    continue
                fft_data = data[key]
                if args.freq_downsample > 1:
                    fft_data = downsample_freq(fft_data, args.freq_downsample)

                fig, ax = plt.subplots(figsize=(16, 9))
                im = plot_spectrogram(
                    ax,
                    fft_data,
                    timestamps,
                    freq_mhz,
                    vmin_p=args.vmin_percentile,
                    vmax_p=args.vmax_percentile,
                    title=f"Channel {ch} Spectrogram",
                )
                plt.colorbar(im, ax=ax, label="Power [dB]")
                fig.savefig(output_dir / f"channel_{ch}_spectrogram.png", dpi=args.dpi)
                plt.close(fig)
                print(f"Channel {ch} spectrogram saved.")

        # Cross-spectrum amplitude
        if not args.no_cross_amplitude:
            for i, j in [(1, 2), (1, 3), (2, 3)]:
                real_key = f"pairs/{i}{j}/real"
                imag_key = f"pairs/{i}{j}/imag"
                if real_key not in data or imag_key not in data:
                    print(f"Warning: cross data missing for {i}{j}, skipping.")
                    continue
                amp = np.sqrt(
                    data[real_key].astype(np.float64) ** 2
                    + data[imag_key].astype(np.float64) ** 2
                ).astype(np.float32)
                if args.freq_downsample > 1:
                    amp = downsample_freq(amp, args.freq_downsample)

                fig, ax = plt.subplots(figsize=(16, 9))
                im = plot_cross_amplitude(
                    ax,
                    amp,
                    timestamps,
                    freq_mhz,
                    title=f"Cross-spectrum Amplitude: Ch {i} vs Ch {j}",
                )
                plt.colorbar(im, ax=ax, label="Amplitude [dB]")
                fig.savefig(output_dir / f"cross_amplitude_{i}_{j}.png", dpi=args.dpi)
                plt.close(fig)
                print(f"Cross amplitude {i}-{j} saved.")

        print(f"Plots saved to {output_dir}")

    elif not args.phase_csv:
        print("Nothing to plot. Provide --input and/or --phase-csv.")
        return


if __name__ == "__main__":
    main()
