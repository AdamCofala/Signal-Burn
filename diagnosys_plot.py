#!/usr/bin/env python3
"""
Render spectrograms and coherence plots from minute HDF5 snapshots.
Additionally plot phase vs time from the live processor's phase.csv
and fast phase/amplitude samples from phase_fast.csv.

Usage:
    python3 diagnosys_plot.py --input /path/to/minute_h5/ --output ./images/
    python3 diagnosys_plot.py --phase-csv /path/to/phase.csv --output ./images/
    python3 diagnosys_plot.py --phase-fast-csv /path/to/phase_fast.csv --output ./images/
"""

import argparse
import csv
import re
import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

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
DEFAULT_FAST_DOWNSAMPLE = 1
DEFAULT_VMIN = 5
DEFAULT_VMAX = 95
DEFAULT_DPI = 300
# ------------------------------

# All timestamps in the HDF5/CSV files are UNIX time. Every plot is rendered
# in Poland local time, regardless of the timezone configured on the machine
# running this script.
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
    """Return sorted list of minute HDF5 files."""
    input_dir = Path(input_dir)
    files = sorted(
        input_dir.glob("minute_*.h5"),
        key=lambda p: int(re.search(r"minute_(\d+)\.h5", p.name).group(1)),
    )
    return files


def read_timestamps(files, time_downsample=1):
    """Read only the (tiny) timestamp value from each file - cheap even for
    thousands of files, and lets every per-channel/per-pair load below share
    the same time axis without re-reading it."""
    files_to_load = files[::time_downsample]
    timestamps = np.empty(len(files_to_load), dtype=np.float64)
    for idx, fpath in enumerate(files_to_load):
        with h5py.File(fpath, "r") as f:
            timestamps[idx] = f["timestamps"][0]
    return timestamps


def load_channel_fft(files, ch, fft_size=DEFAULT_FFT_SIZE, time_downsample=1):
    """Load only channel `ch`'s FFT data across files - one channel at a
    time, instead of holding all 3 channels (plus all 3 coherence pairs) in
    memory simultaneously like the old load_data() did."""
    files_to_load = files[::time_downsample]
    n_files = len(files_to_load)
    arr = np.empty((n_files, fft_size), dtype=np.float32)
    for idx, fpath in enumerate(files_to_load):
        with h5py.File(fpath, "r") as f:
            arr[idx] = f[f"cha{ch}/fft"][:].squeeze().astype(np.float32)
    return arr


def load_pair_coherence(files, i, j, fft_size=DEFAULT_FFT_SIZE, time_downsample=1):
    """Load only the (i, j) coherence pair across files - one pair at a
    time, for the same reason as load_channel_fft above."""
    files_to_load = files[::time_downsample]
    n_files = len(files_to_load)
    arr = np.empty((n_files, fft_size), dtype=np.float32)
    for idx, fpath in enumerate(files_to_load):
        with h5py.File(fpath, "r") as f:
            arr[idx] = f[f"pairs/{i}{j}/coherence"][:].squeeze().astype(np.float32)
    return arr


def build_freq_axis(fft_size=DEFAULT_FFT_SIZE, fs=DEFAULT_FS):
    """Return frequency axis in MHz (after fftshift)."""
    freq = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1 / fs))
    return freq.astype(np.float32) / 1e6


def downsample_freq(arr, factor):
    """Downsample frequency axis by averaging bins."""
    if factor <= 1:
        return arr
    n_freq = arr.shape[-1]
    new_len = n_freq // factor
    arr = arr[..., : new_len * factor]
    return arr.reshape(arr.shape[:-1] + (new_len, factor)).mean(axis=-1)


def build_time_locator(timestamps):
    """
    Pick an evenly-spaced major tick locator for a time axis based on the
    total time span, aligned to clean clock marks (e.g. :00, :30) in Poland
    local time - not offset from the data's exact start time. Shared by
    every plot type (spectrograms, coherence, phase, fast phase).
    """
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
        return mdates.HourLocator(byhour=range(0, 24, 1), tz=LOCAL_TZ)
    elif span_sec <= 3 * 24 * 60 * 60:
        return mdates.HourLocator(byhour=range(0, 24, 3), tz=LOCAL_TZ)
    else:
        return mdates.HourLocator(byhour=range(0, 24, 6), tz=LOCAL_TZ)


def add_date_top_axis(ax, timestamps):
    """
    Add a secondary x-axis on top showing the calendar date (Poland local),
    with a tick at the data's start and at every midnight crossed - so
    overnight data spanning two dates shows both dates, not only a single
    "00:00 -> next day" tick.
    """
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
    """Shared time-axis treatment: aligned ticks + date top axis."""
    ax.xaxis.set_major_locator(build_time_locator(timestamps))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt, tz=LOCAL_TZ))
    ax.set_xlabel("Time [HH:MM]" if fmt == "%H:%M" else "Time")
    add_date_top_axis(ax, timestamps)


def style_freq_axis(ax, freq_mhz):
    """Force the frequency axis out to the true Nyquist edge on both ends,
    with clean 2.5 MHz ticks - the data's last FFT bin falls one bin short
    of the exact Nyquist frequency, so without this the top-edge (e.g. 25)
    MHz label doesn't reliably appear."""
    nyquist_mhz = -freq_mhz[0]
    y_ticks = np.arange(freq_mhz[0], nyquist_mhz + 1e-6, 2.5)
    ax.set_ylim(freq_mhz[0], nyquist_mhz)
    ax.yaxis.set_major_locator(ticker.FixedLocator(y_ticks))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x + nyquist_mhz:.1f}")
    )
    ax.set_ylabel("Frequency [MHz]")


def plot_spectrogram(
    ax, data, timestamps, freq_mhz, vmin_p=5, vmax_p=95, title="", cmap="jet"
):
    """Plot a 2D spectrogram (dB) with rasterization."""
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

    return im


def plot_coherence(ax, coh_data, timestamps, freq_mhz, title=""):
    """Plot coherence with rasterization."""
    dt = [datetime.datetime.fromtimestamp(t, tz=LOCAL_TZ) for t in timestamps]
    extent = [
        mdates.date2num(dt[0]),
        mdates.date2num(dt[-1]),
        freq_mhz[0],
        freq_mhz[-1],
    ]

    im = ax.imshow(
        coh_data.T,
        aspect="auto",
        origin="lower",
        extent=extent,
        vmin=0,
        vmax=1,
        cmap="plasma",
        interpolation="nearest",
        rasterized=True,
    )

    style_freq_axis(ax, freq_mhz)
    style_time_axis(ax, timestamps)
    ax.set_title(title)
    ax.grid(which="major", axis="both", linestyle=":", linewidth=0.4, alpha=0.4)

    return im


def plot_phases(phase_csv: Path, output_dir: Path):
    """Read phase.csv and produce phase plots for each channel with proper time axis."""
    if not phase_csv.exists():
        print(f"File {phase_csv} not found.")
        return

    timestamps = []
    data = {ch: {"real": [], "imag": []} for ch in range(1, 4)}

    with open(phase_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = int(row["second"])
            timestamps.append(ts)
            for ch in range(1, 4):
                data[ch]["real"].append(float(row[f"cha{ch}_real"]))
                data[ch]["imag"].append(float(row[f"cha{ch}_imag"]))

    if not timestamps:
        print("No data in phase CSV.")
        return

    timestamps = np.array(timestamps, dtype=np.float64)
    times_dt = [datetime.datetime.fromtimestamp(ts, tz=LOCAL_TZ) for ts in timestamps]

    for ch in range(1, 4):
        z = np.array(data[ch]["real"]) + 1j * np.array(data[ch]["imag"])
        phase = np.angle(z, deg=True)

        fig, ax = plt.subplots(figsize=(16, 9))
        ax.plot(
            times_dt, phase, color="tab:blue", linewidth=0.5, alpha=0.9, rasterized=True
        )
        ax.set_ylabel("Phase [deg]")
        ax.set_ylim(-180, 180)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(90))
        ax.set_title(f"Channel {ch} - Phase (1 s average)")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)

        style_time_axis(ax, timestamps)

        fig.tight_layout()
        fig.savefig(output_dir / f"phase_cha{ch}.png", dpi=DEFAULT_DPI)
        plt.close(fig)

    print(f"Phase plots saved to {output_dir}")


def plot_fast_phases(fast_csv: Path, output_dir: Path, time_downsample=1):
    """Read phase_fast.csv and produce high-speed phase and amplitude plots
    with proper time axis.

    time_downsample: plot only every Nth sample. At 500 Hz sampling, a few
    hours of data is millions of points - drawing every single one is the
    main reason these plots crawl. Downsampling for the plot doesn't touch
    the underlying CSV data, just what gets rendered.
    """
    if not fast_csv.exists():
        print(f"File {fast_csv} not found.")
        return

    times_ms = []
    data = {}

    with open(fast_csv) as f:
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
        print("No data in fast phase CSV.")
        return

    timestamps_unix = np.array(times_ms, dtype=np.float64) / 1000.0

    step = max(1, time_downsample)
    plot_idx = slice(None, None, step)
    timestamps_plot = timestamps_unix[plot_idx]
    times_dt = [
        datetime.datetime.fromtimestamp(ts, tz=LOCAL_TZ) for ts in timestamps_plot
    ]
    if step > 1:
        print(
            f"Fast phase: rendering every {step}th sample "
            f"({len(timestamps_plot)}/{len(timestamps_unix)} points) to keep plotting fast."
        )

    for ch in range(1, num_channels + 1):
        phase_vals = np.array(data[ch]["phase"])[plot_idx]
        amp_vals = np.array(data[ch]["amplitude"])[plot_idx]

        # --- Fast phase ---
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(
            times_dt,
            phase_vals,
            color="tab:blue",
            linewidth=0.3,
            alpha=0.8,
            rasterized=True,
        )
        ax.set_ylabel("Phase [deg]")
        ax.set_ylim(-180, 180)
        ax.yaxis.set_major_locator(ticker.MultipleLocator(90))
        ax.set_title(f"Channel {ch} - Fast Phase (50 Hz sampling)")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)

        style_time_axis(ax, timestamps_plot, fmt="%H:%M:%S")

        fig.tight_layout()
        fig.savefig(output_dir / f"fast_phase_cha{ch}.png", dpi=DEFAULT_DPI)
        plt.close(fig)

        # --- Fast amplitude (collected but never plotted before) ---
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(
            times_dt,
            amp_vals,
            color="tab:red",
            linewidth=0.3,
            alpha=0.8,
            rasterized=True,
        )
        ax.set_ylabel("Amplitude")
        ax.set_title(f"Channel {ch} - Fast Amplitude (500 Hz sampling)")
        ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.4)

        style_time_axis(ax, timestamps_plot, fmt="%H:%M:%S")

        fig.tight_layout()
        fig.savefig(output_dir / f"fast_amplitude_cha{ch}.png", dpi=DEFAULT_DPI)
        plt.close(fig)

    print(f"Fast phase/amplitude plots saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Render spectrograms and phase plots.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fs", type=float, default=DEFAULT_FS)
    parser.add_argument("--fft-size", type=int, default=DEFAULT_FFT_SIZE)
    parser.add_argument("--freq-downsample", type=int, default=1)
    parser.add_argument("--time-downsample", type=int, default=1)
    parser.add_argument("--vmin-percentile", type=float, default=DEFAULT_VMIN)
    parser.add_argument("--vmax-percentile", type=float, default=DEFAULT_VMAX)
    parser.add_argument("--phase-csv", type=Path, help="Path to phase.csv")
    parser.add_argument(
        "--phase-fast-csv",
        type=Path,
        help="Path to phase_fast.csv for fast phase/amplitude plots",
    )
    parser.add_argument(
        "--fast-time-downsample",
        type=int,
        default=DEFAULT_FAST_DOWNSAMPLE,
        help="Render only every Nth sample of the fast phase CSV, to keep "
        "plotting fast for long high-rate captures (default: %(default)s, "
        "i.e. no downsampling)",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase_fast_csv:
        plot_fast_phases(
            args.phase_fast_csv, output_dir, time_downsample=args.fast_time_downsample
        )

    if args.phase_csv:
        plot_phases(args.phase_csv, output_dir)
        if not args.input:
            return

    if not args.input:
        if not args.phase_csv and not args.phase_fast_csv:
            print("Nothing to plot. Use --input, --phase-csv, or --phase-fast-csv.")
        return

    files = find_h5_files(args.input)
    if not files:
        print("No minute_*.h5 files found.")
        return

    print(f"Loading {len(files)} files...")
    timestamps = read_timestamps(files, time_downsample=args.time_downsample)

    freq_mhz_full = build_freq_axis(args.fft_size, args.fs)
    if args.freq_downsample > 1:
        freq_mhz = downsample_freq(
            freq_mhz_full[np.newaxis, :], args.freq_downsample
        ).squeeze()
        print(
            f"Frequency axis downsampled by {args.freq_downsample}x "
            f"({freq_mhz_full.shape[0]} -> {freq_mhz.shape[0]} bins)"
        )
    else:
        freq_mhz = freq_mhz_full

    # One channel at a time: load only this channel's FFT data, plot it,
    # save, then drop the array before moving to the next one. Peak memory
    # is now one channel's worth of data instead of 3 channels + 3
    # coherence pairs held simultaneously.
    for ch in range(1, 4):
        print(f"Loading channel {ch} FFT data...")
        fft_ch = load_channel_fft(
            files, ch, fft_size=args.fft_size, time_downsample=args.time_downsample
        )
        if args.freq_downsample > 1:
            fft_ch = downsample_freq(fft_ch, args.freq_downsample)

        fig, ax = plt.subplots(figsize=(16, 9))
        im = plot_spectrogram(
            ax,
            fft_ch,
            timestamps,
            freq_mhz,
            vmin_p=args.vmin_percentile,
            vmax_p=args.vmax_percentile,
            title=f"Channel {ch} Spectrogram",
        )
        plt.colorbar(im, ax=ax, label="Power [dB]")
        fig.savefig(output_dir / f"channel_{ch}_spectrogram.png", dpi=args.dpi)
        plt.close(fig)
        del fft_ch
        print(f"Channel {ch} spectrogram saved.")

    # Same one-at-a-time treatment for coherence pairs.
    for i, j in [(1, 2), (1, 3), (2, 3)]:
        print(f"Loading coherence {i}-{j} data...")
        coh_ij = load_pair_coherence(
            files, i, j, fft_size=args.fft_size, time_downsample=args.time_downsample
        )
        if args.freq_downsample > 1:
            coh_ij = downsample_freq(coh_ij, args.freq_downsample)

        fig, ax = plt.subplots(figsize=(12, 6))
        im = plot_coherence(
            ax,
            coh_ij,
            timestamps,
            freq_mhz,
            title=f"Coherence: Channel {i} vs Channel {j}",
        )
        plt.colorbar(im, ax=ax, label="Coherence")
        fig.savefig(output_dir / f"coherence_{i}_{j}.png", dpi=args.dpi)
        plt.close(fig)
        del coh_ij
        print(f"Coherence {i}-{j} saved.")

    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
