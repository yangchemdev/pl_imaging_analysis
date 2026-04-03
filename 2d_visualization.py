"""
2d_visualization.py
---------------
Load a set of tab-delimited .txt files representing a 3-D dataset (t, x, y),
bin the time axis by averaging, find t=0 from the global intensity peak, and
save each binned time frame as a separate PNG file.

Layout convention
-----------------
  - Each file  → one y-slice   (shape: n_t_raw × n_x)
  - Each row   → one time step
  - Each column→ one x position

Files are sorted by name so they map to y in ascending order.

Usage
-----
Edit the PARAMETERS block below, then run:
    python 2d_visualization.py
"""

import glob
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from hytools.hy_basic import check_make_dir

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  –  edit these
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR   = "F:\\OneDrive - purdue.edu\\Data\\Optical Spectra\\PL\\2026\\260327 1DLOC PL_fin\\plimg\\11"
OUTPUT_DIR = check_make_dir(f"{DATA_DIR}\\fig")
FILE_GLOB  = "*.txt"

T_BIN      = 1024        # raw time rows averaged per output frame
DT         = 0.016      # time step (ns, ps, … — whatever your instrument uses)

# T0 detection: the spatially-summed trace is smoothed with a boxcar of this
# width (in raw rows) before argmax.  Increase if the peak is very noisy;
# decrease if the rise is sharp and you want finer localisation.
T0_SMOOTH  = 64

DX         = 10 / 182    # x step (same units as your spatial axis)
DY         = 10 / 182     # y step (same units as your spatial axis)

# Colormap / display
CMAP        = "RdBu_r"
CLIM_MODE   = "frame"   # "global" | "frame"
CLIM_PCTILE = (1, 99)

# Output
SAVE_DPI    = 100
# ─────────────────────────────────────────────────────────────────────────────


def _natural_sort_key(s: str) -> list:
    """
    Split a string into alternating text/integer chunks so that filenames
    with embedded numbers sort in human-intuitive order:
        xx2.txt < xx10.txt < xx11.txt  (not xx10 < xx11 < xx2)
    """
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", s)]


def load_files(data_dir: str, pattern: str) -> tuple[np.ndarray, list[str]]:
    """
    Load all matching files into a 3-D array (n_y, n_t_raw, n_x).
    Files are sorted by natural (numeric) order of their filename so that
    e.g. xx2.txt comes before xx10.txt, not after it.
    """
    paths = sorted(
        glob.glob(os.path.join(data_dir, pattern)),
        key=_natural_sort_key,
    )
    if not paths:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in '{data_dir}'."
        )

    print("  File → y-index mapping:")
    slices = []
    for i, p in enumerate(paths):
        arr = np.loadtxt(p, delimiter="\t")   # (n_t_raw, n_x)

        # Find the last row that is not entirely zero and trim everything after it.
        # Rows in the middle of the file that happen to be all-zero are kept.
        nonzero_rows = np.where(arr.any(axis=1))[0]
        if nonzero_rows.size == 0:
            raise ValueError(f"File '{os.path.basename(p)}' contains no non-zero rows.")
        last_nonzero = int(nonzero_rows[-1])
        n_trimmed = len(arr) - (last_nonzero + 1)
        arr = arr[: last_nonzero + 1]

        slices.append(arr)
        trim_note = f"  (trimmed {n_trimmed} trailing zero rows)" if n_trimmed else ""
        print(f"    y[{i}] ← {os.path.basename(p)}  [{len(arr)} rows]{trim_note}")

    # All slices must have the same number of rows to stack.
    # If trimming produced unequal lengths, truncate to the shortest.
    lengths = [s.shape[0] for s in slices]
    if len(set(lengths)) > 1:
        min_len = min(lengths)
        print(f"  WARNING: files have unequal row counts after trimming "
              f"({min_len}–{max(lengths)}). Truncating all to {min_len} rows.")
        slices = [s[:min_len] for s in slices]

    data = np.stack(slices, axis=0)           # (n_y, n_t_raw, n_x)
    return data, paths


def find_t0_index(data: np.ndarray, smooth_width: int) -> int:
    """
    Find the raw time index of t=0 (excitation pulse arrival) by locating the
    peak of the total spatially-summed intensity trace.

    Summing over all (x, y) pixels collapses spatial noise and gives a 1-D
    trace of length n_t_raw.  A uniform (boxcar) filter is applied before
    argmax to suppress residual shot noise while preserving the peak position.

    Parameters
    ----------
    data         : (n_y, n_t_raw, n_x)
    smooth_width : width of the boxcar kernel in raw time rows

    Returns
    -------
    i0 : int — raw time index of the intensity peak (= t=0)
    """
    trace    = data.sum(axis=(0, 2))                          # (n_t_raw,)
    smoothed = uniform_filter1d(trace, size=smooth_width, mode="nearest")
    return int(np.argmax(smoothed))


def bin_time(data: np.ndarray, bin_size: int) -> np.ndarray:
    """
    Average the time axis in non-overlapping blocks of `bin_size`.
    Trailing rows that do not fill a complete block are discarded.

    Parameters
    ----------
    data     : (n_y, n_t_raw, n_x)
    bin_size : number of raw time rows per output frame

    Returns
    -------
    binned   : (n_y, n_t_binned, n_x)
    """
    n_y, n_t_raw, n_x = data.shape
    n_t_binned = n_t_raw // bin_size
    trimmed    = data[:, : n_t_binned * bin_size, :]
    binned     = trimmed.reshape(n_y, n_t_binned, bin_size, n_x).mean(axis=2)
    return binned


def build_axes(
    n_t_raw: int,
    n_t_binned: int,
    i0_raw: int,
    dt: float,
    bin_size: int,
    dx: float,
    dy: float,
    n_x: int,
    n_y: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build physical coordinate arrays.

    Raw time axis: t_raw = np.arange(n_t_raw) * dt  (instrument counts from 0).
    After binning, each frame's representative time is the mean of its raw rows.
    The axis is shifted so that the bin containing i0_raw falls at t = 0.

    x and y are built from 0 using the supplied step sizes and the actual
    data dimensions, so they always match the loaded data exactly.

    Returns
    -------
    t : (n_t_binned,)  — time axis, zero-referenced to the intensity peak
    x : (n_x,)         — starts at 0, step dx
    y : (n_y,)         — starts at 0, step dy
    """
    t_raw  = np.arange(n_t_raw) * dt
    n_use  = n_t_binned * bin_size
    t_trim = t_raw[:n_use].reshape(n_t_binned, bin_size)
    t      = t_trim.mean(axis=1)

    i0_binned = min(i0_raw // bin_size, n_t_binned - 1)
    t        -= t[i0_binned]

    x = np.arange(n_x) * dx
    y = np.arange(n_y) * dy

    return t, x, y


def compute_clim(data: np.ndarray, pctile: tuple[float, float]) -> tuple[float, float]:
    """Robust colour limits via percentile clipping."""
    lo, hi = np.nanpercentile(data, pctile)
    return lo, hi


# ── Analysis functions ────────────────────────────────────────────────────────

def save_time_averaged_image(
    data_binned: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    cmap: str,
    clim_pctile: tuple[float, float],
    dpi: int,
) -> np.ndarray:
    """
    Compute and save the time-averaged spatial image (mean over all binned
    frames) as  time_average.png.

    Parameters
    ----------
    data_binned : (n_y, n_t_binned, n_x)

    Returns
    -------
    avg_image : (n_y, n_x)
    """
    avg_image = data_binned.mean(axis=1)          # (n_y, n_x)

    X, Y = np.meshgrid(pad_edges(x), pad_edges(y))
    vmin, vmax = compute_clim(avg_image[np.newaxis], clim_pctile)

    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(X, Y, avg_image, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="flat")
    fig.colorbar(mesh, ax=ax, label="Mean intensity")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Time-averaged intensity")

    fname = os.path.join(output_dir, "time_average.png")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved time-averaged image → {fname}")

    return avg_image


def _gaussian1d(coord: np.ndarray, amp: float, x0: float,
                sigma: float, bg: float) -> np.ndarray:
    """1-D Gaussian with flat background:  bg + amp * exp(-(coord-x0)²/(2σ²))"""
    return bg + amp * np.exp(-0.5 * ((coord - x0) / sigma) ** 2)


def _fit_axis(profile: np.ndarray, coord: np.ndarray) -> np.ndarray:
    """
    Fit a 1-D Gaussian to `profile` sampled at `coord`.

    Initial guess: background = min, amplitude = range,
    centre = centre-of-mass, sigma = quarter of axis span.

    Returns
    -------
    popt : (4,) array [amp, x0, sigma, bg], or NaN array on failure.
    """
    from scipy.optimize import curve_fit

    bg0  = float(np.min(profile))
    amp0 = float(np.max(profile) - bg0)
    weights = np.clip(profile - bg0, 0, None)
    denom   = weights.sum()
    x0_0    = float((coord * weights).sum() / denom) if denom > 0 else float(coord.mean())
    sig0    = float((coord[-1] - coord[0]) / 4)

    try:
        popt, _ = curve_fit(
            _gaussian1d, coord, profile,
            p0=[amp0, x0_0, sig0, bg0],
            bounds=(
                [0,      coord[0],  1e-9,                    -np.inf],
                [np.inf, coord[-1], coord[-1] - coord[0],     np.inf],
            ),
            maxfev=10_000,
        )
    except Exception:
        popt = np.full(4, np.nan)

    return np.asarray(popt)


def fit_gaussians_vs_time(
    data_binned: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    dpi: int,
) -> dict[str, np.ndarray]:
    """
    For every binned time frame, project the 2-D image onto each spatial axis
    by summation and fit a 1-D Gaussian independently in x and in y.

    Summing along the orthogonal axis (rather than slicing) maximises SNR for
    spatially extended or diffuse emission.

    Fit model
    ---------
        I(u) = bg + amp · exp[−(u − u₀)² / (2 σ²)]

    Parameters
    ----------
    data_binned : (n_y, n_t_binned, n_x)
    t           : (n_t_binned,)  — zero-referenced time axis

    Returns
    -------
    results : dict with keys
        'x_amp', 'x_x0', 'x_sigma', 'x_bg',
        'y_amp', 'y_x0', 'y_sigma', 'y_bg'
        each a (n_t_binned,) float array.

    Side-effect
    -----------
    Saves  gaussian_fit_params.png  to output_dir: a 2×2 superfigure with one
    panel per parameter (amp, x0/centre, sigma/width, bg), each showing the
    x-axis and y-axis fit results vs time on the same axes.
    """
    n_y, n_t_binned, n_x = data_binned.shape
    param_names = ["amp", "x0", "sigma", "bg"]
    results: dict[str, np.ndarray] = {
        f"{axis}_{p}": np.full(n_t_binned, np.nan)
        for axis in ("x", "y") for p in param_names
    }

    print("  Fitting 1-D Gaussians per frame …")
    for i in range(n_t_binned):
        frame = data_binned[:, i, :]           # (n_y, n_x)

        # Project: sum along the orthogonal direction
        profile_x = frame.sum(axis=0)          # sum over y  → f(x)
        profile_y = frame.sum(axis=1)          # sum over x  → f(y)

        popt_x = _fit_axis(profile_x, x)
        popt_y = _fit_axis(profile_y, y)

        for j, p in enumerate(param_names):
            results[f"x_{p}"][i] = popt_x[j]
            results[f"y_{p}"][i] = popt_y[j]

        if (i + 1) % 10 == 0 or (i + 1) == n_t_binned:
            print(f"    Fitted {i + 1}/{n_t_binned} frames …")

    # ── Superfigure: 2×2, one panel per parameter ─────────────────────────────
    panel_meta = [
        ("amp",   "Amplitude (a.u.)"),
        ("x0",    "Centre (axis units)"),
        ("sigma", "Width σ (axis units)"),
        ("bg",    "Background (a.u.)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    for ax_panel, (p, ylabel) in zip(axes.flatten(), panel_meta):
        ax_panel.plot(t, results[f"x_{p}"], "o-", ms=3, lw=1, label="x projection")
        ax_panel.plot(t, results[f"y_{p}"], "s-", ms=3, lw=1, label="y projection")
        ax_panel.axvline(0, color="k", lw=0.8, ls="--", label="t = 0")
        ax_panel.set_ylabel(ylabel)
        ax_panel.set_title(p)
        ax_panel.legend(fontsize=8)

    for ax_panel in axes[1, :]:             # x-label only on bottom row
        ax_panel.set_xlabel("Time")

    fig.suptitle("1-D Gaussian fit parameters vs time", fontsize=13)
    fig.tight_layout()

    fname = os.path.join(output_dir, "gaussian_fit_params.png")
    fig.savefig(fname, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved Gaussian fit parameter plot → {fname}")

    return results


def pad_edges(arr: np.ndarray) -> np.ndarray:
    """
    Convert N cell-centre coordinates to N+1 cell-edge coordinates by
    extrapolating half a grid-spacing at each end.

    Raises a descriptive ValueError for degenerate inputs (length 0 or 1)
    rather than an opaque IndexError.
    """
    if len(arr) == 0:
        raise ValueError("pad_edges received an empty coordinate array. "
                         "Check that X_AXIS / Y_AXIS match the data dimensions.")
    if len(arr) == 1:
        raise ValueError("pad_edges received a length-1 coordinate array — "
                         "cannot infer grid spacing. "
                         "Check that X_AXIS / Y_AXIS match the data dimensions.")
    d = np.diff(arr)
    return np.concatenate([[arr[0] - d[0] / 2], arr[:-1] + d / 2, [arr[-1] + d[-1] / 2]])


def save_frames(
    data_binned: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    output_dir: str,
    clim_mode: str,
    clim_pctile: tuple[float, float],
    cmap: str,
    dpi: int,
) -> None:
    """
    Save each binned time frame as an individual PNG file.

    Files are named  frame_XXXXX_t{value:.4g}.png  so they sort naturally.

    data_binned : (n_y, n_t_binned, n_x)
    t           : (n_t_binned,)  — zero-referenced to the excitation peak
    x, y        : physical coordinate arrays
    """
    os.makedirs(output_dir, exist_ok=True)

    n_y, n_t_binned, n_x = data_binned.shape
    X, Y = np.meshgrid(pad_edges(x), pad_edges(y))   # (n_y+1, n_x+1)

    if clim_mode == "global":
        vmin, vmax = compute_clim(data_binned, clim_pctile)

    n_digits = len(str(n_t_binned - 1))

    for i in range(n_t_binned):
        z = data_binned[:, i, :]               # (n_y, n_x)

        if clim_mode == "frame":
            vmin, vmax = compute_clim(z[np.newaxis], clim_pctile)

        fig, ax = plt.subplots(figsize=(7, 5))
        mesh = ax.pcolormesh(X, Y, z, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat")
        fig.colorbar(mesh, ax=ax, label="Intensity")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"t = {t[i]:.4g}  (frame {i + 1}/{n_t_binned})")

        fname = os.path.join(output_dir, f"frame_{i:0{n_digits}d}_t{t[i]:.4g}.png")
        fig.savefig(fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if (i + 1) % 10 == 0 or (i + 1) == n_t_binned:
            print(f"  Saved {i + 1}/{n_t_binned} frames …")


def main():
    print("Loading files …")
    data_raw, paths = load_files(DATA_DIR, FILE_GLOB)
    n_y, n_t_raw, n_x = data_raw.shape
    print(f"  {n_y} files  ×  {n_t_raw} rows  ×  {n_x} columns  →  shape {data_raw.shape}")

    # ── Find t0 on raw data before binning ───────────────────────────────────
    print(f"Detecting t=0 (boxcar smooth width = {T0_SMOOTH} raw rows) …")
    i0_raw = find_t0_index(data_raw, smooth_width=T0_SMOOTH)
    print(f"  Peak at raw index {i0_raw}  (raw time = {i0_raw * DT:.4g})")

    # ── Bin time axis ────────────────────────────────────────────────────────
    print(f"Binning time axis by {T_BIN} …")
    data_binned = bin_time(data_raw, T_BIN)
    n_t_binned  = data_binned.shape[1]
    n_discarded = n_t_raw - n_t_binned * T_BIN
    print(f"  Binned shape: {data_binned.shape}  ({n_discarded} trailing rows discarded)")

    # ── Build axes with t shifted to zero at the peak ────────────────────────
    t, x, y = build_axes(
        n_t_raw=n_t_raw,
        n_t_binned=n_t_binned,
        i0_raw=i0_raw,
        dt=DT,
        bin_size=T_BIN,
        dx=DX,
        dy=DY,
        n_x=n_x,
        n_y=n_y,
    )
    print(f"  t range after shift: [{t[0]:.4g}, {t[-1]:.4g}]")
    print(f"  x range: [{x[0]:.4g}, {x[-1]:.4g}]  ({len(x)} points)")
    print(f"  y range: [{y[0]:.4g}, {y[-1]:.4g}]  ({len(y)} points)")

    # ── Save per-frame PNGs ───────────────────────────────────────────────────
    print(f"Saving {n_t_binned} PNG frames to '{OUTPUT_DIR}' …")
    save_frames(
        data_binned, t, x, y,
        output_dir=OUTPUT_DIR,
        clim_mode=CLIM_MODE,
        clim_pctile=CLIM_PCTILE,
        cmap=CMAP,
        dpi=SAVE_DPI,
    )

    # ── Time-averaged image ───────────────────────────────────────────────────
    print("Computing time-averaged image …")
    save_time_averaged_image(
        data_binned, x, y,
        output_dir=OUTPUT_DIR,
        cmap=CMAP,
        clim_pctile=CLIM_PCTILE,
        dpi=SAVE_DPI,
    )

    # ── 1-D Gaussian fits vs time ─────────────────────────────────────────────
    print("Fitting 1-D Gaussians vs time …")
    fit_gaussians_vs_time(
        data_binned, t, x, y,
        output_dir=OUTPUT_DIR,
        dpi=SAVE_DPI,
    )

    print("Done.")


if __name__ == "__main__":
    main()