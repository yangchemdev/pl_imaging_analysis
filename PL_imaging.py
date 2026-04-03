'''
Made by: Hanjun Yang, 2025
This module imports and analyzes PL imaging data.

Input:
    1. .txt files containing PL imaging data. Each column represents a TRPL trace.
       Each row represents a time delay. ([time, pixel] indexing)
         - data[m, n] is the PL intensity at time delay m and pixel n.
    2. time step (ns)
    3. pixel step (um)

Output:
    Normalized PL imaging data and fit results.
'''

#%% imports
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tqdm import tqdm

from hytools import (
    hy_basic as hyb,
    hy_fit as hyf,
    hy_plot as hyp,
    hy_configclass as hyconfig,
)


#%% helper functions

def adjacent_average(arr, window=1):
    """
    Compute a centered moving average over a 1D NumPy array.

    Parameters
    ----------
    arr : array_like
        1D input array.
    window : int
        Half-width of the averaging window. The full window spans
        [i - window, i + window], clipped at array boundaries.

    Returns
    -------
    out : np.ndarray (float64)
        Array of the same length as `arr`.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n = len(arr)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        out[i] = arr[lo:hi].mean()
    return out


def iterative_argmax_find(arr, n_peaks, min_distance=0):
    """Iteratively find the n_peaks maxima in arr with a minimum distance constraint.

    Returns the indices of the peaks in sorted order.
    """
    arr = np.asarray(arr)
    peaks = np.zeros(n_peaks, dtype=int)
    temp_arr = arr.copy()
    for i in range(n_peaks):
        idx = np.nanargmax(temp_arr)
        peaks[i] = idx
        lo = max(0, idx - min_distance)
        hi = min(len(temp_arr), idx + min_distance + 1)
        temp_arr[lo:hi] = np.nan
    return np.sort(peaks)


def first_with_w_idx(strings):
    """Return the index of the first string in `strings` containing the character 'w'."""
    for idx, s in enumerate(strings):
        if 'w' in s:
            return idx
    return None


def bin_rows_2D(data: np.ndarray, w: int) -> np.ndarray:
    """Average non-overlapping row blocks of width `w` in a 2D array."""
    if data.ndim != 2:
        raise ValueError("data must be 2D")
    if w <= 0:
        raise ValueError("w must be positive")
    N = data.shape[0]
    K = N // w
    trimmed = data[:K * w]
    return trimmed.reshape(K, w, -1).mean(axis=1)


def bin_rows_1D(data: np.ndarray, w: int) -> np.ndarray:
    """Average non-overlapping blocks of width `w` in a 1D array."""
    if data.ndim != 1:
        raise ValueError("data must be 1D")
    if w <= 0:
        raise ValueError("w must be positive")
    N = data.shape[0]
    K = N // w
    trimmed = data[:K * w]
    return trimmed.reshape(K, w).mean(axis=1)


def fold_trpl(data: np.ndarray, row0: int) -> np.ndarray:
    """Roll a 1D or 2D array so that row `row0` becomes the first row."""
    if row0 < 0:
        raise ValueError("row0 must be non-negative")
    if row0 >= data.shape[0]:
        raise ValueError("row0 must be smaller than data size")
    if data.ndim == 2:
        return np.vstack((data[row0:], data[:row0]))
    elif data.ndim == 1:
        return np.concatenate((data[row0:], data[:row0]))
    else:
        raise ValueError("data must be 1D or 2D")


def last_nonzero_row_idx(data):
    """Return the index of the last row with at least one non-zero element."""
    if data.ndim != 2:
        raise ValueError("Input must be 2D.")
    mask = np.any(data != 0, axis=1)
    idx = np.where(mask)[0]
    return idx[-1] if idx.size > 0 else 0


def fix_replica(data, t_step, replica_n, min_dt, t0_buffer=1, debug=False):
    """
    Fix 447 nm laser replicas by finding replica peaks, aligning them, and averaging.

    Parameters
    ----------
    data : np.ndarray, shape (time, pixel)
    t_step : float
        Time step in ns.
    replica_n : int
        Number of replicas to find.
    min_dt : float
        Minimum separation between replicas in ns.
    t0_buffer : float
        Buffer before t0 used for background subtraction, in ns.
    debug : bool
        If True, return intermediate arrays.
    """
    data_xavg = np.mean(data, axis=1)
    data_xavg_sm = adjacent_average(data_xavg, window=6)  # ~0.1 ns smoothing window
    n_min_dt = int(min_dt // t_step)
    replica_peaks = iterative_argmax_find(data_xavg_sm, n_peaks=replica_n, min_distance=n_min_dt)
    replica_peaks_diff = np.diff(replica_peaks)
    assert np.max(replica_peaks_diff) - np.min(replica_peaks_diff) <= 12, \
        f"Replica peaks are not roughly equally spaced. peaks are row #: {replica_peaks}."
    replica_length = int(np.median(replica_peaks_diff))
    n_t0buffer = int(t0_buffer // t_step)
    data_replica = np.full((replica_length, data.shape[1], replica_n), np.nan)  # [time, pixel, replica]
    for i in range(replica_n):
        data_replica[:, :, i] = data[replica_peaks[i] - n_t0buffer : replica_peaks[i] + replica_length - n_t0buffer, :]
    for i in range(replica_n):
        last_nonzero_idx = last_nonzero_row_idx(data_replica[:, :, i])
        if last_nonzero_idx is not None and last_nonzero_idx < replica_length - 1:
            data_replica[last_nonzero_idx + 1:, :, i] = np.nan
    data_fixed = np.nanmean(data_replica, axis=2)
    data_fixed[np.isnan(data_fixed)] = 0  # HACK: set remaining NaN to 0
    if debug:
        return data_fixed, data_replica, replica_peaks
    else:
        return data_fixed


def natural_sort_key(s):
    """Sort key for strings containing numbers in human order (e.g., file2 before file10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


#%% load function

def load_data(f_ins: list, t_reso: float, x_reso: float, source_type: str = '1', trig_preview=False, **kwargs):
    """
    Format raw data from disk into a processable [t, x] array.

    Parameters
    ----------
    f_ins : list of str
        Input file paths.
    t_reso : float
        Time resolution in ns.
    x_reso : float
        Spatial resolution in um.
    source_type : str
        How to reduce multi-file 2D data to 1D:
        '1'      – use first file only (single file expected),
        'mean'   – average all files,
        'max'    – take element-wise max,
        'radial' – radial average (isotropic),
        'interp' – interpolate along a line at angle theta.
    trig_preview : bool
        If True, show a preview figure. Only meaningful for multi-file modes.

    Optional keyword arguments
    --------------------------
    y_reso : float
        Spatial resolution in y, in um (for 'radial' and 'interp').
    theta : float
        Angle of interpolation line in radians (for 'interp').

    Returns
    -------
    data : np.ndarray, shape [t, x]
    t : np.ndarray
        Time axis in ns, zeroed at t0.
    x_out : np.ndarray
        Spatial axis in um (or radial/interpolation distance).
    fig, axes : (optional)
        Preview figure, returned only when trig_preview=True.
    """
    assert source_type in ['1', 'interp', 'radial', 'mean', 'max'], \
        f"Unknown source_type: {source_type}. Should be '1', 'interp', 'radial', 'mean', or 'max'."
    if source_type == '1':
        assert len(f_ins) == 1, "For source_type '1', only one file should be provided."

    nf = len(f_ins)
    data_raws = [None] * nf
    idlast = np.zeros(nf, dtype=int)
    for i, f in enumerate(f_ins):
        data_raws[i] = np.loadtxt(f, delimiter="\t")  # [t, x]
        idlast[i] = last_nonzero_row_idx(data_raws[i])
    assert all(d.shape == data_raws[0].shape for d in data_raws), \
        "All input files must have the same array shape."
    data_raw = np.stack(data_raws, axis=2)  # [t, x, n]
    trailing_0_id = np.max(idlast)
    data_raw = data_raw[:trailing_0_id + 1, :, :]

    # build axes and align t0
    t = np.arange(data_raw.shape[0]) * t_reso
    x = np.arange(data_raw.shape[1]) * x_reso
    t0 = t[np.argmax(np.mean(data_raw, axis=(1, 2)))]
    t = t - t0

    # background subtraction
    BG_T0_BUFFER = 1  # ns before t0
    bg_mask = t < -BG_T0_BUFFER
    if not np.any(bg_mask):
        print("WARNING: not enough data points to calculate background; setting bg = 0. "
              "Consider decreasing BG_T0_BUFFER.")
        bg = 0
    else:
        bg = np.median(data_raw[bg_mask, :, :])
    data_raw = data_raw - bg

    # reduce to [t, x]
    if source_type == '1':
        data = data_raw[:, :, 0]
        x_out = x
    elif source_type == 'mean':
        data = np.mean(data_raw, axis=2)
        x_out = x
    elif source_type == 'max':
        data = np.max(data_raw, axis=2)
        x_out = x
    else:
        y_reso = kwargs.get('y_reso', x_reso)
        y = np.arange(data_raw.shape[2]) * y_reso
        x0id = np.argmax(np.mean(data_raw, axis=(0, 2)))
        y0id = np.argmax(np.mean(data_raw, axis=(0, 1)))
        x0, y0 = x[x0id], y[y0id]
        if source_type == 'radial':
            r_max = min(x[x0id] - x[0], x[-1] - x[x0id],
                        y[y0id] - y[0], y[-1] - y[y0id])
            X, Y = np.meshgrid(x, y, indexing='ij')
            R = np.sqrt((X - x0)**2 + (Y - y0)**2)
            r_reso = max(x_reso, y_reso)
            r = np.arange(0, r_max, r_reso)
            data = np.zeros((data_raw.shape[0], r.shape[0]))
            for i, ri in enumerate(r):
                mask = (R >= ri) & (R < ri + r_reso)
                if mask.any():
                    data[:, i] = np.mean(data_raw[:, mask], axis=1)
            x_out = r
        elif source_type == 'interp':
            from scipy.interpolate import RegularGridInterpolator
            theta = kwargs.get('theta', 0)
            r_candidates = []
            if np.cos(theta) != 0:
                r_candidates.append((x[-1] - x0) / np.cos(theta))
                r_candidates.append((x[0] - x0) / np.cos(theta))
            if np.sin(theta) != 0:
                r_candidates.append((y[-1] - y0) / np.sin(theta))
                r_candidates.append((y[0] - y0) / np.sin(theta))
            r_candidates_neg = [rc for rc in r_candidates if rc <= 0]
            r_candidates_pos = [rc for rc in r_candidates if rc >= 0]
            r_start = max(r_candidates_neg) if r_candidates_neg else 0.0
            r_end   = min(r_candidates_pos) if r_candidates_pos else 0.0
            r_reso = max(x_reso, y_reso)
            EPSILON = 1e-6
            r = np.arange(r_start + EPSILON, r_end - EPSILON, r_reso)
            x_sample = x0 + r * np.cos(theta)
            y_sample = y0 + r * np.sin(theta)
            points = np.column_stack([x_sample, y_sample])
            data = np.zeros((data_raw.shape[0], r.size))
            for i in range(data_raw.shape[0]):
                interpolator = RegularGridInterpolator((x, y), data_raw[i, :, :], bounds_error=True)
                data[i, :] = interpolator(points)
            x_out = r

    if trig_preview:
        if source_type in ['1', 'mean', 'max']:
            print("Preview skipped: source_type is 1D, no 2D spatial data to display.")
        else:
            fig, axes = plt.subplots(1, 2, figsize=(11, 5))
            X, Y = np.meshgrid(x, y, indexing='ij')
            axes[0].pcolormesh(X, Y, np.sum(data_raw, axis=0), cmap='viridis')
            if source_type == 'radial':
                r_display = r[::max(1, len(r) // 5)]
                for ri in r_display:
                    circle = plt.Circle((x0, y0), ri, color='red', fill=False, linestyle='--', alpha=0.6)
                    axes[0].add_artist(circle)
                axes[0].set_title(f'Time-summed data preview\nRadial mode, r_reso = {r_reso:.2f} um')
            elif source_type == 'interp':
                axes[0].plot(x_sample, y_sample, 'r--', alpha=0.6)
                axes[0].set_title(f'Time-summed data preview\nInterpolation mode, theta = {np.degrees(theta):.1f} deg')
            axes[0].set_xlabel('x (um)')
            axes[0].set_ylabel('y (um)')
            axes[0].set_aspect('equal')
            axes[1].plot(x_out, np.mean(data, axis=0))
            axes[1].set_title('Spatial profile preview')
            axes[1].set_xlabel('x (um)')
            axes[1].set_ylabel('Intensity (a.u.)')
            fig.tight_layout()
        return data, t, x_out, fig, axes
    else:
        return data, t, x_out


#%% config
##### data params #####
source_mode = 'radial'      # '1', 'mean', 'max', 'radial', 'interp'
f_in = None                 # path(s) to .dat file(s). If None, GUI prompt is used.
t_step = 0.016              # time step in ns
motor_step_x = 10           # motor step in um (x)
motor_step_y = 10           # motor step in um (y; ignored in '1', 'mean', 'max' modes)
mag = 182                   # microscope magnification
trig_447_replica_fix = False  # align and average 447 nm laser replicas
replica_n = 4               # number of replicas
min_dt = 5                  # minimum separation between replicas in ns
theta = 30 / 180 * np.pi   # interpolation angle in radians (interp mode only)
fold_row = None             # row index to fold data to end; None to skip

##### process params #####
x_range = None              # spatial range to analyze in um; None = full range
t_range = [0, 50]           # time range to analyze in ns; None = full range
t0_buffer = -1              # buffer before t0 for background subtraction in ns
t_binning_width = 32        # time binning factor; None = no binning
x_fit_model = hyf.func_class_gaussian
t_fit_model = hyf.exp_ne_wrapper(1, np.array([10]), trig_non_negative_A=True, trig_non_negative_c=True)
trig_MSD_rezero = False     # re-zero MSD by subtracting initial MSD value
displacement_source = 'fit' # 'fit' (use fitted Gaussian width) or 'MSD'
sigma_correction = True     # apply sigma correction in D fitting

##### visualize params #####
param_units = ['a.u.', 'um', 'um', 'a.u.']  # units for each Gaussian fit parameter
representative_t = [0, 5, 10, 40]           # time points for representative spatial plots

##### output params #####
f_out = None                # output path; None = use input file directory
overwrite_mode = False      # overwrite existing output directory

##### END OF CONFIG #####

# file selection
if source_mode == '1':
    if f_in is None:
        f_in = hyb.GUI_qt_get_file("Select PL imaging .dat file", False, remember=True)
    dir_in, name, _ = hyb.get_file_name_from_dir(f_in[0])
else:
    f_in = hyb.GUI_qt_get_file("Select PL imaging .dat file", False, remember=True)
    f_in = sorted(f_in, key=lambda f: natural_sort_key(hyb.get_file_name_from_dir(f)[1]))
    dir_in, name, _ = hyb.get_file_name_from_dir(f_in[0])

formatted_date = datetime.today().strftime('%y%m%d')
dir_out = hyb.check_make_dir(f"{dir_in}\\{name}_output_{formatted_date}", auto_rename=not overwrite_mode)

# save config
params_data = hyconfig.code_section({
    "f_in": f_in,
    "dir_in": dir_in,
    "t_step": t_step,
    "motor_step": motor_step_x,
    "mag": mag,
})
params_process = hyconfig.code_section({
    "x_range": x_range,
    "t_range": t_range,
    "t0_buffer": t0_buffer,
    "t_binning_width": t_binning_width,
    "fold_row": fold_row,
    "x_fit_model": x_fit_model.funcname,
    "t_fit_model": t_fit_model.funcname,
    "trig_MSD_rezero": trig_MSD_rezero,
    "displacement_source": displacement_source,
    "sigma_correction": sigma_correction,
})
params_visualize = hyconfig.code_section({
    "param_units": param_units,
    "representative_t": representative_t,
})
params_output = hyconfig.code_section({
    "dir_out": dir_out,
    "overwrite_mode": overwrite_mode,
})
params = hyconfig.config_class([params_data, params_process, params_visualize, params_output])
params.to_json(f"{params_output.dir_out}\\config_files_{formatted_date}.json")
params.to_pickle(f"{params_output.dir_out}\\config_files_{formatted_date}.pkl")


#%% load
if source_mode in ['1', 'mean', 'max']:
    data, t, x = load_data(f_in, t_step, motor_step_x / mag, source_type=source_mode)
elif source_mode in ['interp', 'radial']:
    data, t, x, fig_preview, axes_preview = load_data(
        f_in, t_step, motor_step_x / mag,
        source_type=source_mode,
        y_reso=motor_step_y / mag,
        theta=theta,
        trig_preview=True,
    )
    plt.savefig(f"{dir_out}\\data_preview.png")

# optional: fix 447 nm replicas
if trig_447_replica_fix:
    data = fix_replica(data, t_step, replica_n, min_dt)

# optional: fold data
if fold_row is not None:
    data = fold_trpl(data, fold_row)

# time binning
if t_binning_width is not None:
    data = bin_rows_2D(data, t_binning_width)
    t = bin_rows_1D(t, t_binning_width)

nt = data.shape[0]

# locate peak and define relative axes
data_tavg_temp = np.mean(data, axis=0)
data_xavg_temp = np.mean(data, axis=1)
data_tavg_temp = data_tavg_temp / np.max(data_tavg_temp)
data_xavg_temp = data_xavg_temp / np.max(data_xavg_temp)

x0 = x[hyb.numpy_nearest(data_tavg_temp, np.max(data_tavg_temp), 'id')]
t0 = t[hyb.numpy_nearest(data_xavg_temp, np.max(data_xavg_temp), 'id')]
print(f"t0 is at {t0:.4f} ns, or row {np.round(t0 / t_step):.0f}")

dx = x - x0
dt = t - t0

# background subtraction
bg_mask = dt < -1 * t0_buffer
if not np.any(bg_mask):
    raise ValueError(
        f"No time points found for background (dt < {t0_buffer}). "
        f"Min dt = {dt.min():.3f}"
    )
bg = np.percentile(data[bg_mask, :], 45, axis=0)
data = data - bg[np.newaxis, :]
data[data <= 0] = 0.1  # clamp non-positive values to epsilon

# spatial and temporal range cuts
if x_range is not None:
    x_mask = (dx >= x_range[0]) & (dx <= x_range[1])
    dx = dx[x_mask]
    x = x[x_mask]
    data = data[:, x_mask]

if t_range is not None:
    t_mask = (dt >= t_range[0]) & (dt <= t_range[1])
    dt = dt[t_mask]
    t = t[t_mask]
    nt = t.shape[0]
    data = data[t_mask, :]

print(f"Time axis: {dt[0]:.3f} to {dt[-1]:.3f} ns")

# normalization
data_max_per_frame = np.max(data, axis=1)
data_max_per_pixel = np.max(data, axis=0)
data_norm_t  = data / data_max_per_frame[:, np.newaxis]
data_norm_x  = data / data_max_per_pixel[np.newaxis, :]
data_normall = data / np.max(data)


#%% fitting: time-averaged TRPL and per-frame spatial profiles

data_tavg = np.mean(data, axis=0)
data_xavg = np.mean(data, axis=1)
data_xavg_norm = data_xavg / np.max(data_xavg)

# spatially averaged TRPL fit
t_fitter = hyf.fitter_1D('tfit_avg', t_fit_model, dt, data_xavg_norm)
t_fitter.fit()
tfit_data = t_fitter.y_fit

# per-frame Gaussian spatial fit
n_xfit_params  = x_fit_model.n_params
xfit_param_names = x_fit_model.param_names
xfit_params   = np.zeros((nt, n_xfit_params))
xfit_stds     = np.zeros((nt, n_xfit_params))
xfit_data     = np.zeros_like(data)
xfit_residual = np.zeros_like(data)
xfit_r2       = np.zeros(nt)
xfit_status   = np.zeros(nt)

for idt in tqdm(range(nt)):
    x_fitter = hyf.fitter_1D(
        f"xfit_#{idt}", x_fit_model, x, data[idt, :],
        lb=np.array([0, -np.inf, 0.1, -np.inf]),
        hb=np.array([np.inf, np.inf, 1, np.inf]),
        p0=np.array([np.max(data[idt, :]), x0, 0.5, 0]),  # [A, x0, w, offset]
    )
    x_fitter.fit()
    xfit_params[idt, :]   = x_fitter.params['value']
    xfit_stds[idt, :]     = x_fitter.params['std']
    xfit_data[idt, :]     = x_fitter.y_fit
    xfit_residual[idt, :] = x_fitter.residue
    xfit_r2[idt]          = x_fitter.r2
    xfit_status[idt]      = x_fitter.status

# MSD calculation
MSD = np.zeros(nt)
for idt in range(nt):
    dx_msd = dx - np.sum(data[idt, :] * dx) / np.sum(data[idt, :]) if trig_MSD_rezero else dx
    weight = data[idt, :] / np.sum(data[idt, :])
    MSD[idt] = np.sum(weight * dx_msd**2)

# displacement² for D fitting
if displacement_source == 'fit':
    idx_w = first_with_w_idx(xfit_param_names)
    dis2 = 2 * xfit_params[:, idx_w]**2
elif displacement_source == 'MSD':
    dis2 = MSD
else:
    raise ValueError(f"Unknown displacement_source: {displacement_source}. Should be 'fit' or 'MSD'.")

valid_mask  = ~np.isnan(dis2)
dis2_tofit  = dis2[valid_mask]
dt_tofit    = dt[valid_mask]

dis2_fitter = hyf.fitter_1D('D_fit', hyf.func_class_linear, dt_tofit, dis2_tofit)

# fitting weights / sigma
if sigma_correction:
    if displacement_source == 'fit':
        dis2_sigma = np.abs(4 * xfit_params[:, idx_w]) * xfit_stds[:, idx_w]
        dis2_sigma = dis2_sigma[valid_mask]
    elif displacement_source == 'MSD':
        dis2_sigma = 1 / np.abs(data_xavg)
        dis2_sigma = dis2_sigma[valid_mask]
    t0id0 = hyb.numpy_nearest(dt_tofit, 0.1, 'idx', direction=1)
    t1id0 = hyb.numpy_nearest(dt_tofit, 1,   'idx', direction=1)
    t1id0 = max(t0id0 + 5, t1id0)
    print(f"Using rows {t0id0} to {t1id0} to calculate sigma floor.")
    dis2_sigma_floor = np.median(dis2_sigma[t0id0:t1id0]) * 0.5
    dis2_sigma += dis2_sigma_floor
else:
    dis2_sigma = np.ones_like(dis2_tofit) * np.mean(dis2_tofit) * 0.1

dis2_fitter.fit(sigma=dis2_sigma)
D     = dis2_fitter.params['value'][0] / 4 * 10  # cm²/s (2D Einstein relation)
D_std = dis2_fitter.params['std'][0]    / 4 * 10
dis2_fit      = dis2_fitter.y_fit
dis2_residual = dis2_fitter.residue
dis2_r2       = dis2_fitter.r2


#%% visualize

matplotlib.rcParams['font.size'] = 7

# figure 1: raw data overview
fig, axes = plt.subplots(2, 2, figsize=(6, 5), dpi=150)
axes = axes.flatten()

im0 = axes[0].imshow(data_normall, extent=[x[0], x[-1], dt[0], dt[-1]],
                     aspect='auto', cmap='viridis', origin='lower')
axes[0].set_xlabel('x (um)')
axes[0].set_ylabel('Time (ns)')
hyp.colorbar_magic(im0)

im1 = axes[1].imshow(data_norm_t, extent=[x[0], x[-1], dt[0], dt[-1]],
                     aspect='auto', cmap='viridis', origin='lower')
axes[1].set_xlabel('x (um)')
axes[1].set_ylabel('Time (ns)')
hyp.colorbar_magic(im1)

idcenter = hyb.numpy_nearest(dx, 0, 'idx')
axes[2].semilogy(dt, data_xavg_norm,         label='spatial average')
axes[2].semilogy(dt, tfit_data,              label='fit',    linestyle='--')
axes[2].semilogy(dt, data_normall[:, idcenter], label='center')
lifetime_1e = dt[hyb.numpy_nearest(data_xavg, np.max(data_xavg) / np.e, 'idx')]
axes[2].axvline(lifetime_1e, color='gray', linestyle=':', label=f'1/e: {lifetime_1e:.2f} ns')
axes[2].set_xlabel('Time (ns)')
axes[2].set_ylabel('Intensity (a.u.)')
axes[2].legend(fontsize=8)

for tplot in representative_t:
    idt = hyb.numpy_nearest(dt, tplot, 'idx')
    axes[3].plot(dx, data_norm_t[idt, :], label=f'{dt[idt]:.2f} ns', alpha=0.5)
axes[3].set_xlabel('x (um)')
axes[3].set_ylabel('Intensity (a.u.)')
axes[3].legend(fontsize=8)

fig.suptitle(f'PL imaging raw data\n{name}')
axes[0].set_title('Normalized to global max')
axes[1].set_title('Normalized to each time frame')
axes[2].set_title('Spatially averaged data and fit')
axes[3].set_title('Representative spatial distribution')
plt.tight_layout()
fig.savefig(f"{dir_out}\\PL_imaging_raw_{formatted_date}.png", dpi=300)
plt.close(fig)

# figure 2: fitting parameters over time
n_params_to_draw = n_xfit_params + 2  # +1 R², +1 D²
n_cols = int(np.ceil(np.sqrt(n_params_to_draw)))
n_rows = int(np.ceil(n_params_to_draw / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.3, n_rows * 2.3), dpi=300)
axes = axes.flatten()
fig.suptitle(f'PL fit parameters over time\n{name}')
color_palette = plt.cm.Dark2.colors

for idplot in range(n_xfit_params):
    c = color_palette[idplot % len(color_palette)]
    if idplot == 0:     #HACK for amplitude.
        axes[idplot].semilogy(dt, xfit_params[:, idplot], linewidth=1.5, color=c, zorder=2)
    else:
        axes[idplot].plot(dt, xfit_params[:, idplot], linewidth=1.5, color=c, zorder=2)
    ylim = axes[idplot].get_ylim()
    axes[idplot].fill_between(
        dt,
        xfit_params[:, idplot] - xfit_stds[:, idplot],
        xfit_params[:, idplot] + xfit_stds[:, idplot],
        color=c, alpha=0.3,
    )
    axes[idplot].set_title(xfit_param_names[idplot])
    axes[idplot].set_xlabel('Time (ns)')
    axes[idplot].set_ylabel(f'{xfit_param_names[idplot]} ({param_units[idplot]})')
    axes[idplot].set_ylim(ylim)

# R² panel
xfit_r2[xfit_status != 1] = -1
axes[idplot + 1].plot(dt, xfit_r2, color='orange', linewidth=1)
axes[idplot + 1].set_xlabel('Time (ns)')
axes[idplot + 1].set_ylabel('R²')
axes[idplot + 1].set_title('Fitting R². −1 for failed fits.')

# displacement² and D panel
axes[idplot + 2].plot(dt_tofit, dis2_tofit, linewidth=1, label='Displacement²')
if dis2_fitter.status == 1:
    axes[idplot + 2].plot(dt_tofit, dis2_fit, color='r', linestyle='--', alpha=0.5,
                          label=hyb.format_SF_err(D, D_std) + ' cm²/s')
axes[idplot + 2].set_xlabel('Time (ns)')
axes[idplot + 2].set_ylabel('Displacement² (um²)')
axes[idplot + 2].set_title('Displacement² (um²)')
axes[idplot + 2].legend(fontsize=6)
ymin = np.nanpercentile(dis2_tofit, 5)
ymax = np.nanpercentile(dis2_tofit, 95)
axes[idplot + 2].set_ylim(ymin * 0.8, ymax * 1.2)

xfit_weights = 1 / dis2_sigma**2
xfit_weights = xfit_weights / np.max(xfit_weights)
ax2 = axes[idplot + 2].twinx()
ax2.plot(dt_tofit, xfit_weights, color='k', alpha=0.4, linewidth=0.5)
ax2.set_ylabel("Fitting weight (0–1)")

plt.tight_layout()
fig.savefig(f"{dir_out}\\PL_imaging_fitparams_{formatted_date}.png", dpi=300)
plt.close(fig)


#%% save text output

short_name = re.sub(r"p\d+u\d+$", "", name)
dir_txtsave = hyb.check_make_dir(os.path.normpath(f"{dir_out}\\txtdata"))

hyb.save_combined_matrix(data,          dt, dx, f"{dir_txtsave}\\{short_name}_data_raw.txt",     notice=True)
hyb.save_combined_matrix(data_normall,  dt, dx, f"{dir_txtsave}\\{short_name}_data_normall.txt", notice=False)
hyb.save_combined_matrix(data_norm_t,   dt, dx, f"{dir_txtsave}\\{short_name}_data_normt.txt",   notice=False)
hyb.save_combined_matrix(data_norm_x,   dt, dx, f"{dir_txtsave}\\{short_name}_data_normx.txt",   notice=False)

hyb.save_combined_matrix(data.T,         dx, dt, f"{dir_txtsave}\\{short_name}_data_raw_T.txt",     notice=False)
hyb.save_combined_matrix(data_normall.T, dx, dt, f"{dir_txtsave}\\{short_name}_data_normall_T.txt", notice=False)
hyb.save_combined_matrix(data_norm_t.T,  dx, dt, f"{dir_txtsave}\\{short_name}_data_normt_T.txt",   notice=False)
hyb.save_combined_matrix(data_norm_x.T,  dx, dt, f"{dir_txtsave}\\{short_name}_data_normx_T.txt",   notice=False)

hyb.save_combined_matrix(xfit_data,   dt, dx, f"{dir_txtsave}\\{short_name}_fitted_data.txt",   notice=False)
hyb.save_combined_matrix(xfit_data.T, dx, dt, f"{dir_txtsave}\\{short_name}_fitted_data_T.txt", notice=False)

dir_trpl_save = hyb.check_make_dir(f"{dir_out}\\trpl")
pd.DataFrame({'Time (ns)': dt, 'Intensity (a.u.)': data_xavg, 'Intensity_norm (a.u.)': data_xavg_norm}) \
  .to_csv(f"{dir_trpl_save}\\{short_name}_x_avg_trpl.csv", sep=',')
t_fitter.params \
  .to_csv(f"{dir_trpl_save}\\{short_name}_x_avg_trpl_fit_params.csv", sep=',')
pd.DataFrame({'Position (um)': dx, 'Intensity (a.u.)': data_tavg}) \
  .to_csv(f"{dir_trpl_save}\\{short_name}_t_avg_profile.csv", sep=',')

dir_fitsave = hyb.check_make_dir(f"{dir_out}\\fitparams")
param_pd = pd.DataFrame(xfit_params, columns=xfit_param_names, index=dt)
param_pd.index.name = "time_ns"
std_pd = pd.DataFrame(xfit_stds, columns=[n + '_std' for n in xfit_param_names], index=dt)
std_pd.index.name = "time_ns"
pd.concat([param_pd, std_pd], axis=1) \
  .to_csv(f"{dir_fitsave}\\{short_name}_fit_params_combined.csv", sep=',', index=True)

print("Job done!")