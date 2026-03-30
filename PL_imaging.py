'''
Made by; Hanjun Yang, 2025
Thia module import and analyze PL imaging data.
Dependencies:
    TBD

input: 
    1. .dat files containing PL imaging data. Each column represent a TRPL trace. Each row represent a time delay. ([time, pixel] indexing)
        - so data[m,n] is the PL intensity at time delay m for pixel n.
    2. time step
    3. pixel step
    4. 

output:
    normalized PL imaging data
    fit result
'''
#%% import
import os
import numpy as np
from hytools import (
    hy_basic as hyb,
    hy_fit as hyf,
    hy_plot as hyp,
    hy_configclass as hyconfig,
)
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm as tqdm
import matplotlib
from scipy.signal import savgol_filter

#%% define
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

def iterative_argmax_find(arr, n_peaks, min_distance = 0):
    ''' Iteratively find the n_peak maximum in arr, with a minimum distance constraint. Returns the indices of the peaks. '''
    arr = np.asarray(arr)
    peaks = np.zeros(n_peaks, dtype=int)
    temp_arr = arr.copy()
    for i in range(n_peaks):
        idx = np.nanargmax(temp_arr)
        peaks[i] = idx
        # cut 
        lo = max(0, idx - min_distance)
        hi = min(len(temp_arr), idx + min_distance + 1)
        temp_arr[lo:hi] = np.nan
    return np.sort(peaks)

def first_with_w_idx(strings):
    for idx, s in enumerate(strings):
        if 'w' in s:
            return idx
    return None   # if nothing contains 'w'

def bin_rows_2D(data: np.ndarray, w: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("data must be 2D")
    if w <= 0:
        raise ValueError("w must be positive")
    N = data.shape[0]
    K = N // w  # number of full bins
    trimmed = data[:K * w]  # discard leftover rows

    # reshape to (K, m, D) then average along axis 1
    return trimmed.reshape(K, w, -1).mean(axis=1)

def bin_rows_1D(data: np.ndarray, w: int) -> np.ndarray:
    if data.ndim != 1:
        raise ValueError("data must be 1D")
    if w <= 0:
        raise ValueError("w must be positive")

    N = data.shape[0]
    K = N // w  # number of full bins
    trimmed = data[:K * w]  # discard leftover rows

    # reshape to (K, w) then average along axis 1
    return trimmed.reshape(K, w).mean(axis=1)

def fold_trpl(data: np.ndarray, row0: int) -> np.ndarray:
    '''
    fold numpy array    
    '''
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
    if data.ndim != 2:
        raise ValueError("Input must be 2D.")

    # Boolean mask: True if row has any non-zero element
    mask = np.any(data != 0, axis=1)

    # Find indices where mask is True
    idx = np.where(mask)[0]

    return idx[-1] if idx.size > 0 else None   # None means all-zero matrix

def fix_replica(data, t_step, replica_n, min_dt, t0_buffer = 1, debug = False):
    '''
    Fix 447 replica by finding the replica peaks, aligning them, and averaging over them.
    Arguments:
        data: 2D array of shape (time, pixel)
        t_step: float, time step in ns
        replica_n: int, number of replica to find
        min_dt: float, minimum dt between replica in ns
        t0_buffer: float, buffer before t0 to calculate background, in ns. Used when subtracting background
    '''
    # xavg and smooth
    data_xavg = np.mean(data, axis=1)  # spatially averaged data
    data_xavg_sm = adjacent_average(data_xavg, window=6) # 447 t-reso is usually 8, makes window length ~13*0.008 = ~0.1 ns
    # find replica peaks.
    n_min_dt = int(min_dt // t_step)  # convert min_dt to number of rows
    replica_peaks = iterative_argmax_find(data_xavg_sm, n_peaks=replica_n, min_distance= n_min_dt)
    # sanity check: replica peaks should be roughly equally spaced
    replica_peaks_diff = np.diff(replica_peaks)
    assert np.max(replica_peaks_diff) - np.min(replica_peaks_diff) <= 12 , f"Replica peaks are not roughly equally spaced. peaks are row #: {replica_peaks}."
    # 12 point is for ~0.1 ns error.
    replica_length = int(np.median(replica_peaks_diff))  
    # align and sum-over the replica
    n_t0buffer = int(t0_buffer // t_step)
    data_replica = np.full((replica_length, data.shape[1], replica_n), np.nan)     # indexing: [time, pixel, replica]
    for i in range(replica_n):
        data_replica[:, :, i] = data[replica_peaks[i]-n_t0buffer : replica_peaks[i]+replica_length-n_t0buffer, :]
    # some replica have trailing zeros.
    for i in range(replica_n):
        last_nonzero_idx = last_nonzero_row_idx(data_replica[:, :, i])
        if last_nonzero_idx is not None and last_nonzero_idx < replica_length - 1:
            data_replica[last_nonzero_idx+1:, :, i] = np.nan  # set to nan for later cut-off
    data_fixed = np.nanmean(data_replica, axis=2)   # average over replica
    data_fixed[np.isnan(data_fixed)] = 0  #HACK: set nan to 0.
    if debug:
        return data_fixed, data_replica, replica_peaks
    else:
        return data_fixed

#%% config
##### data params #####
f_in = None  # path to the .dat file. if None, will prompt user to select file
t_step = 0.008  # time step in ns
motor_step = 10  # motor step in um
mag = 182  # microscopy magnification
trig_447_replica_fix = True  # When 447 laser is used, it sometimes create 4 replica. This trigger will align and sum-over the 4 replica.
replica_n = 4 # number of replica.
min_dt = 5  # minimum dt between replica, in ns.
##### process params #####
x_range = None   # spatial range to analyze, in um. If None, will use full range
t_range = [-0.5, 10]  # time range to analyze, in ns. If None, will use full range
t0_buffer = 1  # buffer before t0 to calculate background, in positive ns.
t_binning_width = 10 # time binning factor. If None, no binning.
fold_row = None   # end of rows to be folded to the end of data, use None to skip. Unit in row Useful when total measurement time is short.
x_fit_model = hyf.func_class_gaussian  # model to fit spatial profile
t_fit_model = hyf.exp_ne_wrapper(2, np.array([1, 5]), trig_non_negative=True)  # model to fit time profile. Currently only supports hyf.exp_ne_wrapper
trig_MSD_rezero = False # whether to re-zero MSD calculation by subtracting initial MSD value
displacement_source = 'fit' # source of diffusion coefficient calculation. 'fit' to use fitted w, 'MSD' to use MSD
sigma_correction = True # whether to apply sigma correction in D fitting
##### visualize params #####
param_units = ['a.u.', 'um', 'um', 'a.u.'] # units for each fitted param, in order
representative_t = [0, 1, 4, 8]     # representative frames to be plotted.
##### output params #####
f_out = None  # path to save output files. If None, will use input file directory
overwrite_mode = False # whether to overwrite existing output files
##### END OF CONFIG #####
if f_in is None:
    f_in = hyb.GUI_qt_get_file("Select PL imaging .dat file", False, remember=True)
    f_in = f_in[0]
# get dir_in
dir_in, name, _ = hyb.get_file_name_from_dir(f_in)

# get time data
todaydate = datetime.today()
formatted_date = todaydate.strftime('%y%m%d')
# set output dir
dir_out = hyb.check_make_dir(f"{dir_in}\\{name}_output_{formatted_date}", auto_rename = not overwrite_mode)

# make config class
params_data = hyconfig.code_section({
    "f_in": f_in,  # path to the .dat file. if None, will prompt user to select file
    "dir_in": dir_in,  # directory of input file  
    "t_step": t_step,  # time step in ns
    "motor_step": motor_step,  # motor step in um
    "mag": mag,  # microscopy magnification 
})
params_process = hyconfig.code_section({
    "x_range": x_range,   # spatial range to analyze, in um. If None, will use full range
    "t_range": t_range,  # time range to analyze, in ns. If None, will use full range
    "t0_buffer": t0_buffer,   # buffer before t0 to calculate background, in ns. Used when subtracting background
    "t_binning_width": t_binning_width, # time binning factor. If None, no binning.
    "fold_row": fold_row,
    "x_fit_model": x_fit_model.funcname,  # model to fit spatial profile
    "t_fit_model": t_fit_model.funcname,  # model to fit time profile. Currently only supports exp decay
    "trig_MSD_rezero": trig_MSD_rezero, # whether to re-zero MSD calculation by subtracting initial MSD value
    "displacement_source": displacement_source, # source of diffusion coefficient calculation. 'fit' to use fitted w, 'MSD' to use MSD
    "sigma_correction": sigma_correction # whether to apply sigma correction in D fitting
})
params_visualize = hyconfig.code_section({
    "param_units": param_units,
    "representative_t": representative_t
})
params_output = hyconfig.code_section({
    "dir_out": dir_out,  # path to save output files. If None, will use input file directory
    "overwrite_mode": overwrite_mode # whether to overwrite existing output files
})

params = hyconfig.config_class([params_data, params_process, params_visualize, params_output])
params.to_json(f"{params_output.dir_out}\\config_files_{formatted_date}.json")
params.to_pickle(f"{params_output.dir_out}\\config_files_{formatted_date}.pkl")
#%% load
data = np.loadtxt(params_data.f_in, delimiter="\t") # I need to check the delimiter

# 447 replica fix
if trig_447_replica_fix:
    data, data_replica, n_replica = fix_replica(data, t_step, replica_n, min_dt, 2 * t0_buffer, debug = True)   # data_replica is only for debug
    print(f"447 replica fixed. Found replica peaks at rows {n_replica}. Averaged over {replica_n} replica.")
# delete trailing 0s
data = data[:last_nonzero_row_idx(data)+1]

nt = data.shape[0]
nx = data.shape[1]

t = np.arange(nt) * params_data.t_step
x_step = params_data.motor_step / params_data.mag  # in um
x = np.arange(nx) * x_step  # in um
# fold
if fold_row is not None:
    data = fold_trpl(data, fold_row)
    t = np.arange(nt) * params_data.t_step


# t binning
if t_binning_width is not None:
    half_bin = t_binning_width // 2
    data = bin_rows_2D(data, t_binning_width)
    t = bin_rows_1D(t, t_binning_width)
    nt = data.shape[0]

# make average 
data_tavg_temp = np.mean(data, axis = 0)     # average over time. Now only have x axis. Note this is before processing so shouldn't be used later
data_xavg_temp = np.mean(data, axis = 1)     # average over space. Now only have t axis
data_tavg_temp = data_tavg_temp / np.max(data_tavg_temp)   # normalize 
data_xavg_temp = data_xavg_temp / np.max(data_xavg_temp)   # normalize 

# find center
# NOTE: weird logic
x0 = x[hyb.numpy_nearest(data_tavg_temp, np.max(data_tavg_temp), 'id')]
t0 = t[hyb.numpy_nearest(data_xavg_temp, np.max(data_xavg_temp), 'id')]
print(f"t0 is at {t0} ns, or {np.round(t0/params_data.t_step)} row")

# find net x and net t
dx = x - x0
dt = t - t0

# debg
# use intensity before t0 as background. 5 ns buffer before t0
bg_mask = dt < -1*t0_buffer   # NOTE: the buffer is subject to change
if not np.any(bg_mask):
    raise ValueError(f"No time points found for background (dt < {t0_buffer}). "
                     f"Min dt = {dt.min():.3f}")
bg = np.percentile(data[bg_mask, :], 45, axis=0)
data = data - bg[np.newaxis, :]
data[data <= 0] = 0.1  # set non-positive values to epsilon

# cut
if x_range is not None:
    x_mask = (dx >= x_range[0]) & (dx <= x_range[1])
    dx = dx[x_mask]
    x = x[x_mask]
    nx = x.shape[0]
    data = data[:, x_mask]

if t_range is not None:
    t_mask = (dt >= t_range[0]) & (dt <= t_range[1])
    dt = dt[t_mask]
    t = t[t_mask]
    nt = t.shape[0]
    data = data[t_mask, :]

print(f"New t is {dt[0]} to {dt[-1]}")  # debug only

# smooth
pass  # TODO: to be implemented

# norm
data_max_per_frame = np.max(data, axis=1)
data_max_per_pixel = np.max(data, axis=0)

data_norm_t = data / data_max_per_frame[:, np.newaxis]
data_norm_x = data / data_max_per_pixel[np.newaxis, :]
data_normall = data / np.max(data)
#%% average time fit
# remake average 
data_tavg = np.mean(data, axis = 0)
data_xavg = np.mean(data, axis = 1)
data_tavg_norm = data_tavg / np.max(data_tavg)   # normalize 
data_xavg_norm = data_xavg / np.max(data_xavg)   # normalize 

# fit the data_xavg
t_fitter = hyf.fitter_1D('tfit_avg', t_fit_model, dt, data_xavg_norm)
t_fitter.fit()
tfit_params = t_fitter.params['value']
tfit_stds = t_fitter.params['std']
tfit_data = t_fitter.y_fit
tfit_residual = t_fitter.residue
# loop over t to fit each spatial profile
# make data container
n_xfit_params = x_fit_model.n_params
xfit_param_names = x_fit_model.param_names

xfit_params = np.zeros((nt, n_xfit_params))
xfit_stds = np.zeros((nt, n_xfit_params))
xfit_data = np.zeros_like(data)
xfit_residual = np.zeros_like(data)
xfit_r2 = np.zeros(nt)
xfit_status = np.zeros(nt)
# fit
for idt in tqdm(range(nt)):
    x_fitter = hyf.fitter_1D(f"xfit_#{idt}", x_fit_model, x, data[idt, :], 
                             lb = np.array([0, -np.inf, 0.1, -np.inf]), 
                             hb = np.array([np.inf, np.inf, 1, np.inf]),
                             p0 = np.array([np.max(data[idt, :]), x0, 0.5, 0]))  # [A, x0, w, offset]
    x_fitter.fit()
    xfit_params[idt, :] = x_fitter.params['value']
    xfit_stds[idt, :] = x_fitter.params['std']
    xfit_data[idt, :] = x_fitter.y_fit
    xfit_residual[idt, :] = x_fitter.residue
    xfit_r2[idt] = x_fitter.r2
    xfit_status[idt] = x_fitter.status

# calculate MSD
MSD = np.zeros(nt)
for idt in range(nt):
    if trig_MSD_rezero == True:
        meanx = np.sum(data[idt, :] * dx) / np.sum(data[idt, :])
        dx_msd = dx - meanx
    else:
        dx_msd = dx
    weight = data[idt, :] / np.sum(data[idt, :])
    MSD[idt] = np.sum(weight * dx_msd**2)

# calculate D
if displacement_source == 'fit':
    idx_w = first_with_w_idx(xfit_param_names)  # find the width parameter
    dis2 = 2*xfit_params[:, idx_w]**2  # squared displacement in 2D
elif displacement_source == 'MSD':
    dis2 = MSD
else: 
    raise ValueError(f"Unknown D_source: {displacement_source}. Should be 'fit' or 'MSD'.")
# linear fit to dis2 vs time
# remove nan
valid_mask = ~np.isnan(dis2)
dis2_tofit = dis2[valid_mask]
dt_tofit = dt[valid_mask]

# make fitter
dis2_fitter = hyf.fitter_1D('D_fit', hyf.func_class_linear, dt_tofit, dis2_tofit)
# make sigma for nan and other illegal value
if sigma_correction is True:
    if displacement_source == 'fit':
        # dis2_sigma = np.abs(xfit_data[:, idx_w]) * xfit_stds[:, idx_w] * 4    # old
        dis2_sigma = np.abs(4 * xfit_params[:, idx_w]) * xfit_stds[:, idx_w]    # new error prop
        dis2_sigma = dis2_sigma[valid_mask]
    elif displacement_source == 'MSD':
        dis2_sigma = 1 / np.abs(data_xavg)  # rough estimate of noise level
        dis2_sigma = dis2_sigma[valid_mask]
    # reduce very small sigma. 0.1~1 ns usually fit well, so I will set 50% of that as floor 
    t0id0 = hyb.numpy_nearest(dt_tofit, 0.1, 'idx',direction=1)
    t1id0 = hyb.numpy_nearest(dt_tofit, 1, 'idx', direction=1)
    t1id0 = max(t0id0+5, t1id0)  # ensure at least 5 points
    print(f"Using {t0id0} to {t1id0} rows to calculate sigma floor.")   # debug
    dis2_sigma_floor = np.median(dis2_sigma[t0id0:t1id0]) * 0.5  # set floor to 50% of median of early sections
    dis2_sigma += dis2_sigma_floor  # add floor
else:
    dis2_sigma = np.ones_like(dis2_tofit) * np.mean(dis2_tofit) * 0.1  # same weight, normalized to average value so that it is easy to plot together with dis2
dis2_fitter.fit(sigma=dis2_sigma)   
# dis2_fitter.fit() 
D = dis2_fitter.params['value'][0] / 4  # diffusion coefficient in 2D, unit in um2/ns
D = D * 10  # convert to cm2/s
D_std = dis2_fitter.params['std'][0] / 4
D_std = D_std * 10  # convert to cm2/s
dis2_fit = dis2_fitter.y_fit
dis2_residual = dis2_fitter.residue
dis2_r2 = dis2_fitter.r2

#%% visualize
matplotlib.rcParams['font.size'] = 7   # size too large... HACK: global setting.
# plot raw data
fig, axes = plt.subplots(2,2, figsize=(6, 5), dpi=150)
axes = axes.flatten()
# normalized data
# data_normall_log = np.log10(data_normall)
im0 = axes[0].imshow(data_normall, extent=[x[0], x[-1], dt[0], dt[-1]], aspect='auto', cmap='viridis', origin = 'lower')
axes[0].set_xlabel('x (um)')
axes[0].set_ylabel('Time (ns)')
cb0 = hyp.colorbar_magic(im0)

# t-normalized data
# data_norm_t_log = np.log10(data_norm_t)
img1 = axes[1].imshow(data_norm_t, extent=[x[0], x[-1], dt[0], dt[-1]], aspect='auto', cmap='viridis', origin = 'lower')
axes[1].set_xlabel('x (um)')
axes[1].set_ylabel('Time (ns)')
cb1 = hyp.colorbar_magic(img1)

# TRPL 
plot_raw = axes[2].semilogy(dt, data_xavg_norm, label='spatial averaged data')
idcenter = hyb.numpy_nearest(dx, 0, 'idx')
plot_fitted = axes[2].semilogy(dt, tfit_data, label='fit', linestyle='--')
plot_center = axes[2].semilogy(dt, data_normall[:, idcenter], label='center')
axes[2].set_xlabel('Time (ns)')
axes[2].set_ylabel('Intensity (a.u.)')
# get phenmonlogical lifetime form 1/e
lifetime_1e = dt[hyb.numpy_nearest(data_xavg, np.max(data_xavg)/np.e, 'idx')]
axes[2].axvline(lifetime_1e, color='gray', linestyle=':', label=f'1/e lifetime: {lifetime_1e:.2f} ns')
axes[2].legend(fontsize = 8)

# representative spatial
plot_x = []
for idtplot, tplot in enumerate(representative_t):
    idt = hyb.numpy_nearest(dt, tplot, 'idx')
    plot_x.append(axes[3].plot(dx, data_norm_t[idt, :], label = f'{dt[idt]:.2f} ns', alpha = 0.5)[0])
axes[3].set_xlabel('x (um)')
axes[3].set_ylabel('Intensity (a.u.)')
axes[3].legend(fontsize = 8)

fig.suptitle(f'PL imaging raw data\n{name}')
axes[0].set_title('Normalized to global max')
axes[1].set_title('Normalized to each time frame')
axes[2].set_title('Spatially averaged data and fit')
axes[3].set_title('Representative spatial distribution')

plt.tight_layout()
fig.savefig(f"{dir_out}\\PL_imaging_raw_{formatted_date}.png", dpi=300)
plt.close(fig)

# plot fitting result
# find optimal row and col numbers
n_params_to_draw = n_xfit_params+2  # plus one for R2 and D2
n_cols = int(np.ceil(np.sqrt(n_params_to_draw)))
n_rows = int(np.ceil(n_params_to_draw / n_cols))
# make figure
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*2.3, n_rows*2.3), dpi=300)
axes = axes.flatten()
fig.suptitle(f'PL  parameters over time\n{name}')
# plot fitting params
# get palattes
color_palatte = plt.cm.Dark2.colors

for idplot in range(n_xfit_params):
    axes[idplot].plot(dt, xfit_params[:, idplot], label=f'Param {idplot}', linewidth = 1.5, color = color_palatte[idplot % len(color_palatte)],zorder=2)
    # axes[idplot].scatter(dt, xfit_params[:, idplot], label=f'Param {idplot}', s = 8, edgecolor = color_palatte[idplot % len(color_palatte)], facecolors = (1,1,1),zorder=3)
    ylim = axes[idplot].get_ylim()  # freeze current ylim
    lower_bound = xfit_params[:, idplot] - xfit_stds[:, idplot]
    upper_bound = xfit_params[:, idplot] + xfit_stds[:, idplot]
    axes[idplot].fill_between(dt, lower_bound, upper_bound, color = color_palatte[idplot % len(color_palatte)], alpha=0.3)
    axes[idplot].set_title(f'{xfit_param_names[idplot]}')
    axes[idplot].set_xlabel('Time (ns)')
    axes[idplot].set_ylabel(f'{xfit_param_names[idplot]} ({param_units[idplot]})')
    axes[idplot].set_ylim(ylim)         # reapply ylim
#plot r2
xfit_r2[xfit_status != 1] = -1  # set R2 to -1 if fit failed
axes[idplot+1].plot(dt, xfit_r2, label='R2', color='orange', linewidth=1)
axes[idplot+1].set_xlabel('Time (ns)')
axes[idplot+1].set_ylabel('r2')
axes[idplot+1].set_title('Fitting R2. -1 for failed fit')
# plot dis2
# axes[idplot+2].fill_between(dt_tofit, dis2_tofit-dis2_sigma, dis2_tofit+dis2_sigma, alpha = 0.3, linewidth=0.5)    # sigma for D fitting
axes[idplot+2].plot(dt_tofit, dis2_tofit, label='Displacement^2', linewidth=1)  # input for D fitting
if dis2_fitter.status == 1:
    axes[idplot+2].plot(dt_tofit, dis2_fit, label=hyb.format_SF_err(D, D_std) + ' cm2/s', color='r', linestyle='--', alpha = 0.5)  # fitted line
axes[idplot+2].set_xlabel('Time (ns)')
axes[idplot+2].set_ylabel('Displacement^2 (um^2)')
axes[idplot+2].set_title('Displacement^2 (um^2)')
axes[idplot+2].legend(fontsize=6)
# exclude outliers
ymin = np.nanpercentile(dis2_tofit, 5)
ymax = np.nanpercentile(dis2_tofit, 95)
axes[idplot+2].set_ylim(ymin*0.8, ymax*1.2)

# estimate and plot fitting weight
xfit_weights = 1 / dis2_sigma**2
xfit_weights = xfit_weights / np.max(xfit_weights)  # normalize to max 0.3 for plotting

ax2 = axes[idplot+2].twinx()
ax2.plot(dt_tofit, xfit_weights, color='k', alpha=0.4, linewidth=0.5)
ax2.set_ylabel("fitting weight (0–1)")

plt.tight_layout()
fig.savefig(f"{dir_out}\\PL_imaging_fitparams_{formatted_date}.png", dpi=300)
plt.close(fig)

#%% save text
# save normalized data
import re
short_name = re.sub(r"p\d+u\d+$", "", name) # remove pXXuXX at the end of the name
dir_txtsave = os.path.normpath(f"{dir_out}\\txtdata")
dir_txtsave = hyb.check_make_dir(dir_txtsave)
hyb.save_combined_matrix(data, dt, dx, f"{dir_txtsave}\\{short_name}_data_raw.txt", notice=True)    # only proc notice for the 1st save.
hyb.save_combined_matrix(data_normall, dt, dx, f"{dir_txtsave}\\{short_name}_data_normall.txt", notice=False)
hyb.save_combined_matrix(data_norm_t, dt, dx, f"{dir_txtsave}\\{short_name}_data_normt.txt", notice=False)
hyb.save_combined_matrix(data_norm_x, dt, dx, f"{dir_txtsave}\\{short_name}_data_normx.txt", notice=False)

hyb.save_combined_matrix(data.T, dx, dt, f"{dir_txtsave}\\{short_name}_data_raw_T.txt", notice=False)
hyb.save_combined_matrix(data_normall.T, dx, dt, f"{dir_txtsave}\\{short_name}_data_normall_T.txt", notice=False)
hyb.save_combined_matrix(data_norm_t.T, dx, dt, f"{dir_txtsave}\\{short_name}_data_normt_T.txt", notice=False)
hyb.save_combined_matrix(data_norm_x.T, dx, dt, f"{dir_txtsave}\\{short_name}_data_normx_T.txt", notice=False)

hyb.save_combined_matrix(xfit_data, dt, dx, f"{dir_txtsave}\\{short_name}_fitted_data.txt", notice=False)
hyb.save_combined_matrix(xfit_data.T, dx, dt, f"{dir_txtsave}\\{short_name}_fitted_data_T.txt", notice=False)

dir_trpl_save = hyb.check_make_dir(f"{dir_out}\\trpl")

# save spatial averaged trpl
out_pd = pd.DataFrame({'Time (ns)': dt, 'Intensity (a.u.)': data_xavg, 'Intensity_norm (a.u.)': data_xavg_norm})
out_pd.to_csv(f"{dir_trpl_save}\\{short_name}_x_avg_trpl.csv", sep=',')
# save spatial averaged fit
out_pd = t_fitter.params
out_pd.to_csv(f"{dir_trpl_save}\\{short_name}_x_avg_trpl_fit_params.csv", sep=',')

# save temporal averaged data
out_pd = pd.DataFrame({'Position (um)': dx, 'Intensity (a.u.)': data_tavg})
out_pd.to_csv(f"{dir_trpl_save}\\{short_name}_t_avg_profile.csv", sep=',')

# save fit params    
dir_fitsave = f"{dir_out}\\fitparams"    
hyb.check_make_dir(dir_fitsave)

param_pd = pd.DataFrame(xfit_params, columns=xfit_param_names, index=dt)
param_pd.index.name = "param_name"
# param_pd.to_csv(f"{dir_fitsave}\\{short_name}_fit_params.csv", sep=',', index=True)

std_pd = pd.DataFrame(xfit_stds, columns=[name + '_std' for name in xfit_param_names], index=dt)
std_pd.index.name = "param_name"
# std_pd.to_csv(f"{dir_fitsave}\\{short_name}_fit_params_std.csv", sep=',', index=True)     # only save combined matrix 

combined_pd = pd.concat([param_pd, std_pd], axis=1)
combined_pd.to_csv(f"{dir_fitsave}\\{short_name}_fit_params_combined.csv", sep=',', index=True)
#%% finish
print("Job done!")