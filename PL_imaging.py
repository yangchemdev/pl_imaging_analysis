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
import matplotlib   # debug

#%% define
def first_with_w_idx(strings):
    for idx, s in enumerate(strings):
        if 'w' in s:
            return idx
    return None   # if nothing contains 'w'
import numpy as np

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

#%% config
##### data params #####
f_in = None  # path to the .dat file. if None, will prompt user to select file
t_step = 0.016  # time step in ns
motor_step = 5 # motor step in um
mag = 180  # microscopy magnification
##### process params #####
x_range = [-1.5, 1.5]   # spatial range to analyze, in um. If None, will use full range
t_range = [0, 400]  # time range to analyze, in ns. If None, will use full range
t_binning_width = 31 # time binning factor. If None, no binning.
fold_row = None   # end of rows to be folded to the end of data, use None to skip. Unit in row Useful when total measurement time is short.
smooth_x = None  # smoothing boxcar in x direction, no unit. If None, no smoothing
smooth_t = None  # smoothing boxcar in t direction, no unit. If None, no smoothing
x_fit_model = hyf.func_class_gaussian  # model to fit spatial profile
t_fit_model = hyf.exp_ne_wrapper(1, np.array([1]), trig_non_negative=True)  # model to fit time profile. Currently only supports exp decay
trig_MSD_rezero = False # whether to re-zero MSD calculation by subtracting initial MSD value
displacement_source = 'fit' # source of diffusion coefficient calculation. 'fit' to use fitted w, 'MSD' to use MSD
##### visualize params #####
param_units = ['a.u.', 'um', 'um', 'a.u.'] # units for each fitted param, in order
representative_t = [0, 10, 100, 200]     # representative frames to be plotted.
##### output params #####
f_out = None  # path to save output files. If None, will use input file directory

if f_in is None:
    f_in = hyb.GUI_qt_get_file("Select PL imaging .dat file", False, remember=True)
    f_in = f_in[0]
# get dir_in
dir_in, name, _ = hyb.get_file_name_from_dir(f_in)

# get time data
todaydate = datetime.today()
formatted_date = todaydate.strftime('%y%m%d')
# set output dir
dir_out = hyb.check_make_dir(f"{dir_in}\\{name}_output_{formatted_date}", auto_rename=True)

# make config class
params_data = hyconfig.code_section([
    ("f_in", f_in),  # path to the .dat file. if None, will prompt user to select file
    ("dir_in", dir_in),  # directory of input file  
    ("t_step", t_step),  # time step in ns
    ("motor_step", motor_step),  # motor step in um
    ("mag", mag),  # microscopy magnification 
])
params_process = hyconfig.code_section([
    ("x_range", x_range),   # spatial range to analyze, in um. If None, will use full range
    ("t_range", t_range),  # time range to analyze, in ns. If None, will use full range
    ("t_binning_width", t_binning_width), # time binning factor. If None, no binning.
    ('fold_row', fold_row),
    ("smooth_x", smooth_x),  # smoothing boxcar in x direction, no unit. If None, no smoothing
    ("smooth_t", smooth_t),  # smoothing boxcar in t direction, no unit. If None, no smoothing
    ("x_fit_model", x_fit_model.funcname),  # model to fit spatial profile
    ("t_fit_model", t_fit_model.funcname),  # model to fit time profile. Currently only supports exp decay
    ("trig_MSD_rezero", trig_MSD_rezero), # whether to re-zero MSD calculation by subtracting initial MSD value
    ("displacement_source", displacement_source) # source of diffusion coefficient calculation. 'fit' to use fitted w, 'MSD' to use MSD
])
params_visualize = hyconfig.code_section([
    ("param_units", param_units),
    ("representative_t", representative_t)
])
params_output = hyconfig.code_section([
    ("dir_out", dir_out)  # path to save output files. If None, will use input file directory
])

params = hyconfig.config_class([params_data, params_process, params_visualize, params_output])
params.to_json(f"{params_output.dir_out}\\config_files_{formatted_date}.json")
params.to_pickle(f"{params_output.dir_out}\\config_files_{formatted_date}.pkl")
#%% load
data = np.loadtxt(params_data.f_in, delimiter="\t") # I need to check the delimiter
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
data_tavg = np.mean(data, axis = 0)     # average over time. Now only have x axis
data_xavg = np.mean(data, axis = 1)     # average over space. Now only have t axis

# find center
x0 = x[hyb.numpy_nearest(data_tavg, np.max(data_tavg), 'id')]
t0 = t[hyb.numpy_nearest(data_xavg, np.max(data_xavg), 'id')]
print(f"t0 is at {t0} ns, or {np.round(t0/params_data.t_step)} row")

# find net x and net t
dx = x - x0
dt = t - t0

# debg
# use intensity before t0 as background. 5 ns buffer before t0
bg_mask = dt < -2   # NOTE: the buffer is subject to change
if not np.any(bg_mask):
    raise ValueError(f"No time points found for background (dt < -5). "
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
data_max_center = np.max(data, axis=1)
data_norm_t = data / data_max_center[:, np.newaxis]
data_normall = data / np.max(data)
#%% average time fit
# remake average 
data_tavg = np.mean(data, axis = 0)
data_xavg = np.mean(data, axis = 1)
# fit the data_xavg
t_fitter = hyf.fitter_1D('tfit_avg', t_fit_model, dt, data_xavg)
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
    x_fitter = hyf.fitter_1D(f"xfit_#{idt}", x_fit_model, dx, data[idt, :], lb=np.array([0, -0.2, 0.1, 0]), hb=np.array([np.inf, 0.2, 1, np.inf]))
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
if displacement_source == 'fit':
    dis2_sigma = np.abs(xfit_data[:, idx_w]) * xfit_stds[:, idx_w] * 4    # use error propagation to estimate sigma of dis2
    dis2_sigma[np.isnan(dis2_sigma)] = np.nanmax(dis2_sigma)  # rough estimate of noise level
elif displacement_source == 'MSD':
    dis2_sigma = 1 / np.abs(data_xavg)  # rough estimate of noise level
dis2_fitter.fit(sigma=dis2_sigma)    
D = dis2_fitter.params['value'][0] / 4  # diffusion coefficient in 2D, unit in um2/ns
D = D * 10  # convert to cm2/s
D_std = dis2_fitter.params['std'][0] / 4
D_std = D_std * 10  # convert to cm2/s
dis2_fit = dis2_fitter.y_fit
dis2_residual = dis2_fitter.residue
dis2_r2 = dis2_fitter.r2

#%% visualize
# plot raw data
fig, axes = plt.subplots(2,2, figsize=(8,6), dpi=300)
axes = axes.flatten()
# normalized data
# data_normall_log = np.log10(data_normall)
im0 = axes[0].imshow(data_normall, extent=[dx[0], dx[-1], dt[0], dt[-1]], aspect='auto', cmap='viridis', origin = 'lower')
axes[0].set_xlabel('x (um)')
axes[0].set_ylabel('Time (ns)')
cb0 = hyp.colorbar_magic(im0)

# t-normalized data
# data_norm_t_log = np.log10(data_norm_t)
img1 = axes[1].imshow(data_norm_t, extent=[dx[0], dx[-1], dt[0], dt[-1]], aspect='auto', cmap='viridis', origin = 'lower')
axes[1].set_xlabel('x (um)')
axes[1].set_ylabel('Time (ns)')
cb1 = hyp.colorbar_magic(img1)

# spatial-averaged trpl 
plot_raw = axes[2].semilogy(dt, data_xavg, label='spatial averaged data')
plot_fitted = axes[2].semilogy(dt, tfit_data, label='fit', linestyle='--')
axes[2].set_xlabel('Time (ns)')
axes[2].set_ylabel('Intensity (a.u.)')

# representative spatial
plot_x = []
for idtplot, tplot in enumerate(representative_t):
    idt = hyb.numpy_nearest(dt, tplot, 'idx')
    plot_x.append(axes[3].plot(dx, data_norm_t[idt, :], label = f'{dt[idt]:.2f} ns', alpha = 0.5)[0])
axes[3].set_xlabel('x (um)')
axes[3].set_ylabel('Intensity (a.u.)')
axes[3].legend()

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
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3), dpi=300)
axes = axes.flatten()
fig.suptitle(f'PL  parameters over time\n{name}')
# plot fitting params
for idplot in range(n_xfit_params):
    axes[idplot].plot(dt, xfit_params[:, idplot], label=f'Param {idplot}', linewidth=2)
    ylim = axes[idplot].get_ylim()  # freeze current ylim
    lower_bound = xfit_params[:, idplot] - xfit_stds[:, idplot]
    upper_bound = xfit_params[:, idplot] + xfit_stds[:, idplot]
    axes[idplot].fill_between(dt, lower_bound, upper_bound, alpha=0.3)
    axes[idplot].set_title(f'Fitted parameter {xfit_param_names[idplot]}')
    axes[idplot].set_xlabel('Time (ns)')
    axes[idplot].set_ylabel(f'{xfit_param_names[idplot]} ({param_units[idplot]})')
    axes[idplot].set_ylim(ylim)         # reapply ylim
#plot r2
xfit_r2[xfit_status != 1] = -1  # set R2 to -1 if fit failed
axes[idplot+1].plot(dt, xfit_r2, label='R2', color='orange', linewidth=2)
axes[idplot+1].set_xlabel('Time (ns)')
axes[idplot+1].set_ylabel('r2')
axes[idplot+1].set_title('Fitting R2. -1 for failed fit')
# plot dis2
axes[idplot+2].plot(dt, dis2, label='Displacement^2', linestyle = '-')
# axes[idplot+2].plot(dt, dis2_fit, label=hyb.format_SF_err(D, D_std) + 'um2/ns', linestyle='--')
axes[idplot+2].plot(dt_tofit, dis2_fit, label = f'fit,D = {D:.2g} cm2/s', linestyle='--')

axes[idplot+2].set_xlabel('Time (ns)')
axes[idplot+2].set_ylabel('Displacement^2')
axes[idplot+2].set_title('Displacement^2 and linear fit')
axes[idplot+2].legend()

plt.tight_layout()
fig.savefig(f"{dir_out}\\PL_imaging_fitparams_{formatted_date}.png", dpi=300)
plt.close(fig)

#%% save text
# save normalized data
dir_txtsave = f"{dir_out}\\txtdata"
hyb.check_make_dir(dir_txtsave)
hyb.save_combined_matrix(data_normall, dt, dx, f"{dir_txtsave}\\PL_imaging_data_normall_{formatted_date}.txt")
hyb.save_combined_matrix(data_norm_t, dt, dx, f"{dir_txtsave}\\PL_imaging_data_norm_by_t_{formatted_date}.txt")
hyb.save_combined_matrix(xfit_data, dt, dx, f"{dir_txtsave}\\PL_imaging_xfit_data_{formatted_date}.txt")

# save spatial averaged trpl
out_pd = pd.DataFrame({'Time (ns)': dt, 'Intensity (a.u.)': data_xavg})
out_pd.to_csv(f"{dir_txtsave}\\PL_imaging_spatial_avg_trpl_{formatted_date}.csv", sep=',')

# save temporal averaged data
out_pd = pd.DataFrame({'Position (um)': dx, 'Intensity (a.u.)': data_tavg})
out_pd.to_csv(f"{dir_txtsave}\\PL_imaging_temporal_avg_profile_{formatted_date}.csv", sep=',')

# save fit params        
param_pd = pd.DataFrame(xfit_params, columns=xfit_param_names, index=dt)
param_pd.index.name = "param_name"
param_pd.to_csv(f"{dir_txtsave}\\PL_imaging_fit_params_{formatted_date}.csv", sep=',', index=True)

std_pd = pd.DataFrame(xfit_stds, columns=[name + '_std' for name in xfit_param_names], index=dt)
std_pd.index.name = "param_name"
std_pd.to_csv(f"{dir_txtsave}\\PL_imaging_fit_params_std_{formatted_date}.csv", sep=',', index=True)

combined_pd = pd.concat([param_pd, std_pd], axis=1)
combined_pd.to_csv(f"{dir_txtsave}\\PL_imaging_fit_params_combined_{formatted_date}.csv", sep=',', index=True)
#%% finish
print("Job done!")