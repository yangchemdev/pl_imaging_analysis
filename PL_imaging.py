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


#%% config
##### data params #####
f_in = r"F:\OneDrive - purdue.edu\Data\Optical Spectra\PL\PL_image_test\token_data.dat"  # path to the .dat file. if None, will prompt user to select file
t_step = 0.064  # time step in ns
motor_step = 20 # motor step in um
mag = 100  # microscopy magnification
##### process params #####
x_range = [-1.5, 1.5]   # spatial range to analyze, in um. If None, will use full range
t_range = [0, 50]  # time range to analyze, in ns. If None, will use full range
t_binning_width = 11 # time binning factor. If None, no binning.
smooth_x = None  # smoothing boxcar in x direction, no unit. If None, no smoothing
smooth_t = None  # smoothing boxcar in t direction, no unit. If None, no smoothing
x_fit_model = hyf.func_class_gaussian  # model to fit spatial profile
t_fit_model = hyf.exp_ne_wrapper(2, np.array([1, 100]), trig_non_negative=True)  # model to fit time profile. Currently only supports exp decay
trig_MSD_rezero = False # whether to re-zero MSD calculation by subtracting initial MSD value
displacement_source = 'MSD' # source of diffusion coefficient calculation. 'fit' to use fitted w, 'MSD' to use MSD
##### visualize params #####
pass
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
dir_out = hyb.check_make_dir(f"{dir_in}\\output_{formatted_date}", auto_rename=True)

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
    ("smooth_x", smooth_x),  # smoothing boxcar in x direction, no unit. If None, no smoothing
    ("smooth_t", smooth_t),  # smoothing boxcar in t direction, no unit. If None, no smoothing
    ("x_fit_model", x_fit_model.funcname),  # model to fit spatial profile
    ("t_fit_model", t_fit_model.funcname),  # model to fit time profile. Currently only supports exp decay
    ("trig_MSD_rezero", trig_MSD_rezero), # whether to re-zero MSD calculation by subtracting initial MSD value
    ("displacement_source", displacement_source) # source of diffusion coefficient calculation. 'fit' to use fitted w, 'MSD' to use MSD
])
params_visualize = hyconfig.code_section([
    # to be implemented
])
params_output = hyconfig.code_section([
    ("dir_out", dir_out)  # path to save output files. If None, will use input file directory
])

params = hyconfig.config_class([params_data, params_process, params_visualize, params_output])
params.to_json(f"{params_output.dir_out}\\config_files_{formatted_date}.json")
params.to_pickle(f"{params_output.dir_out}\\config_files_{formatted_date}.pkl")
#%% load
data = np.loadtxt(params_data.f_in, delimiter="\t") # I need to check the delimiter
nt = data.shape[0]
nx = data.shape[1]
t = np.arange(nt) * params_data.t_step
x_step = params_data.motor_step / params_data.mag  # in um
x = np.arange(nx) * x_step  # in um

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

# find net x and net t
dx = x - x0
dt = t - t0

# debg
# use intensity before t0 as background. 5 ns buffer before t0
bg_mask = dt < -5   # NOTE: the buffer is subject to change
if not np.any(bg_mask):
    raise ValueError(f"No time points found for background (dt < -5). "
                     f"Min dt = {dt.min():.3f}")
bg = np.median(data[bg_mask, :], axis=0)
data = data - bg[np.newaxis, :]
data[data <= 0] = 1e-3  # set non-positive values to epsilon

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
    x_fitter = hyf.fitter_1D(f"xfit_#{idt}", x_fit_model, dx, data[idt, :])
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
dis2_fitter = hyf.fitter_1D('D_fit', hyf.func_class_linear, dt, dis2)
dis2_fitter.fit()
D = dis2_fitter.params['value'][0] / 4  # diffusion coefficient in 2D
D_std = dis2_fitter.params['std'][0] / 4
dis2_fit = dis2_fitter.y_fit
dis2_residual = dis2_fitter.residue
dis2_r2 = dis2_fitter.r2

#%% visualize
# plot 2D map
fig, axes = plt.subplots(1,3, figsize=(12,4), dpi=300)
data_normall_log = np.log10(data_normall)
im0 = axes[0].imshow(data_normall_log, extent=[dx[0], dx[-1], dt[-1], dt[0]], aspect='auto', cmap='viridis')
axes[0].set_xlabel('x (um)')
axes[0].set_ylabel('Time (ns)')
cb0 = hyp.colorbar_magic(im0)

data_norm_t_log = np.log10(data_norm_t)
img1 = axes[1].imshow(data_norm_t_log, extent=[dx[0], dx[-1], dt[-1], dt[0]], aspect='auto', cmap='viridis')
axes[1].set_xlabel('x (um)')
axes[1].set_ylabel('Time (ns)')
cb1 = hyp.colorbar_magic(img1)

plot_raw = axes[2].semilogy(dt, data_xavg, label='spatial averaged data')
plot_fitted = axes[2].semilogy(dt, tfit_data, label='fit', linestyle='--')
axes[2].set_xlabel('Time (ns)')
axes[2].set_ylabel('Intensity (a.u.)')


fig.suptitle(f'PL imaging raw data\n{name}')
axes[0].set_title('Normalized to global max')
axes[1].set_title('Normalized to each time frame')
axes[2].set_title('Spatially averaged data and fit')

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
    axes[idplot].set_ylabel(f'{idplot} ')
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
axes[idplot+2].plot(dt, dis2_fit, label = 'fit', linestyle='--')

axes[idplot+2].set_xlabel('Time (ns)')
axes[idplot+2].set_ylabel('Displacement^2')
axes[idplot+2].set_title('Displacement^2 and linear fit')
axes[idplot+2].legend()

plt.tight_layout()
fig.savefig(f"{dir_out}\\PL_imaging_fitparams_{formatted_date}.png", dpi=300)
plt.close(fig)

#%% save text
pass
#%% finish
print("Job done!")