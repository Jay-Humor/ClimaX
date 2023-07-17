import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.score import *
from collections import OrderedDict

import pickle
def to_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def read_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)
sns.set_style('darkgrid')
sns.set_context('notebook')

res = '5.625'
DATADIR = f'/home/humor/sugon/ClimaX/test/weatherbench/'
PREDDIR = '/home/humor/sugon/ClimaX/test/preds/'
OTHERDIR = '/home/humor/sugon/ClimaX/temp/'

# Load the validation subset of the data: 2017 and 2018
# Ok, actually it's the TEST data but here we will just call it valid
# z500_valid = load_test_data(f'{DATADIR}geopotential_500', 'z')
# t850_valid = load_test_data(f'{DATADIR}temperature_850', 't')
# # for precipitation we are taking 6 hourly accumulations
# tp_valid = load_test_data(f'{DATADIR}total_precipitation', 'tp').rolling(time=6).sum()
# tp_valid.name = 'tp'
t2m_valid = load_test_data(f'{DATADIR}', 't2m')
valid_data = xr.merge([t2m_valid])

# to speed things up, let's only evaluate every 6th hour
valid_data = valid_data.isel(time=slice(0, None, 5*24))

persistence = xr.open_dataset(f'{OTHERDIR}persistence_{res}.nc')
climatology = xr.open_dataset(f'{OTHERDIR}climatology_{res}.nc')
weekly_climatology = xr.open_dataset(f'{OTHERDIR}weekly_climatology_{res}.nc')

variables = ['t2m']
# lr_3d = xr.merge([xr.open_dataset(f'{OTHERDIR}lr_3d_{v}_{v}.nc') for v in variables])
lr_5d = xr.merge([xr.open_dataset(f'{OTHERDIR}lr_5d_{v}_{v}.nc') for v in variables])
lr_6h_iter = xr.open_dataset(f'{OTHERDIR}lr_6h_iter.nc')


# cnn_3d = xr.open_dataset(f'{OTHERDIR}fccnn_3d.nc')
cnn_5d = xr.open_dataset(f'{OTHERDIR}fccnn_5d.nc')
cnn_6h_iter = xr.open_dataset(f'{OTHERDIR}fccnn_6h_iter.nc')

# See next section on how to compute these from the raw data.
tigge = xr.open_dataset(f'{OTHERDIR}tigge_{res}deg.nc')
t42 = xr.open_dataset(f'{OTHERDIR}t42_5.625deg.nc')
t63 = xr.open_dataset(f'{OTHERDIR}t63_5.625deg.nc')

variables = ['t2m']
climax_5d = xr.merge([xr.open_dataset(f'{PREDDIR}climax-5d.nc') for v in variables])
# climax_12h = xr.merge([xr.open_dataset(f'{PREDDIR}climax-12h_.nc') for v in variables])
climax_12h = xr.open_dataset(f'{PREDDIR}climax-12h_.nc')

target_12h = xr.open_dataset(f'{PREDDIR}target-12h_.nc')

func = compute_weighted_rmse
rmse = OrderedDict({
    'Persistence': evaluate_iterative_forecast(persistence, valid_data, func).load(),
    'Climatology': func(climatology, valid_data).load(),
    'Weekly clim.': func(weekly_climatology, valid_data).load(),
    # 'Operational': evaluate_iterative_forecast(tigge, valid_data, func).load(),
    # 'IFS T42': evaluate_iterative_forecast(t42, valid_data, func).load(),
    # 'IFS T63': evaluate_iterative_forecast(t63, valid_data, func).load(),
    'LR (iterative)': evaluate_iterative_forecast(lr_6h_iter, valid_data, func).load(),
    'CNN (iterative)': evaluate_iterative_forecast(cnn_6h_iter, valid_data, func).load(),
    'LR (direct)': xr.concat(
        [
            # func(lr_3d, valid_data),
            func(lr_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
    'CNN (direct)': xr.concat(
        [
            # compute_weighted_rmse(cnn_3d, valid_data),
            compute_weighted_rmse(cnn_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
    'ClimaX (direct)': xr.concat(
        [
            # func(lr_3d, valid_data),
            func(climax_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
    'ClimaX (12h)': xr.concat(
        [
            # func(lr_3d, valid_data),
            func(climax_12h, valid_data)
        ],
        dim=pd.Index([12], name='lead_time')
    ).load(),
})


func = compute_weighted_acc
acc = OrderedDict({
    'Persistence': evaluate_iterative_forecast(persistence, valid_data, func).load(),
    'Climatology': func(climatology, valid_data).load(),
    'Weekly clim.': func(weekly_climatology, valid_data).load(),
    # 'Operational': evaluate_iterative_forecast(tigge, valid_data, func).load(),
    # 'IFS T42': evaluate_iterative_forecast(t42, valid_data, func).load(),
    # 'IFS T63': evaluate_iterative_forecast(t63, valid_data, func).load(),
    'LR (iterative)': evaluate_iterative_forecast(lr_6h_iter, valid_data, func).load(),
    'CNN (iterative)': evaluate_iterative_forecast(cnn_6h_iter, valid_data, func).load(),
    'LR (direct)': xr.concat(
        [
            # func(lr_3d, valid_data),
            func(lr_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
    'CNN (direct)': xr.concat(
        [
            # func(cnn_3d, valid_data),
            func(cnn_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
    'ClimaX (direct)': xr.concat(
        [
            # func(lr_3d, valid_data),
            func(climax_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
})

func = compute_weighted_mae
mae = OrderedDict({
    'Persistence': evaluate_iterative_forecast(persistence, valid_data, func).load(),
    'Climatology': func(climatology, valid_data).load(),
    'Weekly clim.': func(weekly_climatology, valid_data).load(),
    # 'Operational': evaluate_iterative_forecast(tigge, valid_data, func).load(),
    # 'IFS T42': evaluate_iterative_forecast(t42, valid_data, func).load(),
    # 'IFS T63': evaluate_iterative_forecast(t63, valid_data, func).load(),
    'LR (iterative)': evaluate_iterative_forecast(lr_6h_iter, valid_data, func).load(),
    'CNN (iterative)': evaluate_iterative_forecast(cnn_6h_iter, valid_data, func).load(),
    'LR (direct)': xr.concat(
        [
            # func(lr_3d, valid_data),
            func(lr_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
    'CNN (direct)': xr.concat(
        [
            # func(cnn_3d, valid_data),
            func(cnn_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
    'ClimaX (direct)': xr.concat(
        [
            # func(lr_3d, valid_data),
            func(climax_5d, valid_data)
        ],
        dim=pd.Index([120], name='lead_time')
    ).load(),
})

colors = {
    'Persistence': '0.2',
    'Climatology': '0.5',
    'Weekly clim.': '0.7',
    'Operational': '#984ea3',
    'IFS T42': '#4daf4a',
    'IFS T63': '#377eb8',
    'LR (iterative)': '#ff7f00',
    'LR (direct)': '#ff7f00',
    'CNN (iterative)': '#e41a1c',
    'CNN (direct)': '#e41a1c',
    'ClimaX (direct)': '#984ea3',
}

import cartopy.crs as ccrs

sns.set_style('dark')

cmap_t2m = 'cividis'
cmap_t = 'RdYlBu_r'
cmap_diff = 'bwr'
cmap_error = 'BrBG'

def imcol(ax, data, title='', **kwargs):
    # if not 'vmin' in kwargs.keys():
    #     mx = np.abs(data.max().values)
    #     kwargs['vmin'] = -mx; kwargs['vmax'] = mx
#     I = ax.imshow(data, origin='lower',  **kwargs)
    I = data.plot(ax=ax, transform=ccrs.PlateCarree(),
                  rasterized=True, **kwargs)
    # cb = fig.colorbar(I, ax=ax, orientation='horizontal', pad=0.01, shrink=0.90)
    ax.set_title(title)
    ax.coastlines(alpha=0.5)

fig, axs = plt.subplots(3, 3, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})
# True
for iax, var, cmap, r, t in zip(
    [0], ['t2m'], [cmap_t2m], [[240, 310]], ['T2M [K]']):
    imcol(axs[iax,0], valid_data[var].isel(time=0),  
           title=f'ERA5 {t} t=0h')
    imcol(axs[iax,1], valid_data[var].isel(time=5*24),  
           title=f'ERA5 {t} t=5d')
    imcol(axs[iax,2], 
        valid_data[var].isel(time=5*24)-valid_data[var].isel(time=0), cmap=cmap_diff, 
        title=f'ERA5 {t} diff (5d-0h)')

# CNN
for iax, var, cmap, r, t in zip(
    [1], ['t2m'], [cmap_t2m], [[240, 310]], ['T2M [K]']):
    imcol(axs[iax,0], valid_data[var].isel(time=0),  
           title=f'ERA5 {t} t=0h')
    imcol(axs[iax,1], lr_5d[var].isel(time=0),  
           title=f'lr {t} t=5d')
    imcol(axs[iax,2], 
        lr_5d[var].isel(time=0) - valid_data[var].isel(time=5*24), cmap=cmap_error,
        title=f'Error lr - ERA5 {t} t=5d')

# print(climax_12h)
# print(climax_5d)

# ClimaX
for iax, var, cmap, r, t in zip(
    [2], ['t2m'], [cmap_t2m], [[240, 310]], ['T2M [K]']):
    imcol(axs[iax,0], valid_data[var].isel(time=0),  
           title=f'ERA5 {t} t=0h')
    imcol(axs[iax,1], climax_5d[var].isel(time=0),  
           title=f'ClimaX {t} t=5d')
    imcol(axs[iax,2], 
        climax_5d[var].isel(time=0) - valid_data[var].isel(time=5*24), 
        title=f'Error ClimaX - ERA5 {t} t=5d')

for ax in axs.flat:
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(pad=0)
# plt.savefig('./figures/examples.pdf', bbox_inches='tight')
plt.savefig('./figures/examples.jpeg', bbox_inches='tight', dpi=300)