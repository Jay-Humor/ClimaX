import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
cmap_z = 'cividis'
cmap_t = 'RdYlBu_r'
cmap_diff = 'bwr'
cmap_error = 'BrBG'
ds1 = xr.open_mfdataset('/home/humor/sugon/ClimaX/test/preds/climax-12h.nc', combine='by_coords')
# ds2 = xr.open_mfdataset('/home/humor/sugon/ClimaX/temp/baseline (1)/lr_5d_t2m_t2m.nc', combine='by_coords')

# new = xr.DataArray(data=ds1['t'].to_numpy(), dims=['time'])
ds1['t2m'] = ds1['t']
print(ds1)
del ds1['t']
del ds1['z']
ds1.to_netcdf('test/preds/climax-12h_.nc')