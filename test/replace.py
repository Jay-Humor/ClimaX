import xarray as xr
import numpy as np
import pandas as pd

ds1 = xr.open_dataset('/home/humor/sugon/ClimaX/temp/lr_6h_iter.nc')

ds2 = np.load('/home/humor/sugon/ClimaX/test/preds/climax-12h.npy')

mean = np.load('/home/humor/sugon/ClimaX/temp/normalize_mean.npz')

std = np.load('/home/humor/sugon/ClimaX/temp/normalize_std.npz')

full_dates = pd.date_range(start='2017-01-01', periods=2*365*24, freq='H')

ds1_dates = pd.DatetimeIndex(ds1.time.values)

predict_time = 12
# print(ds1)
# 创建一个与numpy数据集相对应的日期数据集
ds2_dates = []
num_partitions = 8
hours_per_partition = 365 * 24 // num_partitions
for year in range(2):  # 循环两年
    for i in range(num_partitions):
        start = year * 365 * 24 + i * hours_per_partition + predict_time
        ds2_dates.extend(full_dates[start:start+hours_per_partition - predict_time])
ds2_dates = pd.DatetimeIndex(ds2_dates)

# 找出numpy数据集中不存在的日期
missing_dates = ds1_dates.difference(ds2_dates)

# 打印缺失的日期
print("Missing Dates: ", missing_dates)

# 删除 nc 文件中与 numpy 数组不对应的日期
new_ds = ds1.sel(time=~ds1.time.isin(missing_dates))
ds2 = np.squeeze(ds2)
ds2 = ds2 * std['2m_temperature'] + mean['2m_temperature']
print(new_ds['t'].to_numpy()[0][0])
print(ds2[0][0])
print((new_ds['t'].to_numpy()-ds2).max())

# 将 nc 文件的数据替换为 numpy 数据
new_ds['t'][:] = ds2
print(ds1)
print(new_ds)
new_ds.to_netcdf('test/preds/climax-12h.nc')