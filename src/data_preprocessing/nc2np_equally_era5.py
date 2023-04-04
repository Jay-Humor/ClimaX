# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import glob #导入glob模块，用于查找符合特定规则的文件路径名。
import os #导入os模块，提供了一种与操作系统进行交互的便携式方法。

import click #导入click模块，用于创建命令行界面。
import numpy as np #导入numpy模块，用于进行数值计算。
import xarray as xr #导入xarray模块，用于操作多维数组。
from tqdm import tqdm #导入tqdm模块，用于显示进度条。

from climax.utils.data_utils import DEFAULT_PRESSURE_LEVELS, NAME_TO_VAR #导入自定义的函数和变量。

HOURS_PER_YEAR = 8760 # 365天*24小时

def nc2np(path, variables, years, save_dir, partition, num_shards_per_year):
    '''
    将ERA5资料转换为numpy数组, 并将其存储为.npy文件。
    Args:
        path (string): 资料所在路径
        variables (list): 变量名列表
        years (list): 年份列表
        save_dir (string): 存储路径
        partition (string): "train" 或 "test", 数据集类型
        num_shards_per_year (int): 每年的分片数
    '''
    # 创建目录
    os.makedirs(os.path.join(save_dir, partition), exist_ok=True)

    if partition == "train": # 如果为训练集
        normalize_mean = {} # 创建空字典，用于存放归一化后的均值
        normalize_std = {} # 创建空字典，用于存放归一化后的标准差
    climatology = {} # 创建空字典，用于存放气候资料

    constants = xr.open_mfdataset(os.path.join(path, "constants.nc"), combine="by_coords", parallel=True) # 打开常量资料
    constant_fields = ["land_sea_mask", "orography", "lattitude"] # 常量变量名
    constant_values = {} # 创建空字典，用于存放常量资料
    for f in constant_fields:
        constant_values[f] = np.expand_dims(constants[NAME_TO_VAR[f]].to_numpy(), axis=(0, 1)).repeat(HOURS_PER_YEAR, axis=0) # 获取常量值并将维度扩展
        if partition == "train": # 如果为训练集
            normalize_mean[f] = constant_values[f].mean(axis=(0, 2, 3)) # 计算均值并存储
            normalize_std[f] = constant_values[f].std(axis=(0, 2, 3)) # 计算标准差并存储

    for year in tqdm(years): # 迭代每一年份
        np_vars = {} # 创建空字典，用于存放numpy数组

        # 常量变量
        for f in constant_fields:
            np_vars[f] = constant_values[f]

        # 非常量变量
        for var in variables:
            ps = glob.glob(os.path.join(path, var, f"*{year}*.nc")) # 查找变量所对应的.nc文件路径
            ds = xr.open_mfdataset(ps, combine="by_coords", parallel=True)  # 通过xarray打开.nc文件，合并同名坐标
            code = NAME_TO_VAR[var] # 查找变量名所对应的编码

            if len(ds[code].shape) == 3:  # 表层变量
                ds[code] = ds[code].expand_dims("val", axis=1) # 给变量增加一个维度为"val"
                # remove the last 24 hours if this year has 366 days
                np_vars[var] = ds[code].to_numpy()[:HOURS_PER_YEAR] # 获取变量var的值，取一年的值

                if partition == "train":  # compute mean and std of each var in each year
                    var_mean_yearly = np_vars[var].mean(axis=(0, 2, 3)) # 计算每个年份var的均值
                    var_std_yearly = np_vars[var].std(axis=(0, 2, 3)) # 计算每个年份var的标准差

                    if var not in normalize_mean:
                        normalize_mean[var] = [var_mean_yearly] # 第一次处理该变量，初始化均值数组
                        normalize_std[var] = [var_std_yearly] # 第一次处理该变量，初始化标准差数组
                    else:
                        normalize_mean[var].append(var_mean_yearly) # 将该年份的均值添加到均值数组
                        normalize_std[var].append(var_std_yearly) # 将该年份的标准差添加到标准差数组

                clim_yearly = np_vars[var].mean(axis=0) # 计算一年中变量var的平均值
                if var not in climatology:
                    climatology[var] = [clim_yearly] # 第一次处理该变量，初始化climatology
                else:
                    climatology[var].append(clim_yearly) # 将该年份的climatology添加到数组

            else:  # 多层变量，只使用部分层
                assert len(ds[code].shape) == 4 # 断言该变量的形状是4维
                all_levels = ds["level"][:].to_numpy() # 获取该变量所有的层
                all_levels = np.intersect1d(all_levels, DEFAULT_PRESSURE_LEVELS) # 取所有层与默认压力层的交集
                for level in all_levels:
                    ds_level = ds.sel(level=[level]) # 取出该变量在该层的值
                    level = int(level)
                    # remove the last 24 hours if this year has 366 days
                    np_vars[f"{var}_{level}"] = ds_level[code].to_numpy()[:HOURS_PER_YEAR] # 获取该变量在该层的值，取一年的值

                if partition == "train":  # 计算每年每个变量的均值和标准差
                    var_mean_yearly = np_vars[f"{var}_{level}"].mean(axis=(0, 2, 3)) # 计算每个年份该变量在该层的均值
                    var_std_yearly = np_vars[f"{var}_{level}"].std(axis=(0, 2, 3)) # 计算每个年份该变量在该层的标准差
                    if var not in normalize_mean:
                        normalize_mean[f"{var}_{level}"] = [var_mean_yearly] # 第一次处理该变量，初始化均值数组
                        normalize_std[f"{var}_{level}"] = [var_std_yearly] # 第一次处理该变量，初始化标准差数组
                    else:
                        normalize_mean[f"{var}_{level}"].append(var_mean_yearly) # 将该年份该层的均值添加到均值数组
                        normalize_std[f"{var}_{level}"].append(var_std_yearly) # 将该年份该层的标准差添加到标准差数组

                clim_yearly = np_vars[f"{var}_{level}"].mean(axis=0) # 计算一年中变量在该层的平均值
                if f"{var}_{level}" not in climatology:
                    climatology[f"{var}_{level}"] = [clim_yearly] # 第一次处理该变量该层，初始化climatology
                else:
                    climatology[f"{var}_{level}"].append(clim_yearly) # 将该年份该层的climatology添加到数组

        assert HOURS_PER_YEAR % num_shards_per_year == 0  # 确认HOURS_PER_YEAR（每年的小时数）可以被num_shards_per_year（每年的分片数）整除
        num_hrs_per_shard = HOURS_PER_YEAR // num_shards_per_year  # 计算每个分片包含多少小时
        for shard_id in range(num_shards_per_year):  # 对于每个分片，循环迭代
            start_id = shard_id * num_hrs_per_shard  # 计算该分片的起始时间
            end_id = start_id + num_hrs_per_shard  # 计算该分片的结束时间
            # 将每个变量的值划分到该分片
            sharded_data = {k: np_vars[k][start_id:end_id] for k in np_vars.keys()}
            np.savez(
                os.path.join(save_dir, partition, f"{year}_{shard_id}.npz"),  # 将数据保存到特定文件名的文件中
                **sharded_data,  # 使用关键字参数将sharded_data中的所有项传递给savez函数
            )

    if partition == "train":  # 如果当前分区为训练分区
        # 对于所有变量，如果变量不在constant_fields列表中，则对normalize_mean和normalize_std进行堆叠
        for var in normalize_mean.keys():
            if var not in constant_fields:
                normalize_mean[var] = np.stack(normalize_mean[var], axis=0)
                normalize_std[var] = np.stack(normalize_std[var], axis=0)

        # 对于所有变量，对多年的值进行聚合
        for var in normalize_mean.keys():
            if var not in constant_fields:
                mean, std = normalize_mean[var], normalize_std[var]
                # var(X) = E[var(X|Y)] + var(E[X|Y])
                variance = (std**2).mean(axis=0) + (mean**2).mean(axis=0) - mean.mean(axis=0) ** 2
                std = np.sqrt(variance)
                # E[X] = E[E[X|Y]]
                mean = mean.mean(axis=0)
                normalize_mean[var] = mean
                normalize_std[var] = std

        # 将normalize_mean和normalize_std保存到文件中
        np.savez(os.path.join(save_dir, "normalize_mean.npz"), **normalize_mean)
        np.savez(os.path.join(save_dir, "normalize_std.npz"), **normalize_std)

    # 对于所有变量，计算气候学（climatology）值
    for var in climatology.keys():
        climatology[var] = np.stack(climatology[var], axis=0)
    climatology = {k: np.mean(v, axis=0) for k, v in climatology.items()}
    # 将climatology保存到文件中
    np.savez(
        os.path.join(save_dir, partition, "climatology.npz"),
        **climatology,
    )


@click.command()
@click.option("--root_dir", type=click.Path(exists=True)) # 数据根目录
@click.option("--save_dir", type=str) # 数据处理后的保存目录
@click.option(
    "--variables", # 变量名
    "-v",
    type=click.STRING,
    multiple=True,
    default=[
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "toa_incident_solar_radiation",
        "total_precipitation",
        "geopotential",
        "u_component_of_wind",
        "v_component_of_wind",
        "temperature",
        "relative_humidity",
        "specific_humidity",
    ],
)
@click.option("--start_train_year", type=int, default=1979) # 训练数据起始年份
@click.option("--start_val_year", type=int, default=2016) # 验证数据起始年份
@click.option("--start_test_year", type=int, default=2017) # 测试数据起始年份
@click.option("--end_year", type=int, default=2019) # 数据截止年份
@click.option("--num_shards", type=int, default=8) # 数据切片数
def main(
    root_dir,
    save_dir,
    variables,
    start_train_year,
    start_val_year,
    start_test_year,
    end_year,
    num_shards,
):
    '''
    预处理主函数
    Args:
        root_dir (string): 数据根目录
        save_dir (string): 数据处理后的保存目录
        variables (string): 变量名
        start_train_year (int): 训练数据起始年份
        start_val_year (int): 验证数据起始年份
        start_test_year (int): 测试数据起始年份
        end_year (int): 数据截止年份
        num_shards (int): 数据切片数
    '''
    assert start_val_year > start_train_year and start_test_year > start_val_year and end_year > start_test_year
    train_years = range(start_train_year, start_val_year) # 训练年份列表
    val_years = range(start_val_year, start_test_year) # 验证年份列表
    test_years = range(start_test_year, end_year) # 测试年份列表

    os.makedirs(save_dir, exist_ok=True) # 如果保存目录不存在则创建

    nc2np(root_dir, variables, train_years, save_dir, "train", num_shards) # 调用nc2np函数将原始数据转换为numpy格式，并保存为npz文件
    nc2np(root_dir, variables, val_years, save_dir, "val", num_shards) 
    nc2np(root_dir, variables, test_years, save_dir, "test", num_shards)

    # save lat and lon data 保存经纬度数据
    ps = glob.glob(os.path.join(root_dir, variables[0], f"*{train_years[0]}*.nc")) # 获取第一个训练年份的nc文件路径
    x = xr.open_mfdataset(ps[0], parallel=True) # 打开nc文件
    lat = x["lat"].to_numpy() # 获取纬度数组
    lon = x["lon"].to_numpy() # 获取经度数组
    np.save(os.path.join(save_dir, "lat.npy"), lat) # 保存纬度数组
    np.save(os.path.join(save_dir, "lon.npy"), lon) # 保存经度数组


if __name__ == "__main__":
    main() # 执行主函数
