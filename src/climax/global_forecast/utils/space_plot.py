import os
import sys
sys.path.append('/public/home/qindaotest/rendq/pythonlib')
import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib as mpl
from math import *
from mpl_toolkits.basemap import Basemap

def set_lonlat(_m, lon_list, lat_list, lon_labels, lat_labels, lonlat_size):
    lon_dict = _m.drawmeridians(lon_list, labels=lon_labels, color='none', fontsize=lonlat_size)
    lat_dict = _m.drawparallels(lat_list, labels=lat_labels, color='none', fontsize=lonlat_size)
    lon_list = []
    lat_list = []
    for lon_key in lon_dict.keys():
        try:
            lon_list.append(lon_dict[lon_key][1][0].get_position()[0])
        except:
            continue

    for lat_key in lat_dict.keys():
        try:
            lat_list.append(lat_dict[lat_key][1][0].get_position()[1])
        except:
            continue
    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.set_yticks(lat_list)
    ax.set_xticks(lon_list)

    ax.tick_params(labelcolor='none')

def plot_map(ax,data,min_ax,max_ax,color):
    lat1 = 20.05 + area_range[0]*0.25
    lat2 = 59.8 - (160-area_range[1])*0.25
    lat_dim = area_range[1]-area_range[0]

    lon1 = 85.05 + area_range[2]*0.25
    lon2 = 124.8 - (160-area_range[3])*0.25
    lon_dim = area_range[3]-area_range[2]

    map1 = Basemap(llcrnrlon=lon1, llcrnrlat=lat1, urcrnrlon=lon2, urcrnrlat=lat2)
    lons = np.linspace(lon1,lon2,lon_dim)
    lats = np.linspace(lat1,lat2,lat_dim)

    llons, llats = np.meshgrid(lons,lats)
    x, y = map1(llons, llats)

    data = data[area_range[0]:area_range[1],area_range[2]:area_range[3]]

    if color=='1':
       map1.pcolormesh(x,y,data,vmin=min_ax,vmax=max_ax,cmap=mpl.cm.rainbow)
    elif color=='2' :
       map1.pcolormesh(x,y,data,vmin=min_ax,vmax=max_ax,cmap=mpl.cm.RdBu)
    #map1.colorbar()
    cb = map1.colorbar()
    cb.ax.tick_params(labelsize=16)

    map1.drawcoastlines()
    map1.drawcountries(linewidth=0.4)

    set_lonlat(map1, range(85, 125, 5), range(20, 60, 5), [0, 0, 1, 0], [1,0 , 0, 0], 12)

if __name__ == "__main__":
    area_range = [0,160,0,160]
    label = ''

    src = sys.argv[1]   #由/public/home/qindaotest/wym/Fourcastnet_change/FourCastNet_160_160/inference/inference.py生成的.h5文件
    global_stds_path =  '/public/share/sugonhpctest01/user/qindaotest/fourcastnet_data/global_stds.npy'
    global_means_path =  '/public/share/sugonhpctest01/user/qindaotest/fourcastnet_data/global_means.npy'

    fsrc = h5py.File(src, 'r')
    truth = fsrc['ground_truth']        # n,pre_lenth,c,h,w
    truth = truth[()] 

    pred = fsrc['predicted']             # n,pre_lenth,c,h,w 
    pred = pred[()] 

    rmse = fsrc['rmse']               #   n,pre_lenth,c
    rmse = rmse[()] 

    variable_index = int(sys.argv[2])  #要分析的变量的索引        #!
    predict_day = 0    #第几个预测步

    stds = np.load(global_stds_path)
    std = stds[0,variable_index,0,0]
    means = np.load(global_means_path)
    mean = means[0,variable_index,0,0]

    rmse_mean = np.sqrt(np.mean(np.power((truth[:, predict_day, variable_index, :, :] - pred[:, predict_day, variable_index, :, :]),2), axis=0))   #计算160x160的rmse值
    truth_mean = np.mean(truth[:, predict_day, variable_index, :, :],axis=0)    
    pred_mean = np.mean(pred[:, predict_day, variable_index, :, :],axis=0)


    max_data = np.max(truth_mean)
    min_data = np.min(truth_mean)

    fig = plt.figure(figsize = (43,15))

    ax = fig.add_subplot(231)
    plot_map(ax, truth_mean, min_data, max_data,'1')
    ax.set_title(label + '_OBS_day' ,fontsize=16)
    ax.set_xlabel('lons')
    ax.set_ylabel('lats')

    ax = fig.add_subplot(234)
    plot_map(ax, truth_mean*std+mean, np.min(truth_mean*std+mean), np.max(truth_mean*std+mean),'1')
    ax.set_title(label + '_OBS_day' ,fontsize=16)
    ax.set_xlabel('lons')
    ax.set_ylabel('lats')

    ax = fig.add_subplot(232)
    plot_map(ax, pred_mean, min_data, max_data,'1')
    ax.set_title(label + '_Pred_day' ,fontsize=16)
    ax.set_xlabel('lons')
    ax.set_ylabel('lats')

    ax = fig.add_subplot(235)
    plot_map(ax, pred_mean*std+mean, np.min(pred_mean*std+mean), np.max(pred_mean*std+mean),'1')
    ax.set_title(label + '_OBS_day' ,fontsize=16)
    ax.set_xlabel('lons')
    ax.set_ylabel('lats')

    max_rmse = np.max(rmse_mean)
    min_rmse = 0

    ax = fig.add_subplot(233)
    plot_map(ax,rmse_mean, min_rmse, max_rmse,'1')
    ax.set_title(label + '_rmse(OBS-Pred)_day' ,fontsize=16)
    ax.set_xlabel('lons')
    ax.set_ylabel('lats')

    ax = fig.add_subplot(236)
    plot_map(ax,rmse_mean*std,np.min(rmse_mean*std), np.max(rmse_mean*std),'1')
    ax.set_title(label + '_rmse(OBS-Pred)_day' ,fontsize=16)
    ax.set_xlabel('lons')
    ax.set_ylabel('lats')

    plt.tight_layout()

    try:
        save_path = 'result/space_plot.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')
    except:
        os.makedirs('result')
        save_path = 'result/space_plot.png'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close('all')







# with h5py.File('autoregressive_predictions.h5', 'a') as f:
#     try:
#         f.create_dataset("truth_mean", data = truth_mean, shape = (truth_mean.shape[0],truth_mean.shape[1]), dtype = np.float32)
#     except: 
#         del f["truth_mean"]
#         f.create_dataset("truth_mean", data = truth_mean, shape = (truth_mean.shape[0],truth_mean.shape[1]), dtype = np.float32)
#         f["truth_mean"][...] = truth_mean
    
#     try:
#         f.create_dataset("pred_mean", data = pred_mean, shape = (pred_mean.shape[0],pred_mean.shape[1]), dtype = np.float32)
#     except: 
#         del f["pred_mean"]
#         f.create_dataset("pred_mean", data = pred_mean, shape = (pred_mean.shape[0],pred_mean.shape[1]), dtype = np.float32)
#         f["pred_mean"][...] = pred_mean

#     try:
#         f.create_dataset("rmse_mean", data = rmse_mean, shape = (rmse_mean.shape[0],rmse_mean.shape[1]), dtype = np.float32)
#     except: 
#         del f["rmse_mean"]
#         f.create_dataset("rmse_mean", data = rmse_mean, shape = (rmse_mean.shape[0],rmse_mean.shape[1]), dtype = np.float32)
#         f["rmse_mean"][...] = rmse_mean

#     try:
#         f.create_dataset("rmse", data = rmse, shape = (rmse.shape[0],rmse.shape[1],rmse.shape[2]), dtype = np.float32)
#     except: 
#         del f["rmse"]
#         f.create_dataset("rmse", data = rmse, shape = (rmse.shape[0],rmse.shape[1],rmse.shape[2]), dtype = np.float32)
#         f["rmse"][...] = rmse

#     try:
#         f.create_dataset("stds", data = stds, shape = (stds.shape[0],stds.shape[1],stds.shape[2],stds.shape[3]), dtype = np.float32)
#     except: 
#         del f["stds"]
#         f.create_dataset("stds", data = stds, shape = (stds.shape[0],stds.shape[1],stds.shape[2],stds.shape[3]), dtype = np.float32)
#         f["stds"][...] = stds
    
#     try:
#         f.create_dataset("means", data = means, shape = (means.shape[0],means.shape[1],means.shape[2],means.shape[3]), dtype = np.float32)
#     except: 
#         del f["means"]
#         f.create_dataset("means", data = means, shape = (means.shape[0],means.shape[1],means.shape[2],means.shape[3]), dtype = np.float32)
#         f["means"][...] = std
    
#     f.close()





