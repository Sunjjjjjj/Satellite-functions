#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:03:41 2022

@author: kanonyui
"""
import sys
sys.path.insert(0, '/home/sunji/Scripts/')
import os, glob
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from utilities import *
from scipy.optimize import leastsq
from affine import Affine
from pyproj import Proj, transform

def readMODIS(inputfile, ROI):
    data = netCDF4.Dataset(inputfile, 'r')
    temp = pd.DataFrame()
    # read file
    sys.stdout.write('\r Reading %s ...' % (inputfile))
    variables = ['Optical_Depth_Land_And_Ocean']
    other = ['BowTie_Flag', 'Sensor_Azimuth', 'Sensor_Zenith', 'Solar_Azimuth', 'Solar_Zenith', 'Latitude', 'Longitude']
    parameters = variables + other
    for ikey in parameters:
        temp[ikey] = data[ikey][:].filled(np.nan).reshape(-1)
    
    # mask ROI
    mask = ROImask(temp['Latitude'], temp['Longitude'], ROI)
    qamask = temp['BowTie_Flag'] > 0
    
    temp = temp[mask & qamask]
    temp.reset_index(drop = True, inplace = True)
    
    data.close()
    return temp

#%%
# f = '/home/sunji/Data/Himawari-8/L3/daily/H08_20220103_0000_1DARP031_FLDK.02401_02401.nc'
f = '/home/sunji/Data/MODIS/MOD04_3K/MOD04_3K.A2022015.0200.061.2022018172153.hdf'
# ROI = {'S':-50, 'N': 50, 'W': 80, 'E': 200}

# output = readMODIS(f, ROI)


# _, output = readHimawari(f, ROI)
# plt.figure()
# plt.pcolor(output['longitude'], output['latitude'], output['AOT_L3_Merged_Mean'])

# newfile = netCDF4.Dataset('H8_regional.nc', 'w')
# x = newfile.createDimension('longitude', size=output['latitude'].shape[1])
# y = newfile.createDimension('latitude', size=output['latitude'].shape[0])
# # aod = newfile.createDimension('AOD', size=121)
# lon = newfile.createVariable('lon', 'f4', dimensions='longitude')
# lat = newfile.createVariable('lat', 'f4', dimensions='latitude')
# aod = newfile.createVariable('AOD', 'f4', dimensions=('latitude', 'longitude'))

# lat[:] = output['latitude'][:, 0]
# lat.units = 'degrees_north'
# lon[:] = output['longitude'][0, :]
# lon.units = 'degrees_east'
# aod[:] = output['AOT_L3_Merged_Mean']

# newfile.close()





#%%
# # L2 read by file
# # path = '/Users/kanonyui/Downloads/temp_L2/'
# # path = 'home/sunji/Data/Himawari-8/L2/'
# path = '/home/sunji/Data/Himawari-8/L2/'

# for idate in date:
#     year, month, day, hour, minute = idate.year, idate.month, idate.day, idate.hour, idate.minute
#     sys.stdout.write('\r Reading Himawari-8 %04i-%02i-%02i %02i' % (year, month, day, hour))
#     url = 'ftp://ftp.ptree.jaxa.jp/pub/himawari/L2/ARP/030/%04i%02i/%02i/%02i/NC_H08_%04i%02i%02i_%02i%02i_L2ARP030_FLDK.02401_02401.nc' % (year, month, day, hour, year, month, day, hour, minute)
    
#     os.system('/usr/bin/wget --ftp-user=Jiyunting.sun_gmail.com --ftp-password=SP+wari8 -P %s %s' % (path, url))

# #%%
# # path = '/Users/kanonyui/Downloads/temp_L2/'
# filelist = glob.glob(path + '*.nc')
# parameters = ['Hour', 'AOT', 'AOT_uncertainty', 'AE', 'SSA']

# output = pd.DataFrame()
# for f in filelist:
#     data = netCDF4.Dataset(f)    
#     sys.stdout.write('\r %s ...' % (f))
#     temp = {}
#     temp['lat'] = data.variables['latitude'][:]
#     temp['lon'] = data.variables['longitude'][:]
#     lon, lat = np.meshgrid(temp['lon'], temp['lat'])
#     mask, temp['lat'], temp['lon'] = ROImask(lat, lon, ROI)
    
#     for ikey in parameters:
#         temp[ikey] = np.ma.masked_array(data.variables[ikey][:].filled(np.nan), ~mask).data
    
    
#     plt.figure()
#     bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#                 lat_0 = 0, lon_0 = 180, projection='cyl',resolution='c')
#     bm.drawcoastlines(color='gray',linewidth=1)
#     bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
#     bm.drawmeridians(np.arange(0, 361, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
#     plt.scatter(temp['lon'], temp['lat'], c = temp['AOT'], s = 3, vmin = 0, vmax = 3)    
#     plt.title('%s' % (f[f.find('NC_'): f.find('_L2ARP030')]))
    
#     for ikey in temp.keys():
#         temp[ikey] = temp[ikey].reshape(-1)
    
#     temp = pd.DataFrame.from_dict(temp)
    
#     temp.dropna(axis = 0, how = 'any', inplace = True)
#     output = output.append(temp)
    
#%% 
# L3 read by hour
# path = '/Users/kanonyui/Downloads/temp_L3/'
# path = '/home/sunji/Data/Himawari-8/L3/daily/'
# for idate in date:
#     year, month, day, hour, minute = idate.year, idate.month, idate.day, idate.hour, idate.minute
#     sys.stdout.write('\r Reading Himawari-8 %04i-%02i-%02i %02i' % (year, month, day, hour))
#     # url = 'ftp://ftp.ptree.jaxa.jp/pub/himawari/L3/ARP/031/%04i%02i/%02i/H08_%04i%02i%02i_%02i00_1HARP031_FLDK.02401_02401.nc' % (year, month, day, year, month, day, hour)
#     url = 'ftp://ftp.ptree.jaxa.jp/pub/himawari/L3/ARP/031/%04i%02i/daily/H08_%04i%02i%02i_0000_1DARP031_FLDK.02401_02401.nc' % (year, month, year, month, day)
    
#     os.system('/usr/bin/wget --ftp-user=Jiyunting.sun_gmail.com --ftp-password=SP+wari8 -P %s %s' % (path, url))


# #%%
# # path = '/Users/kanonyui/Downloads/temp_L3/'
# filelist = glob.glob(path + '*.nc')
# parameters = ['Hour', 'AOT_Merged']

# output = pd.DataFrame()

# for f in filelist:
#     data = netCDF4.Dataset(f)    
#     sys.stdout.write('\r %s ...' % (f))
#     temp = {}
#     temp['lat'] = data.variables['latitude'][:]
#     temp['lon'] = data.variables['longitude'][:]
#     lon, lat = np.meshgrid(temp['lon'], temp['lat'])
#     mask, temp['lat'], temp['lon'] = ROImask(lat, lon, ROI)
    
#     for ikey in parameters:
#         temp[ikey] = np.ma.masked_array(data.variables[ikey][:].filled(np.nan), ~mask).data
    
    
#     plt.figure()
#     bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#                 lat_0 = 0, lon_0 = 180, projection='cyl',resolution='c')
#     bm.drawcoastlines(color='gray',linewidth=1)
#     bm.drawparallels(np.arange(-45, 46, 45), labels=[False,True,False,False], linewidth = 0, fontsize = 8)
#     bm.drawmeridians(np.arange(0, 361, 90), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
#     plt.scatter(temp['lon'], temp['lat'], c = temp['AOT_Merged'], s = 3, vmin = 0, vmax = 3)    
#     plt.title('%s' % (f[f.find('H08_'): f.find('_1HARP031')]))
    
#     for ikey in temp.keys():
#         temp[ikey] = temp[ikey].reshape(-1)
    
#     temp = pd.DataFrame.from_dict(temp)
    
#     temp.dropna(axis = 0, how = 'any', inplace = True)
#     output = output.append(temp)

def getLatLon(inputfile):
    # get coordinates
    data = gdal.Open(inputfile)
    Metadata = data.GetMetadata()
    
    xdim, ydim = float(Metadata["DATACOLUMNS"]), float(Metadata["DATAROWS"])
    #  获取四个角的维度
    Latitudes = Metadata["GRINGPOINTLATITUDE.1"]
    #  采用", "进行分割
    LatitudesList = Latitudes.split(", ")
    #  获取四个角的经度
    Longitude = Metadata["GRINGPOINTLONGITUDE.1"]
    #  采用", "进行分割
    LongitudeList = Longitude.split(", ")
    # 图像四个角的地理坐标
    GeoCoordinates = np.zeros((4, 2), dtype = "float32")
    GeoCoordinates[0] = np.array([float(LongitudeList[0]),float(LatitudesList[0])])
    GeoCoordinates[1] = np.array([float(LongitudeList[1]),float(LatitudesList[1])])
    GeoCoordinates[2] = np.array([float(LongitudeList[2]),float(LatitudesList[2])])
    GeoCoordinates[3] = np.array([float(LongitudeList[3]),float(LatitudesList[3])])
    #  列数
    Columns = float(Metadata["DATACOLUMNS"])
    #  行数
    Rows = float(Metadata["DATAROWS"])
    #  图像四个角的图像坐标
    PixelCoordinates = np.array([[0, 0],
                    [Columns - 1, 0],
                    [Columns - 1, Rows - 1],
                    [0, Rows - 1]], dtype = "float32")
        
    #  计算仿射变换矩阵
    def func(i):
        Transform0, Transform1, Transform2, Transform3, Transform4, Transform5 = i[0], i[1], i[2], i[3], i[4], i[5]
        return [Transform0 + PixelCoordinates[0][0] * Transform1 + PixelCoordinates[0][1] * Transform2 - GeoCoordinates[0][0],
                Transform3 + PixelCoordinates[0][0] * Transform4 + PixelCoordinates[0][1] * Transform5 - GeoCoordinates[0][1],
                Transform0 + PixelCoordinates[1][0] * Transform1 + PixelCoordinates[1][1] * Transform2 - GeoCoordinates[1][0],
                Transform3 + PixelCoordinates[1][0] * Transform4 + PixelCoordinates[1][1] * Transform5 - GeoCoordinates[1][1],
                Transform0 + PixelCoordinates[2][0] * Transform1 + PixelCoordinates[2][1] * Transform2 - GeoCoordinates[2][0],
                Transform3 + PixelCoordinates[2][0] * Transform4 + PixelCoordinates[2][1] * Transform5 - GeoCoordinates[2][1],
                Transform0 + PixelCoordinates[3][0] * Transform1 + PixelCoordinates[3][1] * Transform2 - GeoCoordinates[3][0],
                Transform3 + PixelCoordinates[3][0] * Transform4 + PixelCoordinates[3][1] * Transform5 - GeoCoordinates[3][1]]
    #  最小二乘法求解
    GeoTransform = leastsq(func,np.asarray((1,1,1,1,1,1)))
    
    
    
    xx, yy = np.meshgrid(np.arange(xdim), np.arange(ydim))
    transform_coeff = Affine(GeoTransform[0][1], GeoTransform[0][2],GeoTransform[0][0], 
                                GeoTransform[0][-2],GeoTransform[0][-1], GeoTransform[0][3])

    T1 = transform_coeff * Affine.translation(0.5, 0.5)
    rc2en = lambda r, c: (c, r) * T1
    northings, eastings  = np.vectorize(rc2en, otypes=[float, float])(yy, xx)
    p = Proj(proj='latlong',datum='WGS84')
    lat, lon = transform(p, p, eastings, northings)
    return lat, lon
    
#  保存为tif
def array2raster(TifName, GeoTransform, array):
    cols = array.shape[1]  # 矩阵列数
    rows = array.shape[0]  # 矩阵行数
    driver = gdal.GetDriverByName('GTiff')
    outRaster = driver.Create(TifName, cols, rows, 1, gdal.GDT_Float32)
    # 括号中两个0表示起始像元的行列号从(0,0)开始
    outRaster.SetGeoTransform(tuple(GeoTransform))
    # 获取数据集第一个波段，是从1开始，不是从0开始
    outband = outRaster.GetRasterBand(1)
    outband.WriteArray(array)
    outRasterSRS = osr.SpatialReference()
    # 代码4326表示WGS84坐标
    outRasterSRS.ImportFromEPSG(4326)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.FlushCache()
    
    TifName = date + ".tif"
    array2raster(TifName, GeoTransform[0], NDVI)
    
    

def readMODIS13A3(inputfile, ROI, save_tiff = False, **kwargs):
    data = netCDF4.Dataset(inputfile, 'r')
    
    parameters = ['1 km monthly NDVI', '1 km monthly EVI', '1 km monthly red reflectance', '1 km monthly NIR reflectance', '1 km monthly blue reflectance', '1 km monthly MIR reflectance', '1 km monthly view zenith angle', '1 km monthly sun zenith angle', '1 km monthly relative azimuth angle']

    temp = pd.DataFrame()
    for ikey in parameters:
        temp[ikey] = data[ikey][:].astype(float).filled(np.nan).reshape(-1) / data[ikey].scale_factor
        
    parameters = ['1 km monthly VI Quality', '1 km monthly pixel reliability']
    for ikey in parameters:
        temp[ikey] = data[ikey][:].astype(float).filled(np.nan).reshape(-1)
    # get coordinate
    lat, lon = getLatLon(inputfile)
    temp['latitude'] = lat.reshape(-1)
    temp['longitude'] = lon.reshape(-1)
    
    # mask ROI
    mask = ROImask(temp['latitude'], temp['longitude'], ROI)
    # qamask = temp['qa_value'] >= qa
    temp = temp[mask]
    temp.reset_index(drop = True, inplace = True)
    
    data.close()
    return temp
