#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 18:29:28 2022


@author: kanonyui
"""

import sys
sys.path.insert(0, '/home/sunji/Scripts/')
import os, glob
import netCDF4
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from pyTROPOMI import *
from pyTEMIS import *
from utilities import *
from shapely.geometry.polygon import LinearRing, Polygon
import time
import seaborn as sns
from matplotlib.colors import LogNorm

# initialization
# path
path = '/home/sunji/Data/TROPOMI/'

# region and period
ROI = {'S':21, 'N': 24, 'W': 111.5, 'E': 115}
ROI_in = {'S':22.1, 'N': 22.9, 'W': 113, 'E': 113.8}
period = [pd.date_range('2022-06-01', '2022-06-30', freq = '1D'),
          ]

# shapefile
china = '/home/sunji/Data/mapshp/china.shp'
province = '/home/sunji/Data/mapshp/guangdong.shp'
city = '/home/sunji/Data/mapshp/zhongshan.shp'
cityname = '中山市'


# other settings
qa = 0.5
res = 0.05
xx, yy = np.meshgrid(np.arange(ROI['W'], ROI['E'], res), np.arange(ROI['S'], ROI['N'], res))

grid = pd.DataFrame()
grid['latitude_g'] = yy.reshape(-1).astype(np.float64).round(2)
grid['longitude_g'] = xx.reshape(-1).astype(np.float64).round(2)
# dimensions of gridded data
ydim, xdim = xx.shape


# plot settings
palette = sns.color_palette("CMRmap_r", 16, as_cmap = True)
palette = mycmap('tropomi')
pelette_o3s = mycmap(name = 'o3sens')

figsetting = {}
parameters = {'figsize': (6, 4), 'axes': [[0., 0.1, 0.8, 0.8]], 'caxes': [[0.775, 0.2, 0.05, 0.6]],
                 'ROI': [ROI], 'bmgrid': [1], 'corientation': 'vertical'}
# NO2
parameters['titles'] = [r'$NO_2$柱浓度分布图']
parameters['cmaps'] = [palette]
parameters['vmin'] = [0e15]
parameters['vmax'] = [0.4e16]
parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 2e15)]
parameters['clabels'] = [r'$[molecules/cm^2]$']
figsetting['OFFL_L2__NO2____'] = parameters.copy()
# O3
parameters['titles'] = [r'$O_3$柱浓度分布图']
parameters['cmaps'] = [palette]
parameters['vmin'] = [7.2e18]
parameters['vmax'] = [8.2e18]
parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 2e17)]
parameters['clabels'] = [r'$[molecules/cm^2]$']
figsetting['OFFL_L2__O3_____'] = parameters.copy()
# SO2
parameters['titles'] = [r'$SO_2$柱浓度分布图']
parameters['cmaps'] = [palette]
parameters['vmin'] = [0]
parameters['vmax'] = [6e16]
parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 1e16)]
parameters['clabels'] = [r'$[molecules/cm^2]$']
figsetting['OFFL_L2__SO2____'] = parameters.copy()
# HCHO
parameters['titles'] = [r'$HCHO$柱浓度分布图']
parameters['cmaps'] = [palette]
parameters['vmin'] = [0.5e16]
parameters['vmax'] = [2e16]
parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 2e15)]
parameters['clabels'] = [r'$[molecules/cm^2]$']
figsetting['OFFL_L2__HCHO___'] = parameters.copy()
# CO
parameters['titles'] = [r'$CO$柱浓度分布图']
parameters['cmaps'] = [palette]
parameters['vmin'] = [1e18]
parameters['vmax'] = [2.2e18]
parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 5e17)]
parameters['clabels'] = [r'$[molecules/cm^2]$']
figsetting['OFFL_L2__CO_____'] = parameters.copy()
# CH4
parameters['titles'] = [r'$CH_4$混合比分布图']
parameters['cmaps'] = [palette]
parameters['vmin'] = [1860]
parameters['vmax'] = [1920]
parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 10)]
parameters['clabels'] = [r'$[-]$']
figsetting['OFFL_L2__CH4____'] = parameters.copy()


#%% save annual data
products = ['OFFL_L2__NO2____', 'OFFL_L2__O3_____', 'OFFL_L2__SO2____', 'OFFL_L2__HCHO___',
            'OFFL_L2__CO_____','OFFL_L2__CH4____']

names = {'OFFL_L2__NO2____':'nitrogendioxide_tropospheric_column',
         'OFFL_L2__O3_____': 'ozone_total_vertical_column',
         'OFFL_L2__SO2____': 'sulfurdioxide_total_vertical_column', 
         'OFFL_L2__HCHO___': 'formaldehyde_tropospheric_vertical_column',
         'OFFL_L2__CO_____': 'carbonmonoxide_total_column', 
         'OFFL_L2__CH4____': 'methane_mixing_ratio_bias_corrected'}


    
# =============================================================================
#     # read data
# =============================================================================
    
TPM = pd.DataFrame()
for ip in period[:]:
    # make fig path
    figpath = '/home/sunji/Documents/中山市大气遥感监测服务/%04i-%02i/' % (ip[0].year, ip[0].month)
    if not os.path.exists(figpath):
        os.mkdir(figpath)
    
    DATA = {}
    for iproduct in products[:]:
        filelist = glob.glob('%s%s/*nc*' % (path, iproduct))
    #     # change unit
        unit = 'molecule'
        if iproduct == 'OFFL_L2__O3_____':
            unit = 'DU'
            
        
        tpm = pd.DataFrame()
        for idate in ip[:]:
            # sys.stdout.write('\r Reading %04i-%02i-%2 ...' % (iyear, imonth))
            for f in sorted(filelist):
                if f.find('%s%04i%02i%02i' % (iproduct, idate.year, idate.month, idate.day)) > 0:
                    try:
                        temp = readTROPOMI(f, ROI, unit = unit, qa = qa)
                        temp['date'] = idate
                        tpm = pd.concat([tpm, temp], axis = 0)
                        TPM = pd.concat([TPM, temp], axis = 0)
                    except:
                        print('\r This file is not read: %s' % f)
                        filelist.append(f)
        DATA[iproduct] = TPM
# =============================================================================
# # grid data
# =============================================================================
        tpm = tpm.reset_index(inplace = False, drop = True)
        temp, (ydim, xdim) = df2grid(tpm, ROI, res = res)
        data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 
                'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
                'parameter': temp[names[iproduct]].values.reshape(ydim, xdim)}
# =============================================================================
# # plot data
# =============================================================================
        _, axes = plotmap(figsetting[iproduct]).pcolormap(data)
        # zoom in
        axins = axes[0][0].inset_axes([0.65, 0, 0.35, 0.4])
        axins.set_xlim(ROI_in['W'], ROI_in['E'])
        axins.set_ylim(ROI_in['S'], ROI_in['N'])
        axins.pcolor(data['longitude'], data['latitude'], data['parameter'],
                      cmap = parameters['cmaps'][0], vmin = parameters['vmin'][0], vmax = parameters['vmax'][0])
        axins.set_xticklabels([])
        axins.set_yticklabels([])
        axes[0][0].indicate_inset_zoom(axins)
        axes[0][0].indicate_inset_zoom(axins, edgecolor="black", )
        # plot shapefile
        for iax in [axes[0][0], axins]:
            df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
            df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
            df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)
        plt.savefig(figpath + '%s %04i-%02i-%02i - %04i-%02i-%02i' % (iproduct, ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)

#%%
# plot O3 sensitivity
    temp, (ydim, xdim) = df2grid(DATA['OFFL_L2__NO2____'], ROI, res = res)
    hcho_g, (ydim, xdim) = df2grid(DATA['OFFL_L2__HCHO___'], ROI, res = res)
    temp['formaldehyde_tropospheric_vertical_column'] = hcho_g['formaldehyde_tropospheric_vertical_column'] 
    
    # temp = grid.merge(no2_m, how = 'left')
    # temp = temp.merge(hcho_m, how = 'left', on = ['latitude_g', 'longitude_g'])
    temp['o3sens'] = temp['formaldehyde_tropospheric_vertical_column'] / temp['nitrogendioxide_tropospheric_column']
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['o3sens'].values.reshape(ydim, xdim)}

    parameters['titles'] = [r'$O_3$敏感度分析(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
    parameters['cmaps'] = [cmap]
    parameters['vmin'] = [0]
    parameters['vmax'] = [3]
    parameters['cticks'] = [[0.5,1.5,2.5]]
    parameters['clabels'] = [r'[-]']
    
    
    # axes = plotmap(parameters).contourfmap(data, levels = np.arange(0, 3.1, 0.5))
    _, axes = plotmap(parameters).pcolormap(data)
    axes[0][3].ax.set_yticklabels([r'$VOCs$', r'$VOCs-NO_x$',r'$NO_x$'], fontsize = 8)
    # zoom in
    axins = axes[0][0].inset_axes([0.65, 0, 0.35, 0.4])
    axins.set_xlim(ROI_in['W'], ROI_in['E'])
    axins.set_ylim(ROI_in['S'], ROI_in['N'])
    axins.pcolor(data['longitude'], data['latitude'], data['parameter'],
                  cmap = parameters['cmaps'][0], vmin = parameters['vmin'][0], vmax = parameters['vmax'][0])
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axes[0][0].indicate_inset_zoom(axins)
    axes[0][0].indicate_inset_zoom(axins, edgecolor="black")
    
    for iax in [axes[0][0], axins]:
        df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)

    plt.savefig(figpath + 'O3 sens %04i-%02i-%02i - %04i-%02i-%02i' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)
