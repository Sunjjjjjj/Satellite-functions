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

# plot settings
palette = sns.color_palette("CMRmap_r", 16, as_cmap = True)
palette = mycmap('tropomi')

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


#%% save annual data
product = 'OFFL_L2__NO2____'
filelist_no2 = glob.glob('%s%s/*nc*' % (path, product))
product = 'OFFL_L2_HCHO___'
filelist_hcho = glob.glob('%s%s/*nc*' % (path, product))
product = 'OFFL_L2__SO2____'
filelist_so2 = glob.glob('%s%s/*nc*' % (path, product))
product = 'OFFL_L2__CO_____'
filelist_co = glob.glob('%s%s/*nc*' % (path, product))
product = 'OFFL_L2_CH4_____'
filelist_ch4 = glob.glob('%s%s/*nc*' % (path, product))
product = 'OFFL_L2__O3_____'
filelist_o3 = glob.glob('%s%s/*nc*' % (path, product))

NO2 = pd.DataFrame()
HCHO = pd.DataFrame()
SO2 = pd.DataFrame()
CO = pd.DataFrame()
CH4 = pd.DataFrame()
O3 = pd.DataFrame()

filelist = []
for ip in period[:]:
    no2 = pd.DataFrame()
    hcho = pd.DataFrame()
    so2 = pd.DataFrame()
    co = pd.DataFrame()
    ch4 = pd.DataFrame()
    o3 = pd.DataFrame()

    figpath = '/home/sunji/Documents/中山市大气遥感监测服务/%04i-%02i/' % (ip[0].year, ip[0].month)
    if not os.path.exists(figpath):
        os.mkdir(figpath)

    
    fnr = []
    for idate in ip[:]:
        # sys.stdout.write('\r Reading %04i-%02i-%2 ...' % (iyear, imonth))
        no2_temp = pd.DataFrame()
        hcho_temp = pd.DataFrame()
        so2_temp = pd.DataFrame()
        co_temp = pd.DataFrame()
        ch4_temp = pd.DataFrame()
        o3_temp = pd.DataFrame()

        for f in sorted(filelist_o3):
            if f.find('OFFL_L2__O3_____%04i%02i%02i' % (idate.year, idate.month, idate.day)) > 0:
                try:
                    temp = readTROPOMI(f, ROI, unit = 'molecule', qa = qa)
                    temp['date'] = idate
                    o3_temp = pd.concat([o3, temp], axis = 0)
                    o3 = pd.concat([o3, temp], axis = 0)
                    O3 = pd.concat([o3, temp], axis = 0)
                except:
                    print('\r This file is not read: %s' % f)
                    filelist.append(f)
                    
        for f in sorted(filelist_ch4):
            if f.find('OFFL_L2__CH4____%04i%02i%02i' % (idate.year, idate.month, idate.day)) > 0:
                try:
                    temp = readTROPOMI(f, ROI, qa = qa)
                    temp['date'] = idate
                    ch4_temp = pd.concat([ch4, temp], axis = 0)
                    ch4 = pd.concat([ch4, temp], axis = 0)
                    CH4 = pd.concat([ch4, temp], axis = 0)
                except:
                    print('\r This file is not read: %s' % f)
                    filelist.append(f)

        for f in sorted(filelist_co):
            if f.find('OFFL_L2__CO_____%04i%02i%02i' % (idate.year, idate.month, idate.day)) > 0:
                try:
                    temp = readTROPOMI(f, ROI, unit = 'molecule', qa = qa)
                    temp['date'] = idate
                    co_temp = pd.concat([co, temp], axis = 0)
                    co = pd.concat([co, temp], axis = 0)
                    CO = pd.concat([co, temp], axis = 0)
                except:
                    print('\r This file is not read: %s' % f)
                    filelist.append(f)
                    
        for f in sorted(filelist_no2):
            if f.find('OFFL_L2__NO2____%04i%02i%02i' % (idate.year, idate.month, idate.day)) > 0:
                try:
                    temp = readTROPOMI(f, ROI, unit = 'molecule', qa = qa)
                    no2_temp = pd.concat([no2, temp], axis = 0)
                    no2 = pd.concat([no2, temp], axis = 0)
                    NO2 = pd.concat([NO2, temp], axis = 0)
                except:
                    print('\r This file is not read: %s' % f)
                    filelist.append(f)

        for f in sorted(filelist_hcho):
            if f.find('OFFL_L2__HCHO___%04i%02i%02i' % (idate.year, idate.month, idate.day)) > 0:
                try:
                    temp = readTROPOMI(f, ROI, unit = 'molecule', qa = qa)
                    temp['date'] = idate
                    hcho_temp = pd.concat([hcho, temp], axis = 0)
                    hcho = pd.concat([hcho, temp], axis = 0)
                    HCHO = pd.concat([HCHO, temp], axis = 0)
                except:
                    print('\r This file is not read: %s' % f)
                    filelist.append(f)

        for f in sorted(filelist_so2):
            if f.find('OFFL_L2__SO2____%04i%02i%02i' % (idate.year, idate.month, idate.day)) > 0:
                try:
                    temp = readTROPOMI(f, ROI, unit = 'molecule', qa = qa)
                    temp['date'] = idate
                    so2_temp = pd.concat([so2, temp], axis = 0)
                    so2 = pd.concat([so2, temp], axis = 0)
                    SO2 = pd.concat([SO2, temp], axis = 0)
                except:
                    print('\r This file is not read: %s' % f)
                    filelist.append(f)
#%%
# plot CH4
    parameters = {'figsize': (6, 4), 'axes': [[0., 0.1, 0.8, 0.8]], 'caxes': [[0.775, 0.2, 0.05, 0.6]],
                    'ROI': [ROI], 'bmgrid': [1], 'corientation': 'vertical'}
    temp, (ydim, xdim) = df2grid(ch4, ROI, res = res)
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['methane_mixing_ratio_bias_corrected'].values.reshape(ydim, xdim)}
    parameters['titles'] = [r'$CH_4$混合比分布图(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
    parameters['cmaps'] = [palette]
    parameters['vmin'] = [1860]
    parameters['vmax'] = [1920]
    parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 10)]
    parameters['clabels'] = [r'$[-]$']
    _, axes = plotmap(parameters).pcolormap(data)
    # axes[0][1].set_norm(LogNorm(parameters['vmin'][0], parameters['vmax'][0]))
    # zoom in
    axins = axes[0][0].inset_axes([0.65, 0, 0.35, 0.4])
    axins.set_xlim(ROI_in['W'], ROI_in['E'])
    axins.set_ylim(ROI_in['S'], ROI_in['N'])
    axins.pcolor(data['longitude'], data['latitude'], data['parameter'],
                  cmap = parameters['cmaps'][0], norm = LogNorm(parameters['vmin'][0], parameters['vmax'][0]))
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axes[0][0].indicate_inset_zoom(axins)
    axes[0][0].indicate_inset_zoom(axins, edgecolor="black")
    
    for iax in [axes[0][0], axins]:
        df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)
    plt.savefig(figpath + 'CH4 %04i-%02i-%02i - %04i-%02i-%02i' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)

#%%
# plot CO
    parameters = {'figsize': (6, 4), 'axes': [[0., 0.1, 0.8, 0.8]], 'caxes': [[0.775, 0.2, 0.05, 0.6]],
                    'ROI': [ROI], 'bmgrid': [1], 'corientation': 'vertical'}
    temp, (ydim, xdim) = df2grid(co, ROI, res = res)
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['carbonmonoxide_total_column'].values.reshape(ydim, xdim)}
    parameters['titles'] = [r'$CO$柱浓度分布图(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
    parameters['cmaps'] = [palette]
    parameters['vmin'] = [1e18]
    parameters['vmax'] = [2.2e18]
    parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 5e17)]
    parameters['clabels'] = [r'$[molecules/cm^2]$']
    _, axes = plotmap(parameters).pcolormap(data)
    # axes[0][1].set_norm(LogNorm(parameters['vmin'][0], parameters['vmax'][0]))
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
    
    for iax in [axes[0][0], axins]:
        df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)
    plt.savefig(figpath + 'CO %04i-%02i-%02i - %04i-%02i-%02i' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)
#%%
# plot O3
    parameters = {'figsize': (6, 4), 'axes': [[0., 0.1, 0.8, 0.8]], 'caxes': [[0.775, 0.2, 0.05, 0.6]],
                    'ROI': [ROI], 'bmgrid': [1], 'corientation': 'vertical'}
    o3 = o3.reset_index(inplace = False, drop = True)
    temp, (ydim, xdim) = df2grid(o3, ROI, res = res)
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['ozone_total_vertical_column'].values.reshape(ydim, xdim)}
    parameters['titles'] = [r'$O_3$柱浓度分布图(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
    parameters['cmaps'] = [palette]
    parameters['vmin'] = [7.2e18]
    parameters['vmax'] = [8.2e18]
    parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 2e17)]
    parameters['clabels'] = [r'$[molecules/cm^2]$']
    _, axes = plotmap(parameters).pcolormap(data)
    # axes[0][1].set_norm(LogNorm(parameters['vmin'][0], parameters['vmax'][0]))
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
    
    for iax in [axes[0][0], axins]:
        df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)
    plt.savefig(figpath + 'O3 %04i-%02i-%02i - %04i-%02i-%02i' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)

#%%
# plot SO2
    parameters = {'figsize': (6, 4), 'axes': [[0., 0.1, 0.8, 0.8]], 'caxes': [[0.775, 0.2, 0.05, 0.6]],
                    'ROI': [ROI], 'bmgrid': [1], 'corientation': 'vertical'}
    temp, (ydim, xdim) = df2grid(so2, ROI, res = res)
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['sulfurdioxide_total_vertical_column'].values.reshape(ydim, xdim)}
    parameters['titles'] = [r'$SO_2$柱浓度分布图(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
    parameters['cmaps'] = [palette]
    parameters['vmin'] = [0]
    parameters['vmax'] = [6e16]
    parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 1e16)]
    parameters['clabels'] = [r'$[molecules/cm^2]$']
    _, axes = plotmap(parameters).pcolormap(data)
    # axes[0][1].set_norm(LogNorm(parameters['vmin'][0], parameters['vmax'][0]))
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
    
    for iax in [axes[0][0], axins]:
        df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)
    plt.savefig(figpath + 'SO2 %04i-%02i-%02i - %04i-%02i-%02i' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)

#%%
# plot NO2
    parameters = {'figsize': (6, 4), 'axes': [[0., 0.1, 0.8, 0.8]], 'caxes': [[0.775, 0.2, 0.05, 0.6]],
                    'ROI': [ROI], 'bmgrid': [1], 'corientation': 'vertical'}
    no2 = no2.reset_index(inplace = False, drop = True)
    temp, (ydim, xdim) = df2grid(no2, ROI, res = res)
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['nitrogendioxide_tropospheric_column'].values.reshape(ydim, xdim)}
    parameters['titles'] = [r'$NO_2$柱浓度分布图(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
    parameters['cmaps'] = [palette]
    parameters['vmin'] = [0e15]
    parameters['vmax'] = [0.4e16]
    parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 2e15)]
    parameters['clabels'] = [r'$[molecules/cm^2]$']
    _, axes = plotmap(parameters).pcolormap(data)
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
    
    for iax in [axes[0][0], axins]:
        df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)
    plt.savefig(figpath + 'NO2 %04i-%02i-%02i - %04i-%02i-%02i' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)

#%%
# plot HCHO
    hcho = hcho.reset_index(inplace = False, drop = True)
    temp, (ydim, xdim) = df2grid(hcho, ROI, res = res)
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['formaldehyde_tropospheric_vertical_column'].values.reshape(ydim, xdim)}
    parameters['titles'] = [r'$HCHO$柱浓度分布图(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
    parameters['cmaps'] = [palette]
    parameters['vmin'] = [0.5e16]
    parameters['vmax'] = [2e16]
    parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 2e15)]
    parameters['clabels'] = [r'$[molecules/cm^2]$']
    
    _, axes = plotmap(parameters).pcolormap(data)
    axes[0][1].set_norm(LogNorm(parameters['vmin'][0], parameters['vmax'][0]))
    # zoom in
    axins = axes[0][0].inset_axes([0.65, 0, 0.35, 0.4])
    axins.set_xlim(ROI_in['W'], ROI_in['E'])
    axins.set_ylim(ROI_in['S'], ROI_in['N'])
    axins.pcolor(data['longitude'], data['latitude'], data['parameter'],
                  cmap = parameters['cmaps'][0], norm = LogNorm(parameters['vmin'][0], parameters['vmax'][0]))
    axins.set_xticklabels([])
    axins.set_yticklabels([])
    axes[0][0].indicate_inset_zoom(axins)
    axes[0][0].indicate_inset_zoom(axins, edgecolor="black")
    
    for iax in [axes[0][0], axins]:
        df, p = plotshp(china, iax, color = 'k', linewidth = 0.2, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(province, iax, color = 'k', linewidth = 0.4, ROI = [], ROI_linewidth = 0.4)
        df, p = plotshp(city, iax, color = 'k', linewidth = 0.8, ROI = ['鼓楼区'], ROI_linewidth = 0.8)
    plt.savefig(figpath + 'HCHO %04i-%02i-%02i - %04i-%02i-%02i' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day), dpi = 300, transparent = True)

#%%
# plot O3 sensitivity
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors
    c = sns.color_palette('RdYlBu_r', 20)
    cmap = matplotlib.colors.ListedColormap([c[4], c[10], c[14]])
    
    # no2_temp = griddata((no2_m['longitude'], no2_m['latitude']), no2_m['nitrogendioxide_tropospheric_column'], (xx, yy), method = 'linear')
    # hcho_temp = griddata((hcho_m['longitude'], hcho_m['latitude']), hcho_m['formaldehyde_tropospheric_vertical_column'], (xx, yy), method = 'linear')

    temp, (ydim, xdim) = df2grid(no2, ROI, res = res)
    hcho_g, (ydim, xdim) = df2grid(hcho, ROI, res = res)
    temp['formaldehyde_tropospheric_vertical_column'] = hcho_g['formaldehyde_tropospheric_vertical_column'] 
    
    # temp = grid.merge(no2_m, how = 'left')
    # temp = temp.merge(hcho_m, how = 'left', on = ['latitude_g', 'longitude_g'])
    temp['o3sens'] = temp['formaldehyde_tropospheric_vertical_column'] / temp['nitrogendioxide_tropospheric_column']
    data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
            'parameter': temp['o3sens'].values.reshape(ydim, xdim)}

    

    # temp = hcho_temp / no2_temp 
    # data = {'longitude': xx, 'latitude': yy, 'o3sens': temp}
    
    # merged = hcho_m.merge(no2_m, on = ['latitude_g', 'longitude_g'])
    # temp = merged['formaldehyde_tropospheric_vertical_column']/merged['nitrogendioxide_tropospheric_column']
    # data = {'longitude': merged['longitude_g'], 'latitude': merged['latitude_g'], 'o3sens': temp}

    
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
