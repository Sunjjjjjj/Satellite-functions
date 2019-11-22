#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:57:41 2018

@author: sunj
"""
import sys 
sys.path.insert(0, '/usr/people/sunj/Documents/pyvenv/Projects/General_scripts')
import sys, os
import shutil
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import glob
from scipy import ndimage
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
from pylab import cm
import pandas as pd
from pandas import Series, DataFrame, Panel
import pickle
from scipy import spatial
from datetime import datetime
from scipy.interpolate import griddata
from scipy.interpolate import interp1d
from scipy import stats
from otherFunctions import *
import h5py
import netCDF4
from MERRA2_functions import *
from MultiSensor import *
from pyOMI import *
import string
from AERONETtimeSeries_v3 import *
from GOME2 import *

plotidx = string.ascii_lowercase

plt.close('all')
dataOutputDir = '/nobackup/users/sunj/'
dataInputDir = '/nobackup_1/users/sunj/'

figdir = '/usr/people/sunj/Dropbox/Paper_Figure/MERRA2/'


ROIs = np.load(dataOutputDir + 'P3_output/ROIs.npy').reshape(-1)[0]

ROIs_temp = ROIs.keys()


def func(lat, lon):
    return round(lat * 2) / 2, round(lon * 1.6) / 1.6

t1 = time.time()
# =============================================================================
# surafce information
# =============================================================================
path =  dataInputDir + 'MERRA2/M2C0NXASM/' 
filename = 'MERRA2_101.const_2d_asm_Nx.00000000.nc4'
# constant surface height
data = netCDF4.Dataset(path + filename)
lon, lat = data['lon'][:] , data['lat'][:]
XX, YY = np.meshgrid(lon, lat)
geopotential_sfc = np.squeeze(data.variables['PHIS']) # [m2/s2]
g0 = 9.81  # [m/s2]
H0 = geopotential_sfc / g0 / 1e3  # [km]
land = data['FRLAND'][0]
ice = data['FRLANDICE'][0]
lake = data['FRLAKE'][0]
ocean = data['FROCEAN'][0]
data.close()
landmask = np.ones(land.shape) * 0
landmask[land > 0.5] = 1
landmask[ice > 0.5] = 1


#%%
# =============================================================================
# load AerNN ALH
# =============================================================================
startdate = '%4i-%02i-%02i' % (2006, 1, 1)
enddate   = '%4i-%02i-%02i' % (2006, 12, 31)
dates = pd.date_range(startdate, enddate)

AerNN90 = pd.DataFrame()
AerNN95 = pd.DataFrame()
for idate in dates:
    sys.stdout.write('\r Collecting AerNN %4i-%02i-%02i' % (idate.year, idate.month, idate.day))

    temp_AerNN90 = pd.read_pickle(dataOutputDir + 'P3_Validation/OMIAerNN90_%4i-%02i-%02i.pickle'  % (idate.year, idate.month, idate.day))
    temp_AerNN95 = pd.read_pickle(dataOutputDir + 'P3_Validation/OMIAerNN95_%4i-%02i-%02i.pickle'  % (idate.year, idate.month, idate.day))
    # quality control
    try:
        mask = (temp_AerNN90['AOD_MODIS'] >= 0.5) & (temp_AerNN90['SCDO2'] > 0) & (temp_AerNN90['As'] <= 0.15) & \
                (temp_AerNN90['OMMYDCLD'] <= 0.02)  & (temp_AerNN90['CF'] >= 0)  & \
                (temp_AerNN90['CF_OMI'] <= 0.1) & (temp_AerNN90['CF_OMI'] >= 0) & \
                (temp_AerNN90['ALH_MDTDB'] <= 100) & (temp_AerNN90['ALH_MDTDB'] >= 0) & \
                (temp_AerNN90['residue'] >= -10) & (temp_AerNN90['sza'] <= 70)
        AerNN90 = AerNN90.append(temp_AerNN90[mask])
        
        mask = (temp_AerNN95['AOD_MODIS'] >= 0.5) & (temp_AerNN95['SCDO2'] > 0) & (temp_AerNN95['As'] <= 0.15) & \
                (temp_AerNN95['OMMYDCLD'] <= 0.02)  & (temp_AerNN95['CF'] >= 0)  & \
                (temp_AerNN95['CF_OMI'] <= 0.1) & (temp_AerNN95['CF_OMI'] >= 0) & \
                (temp_AerNN95['ALH_MDTDB'] <= 100) & (temp_AerNN95['ALH_MDTDB'] >= 0) & \
                (temp_AerNN90['residue'] >= -10) & (temp_AerNN95['sza'] <= 70)
        AerNN95 = AerNN95.append(temp_AerNN95[mask])
    except:
        print('No AerNN data on %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
# =============================================================================
# load TROPOMI ALH
# =============================================================================
startdate = '%4i-%02i-%02i' % (2018, 11, 1)
enddate   = '%4i-%02i-%02i' % (2019, 8, 31)
dates = pd.date_range(startdate, enddate)

TPM = pd.DataFrame()
for idate in dates:
    sys.stdout.write('\r Collecting TROPOMI %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
    temp_TPM = pd.read_pickle(dataOutputDir + 'P3_Validation/TROPOMI_%4i-%02i-%02i.pickle'  % (idate.year, idate.month, idate.day))
    
    try: 
        mask = (temp_TPM['ALH'] < 100) & (temp_TPM['AI388'] < 100) & (temp_TPM['QF'] == 0) & (temp_TPM['NNstdALH'] <= 0.2) & \
                (temp_TPM['AOD'] <= 5) & (temp_TPM['AOD'] >= 0) & (temp_TPM['NNstdAOD'] <= 10) & (temp_TPM['sza'] <= 70) 
        temp_TPM = temp_TPM[mask]
        TPM = TPM.append(temp_TPM, sort = True)
    except:
        print('No TROPOMI data on %4i-%02i-%02i' % (idate.year, idate.month, idate.day))

## =============================================================================
## load MERRA2_OMAERUV_AERONET ALH
## =============================================================================
#startdate = '%4i-%02i-%02i' % (2006, 1, 1)
#enddate   = '%4i-%02i-%02i' % (2016, 12, 31)
#dates = pd.date_range(startdate, enddate)
#
#MR2OMI = pd.DataFrame()
#
#for idate in dates:
#    sys.stdout.write('\r # %04i-%02i-%02i' % (idate.year, idate.month, idate.day))
#    temp = pd.read_pickle(dataOutputDir + 'MERRA-2_OMAERUV_AERONET_collocation/MERRA-2_OMAERUV_AERONET_collocation_%4i-%02i-%02i.pickle' \
#                         % (idate.year, idate.month, idate.day))
#   # Quality control
#    try:
#        temp['Single_Scattering_Albedo[550nm]'] = 1 - temp['Absorption_AOD[550nm]'] / temp['AOD_550nm']
#        temp['Single_Scattering_Albedo[500nm]'] = (temp['Single_Scattering_Albedo[550nm]'] - temp['Single_Scattering_Albedo[440nm]'])  / (550 - 440) * (500 - 440) + temp['Single_Scattering_Albedo[440nm]']
#
#        land = temp['landoceanMask'] >= 0.5
#        temp['AODdiff'] = np.nan
#        temp['AODdiff1'] = np.nan
#        temp.loc[temp['landoceanMask'] >= 0.5, 'AODdiff'] = abs(temp['AOD_500nm'][land] - temp['AOD500'][land]) <= (0.05 + 0.15 * temp['AOD_500nm'][land])
#        temp.loc[temp['landoceanMask'] < 0.5, 'AODdiff'] = abs(temp['AOD_500nm'][~land] - temp['AOD500'][~land]) <= (0.03 + 0.05 * temp['AOD_500nm'][~land])
#        temp.loc[temp['landoceanMask'] >= 0.5, 'AODdiff1'] = abs(temp['AOD_550nm'][land] - temp['AOD'][land]) <= (0.05 + 0.15 * temp['AOD_550nm'][land])
#        temp.loc[temp['landoceanMask'] < 0.5, 'AODdiff1'] = abs(temp['AOD_550nm'][~land] - temp['AOD'][~land]) <= (0.03 + 0.05 * temp['AOD_550nm'][~land])
#
#        temp['SSA'] = 1 - temp['AAOD'] / temp['AOD']
#        temp['SSAdiff'] = (abs(temp['Single_Scattering_Albedo[500nm]'] - temp['SSA500']) <= 0.03)
#        temp['SSAdiff1'] = (abs(temp['Single_Scattering_Albedo[550nm]'] - temp['SSA']) <= 0.03)
#         
#        mask = (~np.isnan(temp['residue'])) & (~np.isnan(temp['AI388'])) & \
#                (temp['ALH'] > 0) #& temp['AODdiff'] & temp['SSAdiff'] & temp['AODdiff1'] & temp['SSAdiff1']
#        temp = temp[mask].reset_index(drop = True)
#    except:
#        print('No OMAERUV data on %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
#    MR2OMI = MR2OMI.append(temp, sort = True)

#%%
# =============================================================================
# load OMAERUV ALH
# =============================================================================
OMI = pd.read_pickle(dataOutputDir + 'P3_output/OMAERUV_gridded_2006-2016_monthly')
# quality control
mask = (OMI['ALH'] >= 0) & (OMI['AOD500'] >= 0) & (OMI['AOD500'] >= 0) & (OMI['SSA500'] >= 0) & \
        (OMI['CF'] >= 0) #& (OMI['CF'] <= 0.1)
OMI = OMI[mask].reset_index(drop = True)



#startdate = '%4i-%02i-%02i' % (2006, 1, 1)
#enddate   = '%4i-%02i-%02i' % (2006, 1, 31)
#dates = pd.date_range(startdate, enddate)
#OMI = pd.DataFrame()
#ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}
#for idate in dates:
#    sys.stdout.write('\r Collecting OMAERUV %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
#    try:
##    temp_OMI = readOMAERUV(['%04i-%02i-%02i' % (idate.year, idate.month, idate.day)], ROI, -10, grid = False)
#        temp_OMI = pd.read_pickle(dataOutputDir + 'P3_Validation/MR2OMI_%4i-%02i-%02i.pickle'  % (idate.year, idate.month, idate.day))
#        mask = (temp_OMI['ALH'] >= 0) & (temp_OMI['AOD500'] >= 0) & (temp_OMI['AOD500'] >= 0) & (temp_OMI['SSA500'] >= 0) & \
#                (temp_OMI['CF_OMI'] >= 0) #& (OMI['CF'] <= 0.1)
#        OMI = OMI.append(temp_OMI[mask])
#    except:
#        print('No OMAERUV data on %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
#OMI = OMI.reset_index(drop = True)

#%%
# =============================================================================
# load GOME2 AAH
# =============================================================================
startdate = '%4i-%02i-%02i' % (2018, 8, 13)
enddate   = '%4i-%02i-%02i' % (2018, 12, 31)
dates = pd.date_range(startdate, enddate)

GM2= pd.DataFrame()
MR2GM2 = pd.DataFrame()
for idate in dates:
    sys.stdout.write('\r Collecting GOME2 %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
    try:
        temp_GM2 = pd.read_pickle(dataOutputDir + 'P3_Validation/GOME2_AAH_%4i-%02i-%02i.pickle' % (idate.year, idate.month, idate.day))
        mask = (temp_GM2['AAH'] >= 0) & (temp_GM2['AI380'] >= 4)  & (temp_GM2['AI380'] <= 100) & (temp_GM2['AAH_flag'] == 0) & \
                (temp_GM2['CF'] <= 0.25) & (temp_GM2['CF'] <= 0.25) & (temp_GM2['sza'] <= 70)
        temp_GM2 = temp_GM2[mask]
        GM2 = GM2.append(temp_GM2, sort = True)
# =============================================================================
#   read MERRA-2 AAOD
# =============================================================================
        MR2 = pd.DataFrame()
        directory = dataInputDir + 'MERRA2/M2T1NXAER_morning/%4i/' % (idate.year)
        filename = 'MERRA2_100.tavg1_2d_aer_Nx_sub.%4i%02i%02i.nc4' % (idate.year, idate.month, idate.day)
        data = netCDF4.Dataset(directory + filename)
        MR2['lat_g'] = YY.reshape(-1)
        MR2['lon_g'] = XX.reshape(-1)
        MR2['AOD'] = data['TOTEXTTAU'][:].reshape(-1)
        MR2['AAOD'] = (data['TOTEXTTAU'][:] - data['TOTSCATAU'][:]).reshape(-1)
        data.close()   
# =============================================================================
# grid onto MERRA2 coordinate
# =============================================================================
        
        result = np.array(list(map(func, list(temp_GM2.lat), list(temp_GM2.lon))))
        result[:, 1][result[:, 1] > XX.max()] = XX.max()
        temp_GM2['lat_g'], temp_GM2['lon_g']  = result[:, 0], result[:, 1]

        temp_COL = pd.merge(MR2, temp_GM2, on = ['lat_g', 'lon_g'], how = 'right')
        MR2GM2 = MR2GM2.append(temp_COL)
    except:
        print('No GOME2 data on %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
GM2AER = pd.read_pickle(dataOutputDir + 'P3_Validation/GOME2_AERONET_2018-2019.pickle')

#%%
# =============================================================================
# first glance
# =============================================================================
TPM = TPM.reset_index(drop = True)
AerNN90 = AerNN90.reset_index(drop = True)
AerNN95 = AerNN95.reset_index(drop = True)
#MR2OMI.to_pickle(dataOutputDir + 'P3_Validation/MERRA2_OMAERUV_AERONET_2006-2016.pickle')
MR2OMI = pd.read_pickle(dataOutputDir + 'P3_Validation/MERRA2_OMAERUV_AERONET_2006-2016.pickle')
mask = (MR2OMI['CF'] >= 0)
MR2OMI = MR2OMI[mask].reset_index(drop = True)

DATA = {'OMAERUV': OMI, 
        'AerNN90': AerNN90, 
        'AerNN95': AerNN95, 
        'TROPOMI': TPM,
        'GOME2': GM2}

DATA_m = {}
for i, ikey in enumerate(DATA.keys()): 
    idata = DATA[ikey]
    result = np.array(list(map(func, list(idata.lat), list(idata.lon))))
    idata['lat_g'], idata['lon_g'] = result[:, 0], result[:, 1]
    DATA_m[ikey] = idata.groupby(['lat_g', 'lon_g']).mean()
    DATA_m[ikey]['num'] = idata.groupby(['lat_g', 'lon_g']).count()['lat']
    DATA_m[ikey].reset_index(inplace = True)

DATA_m['AerNN90']['ALH'] = DATA_m['AerNN90']['ALH_MDTDB']
DATA_m['AerNN95']['ALH'] = DATA_m['AerNN95']['ALH_MDTDB']
DATA_m['GOME2']['ALH'] = DATA_m['GOME2']['AAH']
# =============================================================================
# data availability
# =============================================================================
fig = plt.figure(figsize = (6, 8))
for i, ikey in enumerate(DATA.keys()): 
    ax = fig.add_axes([0.1, 0.8 - i * 0.175, 0.4, 0.17])
    bm = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90, \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    bm.drawparallels(np.arange(-45, 46, 45), labels = [1,0,1,0], linewidth = 0)
    plt.scatter(XX, YY, c = 'lightgray', s = 100)
    cb = plt.scatter(DATA_m[ikey].lon_g, DATA_m[ikey].lat_g, c = DATA_m[ikey]['ALH'], s = 1, cmap = 'CMRmap_r', \
                     norm=colors.LogNorm(vmin=1, vmax= 10))
    plt.text(-170, -75, '(%s) %s ALH' % (plotidx[i * 2], ikey), bbox=dict(facecolor='w', alpha=0.5))
    if i == 4:
        bm.drawmeridians(np.arange(-90, 91, 90), labels = [0,1,0,1], linewidth = 0)
        cax = fig.add_axes([0.1, 0.07, 0.4, 0.01])
        cbar = plt.colorbar(cb, cax = cax, orientation = 'horizontal', extend = 'both', ticks = (1, 10), fraction=0.2, pad=0.05, shrink = 0.9, aspect = 8)
        cbar.set_label('ALH [km]')
        
    ax = fig.add_axes([0.55, 0.8 - i * 0.175, 0.4, 0.17])
    bm = Basemap(llcrnrlon=-180, llcrnrlat=-90, urcrnrlon=180, urcrnrlat=90, \
                lat_0 = 0, lon_0 = 0, projection='cyl',resolution='c')
    bm.drawcoastlines(color='gray',linewidth=1)
    plt.scatter(XX, YY, c = 'lightgray', s = 100)
    cb = plt.scatter(DATA_m[ikey].lon_g, DATA_m[ikey].lat_g, c = DATA_m[ikey]['num'] , s = 1, cmap = 'CMRmap_r', \
                     norm=colors.LogNorm(vmin=1, vmax= 1e2))
    plt.text(-170, -75, '(%s) %s num' % (plotidx[(i * 2 + 1)], ikey), bbox=dict(facecolor='w', alpha=0.5))
    if i == 4:
        bm.drawmeridians(np.arange(-90, 91, 90), labels = [0,1,0,1], linewidth = 0)
        cax = fig.add_axes([0.55, 0.07, 0.4, 0.01])
        cbar = plt.colorbar(cb, cax = cax, orientation = 'horizontal', extend = 'both', ticks = (1,1e2), fraction=0.2, pad=0.05, shrink = 0.9, aspect = 8)
        cbar.set_label('Number [-]')
plt.savefig(figdir + 'ALH_products_distribution.png', dpi = 300, transparent = True)


#%%
# =============================================================================
# UVAI and MERRA-2_AERONET OMAERUV ALH
# =============================================================================
bins = (0, 0.2, 0.5, 1)

fig = plt.figure(figsize = (12, 3))
mask = (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff']) & (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1'])
a, b = MR2OMI[mask].residue, MR2OMI[mask]['ALH']
ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
cb = plt.scatter(a, b, s = 15, c = MR2OMI[mask].AOD500, norm=colors.LogNorm(vmin=1e-1, vmax=1), cmap = 'CMRmap_r', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
plt.ylabel('OMAERUV ALH [km]')
plt.xlabel('UVAI [-]')
plt.text(0, 5, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = (1e-1, 1), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)

for i, iAOD in enumerate(bins):
    ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
    if i == len(bins) - 1:
        AOD1, AOD2 = bins[i], bins[i]
        mask = (MR2OMI['AOD500'] >= AOD1) & (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff']) & (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1'])
        plt.title(r'(%s) AOD$_{500}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
    else:
        AOD1, AOD2 = bins[i], bins[i + 1]
        mask = (MR2OMI['AOD500'] >= AOD1) & (MR2OMI['AOD500'] < AOD2)  & (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff']) & (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1'])
        plt.title(r'(%s) %1.2f$\leq$AOD$_{500}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)

    a, b = MR2OMI[mask]['residue'], MR2OMI[mask]['ALH']
    plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    plt.yticks([])
    plt.text(0, 5, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
    plt.xlim(-1, 6)
    plt.ylim(0, 6)
    plt.xlabel('UVAI [-]')
plt.savefig(figdir + 'UVAI_ALH_OMAERUV_MR2OMIAER.png', dpi = 300, transparent = True)
#%% 
# =============================================================================
#  UVAI and MERRA-2 ALH
# =============================================================================
bins = (0, 0.2, 0.5, 1)

Hpara = ['Haer_e', 'Haer_a', 'Haer_63', 'Haer_t1']
Hlabels = [r'$H_{aer}^{\beta}$', r'$H_{aer}^{\tau}$', r'$H_{aer}^{63}$', \
           r'$H_{aer}^t$']

for j, ipara in enumerate(Hpara):
    fig = plt.figure(figsize = (12, 3))
    mask = (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1'])  & (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff'])
    a, b = MR2OMI[mask].residue, MR2OMI[mask][ipara]
    ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
    cb = plt.scatter(a, b, s = 15, c = MR2OMI[mask].AOD, norm=colors.LogNorm(vmin=1e-2, vmax=1), cmap = 'CMRmap_r',  marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#    cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
    plt.xlabel('UVAI [-]')
    plt.text(-1, 13, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
    plt.ylabel('%s [km]' % Hlabels[j])
    plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
    plt.xlim(-2, 6)
    plt.ylim(0, 15)

    cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
    cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = (1e-2, 1e-1, 1), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)
    
    for i, iAOD in enumerate(bins):
        ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
        if i == len(bins) -1 :
            AOD1, AOD2 = bins[i], bins[i]
            mask = (MR2OMI['AOD'] >= AOD1) & (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1'])  & (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff'])
            plt.title(r'(%s) AOD$_{550}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
        else:
            AOD1, AOD2 = bins[i], bins[i + 1]
            mask = (MR2OMI['AOD'] >= AOD1) & (MR2OMI['AOD'] < AOD2)  & (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1']) & (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff'])
            plt.title(r'(%s) %1.2f$\leq$AOD$_{550}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)
    
        a, b = MR2OMI[mask]['residue'], MR2OMI[mask][ipara]
        plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    #    handle = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = (30, 30), vmin = 1, vmax = 50)
        plt.yticks([])
        plt.text(-1, 13, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
        plt.xlim(-2, 6)
        plt.ylim(0, 15)
        plt.xlabel('UVAI [-]')
plt.savefig(figdir + 'UVAI_ALH_MERRA-2.png', dpi = 300, transparent = True)
    
#%%
# =============================================================================
# UVAI and OMAERUV ALH monthly
# =============================================================================
bins = (0, 0.2, 0.5, 1)

fig = plt.figure(figsize = (12, 3))
mask = (OMI['AOD500'] > 0)
a, b = OMI[mask].residue, OMI[mask]['ALH']
ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
cb = plt.scatter(a, b, s = 15, c = OMI[mask].AOD500, norm=colors.LogNorm(vmin=1e-1, vmax=1), cmap = 'CMRmap_r', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
plt.ylabel('OMAERUV ALH [km]')
plt.xlabel('UVAI [-]')
plt.text(0, 5, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = (1e-1, 1), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)

for i, iAOD in enumerate(bins):
    ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
    if i == len(bins) - 1:
        AOD1, AOD2 = bins[i], bins[i]
        mask = (OMI['AOD500'] >= AOD1) # & (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff']) & (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1'])
        plt.title(r'(%s) AOD$_{500}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
    else:
        AOD1, AOD2 = bins[i], bins[i + 1]
        mask = (OMI['AOD500'] >= AOD1) & (OMI['AOD500'] < AOD2) # & (MR2OMI['AODdiff']) & (MR2OMI['SSAdiff']) & (MR2OMI['AODdiff1']) & (MR2OMI['SSAdiff1'])
        plt.title(r'(%s) %1.2f$\leq$AOD$_{500}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)

    a, b = OMI[mask]['residue'], OMI[mask]['ALH']
    plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    plt.yticks([])
    plt.text(0, 5, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
    plt.xlim(-1, 6)
    plt.ylim(0, 6)
    plt.xlabel('UVAI [-]')
plt.savefig(figdir + 'UVAI_ALH_OMAERUV_monthly.png', dpi = 300, transparent = True)

#%%
# =============================================================================
# UVAI and TROPOMI ALH
# =============================================================================
bins = (0, 1, 2, 3)

fig = plt.figure(figsize = (12, 3))
mask = (TPM.ALH > 0)
a, b = TPM[mask].AI388, TPM[mask]['ALH']
ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
#cb = plt.scatter(a, b, s = 15, c = TPM[mask].H0, cmap = 'CMRmap_r', marker = 's', facecolorsnorm=colors.LogNorm(vmin=1e-2, vmax= 3) = 'none', edgecolors = 'royalblue', alpha = 0.4)

cb = plt.scatter(a, b, s = 15, c = TPM[mask].AOD, norm=colors.LogNorm(vmin=5e-1, vmax= 5), cmap = 'CMRmap_r', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
plt.ylabel('TROPOMI ALH [km]')
plt.xlabel('UVAI [-]')
plt.text(2, 9, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
plt.xlim(1, 8)
plt.ylim(0, 10)
cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = np.array([5e-1, 1, 5]), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)

for i, iAOD in enumerate(bins):
    ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
    if i == len(bins) - 1:
        AOD1, AOD2 = bins[i], bins[i]
        mask = (TPM['AOD'] >= AOD1)
        plt.title(r'(%s) AOD$_{500}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
    else:
        AOD1, AOD2 = bins[i], bins[i + 1]
        mask = (TPM['AOD'] >= AOD1) & (TPM['AOD'] <= AOD2)
        plt.title(r'(%s) %1.2f$\leq$AOD$_{500}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)

    a, b = TPM[mask]['AI388'], TPM[mask]['ALH']
    plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    plt.yticks([])
    plt.text(2, 9, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
    plt.xlim(1, 8)
    plt.ylim(0, 10)
    plt.xlabel('UVAI [-]')
plt.savefig(figdir + 'UVAI_ALH_TROPOMI.png', dpi = 300, transparent = True)

#%%
# =============================================================================
# UVAI and AerNN90
# =============================================================================
bins = (0.5, 1.5, 2, 2.5)
Q1, Q3 = np.log(AerNN90.ALH_MDTDB).quantile(0.25), np.log(AerNN90.ALH_MDTDB).quantile(0.75)
IQR = Q3 - Q1
lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR


fig = plt.figure(figsize = (12, 3))
mask = (AerNN90.AOD_MODIS >=0) #& (AerNN90.CF < 0.02) & (AerNN90.AOD_MODIS >= 1.5) & (AerNN90.As_OMI < 0.05)
#(np.log(AerNN90.ALH_MDTDB) >= lb) & (np.log(AerNN90.ALH_MDTDB) <= ub)
a, b = AerNN90[mask].residue, AerNN90[mask]['ALH_MDTDB']
ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
cb = plt.scatter(a, b, s = 15, c = AerNN90[mask].AOD_MODIS, norm=colors.LogNorm(vmin=5e-1, vmax= 5), cmap = 'CMRmap_r', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
plt.ylabel('AerNN90 ALH [km]')
plt.xlabel('UVAI [-]')
plt.text(2, 55, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
#plt.xlim(-1, 8)
#plt.ylim(0, 10)
cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = np.array([5e-1, 5]), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)

for i, iAOD in enumerate(bins):
    ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
    if i == len(bins) - 1:
        AOD1, AOD2 = bins[i], bins[i]
        mask = (AerNN90['AOD_MODIS'] >= AOD1)
        plt.title(r'(%s) AOD$_{500}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
    else:
        AOD1, AOD2 = bins[i], bins[i + 1]
        mask = (AerNN90['AOD_MODIS'] >= AOD1)
        plt.title(r'(%s) %1.2f$\leq$AOD$_{500}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)

    a, b = AerNN90[mask]['residue'], AerNN90[mask]['ALH_MDTDB']
    plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    if i > 0:
        plt.yticks([])
    plt.text(0, 55, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
    plt.xlim(-1, 8)
    plt.ylim(0, 60)
    plt.xlabel('UVAI [-]')
    if i == 0:
        plt.ylabel('OMAERUV ALH [km]')    
plt.savefig(figdir + 'UVAI_ALH_AerNN90.png', dpi = 300, transparent = True)

#%%
# =============================================================================
# UVAI and AerNN95
# =============================================================================
bins = (0.5, 1.5, 2, 2.5)
Q1, Q3 = np.log(AerNN95.ALH_MDTDB).quantile(0.25), np.log(AerNN95.ALH_MDTDB).quantile(0.75)
IQR = Q3 - Q1
lb = Q1 - 1.5 * IQR
ub = Q3 + 1.5 * IQR


fig = plt.figure(figsize = (12, 3))
mask = (AerNN95.AOD_MODIS >=0) #& (AerNN95.CF < 0.02) & (AerNN95.AOD_MODIS >= 1.5) & (AerNN95.As_OMI < 0.05)
#(np.log(AerNN95.ALH_MDTDB) >= lb) & (np.log(AerNN95.ALH_MDTDB) <= ub)
a, b = AerNN95[mask].residue, AerNN95[mask]['ALH_MDTDB']
ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
cb = plt.scatter(a, b, s = 15, c = AerNN95[mask].AOD_MODIS, norm=colors.LogNorm(vmin=5e-1, vmax= 5), cmap = 'CMRmap_r', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
plt.ylabel('AerNN95 ALH [km]')
plt.xlabel('UVAI [-]')
plt.text(2, 55, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
#plt.xlim(-1, 8)
#plt.ylim(0, 10)
cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = np.array([5e-1, 5]), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)

for i, iAOD in enumerate(bins):
    ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
    if i == len(bins) - 1:
        AOD1, AOD2 = bins[i], bins[i]
        mask = (AerNN95['AOD_MODIS'] >= AOD1)
        plt.title(r'(%s) AOD$_{500}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
    else:
        AOD1, AOD2 = bins[i], bins[i + 1]
        mask = (AerNN95['AOD_MODIS'] >= AOD1)
        plt.title(r'(%s) %1.2f$\leq$AOD$_{500}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)

    a, b = AerNN95[mask]['residue'], AerNN95[mask]['ALH_MDTDB']
    plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    if i > 0:
        plt.yticks([])
    plt.text(0, 55, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
    plt.xlim(-1, 8)
    plt.ylim(0, 60)
    plt.xlabel('UVAI [-]')
    if i == 0:
        plt.ylabel('OMAERUV ALH [km]')    
plt.savefig(figdir + 'UVAI_ALH_AerNN95.png', dpi = 300, transparent = True)

#%%
# =============================================================================
# UVAI and AERONET_GOME2 ALH
# =============================================================================
bins = (0, 0.5, 1, 2)

fig = plt.figure(figsize = (12, 3))
mask = (GM2AER.AI380 >=0)
a, b = GM2AER[mask].AI380, GM2AER[mask]['AAH']
ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
cb = plt.scatter(a, b, s = 15, c = GM2AER[mask].AOD_550nm, norm=colors.LogNorm(vmin=1e-2, vmax= 3), cmap = 'CMRmap_r', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
plt.ylabel('GOME2 AAH [km]')
plt.xlabel('UVAI [-]')
plt.text(2, 9, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
#plt.xlim(1, 8)
plt.ylim(0, 10)
cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = np.array([1e-1, 1, 3]), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)

for i, iAOD in enumerate(bins):
    ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
    if i == len(bins) - 1:
        AOD1, AOD2 = bins[i], bins[i]
        mask = (GM2AER['AOD_550nm'] >= AOD1)
        plt.title(r'(%s) AOD$_{500}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
    else:
        AOD1, AOD2 = bins[i], bins[i + 1]
        mask = (GM2AER['AOD_550nm'] >= AOD1) & (GM2AER['AOD_550nm'] <= AOD2)
        plt.title(r'(%s) %1.2f$\leq$AOD$_{500}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)

    a, b = GM2AER[mask]['AI380'], GM2AER[mask]['AAH']
    plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    plt.yticks([])
    plt.text(2, 9, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
#    plt.xlim(1, 8)
    plt.ylim(0, 10)
    plt.xlabel('UVAI [-]')
plt.savefig(figdir + 'UVAI_ALH_GOME2_GM2AER.png', dpi = 300, transparent = True)

#%%
# =============================================================================
# UVAI and MERRA2_GOME2 ALH
# =============================================================================
#MR2OMI = MR2OMI.groupby(['lat', 'lon']).mean()
#MR2OMI.reset_index(inplace = True)
bins = (0, 0.5, 1, 2)

fig = plt.figure(figsize = (12, 3))
mask = (MR2GM2.AI380 >=0)
a, b = MR2GM2[mask].AI380, MR2GM2[mask]['AAH']
ax = fig.add_axes([0.05, 0.2, 0.15, 0.65])
cb = plt.scatter(a, b, s = 15, c = MR2GM2[mask].AOD, norm=colors.LogNorm(vmin=1e-2, vmax= 3), cmap = 'CMRmap_r', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.4)
#cb = plt.hist2d(a, b, cmap = 'CMRmap_r', bins = 50, vmin = 1, vmax = 50)
plt.ylabel('GOME2 AAH [km]')
plt.xlabel('UVAI [-]')
plt.text(5, 9, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
plt.title('(%s) All samples' % plotidx[0], fontsize = 10)
#plt.xlim(1, 8)
plt.ylim(0, 10)
cax = fig.add_axes([0.21, 0.2, 0.015, 0.65])
cbar = plt.colorbar(cb, cax = cax, orientation = 'vertical', extend = 'both', ticks = np.array([1e-1, 1, 3]), fraction=0.1, pad=0.05, shrink = 0.9, aspect = 10)

for i, iAOD in enumerate(bins):
    ax = fig.add_axes([0.1 + 0.175 * (i + 1), 0.2, 0.15, 0.65])
    if i == len(bins) - 1:
        AOD1, AOD2 = bins[i], bins[i]
        mask = (MR2GM2['AOD'] >= AOD1)
        plt.title(r'(%s) AOD$_{500}\geq$%1.2f' % (plotidx[i + 1], AOD1), fontsize = 10)
    else:
        AOD1, AOD2 = bins[i], bins[i + 1]
        mask = (MR2GM2['AOD'] >= AOD1) & (MR2GM2['AOD'] <= AOD2)
        plt.title(r'(%s) %1.2f$\leq$AOD$_{500}$<%1.2f' % (plotidx[i + 1], AOD1, AOD2), fontsize = 10)

    a, b = MR2GM2[mask]['AI380'], MR2GM2[mask]['AAH']
    plt.scatter(a, b, s = 15, c = 'royalblue', marker = 's', facecolors = 'none', edgecolors = 'royalblue', alpha = 0.1)
    plt.yticks([])
    plt.text(5, 9, 'num:%i R:%1.2f' % (mask.sum(), a.corr(b, 'spearman')))
#    plt.xlim(1, 8)
    plt.ylim(0, 10)
    plt.xlabel('UVAI [-]')
plt.savefig(figdir + 'UVAI_ALH_GOME2_MR2GM2.png', dpi = 300, transparent = True)
    
t2 = time.time()
print('Time: %1.2f' % (t2 - t1))
