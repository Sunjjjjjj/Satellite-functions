#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 11:07:49 2022

@author: sunji
"""

import sys
sys.path.insert(0, '/home/sunji/Scripts/')
import os, glob
import netCDF4
import Nio
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
from utilities import *


def readOCO2(inputfile, ROI):
    # read file
    sys.stdout.write('\r Reading %s ...' % (inputfile))
    
    data = netCDF4.Dataset(inputfile, 'r')
    # data = Nio.open_file(inputfile, 'r')
    temp = pd.DataFrame()
    
    # read XCO2 Lite product
    if inputfile.find('oco2_LtCO2') >= 0:
        parameters = ['xco2', 'sensor_zenith_angle', 'solar_zenith_angle', 'latitude', 'longitude', 'time',
                     'xco2_qf_bitflag', 'xco2_qf_simple_bitflag', 'xco2_quality_flag', 'xco2_uncertainty',
                     ]
        meteo = ['windspeed_u_met', 'windspeed_v_met', 'psurf_apriori_wco2', 'psurf_apriori_sco2', 'psurf_apriori_o2a']
        sounding = ['altitude', 'land_fraction', 'land_water_indicator', 'sensor_azimuth_angle', 'solar_azimuth_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['%s' % ikey][:].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        for ikey in meteo:
            try:
                temp[ikey] = data['Meteorology/%s' % ikey][:].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        for ikey in sounding:
            try:
                temp[ikey] = data['Sounding/%s' % ikey][:].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        temp['time'] = pd.to_datetime('1970-01-01 00:00:00') + pd.to_timedelta(temp['time'], unit = 'seconds')
        # time = np.array(data['time'][:], str)
        # temp['time_utc'] = pd.to_datetime(time.reshape(-1))
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        temp = temp[mask]
        temp.reset_index(drop = True, inplace = True)
        
        
    # read XCO2 standard product
    if inputfile.find('oco2_L2Std') >= 0:
        variables = ['xco2']
        other = ['latitude', 'longitude']
        parameters = variables
        for ikey in parameters:
            try:
                temp[ikey] = data['RetrievalResults/%s' % ikey][:].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        for ikey in other:
            try:
                temp[ikey] = data['RetrievalGeometry/retrieval_%s' % ikey][:]
            except:
                print('No %s' % ikey)
        # time = np.array(data['time'][:], str)
        # temp['time_utc'] = pd.to_datetime(time.reshape(-1))
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        temp = temp[mask]
        temp.reset_index(drop = True, inplace = True)
        
    # read XCO2 IMAPDOAS product
    if inputfile.find('oco2_L2IDP') >= 0:
        variables = ['co2_column_strong_band_idp', 'co2_column_weak_band_idp']
        other = ['latitude', 'longitude']
        parameters = variables
        for ikey in parameters:
            try:
                temp[ikey] = data['DOASCO2/%s' % ikey][:].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        for ikey in other:
            try:
                temp[ikey] = data['SoundingGeometry/sounding_%s' % ikey][:].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        # time = np.array(data['time'][:], str)
        # temp['time_utc'] = pd.to_datetime(time.reshape(-1))
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        temp = temp[mask]
        temp.reset_index(drop = True, inplace = True)
        
    data.close()
    return temp





# ROI = {'S':15, 'N': 55, 'W': 65, 'E': 135}

# inputfile = '/home/sunji/Data/OCO2/OCO2_L2_Lite_FP_10r/oco2_LtCO2_210102_B10206Ar_210922005052s.SUB.nc4'
# temp = readOCO2(inputfile, ROI)
# oco2 = pd.DataFrame()
# filelist = glob.glob('/home/sunji/Data/oco2/oco2_L2Std/*.h5')
# for inputfile in filelist:

#     temp = readOCO2(inputfile, ROI)
#     oco2 = pd.concat([oco2, temp], axis = 0)