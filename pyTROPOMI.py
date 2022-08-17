#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:23:58 2022

@author: sunji
"""
import sys
sys.path.insert(0, '/home/sunji/Scripts/')
import os, glob
import netCDF4
import Nio
import numpy as np
import pandas as pd
import scipy.interpolate
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
from utilities import *

def readTROPOMI(inputfile, ROI, unit = 'mol', qa = 0.5):
    # read file
    sys.stdout.write('\r Reading %s ...' % (inputfile))
    
    data = netCDF4.Dataset(inputfile, 'r')
    # data = Nio.open_file(inputfile, 'r')
    temp = pd.DataFrame()
    
    if unit in ['mol', 'molecule']:
        pass
    else:
        print('Unit label is incorrect! Choose from ''mol', 'molecule''.')
        
    # read CO product
    if inputfile.find('S5P_OFFL_L2__CO_____') >= 0:
        variables = ['carbonmonoxide_total_column', 'carbonmonoxide_total_column_corrected',
                      'carbonmonoxide_total_column_precision']
        other = ['qa_value', 'latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        factor_molecule = data['PRODUCT/carbonmonoxide_total_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                try:
                    temp[ikey] = temp[ikey] * factor_molecule
                except:
                    pass
            
    # read O3 product
    if inputfile.find('S5P_OFFL_L2__O3_____') >= 0:
        variables = ['ozone_total_vertical_column', 'ozone_total_vertical_column_precision'] 
        other = ['qa_value', 'latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        factor_DU = data['PRODUCT/ozone_total_vertical_column'].multiplication_factor_to_convert_to_DU
        factor_molecule = data['PRODUCT/ozone_total_vertical_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'DU':
            for ikey in variables:
                temp[ikey] = temp[ikey] * factor_DU
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * factor_molecule
            
    
    # read NO2 product
    if inputfile.find('L2__NO2____') >= 0:
        variables = ['nitrogendioxide_tropospheric_column', 'nitrogendioxide_tropospheric_column_precision']
        other = ['nitrogendioxide_tropospheric_column_precision_kernel', 
                      'air_mass_factor_troposphere', 'air_mass_factor_total', 'qa_value',
                     'latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        temp['aerosol_index_354_388'] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/aerosol_index_354_388'][0].filled(np.nan).reshape(-1)
        
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        factor_molecule = data['PRODUCT/nitrogendioxide_tropospheric_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * factor_molecule


    # read SO2 product
    if inputfile.find('S5P_OFFL_L2__SO2____') >= 0:
        variables = ['sulfurdioxide_total_vertical_column', 'sulfurdioxide_total_vertical_column_precision']
        other = ['qa_value','latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        temp['aerosol_index_340_380'] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/aerosol_index_340_380'][0].filled(np.nan).reshape(-1)
        
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        factor_molecule = data['PRODUCT/sulfurdioxide_total_vertical_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * factor_molecule

    # read CH4 product
    if inputfile.find('S5P_OFFL_L2__CH4____') >= 0:
        variables = ['methane_mixing_ratio', 'methane_mixing_ratio_bias_corrected', 'methane_mixing_ratio_precision']
        other = ['qa_value','latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))

        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)

    # read HCHO product
    if inputfile.find('L2__HCHO___') >= 0:
        variables = ['formaldehyde_tropospheric_vertical_column', 'formaldehyde_tropospheric_vertical_column_precision']
        other = ['qa_value','latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))

        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        factor_molecule = data['PRODUCT/formaldehyde_tropospheric_vertical_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * factor_molecule

    # read UVAI product
    if inputfile.find('S5P_OFFL_L2__AER_AI_') >= 0:
        variables = ['aerosol_index_340_380', 'aerosol_index_340_380_precision',
                     'aerosol_index_354_388', 'aerosol_index_354_388_precision']
        other = ['qa_value','latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))

        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)

    # read ALH product
    if inputfile.find('S5P_OFFL_L2__AER_LH_') >= 0:
        variables = ['aerosol_mid_height', 'aerosol_mid_height_precision',
                     'aerosol_mid_pressure', 'aerosol_mid_pressure_precision']
        other = ['qa_value','latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))

        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)

    # read O3 profile product
    if inputfile.find('S5P_OFFL_L2__O3__PR_') >= 0:
        variables = ['ozone_profile', 'ozone_profile_precision']
        other = ['ozone_total_column', 'ozone_total_column_precision',
                 'ozone_tropospheric_column', 'ozone_tropospheric_column_precision',
                 'qa_value', 'latitude', 'longitude']
        # read layer information
        levels =  data['PRODUCT/level'][:]
        
        # altitude grid
        z = np.round(data['PRODUCT/altitude'][0].mean(axis = (0, 1)).data / 1e3) * 1e3
        
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        
        temp = {}
        for ikey in variables:
            pr = []
            for i in range(xdim):
                for j in range(ydim):
                    x, y = data['PRODUCT/altitude'][0, i, j, :].data, data['PRODUCT/%s' % ikey][0, i, j, :].data
                    # unit coversion factor
                    factor_molecule = data['PRODUCT/%s' % ikey].multiplication_factor_to_convert_to_molecules_percm3
                    # convert unit
                    if unit == 'molecule':
                        y = y * factor_molecule
                    func = scipy.interpolate.interp1d(x, y, fill_value = 'extrapolate')
                    y_interp = func(z)
                    pr.append(y_interp)
            temp[ikey] = pd.DataFrame(pr, columns = np.around(z/1e3))
       

        temp_ = pd.DataFrame()
        for ikey in other:
            temp_[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
            # temp = pd.concat([temp, temp_], axis = 1)
        
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp_['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))

        for i in range(4):
            temp_['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp_['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        
       # mask ROI
        mask = ROImask(temp_['latitude'], temp_['longitude'], ROI)
        qamask = temp_['qa_value'] >= qa
        for ikey in temp.keys():
            temp[ikey] = temp[ikey][mask & qamask]
            temp[ikey].reset_index(drop = True, inplace = True)

        temp_ = temp_[mask & qamask]
        temp_.reset_index(drop = True, inplace = True)
        # unit coversion factor
        factor_DU = data['PRODUCT/ozone_total_column'].multiplication_factor_to_convert_to_DU
        factor_molecule = data['PRODUCT/ozone_total_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        variables = ['ozone_total_column', 'ozone_total_column_precision',
                 'ozone_tropospheric_column', 'ozone_tropospheric_column_precision']
        if unit == 'DU':
            for ikey in variables:
                temp_[ikey] = temp_[ikey] * factor_DU
        if unit == 'molecule':
            for ikey in variables:
                temp_[ikey] = temp_[ikey] * factor_molecule
        temp['column'] = temp_
        
    data.close()
    return temp


def subTROPOMI(inputfile, ROI):
    # read file
    sys.stdout.write('\r Reading %s ...' % (inputfile))
    data = netCDF4.Dataset(inputfile, 'r')
    
    # ROI mask
    Y, X = data['PRODUCT']['latitude'][0].shape
    mask = ROImask(data['PRODUCT']['latitude'][0], data['PRODUCT']['longitude'][0], ROI)
    y, x = np.where(mask == True)
    latitude = data['PRODUCT']['latitude'][0][y.min():y.max(), x.min():x.max()]
    longitude = data['PRODUCT']['longitude'][0][y.min():y.max(), x.min():x.max()]
    # variable list
    output = {}
    for igroup in ['PRODUCT', 'PRODUCT/SUPPORT_DATA/DETAILED_RESULTS', 'PRODUCT/SUPPORT_DATA/GEOLOCATIONS', 'PRODUCT/SUPPORT_DATA/INPUT_DATA']:
    # for igroup in ['PRODUCT', 'PRODUCT/SUPPORT_DATA/GEOLOCATIONS', 'PRODUCT/SUPPORT_DATA/INPUT_DATA']:
        varlist = list(data[igroup].variables.keys())
    
        for ikey in varlist:
            output[ikey] = {}
            output[ikey]['metainfo'] = str(data[igroup][ikey])
            
            try:
                output['multiplication_factor_to_convert_to_molecules_percm2'] = data[igroup][ikey].multiplication_factor_to_convert_to_molecules_percm2
            except:
                pass
            
            try:
                var = np.squeeze(data[igroup][ikey][:].filled(np.nan))
            except:
                pass
            
            if var.size >= X * Y:
                if len(var.shape) > 2:
                    output[ikey]['data'] = var[y.min():y.max(), x.min():x.max(), :]
                else:
                    output[ikey]['data'] = var[y.min():y.max(), x.min():x.max()]
            elif var.size == Y:
                output[ikey]['data'] = var[y.min():y.max()]
            elif var.size == X:
                output[ikey]['data'] = var[x.min():x.max()]
            else: 
                output[ikey]['data'] = var
    
    # save as pickle
    f = open("%s.SUB.pickle" % (inputfile[:-3]), "wb")
    pickle.dump(output, f)
    data.close()
    return output



def getOrbitNumber(inputfile):
    orb = inputfile.split('_')[-4]
    # check length
    if len(orb) != 5:
        print('This is not an orbit number!')
    return int(orb)
    


# def surfOzone(data_column, data_profile):
    
    
    


#%%

# ROI = {'S': 15, 'N': 55, 'W': 65, 'E': 138}
# inputfile = '/home/sunji/Data/TROPOMI/OFFL_L2__CO_____/S5P_OFFL_L2__CO_____20210101T031206_20210101T045336_16681_01_010400_20210102T170040.nc'
# f = '/home/sunji/Data/TROPOMI/OFFL_L2__O3_____/S5P_OFFL_L2__O3_____20211029T041211_20211029T055341_20952_02_020201_20211030T201719.nc'
# f = '/home/sunji/Data/如皋/S5P_OFFL_L2__NO2____20220115T044821_20220115T062952_22059_02_020301_20220116T205403.nc'
# f = '/home/sunji/Data/TROPOMI/OFFL_L2_SO2_____/S5P_OFFL_L2__NO2____20220115T044821_20220115T062952_22059_02_020301_20220116T205403.nc'
# inputfile = '/home/sunji/Data/TROPOMI/OFFL_L2_CH4_____/S5P_OFFL_L2__CH4____20220101T040939_20220101T055109_21860_02_020301_20220102T201551.nc'
# f = '/home//Data/TROPOMI/OFFL_L2_HCHO___/S5P_OFFL_L2__HCHO___20220101T022809_20220101T040939_21859_02_020201_20220102T182339.nc'
# inputfile = '/home/sunji/Data/TROPOMI/OFFL_L2_HCHO___/S5P_OFFL_L2__HCHO___20220101T040939_20220101T055109_21860_02_020201_20220102T201551.nc'
# f = '/home/sunji/Data/TROPOMI/OFFL_L2__AER_LH_/S5P_OFFL_L2__AER_LH_20220115T044821_20220115T062952_22059_02_020301_20220116T205401.nc'
# f = '/home/sunji/Data/TROPOMI/OFFL_L2__AER_AI_/S5P_OFFL_L2__AER_AI_20220115T044821_20220115T062952_22059_02_020301_20220116T183749.nc'
# inputfile = '/home/sunji/Data/TROPOMI/OFFL_L2__NO2____/S5P_OFFL_L2__NO2____20220131T030802_20220131T044932_22285_02_020301_20220201T190344.nc'
inputfile = '/home/sunji/Data/TROPOMI/OFFL_L2__O3__PR_/S5P_OFFL_L2__O3__PR_20220801T030515_20220801T044645_24867_03_020400_20220803T064306.SUB.nc4'
print(getOrbitNumber(inputfile))
# data = readTROPOMI(inputfile, ROI)
# output = subTROPOMI(inputfile, ROI)

# lat = output['latitude']['data']
# lon = output['longitude']['data']

# data = netCDF4.Dataset(inputfile, 'r')
# # ROI mask
# Y, X = data['PRODUCT']['latitude'][0].shape
# mask = ROImask(data['PRODUCT']['latitude'][0], data['PRODUCT']['longitude'][0], ROI)
# y, x = np.where(mask == True)
# latitude = data['PRODUCT']['latitude'][0][y.min():y.max(), x.min():x.max()]
# longitude = data['PRODUCT']['longitude'][0][y.min():y.max(), x.min():x.max()]
#     # variable list

# data_column = 
# data_profile = 



