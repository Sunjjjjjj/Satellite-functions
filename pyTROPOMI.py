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
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pickle
from utilities import *

def readTROPOMI(inputfile, ROI, unit = 'mol', qa = 0.5, save_csv = False, save_pkl = False, **kwargs):
    # read file
    sys.stdout.write('\r Reading %s ...' % (inputfile))
    
    data = netCDF4.Dataset(inputfile, 'r')
    # data = Nio.open_file(inputfile, 'r')
    temp = pd.DataFrame()
    if unit in ['mol', 'molecule', 'DU']:
        pass
    else:
        print('Unit label is incorrect! Choose from ''mol', 'molecule', 'DU''.')
# =============================================================================
#     # read CO product
# =============================================================================
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
        
        # geometry
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        # detail results
        parameters = ['surface_albedo_2325', 'surface_albedo_2335', 'water_total_column', 'water_total_column_precision']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
        # input data 
        parameters = ['surface_altitude', 'surface_pressure']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        # geometry data
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        temp['factor_molecule'] = data['PRODUCT/carbonmonoxide_total_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                try:
                    temp[ikey] = temp[ikey] * temp['factor_molecule']
                except:
                    pass
# =============================================================================
#     # read O3 product
# =============================================================================
    if inputfile.find('S5P_OFFL_L2__O3_____') >= 0:
        variables = ['ozone_total_vertical_column', 'ozone_total_vertical_column_precision'] 
        other = ['qa_value', 'latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        
        # geometry
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        
        # detail results
        # parameters = []
        # for ikey in parameters:
        #     try:
        #         temp[ikey] = data['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/%s' % ikey][0].filled(np.nan).reshape(-1)
        #     except:
        #         print('No %s' % ikey)
                
        # input data 
        parameters = ['surface_albedo', 'surface_altitude', 'surface_pressure', 'cloud_fraction_crb', 'cloud_albedo_crb', 'cloud_pressure_crb']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        # geometry data
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
                
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        temp['factor_DU'] = data['PRODUCT/ozone_total_vertical_column'].multiplication_factor_to_convert_to_DU
        temp['factor_molecule'] = data['PRODUCT/ozone_total_vertical_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'DU':
            for ikey in variables:
                temp[ikey] = temp[ikey] * temp['factor_DU']
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * temp['factor_molecule']
            
# =============================================================================
#     # read NO2 product
# =============================================================================
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
        
        
        
        # detail results
        parameters = ['nitrogendioxide_slant_column_density', 'nitrogendioxide_stratospheric_column', 'nitrogendioxide_summed_total_column', 'nitrogendioxide_total_column']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
        # input data 
        parameters = ['surface_albedo', 'surface_altitude', 'surface_pressure', 'cloud_fraction_crb', 'cloud_albedo_crb', 'cloud_pressure_crb', 'aerosol_index_354_388']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        # geometry data
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        temp['factor_molecule'] = data['PRODUCT/nitrogendioxide_tropospheric_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * temp['factor_molecule']

# =============================================================================
#     # read SO2 product
# =============================================================================
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
        
        # detail results
        parameters = ['sulfurdioxide_total_vertical_column_15km', 'sulfurdioxide_total_vertical_column_15km_precision', 'sulfurdioxide_total_vertical_column_15km_trueness', 'sulfurdioxide_total_vertical_column_1km', 'sulfurdioxide_total_vertical_column_1km_precision', 'sulfurdioxide_total_vertical_column_1km_trueness', 'sulfurdioxide_total_vertical_column_7km', 'sulfurdioxide_total_vertical_column_7km_precision', 'sulfurdioxide_total_vertical_column_7km_trueness', 'sulfurdioxide_total_vertical_column_trueness']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
        # input data 
        parameters = ['aerosol_index_340_380', 'cloud_albedo_crb', 'cloud_fraction_crb', 'cloud_height_crb', 'cloud_pressure_crb', 'eastward_wind', 'northward_wind', 'sea_ice_cover', 'snow_cover', 'snow_ice_flag', 'surface_albedo_328nm', 'surface_albedo_376nm', 'surface_altitude', 'surface_classification', 'surface_pressure', 'surface_temperature']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        # geometry data
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
       
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        temp['factor_molecule'] = data['PRODUCT/sulfurdioxide_total_vertical_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * temp['factor_molecule']

# =============================================================================
#     # read CH4 product
# =============================================================================
    if inputfile.find('S5P_OFFL_L2__CH4____') >= 0:
        variables = ['methane_mixing_ratio', 'methane_mixing_ratio_bias_corrected', 'methane_mixing_ratio_precision']
        other = ['qa_value','latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        
        # detail results
        parameters = ['aerosol_mid_altitude', 'aerosol_mid_altitude_precision', 'aerosol_number_column', 'aerosol_number_column_precision', 'aerosol_optical_thickness_NIR', 'aerosol_optical_thickness_SWIR', 'surface_albedo_NIR','surface_albedo_SWIR', 'water_total_column', 'fluorescence']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
        # input data 
        parameters = ['surface_altitude', 'surface_pressure', 'surface_classification']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        # geometry data
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
       

        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)

# =============================================================================
#     # read HCHO product
# =============================================================================
    if inputfile.find('L2__HCHO___') >= 0:
        variables = ['formaldehyde_tropospheric_vertical_column', 'formaldehyde_tropospheric_vertical_column_precision']
        other = ['qa_value','latitude', 'longitude']
        parameters = variables + other
        for ikey in parameters:
            temp[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        temp['cloud_fraction_intensity_weighted'] = data['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/cloud_fraction_intensity_weighted'][0].filled(np.nan).reshape(-1)
        
        # geometry
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)

        # detail results
        parameters = ['formaldehyde_tropospheric_vertical_column_correction', 'formaldehyde_tropospheric_vertical_column_trueness']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/DETAILED_RESULTS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
        # input data 
        parameters = ['aerosol_index_340_380', 'cloud_albedo_crb', 'cloud_fraction_crb', 'cloud_height_crb', 'cloud_pressure_crb', 'snow_ice_flag', 'surface_albedo', 'surface_altitude', 'surface_classification', 'surface_pressure']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
        
        # geometry data
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
                
        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)
        # unit coversion factor
        temp['factor_molecule'] = data['PRODUCT/formaldehyde_tropospheric_vertical_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        if unit == 'molecule':
            for ikey in variables:
                temp[ikey] = temp[ikey] * temp['factor_molecule']

# =============================================================================
#     # read UVAI product
# =============================================================================
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
        
        # geometry
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                

        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)

# =============================================================================
#     # read ALH product
# =============================================================================
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

        # geometry
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)

        for i in range(4):
            temp['latitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/latitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
            temp['longitude_bounds%i' %(i + 1)] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/longitude_bounds'][0, :, :, i].filled(np.nan).reshape(-1)
        # mask ROI
        mask = ROImask(temp['latitude'], temp['longitude'], ROI)
        qamask = temp['qa_value'] >= qa
        temp = temp[mask & qamask]
        temp.reset_index(drop = True, inplace = True)

# =============================================================================
#     # read O3 profile product
# =============================================================================
    if inputfile.find('S5P_OFFL_L2__O3__PR_') >= 0:
        variables = ['ozone_profile', 'ozone_profile_precision']
        other = ['ozone_total_column', 'ozone_total_column_precision',
                 'ozone_tropospheric_column', 'ozone_tropospheric_column_precision',
                 'qa_value', 'latitude', 'longitude']
        # read layer information
        levels =  data['PRODUCT/level'][:]
        xdim, ydim = data['PRODUCT/latitude'][0].shape
        m, ydim = data['PRODUCT/latitude'][0].shape
        
        # read 3D variables
        temp = {}
        for ikey in variables:
            # grid profile
            # pr = []
            # for i in range(xdim):
            #     for j in range(ydim):
            #         x, y = data['PRODUCT/altitude'][0, i, j, :].data, data['PRODUCT/%s' % ikey][0, i, j, :].data
            #         # unit coversion factor
            #         temp['factor_molecule'] = data['PRODUCT/%s' % ikey].multiplication_factor_to_convert_to_molecules_percm3
            #         # convert unit
            #         if unit == 'molecule':
            #             y = y * temp['factor_molecule']
            #         func = scipy.interpolate.interp1d(x, y, fill_value = 'extrapolate')
            #         y_interp = func(z)
            #         pr.append(y_interp)
            y = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(xdim * ydim, len(levels))
            temp['factor_molecule'] = data['PRODUCT/%s' % ikey].multiplication_factor_to_convert_to_molecules_percm3
            if unit == 'molecule':
                y = y * temp['factor_molecule']
            temp[ikey] = pd.DataFrame(y, columns = np.array(levels) + 1)
        # read altitude and pressure
        temp['altitude'] = pd.DataFrame(data['PRODUCT/altitude'][0].filled(np.nan).reshape(xdim * ydim, len(levels)), 
                                  columns = np.array(levels) + 1)
        temp['pressure'] = pd.DataFrame(data['PRODUCT/pressure'][0].filled(np.nan).reshape(xdim * ydim, len(levels)), 
                                  columns = np.array(levels) + 1)
        # other 2D variables
        temp_ = pd.DataFrame()
        for ikey in other:
            temp_[ikey] = data['PRODUCT/%s' % ikey][0].filled(np.nan).reshape(-1)
        
        temp_['pressure_at_tropopause'] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/pressure_at_tropopause'][0].filled(np.nan).reshape(-1)        
        temp_['surface_altitude'] = data['PRODUCT/SUPPORT_DATA/INPUT_DATA/surface_altitude'][0].filled(np.nan).reshape(-1)        
        
        
        time = np.array(data['PRODUCT/time_utc'][:], str)[0]
        temp_['time_utc'] = pd.to_datetime(np.tile(time.T, ydim).reshape(-1))
        
        # geometry
        parameters = ['solar_azimuth_angle', 'solar_zenith_angle', 'viewing_azimuth_angle', 'viewing_zenith_angle']
        for ikey in parameters:
            try:
                temp[ikey] = data['PRODUCT/SUPPORT_DATA/GEOLOCATIONS/%s' % ikey][0].filled(np.nan).reshape(-1)
            except:
                print('No %s' % ikey)
                
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
        temp['factor_DU'] = data['PRODUCT/ozone_total_column'].multiplication_factor_to_convert_to_DU
        temp['factor_molecule'] = data['PRODUCT/ozone_total_column'].multiplication_factor_to_convert_to_molecules_percm2
        # convert unit
        variables = ['ozone_total_column', 'ozone_total_column_precision',
                 'ozone_tropospheric_column', 'ozone_tropospheric_column_precision']
        if unit == 'DU':
            for ikey in variables:
                temp_[ikey] = temp_[ikey] * temp['factor_DU']
        if unit == 'molecule':
            for ikey in variables:
                temp_[ikey] = temp_[ikey] * temp['factor_molecule']
        
        # height at tropopause
        height_at_tropopause = []
        # tropospheric ozone column
        tropospheric_column = []
        
        index = []
        for i in range(len(temp['altitude'])):
            x, y = temp['pressure'].iloc[i], temp['altitude'].iloc[i]
            func = scipy.interpolate.interp1d(x, y, fill_value = 'extrapolate')
            y_hat = func(temp_['pressure_at_tropopause'].iloc[0])
            
            # index of tropopause
            idx = y[(y - y_hat) <= 0].argmax()
            index.append(idx)
            profile_t = temp['ozone_profile'].iloc[i][:idx]
            dz = temp['altitude'].iloc[i][1:idx + 1].values - temp['altitude'].iloc[i][:idx].values
            # tropospheric ozone column
            tropospheric_column.append((profile_t * dz * 1e2).sum())
            height_at_tropopause.append(y_hat)
        
        temp_['index_at_tropopause'] = index
        temp_['height_at_tropopause'] = height_at_tropopause
        temp_['ozone_tropospheric_column'] = tropospheric_column
        
        temp['column'] = temp_
        
        # regrid profile
        if len(temp['ozone_profile']) > 0:
            prgrid = []
            # z = altitude.mean(axis = 0)
            # z = round(temp['altitude'].mean(axis = 0) / 1e3)  # convert to km
            # z.iloc[0] = 0.03 # z0 is at 30 meter
            z = [ 0.03,  3.,  5.,  7.,  9., 11., 13., 15., 17., 18., 20., 22., 24.,
                   26., 28., 29., 31., 33., 35., 37., 40., 42., 44., 46., 48., 53.,
                   57., 61., 65., 69., 73., 76., 80.]
            for i in range(len(temp['ozone_profile'])):
                x, y = temp['altitude'].iloc[i] / 1e3, temp['ozone_profile'].iloc[i]
                # interpolate to z-grid
                func = scipy.interpolate.interp1d(x, y, fill_value = 'extrapolate')
                y_interp = func(z)
                prgrid.append(y_interp)
            prgrid = pd.DataFrame(prgrid, columns = z)
            temp['prgrid'] = prgrid
        else:
            temp['prgrid'] = pd.DataFrame()
        
    data.close()
    
    # save and compress data
    # temp.dropna(how = 'any', axis = 0, inplace = True)
    temp.reset_index(drop = True, inplace = True)
    
    if save_csv == True:
        temp.to_csv(inputfile[:inputfile.find(inputfile.split('.')[-1])] + 'csv')
    if save_pkl == True:
        temp.to_pickle(inputfile[:inputfile.find(inputfile.split('.')[-1])] + 'pickle')
    
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



def ds_all_layer(inputfile1, inputfile2, ROI, unit = 'mol', qa = 0.5):
    # inputfile1 is profile data, inputfile2 is columnar data
    data1 = readTROPOMI(inputfile1, ROI, unit = unit, qa = qa)
    data2 = readTROPOMI(inputfile2, ROI, unit = unit, qa = qa)
    altitude = data1['altitude'] / 1e3
    prgrid = data1['prgrid']
    column = data1['column']
    
    # relocated
    res = 0.25
    
    # downscaling
    data2['latitude_g'], data2['longitude_g'] = round(data2['latitude'] / res) * res, round(data2['longitude'] / res) * res

    dsprofile = pd.DataFrame(index = data2.index, columns = prgrid.columns[:10])
    for j, ilayer in enumerate(prgrid.columns[:10]):
        print(j)
        column['c0'] = prgrid.iloc[:, j].values
        data1g, (ydim, xdim) = df2grid(column, ROI, res = res)
        # rescale
        for i in range(len(data1g)):
            temp = data1g.iloc[i]
            lat, lon = temp['latitude_g'], temp['longitude_g']
            
            mask = (data2['latitude_g'] == lat) & (data2['longitude_g'] == lon)
            dsprofile.loc[mask, ilayer] = (data2[mask].ozone_total_vertical_column * temp['c0'] / temp['ozone_total_column']).values

    return dsprofile


def downscaling(data1, data2, para, ROI, res = 0.25):
    # inputfile1 is profile data, inputfile2 is columnar data
    # downscaling
    data1g, (ydim, xdim) = df2grid(data1, ROI, res = res)
    data2['latitude_g'], data2['longitude_g'] = round(data2['latitude'] / res) * res, round(data2['longitude'] / res) * res
    data2[para] = np.nan
    # rescale
    for i in range(len(data1g)):
        temp = data1g.iloc[i]
        lat, lon = temp['latitude_g'], temp['longitude_g']
        
        mask = (data2['latitude_g'] == lat) & (data2['longitude_g'] == lon)
        data2.loc[mask, para] = (data2[mask].ozone_total_vertical_column * temp[para] / temp['ozone_total_column']).values

    return data2


def surfaceOzone(inputfile1, inputfile2, ROI, unit = 'mol', qa = 0.5, res = 0.25):
    # inputfile1 is profile data, inputfile2 is columnar data
    data1 = readTROPOMI(inputfile1, ROI, unit = unit, qa = qa)
    data2 = readTROPOMI(inputfile2, ROI, unit = unit, qa = qa)
    altitude = data1['altitude'] / 1e3
    prgrid = data1['prgrid']
    column = data1['column']
    
    # relocated
    column['c0'] = prgrid.iloc[:, 0].values
    data1g, (ydim, xdim) = df2grid(column, ROI, res = res)
    data2['latitude_g'], data2['longitude_g'] = round(data2['latitude'] / res) * res, round(data2['longitude'] / res) * res
    # rescale
    data2['c0'] = np.nan
    for i in range(len(data1g)):
        temp = data1g.iloc[i]
        lat, lon = temp['latitude_g'], temp['longitude_g']
        
        mask = (data2['latitude_g'] == lat) & (data2['longitude_g'] == lon)
        data2.loc[mask, 'c0'] = (data2[mask].ozone_total_vertical_column * temp['c0'] / temp['ozone_total_column']).values
    return data2

#%%
def main():
    ROI = {'S': -90, 'N': 90, 'W': -180, 'E': 180}
    # ROI = {'S':21, 'N': 24, 'W': 111.5, 'E': 115}
    products = ['OFFL_L2__NO2____', 'OFFL_L2__SO2____', 'OFFL_L2__HCHO___','OFFL_L2__O3_____', 
                'OFFL_L2__CO_____','OFFL_L2__CH4____']
    
    product = 'OFFL_L2__HCHO___'
    path = '/home/sunji/Data/TROPOMI/%s/' % product
    filelist = glob.glob(path + '*nc*')
    
    for idate in pd.date_range('2020-01-01', '2022-01-01', freq = '1D'):
        for f in filelist[:]:
            if f.find('%s%04i%02i%02i' % (product, idate.year, idate.month, idate.day)) > 0:
                temp = readTROPOMI(f, ROI, unit = 'molecule', qa = 0, save_pkl = True)
    return temp

if __name__=='__main__':
    temp = main()
    print(temp)

# filelist = glob.glob('/home/sunji/Data/TROPOMI/OFFL_L2__O3_____/pkl/*')
# for f in filelist:
#     temp = pd.read_pickle(f)
#     print(temp)

