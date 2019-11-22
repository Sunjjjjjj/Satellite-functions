#!/usr/bin/env python3 

"""
Download OMI by wget 

@author: sunj
"""

import sys 
sys.path.insert(0, '/usr/people/sunj/Documents/pyvenv/Projects/General_scripts')
import os
import shutil
import glob
import time
import numpy as np
import netCDF4
import pickle 
from scipy.interpolate import interp1d
#from date_conversion import date2DOY, DOY2date, dateList\
from otherFunctions import *
from netCDF4 import Dataset

#==============================================================================
# Initialization:
#==============================================================================
startdate = {'year': 2018, 'month': 8, 'day': 13}
enddate   = {'year': 2019, 'month': 10, 'day': 31}

'''
Choose one from: 
- M2C0NXASM: time-invariant model constant parameters
- M2C0NXCTM: monthly model constant parameters
- M2I3NVASM: altitude
- M2T1NXFLX: boundary layer height

- M2TMNXAER: 2d monthly AOD 
- M2I3NVGAS: 3d mass mixing ratio analysis increments
- M2I3NVAER: 3d mass minxing ratio
- M2I3NXGAS: 2d AOD

'''
dataset = 'M2T1NXAER'

# Get time series [year, DOY]:
dateid = dateList(startdate, enddate)
        
start=time.time() 
#==============================================================================
# MERRA-2 tavgM_2d_aer_Nx: 2d,Monthly mean,Time-averaged,Single-Level,Assimilation,Aerosol Diagnostics
#==============================================================================
if dataset == 'M2TMNXAER': 
    for iyear in range(startdate['year'], enddate['year'] + 1):
        directory = '/nobackup_1/users/sunj/MERRA2/%s/%4i/' % (dataset, iyear)
        url = "https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2TMNXAER.5.12.4/%4i/" % (iyear)
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s' % url) 
        # (Tutorial source: https://disc.gsfc.nasa.gov/information/howto?title=How%20to%20Download%20Data%20Files%20from%20HTTP%20Service%20with%20wget)
        
        filelist = glob.glob('*.nc4')
        
        if not os.path.exists(directory):
            os.makedirs(directory)   
        
        for f in filelist:  
            if os.path.isfile(directory + f):
                os.remove(directory + f)
            shutil.move(f, directory)

# =============================================================================
# MERRA-2 inst3_3d_gas_Nv (5 aerosol types): mass mixing ratio aerosol analysis increments
# =============================================================================
if dataset == 'M2I3NVGAS': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s/%4i/' % (dataset, iyear)
        url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVGAS.5.12.4/%4i/%02i/MERRA2_400.inst3_3d_gas_Nv.%4i%02i%02i.nc4' % (iyear, imonth, iyear, imonth, iday)
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url, directory)) 
        
        filelist = glob.glob('*.nc4')
        if not os.path.exists(directory):
            os.makedirs(directory)   
            
        for f in filelist:  
            if os.path.isfile(directory + f):
                os.remove(directory + f)
            shutil.move(f, directory)

# =============================================================================
#  MERRA-2 inst3_3d_aer_Nv (15 aerosol subtypes): mass mixing ratio 
# =============================================================================
if dataset == 'M2I3NVAER': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s/%4i/' % (dataset, iyear)
        filename = 'MERRA2_300.inst3_3d_aer_Nv.%4i%02i%02i.nc4' % (iyear, imonth, iday)
        url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVAER.5.12.4/%4i/%02i/%s' % (iyear, imonth, filename)
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url, directory)) 
        
        if not os.path.exists(directory):
            os.makedirs(directory)   

        data = {}   
        parameters = ['lat', 'lon'] 
        
        raw_data = netCDF4.Dataset(directory + filename)
        for ipara in parameters:
            data[ipara] = raw_data[ipara][:] 

        with open('/nobackup_1/users/sunj/MERRA2/MERRA2_beta_ext.pickle', 'rb') as handle:
            beta_ext = pickle.load(handle)
        
        aerosol_types = ['DU001', 'DU002', 'DU003', 'DU004', 'DU005', 
                         'SS001', 'SS002', 'SS003', 'SS004', 'SS005', 
                         'BCPHILIC', 'BCPHOBIC', 'OCPHILIC', 'OCPHOBIC', 'SO4']
        # mass concentration [kg/m3] = mass mixing ratio [kg/kg] * air density [kg/m3]
        # extinction [/m] = mass concentration [kg/m3] * mass extinction coefficient [m2/kg]
        
        # save for all aerosol types together
        data['totEXT'] = 0
#        data['totMass'] = 0
#        data['totRho'] = 0
        for iaer in aerosol_types:
            print(iaer)
            x, y = beta_ext.keys().values, beta_ext.loc[iaer].values
            f = interp1d(x, y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            beta_ext_RH = f(np.array(raw_data['RH'][4:6, :, :, :]))
            data['totEXT'] += raw_data['AIRDENS'][4:6, :, :, :] * raw_data[iaer][4:6, :, :, :] * beta_ext_RH
#            data['totMass'] =+ raw_data[iaer][4:6, :, :, :]
#            data['totRho'] = raw_data['AIRDENS'][4:6, :, :, :] * data['totMass'] 
        # save for each aerosol type seperately
        data['DU'] = 0
        for iaer in ['DU001', 'DU002', 'DU003', 'DU004', 'DU005']:
            print(iaer)
            x, y = beta_ext.keys().values, beta_ext.loc[iaer].values
            f = interp1d(x, y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            beta_ext_RH = f(np.array(raw_data['RH'][4:6, :, :, :]))
            data['DU'] += raw_data['AIRDENS'][4:6, :, :, :] * raw_data[iaer][:][4:6, :, :, :] * beta_ext_RH
            
        data['SS'] = 0
        for iaer in ['SS001', 'SS002', 'SS003', 'SS004', 'SS005']:
            print(iaer)
            x, y = beta_ext.keys().values, beta_ext.loc[iaer].values
            f = interp1d(x, y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            beta_ext_RH = f(np.array(raw_data['RH'][4:6, :, :, :]))
            data['SS'] += raw_data['AIRDENS'][4:6, :, :, :] * raw_data[iaer][:][4:6, :, :, :] * beta_ext_RH

        data['BB'] = 0
        for iaer in ['BCPHILIC', 'BCPHOBIC', 'OCPHILIC', 'OCPHOBIC']:
            print(iaer)
            x, y = beta_ext.keys().values, beta_ext.loc[iaer].values
            f = interp1d(x, y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            beta_ext_RH = f(np.array(raw_data['RH'][4:6, :, :, :]))
            data['BB'] += raw_data['AIRDENS'][4:6, :, :, :] * raw_data[iaer][:][4:6, :, :, :] * beta_ext_RH

        data['SU'] = 0
        for iaer in ['SO4']:
            print(iaer)
            x, y = beta_ext.keys().values, beta_ext.loc[iaer].values
            f = interp1d(x, y, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
            beta_ext_RH = f(np.array(raw_data['RH'][4:6, :, :, :]))
            data['SU'] += raw_data['AIRDENS'][4:6, :, :, :] * raw_data[iaer][:][4:6, :, :, :] * beta_ext_RH
 
#        raw_data.close()
        with open('%s.pickle' % (directory + filename[:-3]), 'wb') as handle:
            pickle.dump(data, handle)
        os.remove(directory + filename)

# =============================================================================
# MERRA-2 inst3_2d_gas_Nx
# =============================================================================
if dataset == 'M2I3NXGAS': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s/%4i/' % (dataset, iyear)

        if not os.path.exists(directory):
            os.makedirs(directory)   
        url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NXGAS.5.12.4/%4i/%02i/MERRA2_400.inst3_2d_gas_Nx.%4i%02i%02i.nc4' % (iyear, imonth, iyear, imonth, iday)
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url, directory)) 
        
        filelist = glob.glob('*.nc4')
            
        for f in filelist:  
            if os.path.isfile(directory + f):
                os.remove(directory + f)
            shutil.move(f, directory)

# =============================================================================
# MERRA-2 inst3_3d_asm_Nv: altitude
# =============================================================================
if dataset == 'M2I3NVASM': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s/%4i/' % (dataset, iyear)
        url = 'https://goldsmr5.gesdisc.eosdis.nasa.gov/data/MERRA2/M2I3NVASM.5.12.4/%4i/%02i/MERRA2_400.inst3_3d_asm_Nv.%4i%02i%02i.nc4' % (iyear, imonth, iyear, imonth, iday)

        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url, directory)) 
        
        
        if not os.path.exists(directory):
            os.makedirs(directory)   
        raw_data = netCDF4.Dataset(directory + 'MERRA2_400.inst3_3d_asm_Nv.%4i%02i%02i.nc4' % (iyear, imonth, iday))
        H = raw_data['H'][:]
        H[H.mask == True] = np.nan
        data = H.data[4:6, :, :, :]
        
        with open('%s.pickle' % (directory + 'MERRA2_400.inst3_3d_asm_Nv.%4i%02i%02i' % (iyear, imonth, iday)), 'wb') as handle:
            pickle.dump(data, handle)
        os.remove(directory + 'MERRA2_400.inst3_3d_asm_Nv.%4i%02i%02i.nc4' % (iyear, imonth, iday))

# =============================================================================
# MERRA-2 inst3_3d_asm_Nv: surface height
# =============================================================================
if dataset == 'M2C0NXASM': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s/' % (dataset)

        if not os.path.exists(directory):
            os.makedirs(directory)   

        url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXASM.5.12.4/1980/MERRA2_101.const_2d_asm_Nx.00000000.nc4'
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url, directory)) 
        
        filelist = glob.glob('*.nc4')
        for f in filelist:  
            if os.path.isfile(directory + f):
                os.remove(directory + f)
            shutil.move(f, directory)

# =============================================================================
# MERRA-2 const_2d_ctm_Nx
# =============================================================================
if dataset == 'M2C0NXCTM': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s/' % (dataset)

        if not os.path.exists(directory):
            os.makedirs(directory)   

        url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2_MONTHLY/M2C0NXCTM.5.12.4/1980/MERRA2_101.const_2d_ctm_Nx.00000000.nc4'
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url, directory)) 
        
        filelist = glob.glob('*.nc4')
        for f in filelist:  
            if os.path.isfile(directory + f):
                os.remove(directory + f)
            shutil.move(f, directory)



# =============================================================================
# MERRA-2 const_2d_ctm_Nx
# =============================================================================
if dataset == 'M2T1NXFLX': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s/' % (dataset)

        if not os.path.exists(directory):
            os.makedirs(directory) 
            
        filelist = pd.read_csv(filename, sep=",", header=None)
        filelist = filelist.sort_values(by = 0).reset_index()
        filelist = list(filelist[0].values)
            
        url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXFLX.5.12.4/%4i/%02i/MERRA2_400.tavg1_2d_flx_Nx.%4i%02i%02i.nc4' % (iyear, imonth, iyear, imonth, iday)
        os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url, directory)) 
        
        raw_data = netCDF4.Dataset(directory + 'MERRA2_400.tavg1_2d_flx_Nx.%4i%02i%02i.nc4' % (iyear, imonth, iday))
        hour = raw_data['time'][:].data
        idx = np.where((hour == 720) | (hour == 900))[0]
        BLH = raw_data['PBLH'][idx, :, :] / 1e3
        
        with open('%s.pickle' % (directory + 'MERRA2_400.tavg1_2d_flx_Nx.%4i%02i%02i' % (iyear, imonth, iday)), 'wb') as handle:
            pickle.dump(data, handle)
        os.remove(directory + 'MERRA2_400.tavg1_2d_flx_Nx.%4i%02i%02i.nc4' % (iyear, imonth, iday))



# =============================================================================
# MERRA-2 inst3_2d_gas_Nx
# =============================================================================
if dataset == 'M2T1NXAER': 
    for idate in dateid:
        iyear, imonth, iday = DOY2date(idate[0], idate[1])
        directory = '/nobackup_1/users/sunj/MERRA2/%s_morning/%4i/' % (dataset, iyear)

        if not os.path.exists(directory):
            os.makedirs(directory)  

        url = 'https://goldsmr4.gesdisc.eosdis.nasa.gov/data/MERRA2/M2T1NXAER.5.12.4/%4i/%02i/' % (iyear, imonth)

        if iyear < 2011:
            filename = 'MERRA2_300.tavg1_2d_aer_Nx.%4i%02i%02i.nc4' % (iyear, imonth, iday)
            os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url + filename, directory)) 
        else:
            filename = 'MERRA2_400.tavg1_2d_aer_Nx.%4i%02i%02i.nc4' % (iyear, imonth, iday)
            os.system('wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -c -nH -nd -np -A nc4 %s -P %s' % ( url + filename, directory)) 
            
            
        data = netCDF4.Dataset(directory + filename)
        lat = data['lat'][:]
        lon = data['lon'][:]
        AOD = np.nanmean(data['TOTEXTTAU'][7: 12, :, :], axis = 0)
        AOD_sca = np.nanmean(data['TOTSCATAU'][7: 12, :, :], axis = 0)
        data.close()
        
        output = Dataset(directory + 'MERRA2_100.tavg1_2d_aer_Nx_sub.%4i%02i%02i.nc4' % (iyear, imonth, iday), mode = 'w')
        output.createDimension('Y', len(lat))
        output.createDimension('X', len(lon))
        
        output.createVariable('lat', 'f',  ('Y'))
        output.createVariable('lon', 'f',  ('X'))
        output.createVariable('TOTEXTTAU', 'f',  ('Y', 'X'))
        output.createVariable('TOTSCATAU', 'f',  ('Y', 'X'))
        
        output.variables['lat'][:] = lat
        output.variables['lon'][:] = lon
        output.variables['TOTEXTTAU'][:] = AOD
        output.variables['TOTSCATAU'][:] = AOD_sca

        output.close()
        os.remove(directory + filename)

end = time.time() 
print('Time of downloading:',end - start,'s')


