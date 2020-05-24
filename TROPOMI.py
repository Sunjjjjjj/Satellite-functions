#!/usr/bin/env python3 

"""
Read OMAERO data 
- rawOMAERO:   return raw data orbit by orbit, organized by list
- gridOMAERO:  return gridded data over ROI, organized by np.2darray
- daytsOMAERO: return time series 

@author: sunj
"""


######### load python packages
import sys, os
import shutil
import time
import numpy as np
import pandas as pd
import numpy.ma as ma
import matplotlib.pyplot as plt
import netCDF4 
import glob
from mpl_toolkits.basemap import Basemap
from scipy import spatial
from scipy import interpolate
import matplotlib.mlab as mlab
from scipy.interpolate import griddata
from otherFunctions import *


def readTROPOMI(year, month, day, ROI, threshold, grid, dataset):    
    # directory 
    maindir = '/nobackup/users/sunj/TROPOMI/'

    path = maindir + 'L2-%s/%4i/%02i/%02i/' % (dataset, year, month, day)
    orbit = glob.glob(path + '*AER_AI*')
    # initialize output output
    AI = pd.DataFrame()
    ALH = pd.DataFrame()
    CLD = pd.DataFrame()
    output = pd.DataFrame()
    for iorbit in sorted(orbit): 
        idx = iorbit.find('S5P')
        orbitnum = iorbit[idx + 52 : idx + 57]
        
        idx = iorbit.find('%4i%02i%02iT' % (year, month, day))
        meaTime = iorbit[idx: idx + 31]
        hour = int(iorbit[idx + 9 : idx + 11])
        dt = 14 - hour
        tz1 = timeZone(ROI['W']) - 1
        tz2 = timeZone(ROI['E']) + 2
        if dt in range(tz1, tz2):
            sys.stdout.write('\r %s # %04i-%02i-%02i %s' % (dataset, year, month, day, orbitnum))
# =============================================================================
#  L2 product: aerosol index         
# =============================================================================
            data = netCDF4.Dataset(iorbit,'r')
            # coordinates
            latAI = data.groups['PRODUCT'].variables['latitude'][0]
            lonAI = data.groups['PRODUCT'].variables['longitude'][0]
            latbd = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['latitude_bounds'][0]
            lonbd = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['longitude_bounds'][0]
            # measurement geometry 
            sza = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['solar_zenith_angle'][0]
            saa = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['solar_azimuth_angle'][0]
            vza = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['viewing_zenith_angle'][0]
            vaa = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['GEOLOCATIONS'].variables['viewing_azimuth_angle'][0] + 180.
            # time
            timeUTC = data.groups['PRODUCT'].variables['time_utc'][0]
            
            timeStamp = np.ones(latAI.shape)
            for i in range(len(timeUTC)): 
                temp = timeUTC[i][:10] + ' ' +timeUTC[i][11: 19]
                timeStamp[i, :] = time.mktime(datetime.datetime.strptime(temp, '%Y-%m-%d %H:%M:%S').timetuple())
            
            # surface
            As380 = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['scene_albedo_380'][0]
            As388 = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['scene_albedo_388'][0]
            Ps = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['INPUT_DATA'].variables['surface_pressure'][0] / 100.  # unit: hPa
            # reflectance
            R340obs = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['reflectance_measured_340'][0]
            R340sim = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['reflectance_calculated_340'][0]
            R380 = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['reflectance_measured_380'][0]
            R354obs = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['reflectance_measured_354'][0]
            R354sim = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['reflectance_calculated_354'][0]
            R388 = data.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['DETAILED_RESULTS'].variables['reflectance_measured_388'][0]
            # aerosol index
            AI380 = data.groups['PRODUCT'].variables['aerosol_index_340_380'][0]
            AI388 = data.groups['PRODUCT'].variables['aerosol_index_354_388'][0]
        #    AI340_380_prec = data.groups['PRODUCT'].variables['aerosol_index_340_380_precision'][0]
        #    AI354_388_prec = data.groups['PRODUCT'].variables['aerosol_index_354_388_precision'][0]
        #    QAfalg = data.groups['PRODUCT'].variables['qa_value'][0]
            data.close()
            setAI = {'lat': latAI, 'lon': lonAI, 'timeStamp': timeStamp,
                     'latb1': latbd[:, :, 0], 'latb2': latbd[:, :, 1], 'latb3': latbd[:, :, 2], 'latb4': latbd[:, :, 3], 
                     'lonb1': lonbd[:, :, 0], 'lonb2': lonbd[:, :, 1], 'lonb3': lonbd[:, :, 2], 'lonb4': lonbd[:, :, 3],
                     'sza': sza, 'saa': saa, 'vza': vza,'vaa': vaa, 'Ps': Ps, 
                     'As380': As380, 'R340obs': R340obs, 'R340sim': R340sim, 'R380': R380, 'AI380': AI380, 
                     'As388': As388, 'R354obs': R354obs, 'R354sim': R354sim, 'R388': R388, 'AI388': AI388}
        
            del latAI, lonAI, latbd, lonbd, sza, saa, vza, vaa, As380, As388, R340obs, R340sim, R380, R354obs, R354sim, R388, AI380, AI388
# =============================================================================
# L2 product: aerosol layer height  
# =============================================================================
            filelist = glob.glob(path + '*AER_LH*')
            setALH = {}
            for ifile in filelist: 
                if (ifile.find(meaTime) > 0):
                    data = netCDF4.Dataset(ifile,'r')
                    # coordinate
                    latLH = data.groups['PRODUCT'].variables['latitude'][0]
                    lonLH = data.groups['PRODUCT'].variables['longitude'][0]
                    # aerosol layer height in pressure
                    ALP = data.groups['PRODUCT'].variables['aerosol_mid_pressure'][0] / 100.  # unit: hPa
        #            ALHB = dataB.groups['PRODUCT'].variables['aerosol_mid_pressure'][0] / 100.  # unit: hPa
                    # aerosol layer height in km, check the reference surface height (is relative to the surface, geoid, etc.)
                #    Zs = dataA.groups['PRODUCT'].groups['SUPPORT_DATA'].groups['INPUT_DATA'].variables['surface_altitude'][0] / 1000.  # unit: km
                    alh = data.groups['PRODUCT'].variables['aerosol_mid_height'][0] / 1000.  # unit: km
                #    aLHB = dataB.groups['PRODUCT'].variables['aerosol_mid_height'][0] / 1000.  # unit: km
                #    ALHA += Zs
                #    aLHB += Zs
                    data.close()
        #            dataB.close()
                    setALH = {'lat': latLH, 'lon': lonLH, 'ALH': alh, 'ALP': ALP}
                    del latLH, lonLH, alh, ALP
            
# =============================================================================
# L2 product: cloud fraction and albedo 
# =============================================================================
            filelist = glob.glob(path + '*FRESCO*')
            setCLD = {} 
            for ifile in filelist:
                if (ifile.find(meaTime) > 0):
                    data = netCDF4.Dataset(ifile,'r')
                    # coordinate
                    latCLD = data.groups['PRODUCT'].variables['latitude'][0]
                    lonCLD = data.groups['PRODUCT'].variables['longitude'][0]
                    # aerosol layer height in pressure
                    CF = data.groups['PRODUCT'].variables['cloud_fraction_crb'][0]
                    CA = data.groups['PRODUCT'].variables['cloud_albedo_crb'][0]
                    data.close()
                    setCLD = {'lat': latCLD, 'lon': lonCLD, 'CF': CF, 'CA': CA}
                    del latCLD, lonCLD, CF, CA
# =============================================================================
# L1 product: radiance
# =============================================================================
    #        path = maindir + 'L1RA/%4i/%02i/%02i/' % (year, month, day)
    #        filelist = glob.glob( path + '*.nc')
    #        for ifile in filelist: 
    #            if ifile.find(orbitnum) > 0:
    #                data = netCDF4.Dataset(ifile,'r')
    #            else:
    #                print('L1 radiance of orbit %s does not exist!' % orbitnum)
    #        # coordinates
    #        latRA = data.groups['BAND3_RADIANCE'].groups['STANDARD_MODE'].groups['GEODATA'].variables['latitude'][0]
    #        lonRA = data.groups['BAND3_RADIANCE'].groups['STANDARD_MODE'].groups['GEODATA'].variables['longitude'][0]
    #        # solar zenith angle
    #        sza = data.groups['BAND3_RADIANCE'].groups['STANDARD_MODE'].groups['GEODATA'].variables['solar_zenith_angle'][0]
    #        # wavelength 
    #        wvls = data.groups['BAND3_RADIANCE'].groups['STANDARD_MODE'].groups['INSTRUMENT'].variables['nominal_wavelength'][0]
    #        # radiance  
    #        RA = np.ones([4, latRA.shape[0], latRA.shape[1]]) * np.nan
    #        for iw, ww in enumerate([340, 380, 354, 388]):
    #            idx = np.argmin(abs(wvls - ww), axis = 1)
    #            radiance = data.groups['BAND3_RADIANCE'].groups['STANDARD_MODE'].groups['OBSERVATIONS'].variables['radiance'][0][:, :, idx.min(): idx.max() + 1]
    #            for icol, iwvl in enumerate(idx - idx.min()): 
    #    #            temp.append(wvls[icol, iwvl + idx.min()])
    #                RA[iw, :, icol] = radiance[:, icol, iwvl]
    #        data.close()
# =============================================================================
# L1 product: irradiance
# =============================================================================
    #        path = maindir + 'L1IR/%4i/%02i/%02i/' % (year, month, day)
    #        filelist = glob.glob( path + '*.nc')
    #        for ifile in filelist: 
    #            if ifile.find(orbitnum) > 0:
    #                data = netCDF4.Dataset(ifile,'r')
    #            else:
    #                print('L1 irradiance of orbit %s does not exist!' % orbitnum)
    #                data = netCDF4.Dataset(path + 'S5P_OFFL_L1B_IR_UVN_20180409T202941_20180409T221112_02532_01_001400_20180410T004434.nc','r')
    #        print('Temporarily use data of other orbits!')
    #        data = netCDF4.Dataset(path + 'S5P_OFFL_L1B_IR_UVN_20180409T202941_20180409T221112_02532_01_001400_20180410T004434.nc', 'r')
    #        # wavelength
    #        wvls = data.groups['BAND3_IRRADIANCE'].groups['STANDARD_MODE'].groups['INSTRUMENT'].variables['calibrated_wavelength'][0]
    #        # irradiance 
    #        irr = data.groups['BAND3_IRRADIANCE'].groups['STANDARD_MODE'].groups['OBSERVATIONS'].variables['irradiance'][0][0]
    #    #    # irradiance
    #        IR = np.ones([4, latRA.shape[0], latRA.shape[1]]) * np.nan
    #        for iw, ww in enumerate([340, 380, 354, 388]):
    #            idx = np.argmin(abs(wvls - ww), axis = 1)
    #            temp = []
    #            for icol, iwvl in enumerate(idx): 
    #                temp.append(wvls[icol, iwvl])
    #                IR[iw, :, icol] = irr[icol, iwvl]
    #        data.close()
    #        
    #        setL1 = {'lat': latRA, 'lon': lonRA}
    #        for iw, ww in enumerate([340, 380, 354, 388]): 
    #            setL1['RA%s' %ww] = RA[iw, :, :]
    #            setL1['IR%s' %ww] = IR[iw, :, :]
    #        del latRA, lonRA, sza, wvls, irr, RA, IR
    
            def ROImask(dataset, ROI): 
                return (dataset['lat'] >= ROI['S']) & (dataset['lat'] <= ROI['N']) & (dataset['lon'] >= ROI['W']) & (dataset['lon'] <= ROI['E']) 
            
            if len(setAI) > 0:
                for ikey in setAI.keys():
                    setAI[ikey] = setAI[ikey].reshape(-1)
                setAI = pd.DataFrame.from_dict(setAI)
                setAI['orbit'] = int(orbitnum) * np.ones(len(setAI))
    
                for ikey in setALH.keys():
                    setALH[ikey] = setALH[ikey].reshape(-1)
                setALH = pd.DataFrame.from_dict(setALH)
    
                for ikey in setCLD.keys():
                    setCLD[ikey] = setCLD[ikey].reshape(-1)
                setCLD = pd.DataFrame.from_dict(setCLD)
                
                
                mask = ROImask(setAI, ROI)
                AI = AI.append(setAI[mask])
                try: 
                    mask = ROImask(setALH, ROI)
                    ALH = ALH.append(setALH[mask])
                except:
                    pass
                mask = ROImask(setCLD, ROI)
                CLD = CLD.append(setCLD[mask])
    try: 
        theta = Theta(AI['sza'], AI['saa'], AI['vza'], AI['vaa'])
        mask =  (abs(AI['sza']) <= 75) & (theta >= 30) & \
                    (AI['As380'] <= 0.3) & (AI['As380'] >= 0) & \
                    (np.isnan(AI['Ps']) == False) & \
                    (AI['AI380'] >= threshold) 
        AI['mask'] = mask

# =============================================================================
# grid data
# =============================================================================
        if not grid:
            output = AI.copy()
            output = output.reset_index()
    #            print('Use orginal TROPOMI L2 AI coordinate')
    #        for ikey in AI.keys():
    #            setAI[ikey] = AI[ikey].reshape(-1)
            try: 
                keys = list(ALH.keys())
                keys.remove('lat')
                keys.remove('lon')
                for ikey in keys:
                    output[ikey] = griddata((ALH['lon'].values.reshape(-1), ALH['lat'].values.reshape(-1)), ALH[ikey].values.reshape(-1), (AI['lon'].values.reshape(-1), AI['lat'].values.reshape(-1)), method = 'nearest')    
            except:
                pass
            try:
                keys = list(CLD.keys())
                keys.remove('lat')
                keys.remove('lon')
                for ikey in keys:
                    output[ikey] = griddata((CLD['lon'].values.reshape(-1), CLD['lat'].values.reshape(-1)), CLD[ikey].values.reshape(-1), (AI['lon'].values.reshape(-1), AI['lat'].values.reshape(-1)), method = 'nearest')    
        #            for ikey in list(setL1.keys())[2:]:
        #                setL1[ikey] = griddata((setL1['lon'].reshape(-1), setL1['lat'].reshape(-1)), setL1[ikey].reshape(-1), (setAI['lon'].reshape(-1), setAI['lat'].reshape(-1)), method = 'nearest')    
            except:
                pass
        else: 
    #            print('Grid TROPOMI data onto input coordinates')
            keys = list(AI.keys())
            keys.remove('lat')
            keys.remove('lon')
            for ikey in AI:
#                sys.stdout.write('\r interpolating %s' % (ikey))
                output[ikey] = griddata((AI['lon'].values.reshape(-1), AI['lat'].values.reshape(-1)), AI[ikey].values.reshape(-1), grid, method = 'nearest')    
                output['lat'] = grid[1]
                output['lon'] = grid[0]
            try: 
                keys = list(ALH.keys())
                keys.remove('lat')
                keys.remove('lon')
    
                for ikey in keys:
#                    sys.stdout.write('\r interpolating %s' % (ikey))
                    output[ikey] = griddata((ALH['lon'].values.reshape(-1), ALH['lat'].values.reshape(-1)), ALH[ikey].values.reshape(-1), grid, method = 'nearest')    
            except:
                pass
            
            try:
                keys = list(CLD.keys())
                keys.remove('lat')
                keys.remove('lon')
                for ikey in keys:
                    output[ikey] = griddata((CLD['lon'].values.reshape(-1), CLD['lat'].values.reshape(-1)), CLD[ikey].values.reshape(-1), grid, method = 'nearest')    
            except:
                pass
#            for ikey in list(setL1.keys())[2:]:
#                setL1[ikey] = griddata((setL1['lon'].reshape(-1), setL1['lat'].reshape(-1)), setL1[ikey].reshape(-1), grid, method = 'nearest')    
# =============================================================================
# combine all date into one data set    
# =============================================================================
        #    del setCLD['lat'], setCLD['lon']
        #    try: 
        #        del setALH['lat'], setALH['lon']
        #    except:
        #        pass
        #    setAI = pd.DataFrame.from_dict(setAI)
        #    setALH = pd.DataFrame.from_dict(setALH)
        #    setCLD = pd.DataFrame.from_dict(setCLD)
        #        setL1 = pd.DataFrame.from_dict(setL1)
        #    idata = pd.concat([setAI, setALH, setCLD], axis = 1)
        #    del setAI, setALH, setCLD
        #    idata['orbit'] = int(orbitnum) * np.ones(len(idata))
# =============================================================================
# select qualified pixels    
# =============================================================================
        theta = Theta(output['sza'], output['saa'], output['vza'], output['vaa'])
        criteria = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E']) & \
                    (abs(output['sza']) <= 75) & (theta >= 30) & \
                    (output['As380'] <= 0.3) & (output['As380'] >= 0) & \
                    (np.isnan(output['Ps']) == False) & \
                    (output['AI380'] >= threshold) & (output['mask'] == True)
        if 'ALP' in output.keys():
            criteria = criteria & (np.isnan(output['ALP']) == False)
        if 'CF' in output.keys():
            criteria = criteria & (output['CF'] <= 0.3) & (output['CF'] >= 0)
        
        
# =============================================================================
# combine orbit together
# =============================================================================
        output = output[criteria]
        
        def timeStamp2dateTime(ts):
            return datetime.datetime.fromtimestamp(ts)
        
        dateTime = list(map(timeStamp2dateTime, output['timeStamp']))
        output['dateTime'] = dateTime

    except:
        pass
    return output.reset_index(drop = True)



# =============================================================================
# case information
# =============================================================================
#caseName = 'BC201808'
## case directory
#
## region of interest
#ROI = {'S':45, 'N': 60, 'W': -120, 'E': -95}
## time period
## AAI threshold
#threshold = 4
## grid resolution
#res = 0.25
#lat= np.arange(ROI['N'], ROI['S'], -res)
#lon= np.arange(ROI['W'], ROI['E'], res)
#XX, YY = np.meshgrid(lon, lat)
#grid = (XX.reshape(-1), YY.reshape(-1))
#grid = False
## wavelength 
#wvlPair = (340, 380)
#
## TROPOMI
#TPM = readTROPOMI(2018, 8, 15, ROI, threshold, grid, 'NRTI')


def main(): 
    # case name
    caseName = 'CA201712'
    # test name 
    # case directory
    casedir = '/nobackup/users/sunj/data_saved_for_cases/%s' %(caseName)
    if not os.path.isdir(casedir):
        os.makedirs(casedir)   
    
    # region of interest
    ROI = {'S':25, 'N': 50, 'W': -130, 'E': -110}
    # time period
    startdate = {'year':2017, 'month': 12, 'day': 12}
    enddate   = {'year':2017, 'month': 12, 'day': 12}
    dates = dateList(startdate, enddate)
    
    # AAI threshold
    threshold = 1.
    # grid resolution
    res = 0.1
    lat= np.arange(ROI['N'], ROI['S'], -res)
    lon= np.arange(ROI['W'], ROI['E'], res)
    XX, YY = np.meshgrid(lon, lat)
    grid = (XX.reshape(-1), YY.reshape(-1))
    
    t1 = time.time()
    for idate in dates: 
        yy, mm, dd = DOY2date(idate[0], idate[1])
        data = readTROPOMI(yy, mm, dd, ROI, threshold, False, 'RPRO')
#        outputname = 'dataTROPOMI_%4i_%02i_%02i' % (yy, mm, dd)
#        data.to_pickle(outputname)
#        if os.path.isfile(casedir + '/' + outputname):
#            os.remove(casedir + '/' + outputname)                                
#        shutil.move(outputname, casedir + '/')  
#        outputname = 'mask_%4i_%02i_%02i' % (yy, mm, dd)
#        mask.to_pickle(outputname)
#        if os.path.isfile(casedir + '/' + outputname):
#            os.remove(casedir + '/' + outputname)                                
#        shutil.move(outputname, casedir + '/') 
        
        fig = plt.figure(figsize=(5,5))
        cbax = fig.add_axes([0.825,0.125,0.05,0.75])  
        fig.add_axes([0.0,0.1,0.7,0.8])
        map = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], lat_0 = 0, lon_0 = 0, projection='cyl',resolution='l')
        map.drawcoastlines(color='black',linewidth=0.6)
        map.drawcountries(color = 'black',linewidth=0.4)
        map.drawmeridians(np.arange(-150,-100,5),labels = [1,0,1,0])
        map.drawparallels(np.arange(0,60,5), labels = [1,0,1,0])
        plt.legend()
        cb = plt.scatter(data['lon'], data['lat'], c = data['AI388'], s = 4, cmap='rainbow')  
        cbar = plt.colorbar(cb, extend = 'both', cax = cbax)
#        cbar.set_label(rotation = 270, labelpad = -50)
        
    t2 = time.time()
    print('Time for preparing satellite data: %1.2f s' % (t2 - t1))
    return data
    
if __name__ == '__main__':
    data = main()