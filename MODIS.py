# -*- coding: utf-8 -*-
"""
Read OMAERO data 
- rawOMAERO: return orbit data, organized by list
- gridMODIS: return gridded data over ROI, organized by np.2darray
- daytsMODIS: return time series 


@author: sunj
"""


######### load python packages
import sys, os
import shutil
import time
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import netCDF4 
import glob
from scipy import ndimage
from mpl_toolkits.basemap import Basemap
import pandas as pd
from pandas import Series, DataFrame, Panel
from scipy import spatial
from datetime import datetime, timedelta
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
from otherFunctions import *



def readMODIS(year, month, day, ROI, data): 
    
    print('**** Reading MODIS %02i-%02i-%4i' % (day, month, year))
    moddir = '/nobackup_1/users/sunj/MODIS/AQUA/MYD04_L2/%4i/%02i/%02i/' % (year, month, day)

    aodmod = []
    latmod = []
    lonmod = [] 
    AEmod = []
    CFmod = []
    orbit = []
    
    filelist = glob.glob( moddir + '*.hdf')
    for io in filelist[:]:  
#        orbitnum = int(io[io.find('MYD04_L2') + 18 : io.find('MYD04_L2') + 22])
        
        data = netCDF4.Dataset(io,'r')   

        lat = data.variables['Latitude'][:]
        lon = data.variables['Longitude'][:]
#        aod = data.variables['Optical_Depth_Land_And_Ocean'][:]
#        aod = data.variables['AOD_550_Dark_Target_Deep_Blue_Combined'][:]  # not used because lose highest data
#        aodOcean = data.variables['Effective_Optical_Depth_Best_Ocean'][1, :, :]   # 3K
#        aodOcean = data.variables['Effective_Optical_Depth_0p55um_Ocean'][:, :]     # CA
        aodLand = data.variables['Deep_Blue_Aerosol_Optical_Depth_550_Land'][:, :]  # CA
        aodOcean = data.variables['Effective_Optical_Depth_Best_Ocean'][:, :, :]     
#        aodLand = data.variables['Deep_Blue_Spectral_Aerosol_Optical_Depth_Land'][:, :, :]
        AEland = data.variables['Deep_Blue_Angstrom_Exponent_Land'][:]

#        idx1, idx2 = np.where(aodOcean[0] > 0)
#        idx3, idx4 = np.where(aodLand[0] > 0)
        
        aod550 = aodOcean[1, :, :].copy()
        mask = (aodLand > 0)
#        aod[aodOcean > 0] = aodOcean
        aod550[mask] = aodLand[mask]
         
        AE = Angstorm(470, aodOcean[0,:,:], 550, aod550)
        AE[mask] = AEland[mask]
#        aod500 = wvldepAOD(550, aod[1,:,:], 500, AE)
        CF = data.variables['Aerosol_Cloud_Fraction_Ocean'][:]
        CF[mask] = data.variables['Aerosol_Cloud_Fraction_Land'][:][mask]
        

        sza = data.variables['Solar_Zenith'][:]
        vza = data.variables['Sensor_Zenith'][:]
        saa = data.variables['Solar_Azimuth'][:]
        vaa = data.variables['Sensor_Azimuth'][:]
        sca = data.variables['Scattering_Angle'][:]
        
        raa = saa + 180 - vaa 
        
        scacal = np.arccos(-np.cos(sza/180.*np.pi)*np.cos(vza/180.*np.pi)+np.sin(sza/180.*np.pi)*np.sin(vza/180.*np.pi)*np.cos(raa/180.*np.pi)) / np.pi * 180
#        print 'scacal - sca :', scacal - sca
      
        
        latmod = latmod + list(np.asarray(lat).reshape(-1))
        lonmod = lonmod + list(np.asarray(lon).reshape(-1))        
        aodmod = aodmod + list(np.asarray(aod550).reshape(-1))
        AEmod = AEmod + list(np.asarray(AE).reshape(-1))
        CFmod = CFmod + list(np.asarray(CF).reshape(-1))
        
        data.close() 
    
    
    latarr = np.array(latmod)
    lonarr = np.array(lonmod)   
    aodarr = np.array(aodmod) 
    AEarr = np.array(AEmod)
    CFarr = np.array(CFmod)
#    orbitarr = np.array(orbit)
    
    output = pd.DataFrame() 
    output['lat'] = latarr
    output['lon'] = lonarr
    output['AOD'] = aodarr
    output['AE'] = AEarr
    output['CF'] = CFarr
#    output['orbit'] = orbitarr
    
    criteria = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E']) & \
                (output['AOD'] >= 0) & (output['AOD'] <= 1e2) 
    output['AE'][output['AE'] < 0] = np.nan
    output = output[criteria]
    return output.reset_index(drop = True)


def readMODIS3K(year, month, day, ROI): 
    
    print('**** Reading MODIS %02i-%02i-%4i' % (day, month, year))
    moddir = '/nobackup/users/sunj/MODIS/AQUA/MYD04_3K/%4i/%02i/%02i/' % (year, month, day)

    aodmod = []
    latmod = []
    lonmod = [] 
    orbit = []
    
    filelist = glob.glob( moddir + '*.hdf')
    for io in filelist[:]:  
#        orbitnum = int(io[io.find('MYD04_L2') + 18 : io.find('MYD04_L2') + 22])
        
        data = netCDF4.Dataset(io,'r')   

        lat = data.variables['Latitude'][:]
        lon = data.variables['Longitude'][:]
        aod = data.variables['Optical_Depth_Land_And_Ocean'][:]
#        aod = data.variables['AOD_550_Dark_Target_Deep_Blue_Combined'][:]  # not used because lose highest data
#        aodOcean = data.variables['Effective_Optical_Depth_Best_Ocean'][1, :, :]   # 3K
#        aodOcean = data.variables['Effective_Optical_Depth_0p55um_Ocean'][:, :]     # CA
#        aodLand = data.variables['Deep_Blue_Aerosol_Optical_Depth_550_Land'][:, :]  # CA
#        aodOcean = data.variables['Effective_Optical_Depth_Best_Ocean'][:, :, :]     
#        aodLand = data.variables['Deep_Blue_Spectral_Aerosol_Optical_Depth_Land'][:, :, :]

#        idx1, idx2 = np.where(aodOcean > 0)
#        idx3, idx4 = np.where(aodLand > 0)
##        
#        aod = np.ones(aodOcean.shape) * (-1e4)
#        aod[:, idx1, idx2] = aodOcean[:, idx1, idx2]
#        aod[:, idx3, idx4] = aodLand[:, idx3, idx4]
        
#        AE = Angstorm(470, aod[0,:,:], 550, aod[1,:,:])
#        aod500 = wvldepAOD(550, aod[1,:,:], 500, AE)
        

        sza = data.variables['Solar_Zenith'][:]
        vza = data.variables['Sensor_Zenith'][:]
        saa = data.variables['Solar_Azimuth'][:]
        vaa = data.variables['Sensor_Azimuth'][:]
        sca = data.variables['Scattering_Angle'][:]
        
        raa = saa + 180 - vaa 
        
        scacal = np.arccos(-np.cos(sza/180.*np.pi)*np.cos(vza/180.*np.pi)+np.sin(sza/180.*np.pi)*np.sin(vza/180.*np.pi)*np.cos(raa/180.*np.pi)) / np.pi * 180
#        print 'scacal - sca :', scacal - sca
      
        
        latmod = latmod + list(np.asarray(lat).reshape(-1))
        lonmod = lonmod + list(np.asarray(lon).reshape(-1))        
        aodmod = aodmod + list(np.asarray(aod).reshape(-1))
#        orbit =  orbit + list(orbitnum * np.ones(np.size(lat)))
        
        data.close() 
    
    
    latarr = np.array(latmod)
    lonarr = np.array(lonmod)   
    aodarr = np.array(aodmod) 
#    orbitarr = np.array(orbit)
    
    output = pd.DataFrame() 
    output['lat'] = latarr
    output['lon'] = lonarr
    output['AOD'] = aodarr
#    output['orbit'] = orbitarr
    
    criteria = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E']) & \
                (output['AOD'] >= 0)
    output = output[criteria]
    return output.reset_index(drop = True)



def readMODISL3(year, month, day, ROI, dataset): 
    
    print('**** Reading MODIS %02i-%02i-%4i' % (day, month, year))
    moddir = '/nobackup/users/sunj/MODIS/AQUA/%s/%4i/%02i/%02i/' % (dataset, year, month, day)

    aodmod = []
    latmod = []
    lonmod = [] 
    orbit = []
    
    filelist = glob.glob( moddir + '*.hdf')
    
    for io in filelist[:]:  
        data = netCDF4.Dataset(io,'r')    

        Y = data.variables['YDim'][:]
        X = data.variables['XDim'][:]
        
        lon, lat = np.meshgrid(X, Y)
        aodOcean = data.variables['Aerosol_Optical_Depth_Average_Ocean_Mean'][1, :, :]     # CA
        aodLand = data.variables['Aerosol_Optical_Depth_Land_Mean'][1, :, :]  # CA
#        
        idx1, idx2 = np.where(aodOcean >= 0)
        idx3, idx4 = np.where(aodLand >= 0)
#        
        aod = np.ones(aodOcean.shape) * (-1e4)
        aod[idx1, idx2] = aodOcean[idx1, idx2]
        aod[idx3, idx4] = aodLand[idx3, idx4]
        
        
        latmod = latmod + list(np.asarray(lat).reshape(-1))
        lonmod = lonmod + list(np.asarray(lon).reshape(-1))        
        aodmod = aodmod + list(np.asarray(aod).reshape(-1))
        
        data.close() 
    
    
    latarr = np.array(latmod)
    lonarr = np.array(lonmod)   
    aodarr = np.array(aodmod) 
    
    output = pd.DataFrame() 
    output['lat'] = latarr
    output['lon'] = lonarr
    output['AOD'] = aodarr
    
    criteria = (output['AOD'] >= 0)
    output = output[criteria]
    return output.reset_index(drop = True)



def gridMODIS(year, month, day, ROI, grid, data): 
    print('**** Reading MODIS %02i-%02i-%4i' % (day, month, year))
    moddir = '/nobackup/users/sunj/MODIS/AQUA/MYD04_L2/%4i/%02i/%02i/' % (year, month, day)

    aodmod = []
    latmod = []
    lonmod = [] 
    timemod = [] 
# =============================================================================
#     
# =============================================================================
    filelist = glob.glob( moddir + '*' + data + '*.hdf')
    for io in filelist[:]:  
        data = netCDF4.Dataset(io,'r')    

        lat = data.variables['Latitude'][:]
        lon = data.variables['Longitude'][:]
#        aod = data.variables['Optical_Depth_Land_And_Ocean'][:]
#        aod = data.variables['AOD_550_Dark_Target_Deep_Blue_Combined'][:]  # not used because lose highest data
#        aodOcean = data.variables['Effective_Optical_Depth_Best_Ocean'][:2, :, :]   # 470, 550 nm
#        aodOcean = data.variables['Effective_Optical_Depth_0p55um_Ocean'][:, :]     
#        aodLand = data.variables['Deep_Blue_Aerosol_Optical_Depth_550_Land'][:, :]
        aodOcean = data.variables['Effective_Optical_Depth_Best_Ocean'][:2, :, :]     
        aodLand = data.variables['Deep_Blue_Spectral_Aerosol_Optical_Depth_Land'][:2, :, :]
        timeStamp = data.variables['Scan_Start_Time'][:, :]

        idx1, idx2 = np.where(aodOcean[0,:,:] > 0)
        idx3, idx4 = np.where(aodLand[0,:,:] > 0)
#
        maskOcean = (aodOcean[:,:,:] > 0)
        maskLand = (aodLand[:,:,:] > 0)
        aod = np.ones(aodOcean.shape) * np.nan
        
        aod[maskOcean] = aodOcean[maskOcean]
        aod[maskLand] = aodLand[maskLand]
        
        AE = Angstorm(470, aod[0,:,:], 550, aod[1,:,:])
        aod500 = wvldepAOD(550, aod[1,:,:], 500, AE)

        sza = data.variables['Solar_Zenith'][:]
        vza = data.variables['Sensor_Zenith'][:]
        saa = data.variables['Solar_Azimuth'][:]
        vaa = data.variables['Sensor_Azimuth'][:]
        sca = data.variables['Scattering_Angle'][:]
        
        raa = saa + 180 - vaa 
        
        latmod = latmod + list(np.asarray(lat).reshape(-1))
        lonmod = lonmod + list(np.asarray(lon).reshape(-1))        
        aodmod = aodmod + list(np.asarray(aod500).reshape(-1))
        timemod = timemod + list(np.asarray(timeStamp).reshape(-1))
        
        data.close() 
        
    latarr = np.array(latmod)   
    lonarr = np.array(lonmod)   
    aodarr = np.array(aodmod) 
    timearr = np.array(timemod) 
    criteria = (latarr >= ROI['S']) & (latarr <= ROI['N']) & (lonarr >= ROI['W']) & (lonarr <= ROI['E']) & \
                (aodarr >= 0)
    latarr = latarr[criteria]
    lonarr = lonarr[criteria]
    aodarr = aodarr[criteria]
    timemod = timearr[criteria]


    aodnew = griddata((lonarr, latarr), aodarr , grid, method = 'nearest')
    timenew = griddata((lonarr, latarr), timemod , grid, method = 'nearest')
#    aodnew[np.isnan(aodnew)] = -32767.
    temp = np.c_[grid[0], grid[1], aodnew, timenew]
    output = pd.DataFrame(temp, columns = ['lon', 'lat', 'AOD', 'timeStamp'])
    return output.reset_index(drop = True)


# =============================================================================
# case information
# =============================================================================
## case name
#caseName = 'CA201712'
## test name 
#testName = 'CCI_test'
## case directory
#casedir = '/nobackup/users/sunj/data_saved_for_cases/%s' %(caseName)
#if not os.path.isdir(casedir):
#    os.makedirs(casedir)   
#
## region of interest
#ROI = {'S':25, 'N': 50, 'W': -130, 'E': -110}
## time period
#year    = {'start': 2017, 'end': 2017}
#month   = {'start': 12, 'end': 12}
#day     = {'start': 12, 'end': 12}
## AAI threshold
#plumeAAI = 2.
## grid resolution
#res = 0.1
#lat= np.arange(ROI['N'], ROI['S'], -res)
#lon= np.arange(ROI['W'], ROI['E'], res)
#XX, YY = np.meshgrid(lon, lat)
#grid = (XX.reshape(-1), YY.reshape(-1))
#
#t1 = time.time()
#for yy in range(year['start'], year['end'] + 1):
#    for mm in range(month['start'], month['end'] + 1):
#        for dd in range(day['start'], day['end'] + 1):
#            data = gridMODIS(yy, mm, dd, ROI, grid, 'L2')
#            outputname = 'dataMODIS'
#            data.to_pickle(outputname)
#            if os.path.isfile(casedir + '/' + outputname):
#                os.remove(casedir + '/' + outputname)                                
#            shutil.move(outputname, casedir + '/')  
#t2 = time.time()
#print('Time for preparing satellite data: %1.2f s' % (t2 - t1))

    

def daytsMODIS(year, month, day, aerlat, aerlon, jetlag):
    aodmod = [] 
    tsmod = [] 
#    for iday in range(19,32):
    print('**** Reading MODIS %02i-%02i-%4i' % (day, month, year))
    moddir = '/nobackup/users/sunj/MODIS/AQUA/MYD04/%4i/%02i/%02i/' % (year, month, day)
    modfile = glob.glob( moddir + '*.txt')[0]
    moddata = pd.read_csv(modfile, delim_whitespace=True, header = 7, \
                        names = ['date','time','longitude','latitude','AOD','Angstrom',\
                        'fine_mode_frac','sza','vza','scattering_angle','sunglint_angle'])
    lat = np.array(moddata['latitude'])
    lon = np.array(moddata['longitude'])
    ts = np.array(moddata['time'])
    aod = np.array(moddata['AOD'])
    
    tree = spatial.KDTree(list(zip(lat, lon)))
    locs = tree.query([aerlat,aerlon], k = 1)
    
    aodmod.append(aod[locs[1]])
    
    
    temp = datetime.strptime(ts[locs[1]],'%H:%M:%S.00')
    tsjulian = iday + ( temp.hour + temp.minute/60. + temp.second/3600. + jetlag) / 24. 
    
    tsmod.append(tsjulian)
    
    return tsmod, aodmod 

    



def MODIS4cfg(year, month, day, ROI, plumemsk, res,data): 
    
    print('**** Reading MODIS %02i-%02i-%4i' % (day, month, year))
    moddir = '/nobackup/users/sunj/MODIS/AQUA/MYD04/%4i/%02i/%02i/' % (year, month, day)

    aodmod = []
    latmod = []
    lonmod = [] 
    angmod = []
    asymod = []
    ssamod = [] 
    tmod = []    
    
    filelist = glob.glob( moddir + '*' + data + '*.hdf')
    for io in filelist[:]:  
#        print io
        data = netCDF4.Dataset(io,'r')    

        lat = data.variables['Latitude'][:]
        lon = data.variables['Longitude'][:]
#        aod = data.variables['Optical_Depth_Land_And_Ocean'][:]
        aodOcean = data.variables['Effective_Optical_Depth_0p55um_Ocean'][:, :] 
        aodLand = data.variables['Deep_Blue_Aerosol_Optical_Depth_550_Land'][:, :]
        
        idx1, idx2 = np.where(aodOcean > 0)
        idx3, idx4 = np.where(aodLand > 0)
        
        aod = np.ones(aodOcean.shape) * np.nan
        aod[idx1, idx2] = aodOcean[idx1, idx2]
        aod[idx3, idx4] = aodLand[idx3, idx4]

        ang = data.variables['Angstrom_Exponent_1_Ocean'][:]
        asy = data.variables['Asymmetry_Factor_Best_Ocean'][1,:,:]              # @ 550 nm 
#        ssa = data.variables['Deep_Blue_Spectral_Single_Scattering_Albedo_Land'][:].mean(axis=0)    
        sza = data.variables['Solar_Zenith'][:]
        vza = data.variables['Sensor_Zenith'][:]
        saa = data.variables['Solar_Azimuth'][:]
        vaa = data.variables['Sensor_Azimuth'][:]
        scantime = data.variables['Scan_Start_Time'][:]
    
        latmod = latmod + list(np.asarray(lat).reshape(-1))
        lonmod = lonmod + list(np.asarray(lon).reshape(-1))        
        aodmod = aodmod + list(np.asarray(aod).reshape(-1))
        angmod = angmod + list(np.asarray(ang).reshape(-1))
        asymod = asymod + list(np.asarray(asy).reshape(-1))
#        ssamod = ssamod + list(np.asarray(ssa).reshape(-1))
        tmod = tmod + list(np.asarray(scantime).reshape(-1))
        
        data.close()
        
    
    latarr = np.array(latmod)
    lonarr = np.array(lonmod)   
    aodarr = np.array(aodmod) 
#    angarr = np.array(angmod) 
#    asyarr = np.array(asymod)
#    ssaarr = np.array(ssamod)
    tarr = np.array(tmod) 
  
    latnew = np.arange(ROI['N'], ROI['S'], -res)
    lonnew = np.arange(ROI['W'], ROI['E'], res)
    
    x,y = np.meshgrid(lonnew,latnew)
    aodnew = griddata((lonarr, latarr), aodarr , (x, y), method = 'nearest')
#    angnew = griddata((lonarr, latarr), angarr , (x, y), method = 'linear')
#    asynew = griddata((lonarr, latarr), asyarr , (x, y), method = 'linear')
#    ssanew = griddata((lonarr, latarr), ssaarr , (x, y), method = 'linear')
    tnew = griddata((lonarr, latarr), tarr , (x, y), method = 'nearest')
    
    
    latm = np.ma.masked_array(y,plumemsk)  
    lonm = np.ma.masked_array(x,plumemsk) 
    aodm = np.ma.masked_array(aodnew,plumemsk)    
#    angm = np.ma.masked_array(angnew,plumemsk)    
#    asym = np.ma.masked_array(asynew,plumemsk)    
#    ssam = np.ma.masked_array(ssanew,plumemsk)    
    tm = np.ma.masked_array(tnew,plumemsk)    

    latrm = np.array(latm[latm.mask == 0])    
    lonrm = np.array(lonm[lonm.mask == 0])  
    aodrm = np.array(aodm[aodm.mask == 0])
#    angrm = np.array(angm[angm.mask == 0])
#    asyrm = np.array(asym[asym.mask == 0])
#    ssarm = np.array(ssam[ssam.mask == 0])
    trm = np.array(tm[tm.mask == 0])
    
    tMDS = []
    for i in range(len(trm)): 
        tMDS.append(datetime(1993,1,1,0,0,0) + timedelta(seconds = trm[i]))
        
    return latrm, lonrm, aodrm, tMDS




def readMODISL3(dates, ROI, product): 
    output = pd.DataFrame()
    for idate in dates:
        sys.stdout.write(r'**** Reading MODIS %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
        moddir = '/nobackup_1/users/sunj/MODIS/%s/' % (product)
        
        temp = pd.DataFrame()
        filelist = glob.glob(moddir + '*.hdf')
        for io in filelist[:]:
            if io.find('D3.A%4i%03i' % (idate.year, idate.dayofyear)) > 0:
                data = netCDF4.Dataset(io,'r')   
                
                lat = data.variables['YDim'][:]
                lon = data.variables['XDim'][:]
                XX, YY = np.meshgrid(lon, lat)
                temp['lon'], temp['lat'] = XX.reshape(-1), YY.reshape(-1)
                temp['AOD550'] = data.variables['AOD_550_Dark_Target_Deep_Blue_Combined_Mean'][:].data.reshape(-1)
                temp['AOD550std'] = data.variables['AOD_550_Dark_Target_Deep_Blue_Combined_Standard_Deviation'][:].data.reshape(-1)
                temp['CFm'] = data.variables['Cloud_Fraction_Mean'][:].data.reshape(-1)
                temp['sza'] = data.variables['Solar_Zenith_Mean'][:].data.reshape(-1)
                temp['vza'] = data.variables['Sensor_Zenith_Mean'][:].data.reshape(-1)
                temp['saa'] = data.variables['Solar_Azimuth_Mean'][:].data.reshape(-1)
                temp['raa'] = data.variables['Sensor_Azimuth_Mean'][:].data.reshape(-1)
                
                data.close() 
# =============================================================================
#         
# =============================================================================
                criteria = (temp['lat'] >= ROI['S']) & (temp['lat'] <= ROI['N']) & (temp['lon'] >= ROI['W']) & (temp['lon'] <= ROI['E']) & \
                            (temp['AOD550'] >= 0) & (temp['AOD550'] <= 1e2) 
                temp['date'] = idate
                output = output.append(temp[criteria])
        if len(temp) == 0:
            print('No data on %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
    return output.reset_index(drop = True)

