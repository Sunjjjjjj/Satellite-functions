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
#import shutil
#import time
import numpy as np
#import numba as nb 
import numexpr as ne
import numpy.ma as ma
import matplotlib.pyplot as plt
import netCDF4 
import glob
import time
import datetime
import math
import pandas as pd
from scipy import spatial
from scipy import stats
import matplotlib.mlab as mlab
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
from scipy import interpolate
#from utilities import Theta
import h5py
from scipy import ndimage
from otherFunctions import *





def rawOMAERO(year, month, day, ROI): 
    """
    Read OMAERO raw data orbit by orbit. Pre-processing contains  
    
    Input:  date and region
    Return: selected raw data orbit by orbit
    """


    
#==============================================================================
# Reading data    
#==============================================================================
    print('**** Reading OMAEROv003 %02i-%02i-%04i' %(day, month, year))  
    
    latgol = []
    longol = []
    aodgol = [] 
    aaigol = [] 
    
    path = '/nobackup/users/sunj/OMI/OMAERO/%4i/%02i/%02i/' %(year, month, day)
    filelist = glob.glob( path + '*.he5')[:]
    
   
    for io in filelist[:]: 
        
        t1 = time.time()
        orbdata = h5py.File(io,'r')
   
        data = orbdata['/HDFEOS/SWATHS/ColumnAmountAerosol/Data Fields']
        geo  = orbdata['/HDFEOS/SWATHS/ColumnAmountAerosol/Geolocation Fields']
#        add  = orbdata.groups['HDFEOS'].groups['ADDITIONAL'].groups['FILE_ATTRIBUTES']

        lat  = geo['Latitude'][:]
        lon  = geo['Longitude'][:]    
#        meatime = geo.variables['Time'][:] 
        
        if np.logical_or(lat < -1e10, lon < -1e10).any():
            pass
        else:
        
            sza  = geo['SolarZenithAngle'][:]
            vza  = geo['ViewingZenithAngle'][:]
            saa  = geo['SolarAzimuthAngle'][:]
            vaa  = geo['ViewingAzimuthAngle'][:]
            
            gpqf = geo['GroundPixelQualityFlags'][:]      # ground pixel flag, uint16 0-2^16-1 
            row  = data['XTrackQualityFlags'][:]          # row anomaly flag, uint8 0-255 
            cf = data['EffectiveCloudFraction'][:]*1e-2
            As = data['TerrainReflectivity'][:]*1e-3
#            dt = add.TAI93At0zOfGranule
    #        wvl = add.OPF_Wavelengths
#            pqfNUV = data['ProcessingQualityFlagsNUV'][:]
    
    #        secday = meatime - dt + jetlag * 3600.                             # - time correction + jetlat second of the day         
    #        tstemp = day + secday / 24. / 3600. 
    #        alpha1 = np.interp(tstemp, tsaer, alpha)
    #        
    #        alpha1 = np.array([alpha1] * lat.shape[1]).transpose()  
    
    
    #        alpha2 = np.interp(ts2,ts1,alpha)
    
    
            aod = data['AerosolOpticalThicknessMW'][:,:,0]*1e-3           # aod [lat, lon, wvl] 
            aai = data['AerosolIndexUV'][:]*1e-2
            
           
    #==============================================================================
    # Pre-processing          
    #==============================================================================
    
            szarad = sza / 180. * np.pi
            vzarad = vza / 180. * np.pi
            saarad = saa / 180. * np.pi
            vaarad = vaa / 180. * np.pi
            
            
            raz = np.fabs(vaa - saa)
    #        raz = raz.data            
            raa = 180 - np.where(raz > 180, 360 - raz, raz)
            raarad = raa / 180. * np.pi
    
            raarad = (saa - vaa + 180.)
    
            theta = np.arccos(np.cos(szarad) * np.cos(vzarad) - np.sin(szarad) * np.sin(vzarad) * np.cos(np.pi - raarad)) 
    #        theta = Theta(sza, saa, vza, vaa)
    #        theta2 = np.arccos(-np.cos(szarad) * np.cos(vzarad) + np.sin(szarad) * np.sin(vzarad) * np.cos(np.pi + saarad - vaarad))     
            
    #        szarad = np.pi / 3.
    #        vzarad = np.pi / 4. 
    #        raarad = np.pi / 4. * 3
    #        saarad = np.pi * 0.
    #        vaarad = np.pi / 4. 
    #        raz = np.fabs(vaarad - saarad)
    #        raarad = np.pi - np.where(raz > np.pi, np.pi *2 - raz, raz)
    #        
    #        
    #        theta = np.arccos(np.cos(szarad) * np.cos(vzarad) - np.sin(szarad) * np.sin(vzarad) * np.cos( np.pi- raarad))      
    #        theta2 = np.arccos(-np.cos(szarad) * np.cos(vzarad) + np.sin(szarad) * np.sin(vzarad) * np.cos(np.pi + saarad - vaarad))     
    #        theta3 = np.arccos(-np.cos(szarad) * np.cos(vzarad) + np.sin(szarad) * np.sin(vzarad) * np.cos(np.pi - raarad)) / np.pi * 180.    
            
    #        print(theta)
    #        print(np.pi - theta2)
    #        print(theta3)
    #        print (theta + theta2)
            waterflag = np.where(np.bitwise_and(gpqf, 15) == 1, False, True) 
            sunglint = np.logical_and(waterflag, np.where(np.rad2deg(theta)<30., True, False))
    #        print(len(np.where(sunglint==False)[0]))
            
    #        sunglint2 = np.logical_and(waterflag,np.where(np.rad2deg(np.pi - theta2)<30., True, False))
    #        print(len(np.where(sunglint2==False)[0]))
            eclipse = np.where(np.bitwise_and(gpqf,32),True,False) 
    # =============================================================================
    #         
    # =============================================================================
            bi_row = np.where(row, 255, row > 0)
            
            bi_rowl = np.zeros(bi_row.shape)
            bi_rowr = np.zeros(bi_row.shape)
            
            edge = 5
            
            row_edge = bi_row
            for iedge in range(1, edge + 1):
                
                bi_rowl[:, 1:] = row_edge[:, :-1]
                bi_rowr[:,:-1] = row_edge[:, 1:]
            
                row_edge = row_edge + bi_rowl + bi_rowr
            
            row_edge = np.where(row_edge, 300, row_edge > 0.)
            
    
    # =============================================================================
    #         
    # =============================================================================
    #        locmask = np.logical_or(lat < -1e10, lon < -1e10)
        
    #        idx1, idx2 = np.where(locmask == 0)
    #        print(io, idx1,idx2)
    #        lat = lat[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        lon = lon[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        sunglint = sunglint[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        eclipse = eclipse[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        row_edge = row_edge[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        sza = sza[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        cf = cf[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        As = As[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        aai = aai[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
    #        aod = aod[idx1.min(): idx1.max(), idx2.min(): idx2.max()]
        
            
            
            
    # =============================================================================
    # 
    # =============================================================================
            
    
            mask = np.logical_or(lat[0:-1,0:-1]>ROI['N'], lat[0:-1,0:-1]<ROI['S'])
            mask = np.logical_or(mask, lon[0:-1,0:-1]<ROI['W'])
            mask = np.logical_or(mask, lon[0:-1,0:-1]>ROI['E'])
    
            mask = np.logical_or(mask, sunglint[0: -1, 0: -1])        # sunglint pixel, sun eclipse pixel                      
            mask = np.logical_or(mask, eclipse[0: -1,0: -1])        
            mask = np.logical_or(mask, row_edge[0: -1, 0: -1] > 0)                        # cross track row anomaly 
            mask = np.logical_or(mask, sza[0: -1, 0: -1] > 75.)
            
    #        if (lon < -1e10).any():
    #            lon[lon < -1e10] = 3e4    
    #            mask = np.logical_or(mask, (abs(lon[0: -1, 0: -1] * lon[1:, 0: -1])) > 1e4)      # dateline pixel
    #            mask = np.logical_or(mask, (abs(lon[0: -1, 0: -1] * lon[0: -1, 1:])) > 1e4)
    #        else:   
            mask = np.logical_or(mask, (lon[0: -1, 0: -1] * lon[1:, 0: -1]) < -100)      # dateline pixel
            mask = np.logical_or(mask, (lon[0: -1, 0: -1] * lon[0: -1, 1:]) < -100)
                
    
            mask = np.logical_or(mask, lat[0: -1, 0: -1] > lat[1:, 0: -1])             # decreasing oribit pixel
            mask = np.logical_or(mask, cf[0:-1,0:-1] > 0.3)
            mask = np.logical_or(mask, As[0:-1,0:-1, 2] > 0.6)
    
            if mask.all() == False: 
                lon = np.ma.masked_array(lon[0: -1, 0: -1], mask)
                lat = np.ma.masked_array(lat[0: -1, 0: -1], mask)
                aod = np.ma.masked_array(aod[0: -1, 0: -1], mask)  
                aai = np.ma.masked_array(aai[0: -1, 0: -1], mask)  
                ma.set_fill_value(aai, -32767.)
        
            longol.append(lon)
            latgol.append(lat)
            aodgol.append(aod)
            aaigol.append(aai)
            
            orbdata.close()
            t2 = time.time()
    #        print(t2 - t1, 'per orbit')
     
    #    plt.figure()
    #    plt.pcolor(row_edge)
    #    
    #    plt.figure()
    #    plt.pcolor(row)
       
    return latgol, longol, aaigol

#ROI = {'S':-75, 'N': 75, 'W': -180, 'E': 180}
#
##tottime = 0
#for i in range(1):
#    t1 = time.time()
#    lat, lon, para= rawOMAERO(2014, 10, 12, ROI)
#    t2 = time.time()
#    print(t2 - t1)
#    tottime += t2 - t1
#print('Total time: ', tottime )
# call test code
#def main(): 
#    lat, lon ,aai = rawOMAERO(2017, 1, 1, ROI)
#
#    
#if __name__ == '__main__':
#    main()
#plt.figure()

#for i in np.arange(0,len(lat)):
#    plt.pcolor(np.ma.masked_invalid(lon[i]),np.ma.masked_invalid(lat[i]),np.ma.masked_invalid(para[i]),cmap = 'rainbow', vmin= -4, vmax = 4)
  
  
#@nb.jit(cache = True)
def gridOMAERO(year, month, day, caseName, ROI, parameter, crival, res): 
    # read OMAERO data, grid over ROI 




#    aerdir = '/nobackup/users/sunj/AERONET/%s/' % caseName     #AERONET Santiago_Beaichef (S33.46 W7066 560m) @340
#    aerfile = glob.glob( aerdir + '*.lev15')[0]
#    aerdata = pd.read_csv(aerfile, sep=",", header = 4)
#    
#    tsaer = aerdata['Julian_Day']
#    alpha = aerdata['440-675Angstrom']


    print('**** Reading OMI %02i-%02i-%04i' %(day,month,year))     
    paragol = []
    latgol = []
    longol = [] 
    maskgol = []   
    
    path = '/nobackup/users/sunj/OMI/OMAERO/%4i/%02i/%02i/' %(year, month, day)
    filelist = glob.glob( path + '*.he5')
   
    t0 = time.time()
    for io in filelist[:]: 
        
        orbdata = h5py.File(io,'r')
       
        data = orbdata['/HDFEOS/SWATHS/ColumnAmountAerosol/Data Fields']
        geo  = orbdata['/HDFEOS/SWATHS/ColumnAmountAerosol/Geolocation Fields']
    
        lat  = geo['Latitude'][:]
        lon  = geo['Longitude'][:]    
    #        meatime = geo.variables['Time'][:] 
        sza  = geo['SolarZenithAngle'][:]
        vza  = geo['ViewingZenithAngle'][:]
        saa  = geo['SolarAzimuthAngle'][:]
        vaa  = geo['ViewingAzimuthAngle'][:]
        
        gpqf = geo['GroundPixelQualityFlags'][:]      # ground pixel flag, uint16 0-2^16-1 
        row  = data['XTrackQualityFlags'][:]          # row anomaly flag, uint8 0-255 
        cf = data['EffectiveCloudFraction'][:]*1e-2
        aod = data['AerosolOpticalThicknessMW'][:,:,0]*1e-3           # aod [lat, lon, wvl] 
        aai = data['AerosolIndexUV'][:]*1e-2
        
        meatime = geo['Time'][:] 
#        dt = add.TAI93At0zOfGranule
     
#        secday = meatime - dt + jetlag * 3600.                             # - time correction + jetlat second of the day         
#        tstemp = day + secday / 24. / 3600. 
#        alpha1 = np.interp(tstemp, tsaer, alpha)
        

#        alpha1 = np.array([alpha1] * lat.shape[1]).transpose()  
       

#        aod = data.variables['AerosolOpticalThicknessMW'][:]*1e-3       # aod [lat, lon, wvl] 
#        aai = data.variables['AerosolIndexUV'][:]*1e-2
        row = data['XTrackQualityFlags'][:]    
        cf = data['EffectiveCloudFraction'][:]*1e-2
        As = data['TerrainReflectivity'][:]*1e-3

        if parameter == 'AOD': 
            aod = data['AerosolOpticalThicknessMW'][:]*1e-3           # aod [lat, lon, wvl] 
#            aod = data.variables['AerosolOpticalThicknessPassedThresholdMean'][:]*1e-3           # aod [lat, lon, wvl] 
#            aodstd = data.variables['AerosolOpticalThicknessPassedThresholdStd'][:]*1e-3           # aod [lat, lon, wvl] 
#            para = aod[:,:,-5] * (550./442)** (-alpha1)
#            para = aod[:,:,0] * (550./342.5)** (-alpha1)
#            para = aod[:,:,:].mean(2)
            para = aod[:,:,-1]          # 483.5 nm 
#            print aod[:,:,0].max()
#            print para.max()
#            print para1.max()
        elif parameter == 'AAI':
            para = data['AerosolIndexUV'][:]*1e-2        
       
        
      
        # print 'Calculating flag'
        szarad = sza/180.*np.pi 
        vzarad = vza/180.*np.pi
        
        raz = np.fabs(vaa - saa)
#        raz = raz.data            
        raa = 180 - np.where(raz>180,360-raz,raz)
        raarad = raa/180.*np.pi
        
        scatangle = np.arccos(np.cos(szarad)*np.cos(vzarad) - np.sin(szarad)*np.sin(vzarad)*np.cos(np.pi-raarad))       
        waterflag = np.where(np.bitwise_and(gpqf,15) == 1, False, True)        
        sunglint = np.logical_and(waterflag,np.where(np.rad2deg(scatangle)<30., True, False))
        eclipse = np.where(np.bitwise_and(gpqf,32),True,False) 

# =============================================================================
#         
# =============================================================================
        bi_row = np.where(row, 255, row > 0)
        
        bi_rowl = np.zeros(bi_row.shape)
        bi_rowr = np.zeros(bi_row.shape)
        
        edge = 5
        
        row_edge = bi_row
        for iedge in range(1, edge + 1):
            
            bi_rowl[:, 1:] = row_edge[:, :-1]
            bi_rowr[:,:-1] = row_edge[:, 1:]
        
            row_edge = row_edge + bi_rowl + bi_rowr
        
        row_edge = np.where(row_edge, 300, row_edge > 0.)
        

# =============================================================================
#         
# =============================================================================

        mask = np.logical_or(lat[0:-1,0:-1] > ROI['N'], lat[0:-1,0:-1] < ROI['S'])
        mask = np.logical_or(mask, lon[0:-1,0:-1] < ROI['W'])
        mask = np.logical_or(mask, lon[0:-1,0:-1] > ROI['E'])
        
        
        mask = np.logical_or(mask, eclipse[0:-1,0:-1])        # sunglint pixel, sun eclipse pixel 
        mask = np.logical_or(mask,sunglint[0:-1,0:-1] )
        mask = np.logical_or(mask, row_edge[0:-1,0:-1]>0)                        # cross track row anomaly 
        mask = np.logical_or(mask, sza[0:-1,0:-1]>75.)
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
        mask = np.logical_or(mask, lat[0:-1,0:-1]>lat[1:,0:-1])             # decreasing oribit pixel 
        mask = np.logical_or(mask, para[0:-1,0:-1] < crival)
        mask = np.logical_or(mask, cf[0:-1,0:-1] > 0.3)
        mask = np.logical_or(mask, As[0:-1,0:-1, 2] > 0.6) 
        

        
        
        latgol = latgol + list(np.asarray(lat[0:-1,0:-1]).reshape(-1))
        longol = longol + list(np.asarray(lon[0:-1,0:-1]).reshape(-1))
        paragol = paragol + list(np.asarray(para[0:-1,0:-1]).reshape(-1))
        maskgol = maskgol + list(np.asarray(mask).reshape(-1))

    latarr = np.array(latgol)
    lonarr = np.array(longol)    
    paraarr = np.array(paragol) 
    maskarr = np.array(maskgol)
    
#    print 'Applying mask'
    paraarr = np.ma.masked_array(paraarr, maskarr)
    ma.set_fill_value(paraarr, -32767.)
    paramask = paraarr.data
    paramask[maskarr == 1] = -32767.    
    
#    maskvld = maskarr

    t1 = time.time()
    print('Time for reading all orbit: ', t1 - t0, 's')

    

    
    latvld = latarr
    lonvld = lonarr
    paravld = paraarr
    
    locmask = np.logical_or(latvld>ROI['N'], latvld<ROI['S'])
    locmask = np.logical_or(locmask, lonvld<ROI['W'])
    locmask = np.logical_or(locmask, lonvld>ROI['E'])
    
    idxvld = np.where(locmask == 0)[0]
#    print 'Grid over ROI'

    latnew = np.arange(ROI['N'], ROI['S'], -res)
    lonnew = np.arange(ROI['W'], ROI['E'], res) 
    latvld = latvld[idxvld]
    lonvld = lonvld[idxvld]
    paravld = paravld[idxvld]
   
     
    x,y = np.meshgrid(lonnew,latnew)
    paranew = griddata((lonvld, latvld), paravld, (x, y), method = 'linear', fill_value = -32767.)
    



    
#    masknew = griddata((lonvld, latvld), maskvld, (x, y), method = 'linear', fill_value = 1)
    
#    paranew = griddata((lonvld, latvld), paramask, (x, y), method = 'linear', fill_value = -32767.)
    
    t2 = time.time()
    print('Time for interpolation: ', t2 - t1, 's')
    masknew = np.zeros(paranew.shape)
    masknew = np.where(masknew, 1, paranew < paraarr.min())
    paranewm = np.ma.masked_array(paranew, masknew)
    
    return latnew, lonnew, paranewm



#ROI = {'S':-75, 'N': 75, 'W': -180, 'E': 180}
#latnew, lonnew, paranewm = gridOMAERO(2014, 10, 12, 'AAI_trend', ROI, 'AAI', -1e2, 0.5) 
#
#
#XX, YY = np.meshgrid(lonnew,latnew)
#
#plt.figure()
#plt.pcolor(XX, YY, paranewm)


def daytsOMAERO(year, month, days, aerlat, aerlon, jetlag): 

#    print 'Calculate time zone'
#    if abs(aerlon)%15 < 7.5:MODIS_OMI_statistics.py
#        jetlag = abs(aerlon)//15
#    else:
#        jetlag = abs(aerlon)//15+1
#        
#    if aerlon<0:
#        jetlag = - jetlag 
#    print jetlag 

    
    aodts1 = []
    aodts2 = []
    ts = []
    
    for iday in days[:]:
        print('**** Reading OMI %02i-%02i-%04i' %(iday, month, year)) 
        aodgol1 = []
        aodgol2 = []
        latgol = []
        longol = [] 
        tsgol = [] 
        
        
        path = '/nobackup/users/sunj/OMI/OMAERO/%4i/%02i/%02i/' %(year, month, iday)
        filelist = glob.glob( path + '*.he5')
    
    
        for io in filelist[:]: 
            
            orbdata = netCDF4.Dataset(io,'r')
            
            data = orbdata.groups['HDFEOS'].groups['SWATHS'].groups['ColumnAmountAerosol'].groups['Data Fields']
            geo = orbdata.groups['HDFEOS'].groups['SWATHS'].groups['ColumnAmountAerosol'].groups['Geolocation Fields']
            add = orbdata.groups['HDFEOS'].groups['ADDITIONAL'].groups['FILE_ATTRIBUTES']
            
            lat = geo.variables['Latitude'][:]
            lon = geo.variables['Longitude'][:] 
            meatime = geo.variables['Time'][:]                                 # measurement time
            aod = data.variables['AerosolOpticalThicknessMW'][:]*1e-3          # aod [lat, lon, wvl] 
            dt = add.TAI93At0zOfGranule            
            wvl = add.OPF_Wavelengths
            
    
    #        print '**** Transferring TAI time to Julian Day'
            secday = meatime - dt + jetlag * 3600.                             # - time correction + jetlat second of the day         
            tstemp = iday + secday / 24. / 3600. 
            tsomi = np.array([tstemp] * lat.shape[1]).transpose()  
    #        print '**** Applying Angstrom exponent to calculate AOD@550nm'
    #        alpha = np.log(aod[:,:,0]/aod[:,:,-1]) / np.log(483.5/342.5)
    #        aod550 = aod[:,:,0] * (550./342.5)**(-alpha) 
            aod342 = aod[:,:,0]
            aod442 = aod[:,:,-5]
            
    #        print '**** Collecting all orbits for one specific day'
            latgol = latgol + list(np.asarray(lat).reshape(-1))
            longol = longol + list(np.asarray(lon).reshape(-1))
            aodgol1 = aodgol1 + list(np.asarray(aod342).reshape(-1))
            aodgol2 = aodgol2 + list(np.asarray(aod442).reshape(-1))
            tsgol = tsgol +list(np.asarray(tsomi).reshape(-1))
            
            orbdata.close()
    #    print '**** Collecting all orbits for one specific day'
        latarr = np.array(latgol)
        lonarr = np.array(longol)    
        aodarr1 = np.array(aodgol1) 
        aodarr2 = np.array(aodgol2) 
        tsarr = np.array(tsgol)
        
    #    print '**** Searching the nearest pixel near Santiago_Beauchef AERONET station'
        tree = spatial.KDTree(list(zip(latarr, lonarr)))
        locs = tree.query([aerlat,aerlon], k = 1)
        
        aodts1.append(aodarr1[locs[1]])
        aodts2.append(aodarr2[locs[1]])
        ts.append(tsarr[locs[1]])
    #    print latarr[locs[1]], lonarr[locs[1]]
        
    
    #print '**** Masking data'
    aodts1 = np.array(aodts1)
    aodts2 = np.array(aodts2)
    mask = np.zeros(aodts1.shape)
    mask[aodts1<0]=1
    aod342 = np.ma.masked_array(aodts1,mask)
    aod442 = np.ma.masked_array(aodts2,mask)    
    
    return ts, aod342, aod442

#aerlat = -33.46
#aerlon = -70.66
#ts, aod342, aod442 = daytsOMAERO(2017, 1, [30], aerlat, aerlon, -5)



def OMAERO4cfg(year, month, day, caseName, ROI, plumemsk, crival, res):     
    
    aerdir = '/nobackup/users/sunj/AERONET/%s/' %(caseName)     #AERONET Santiago_Beaichef (S33.46 W7066 560m) @340
    aerfile = glob.glob( aerdir + '*.lev15')[0]
    aerdata = pd.read_csv(aerfile, sep=",", header = 4)
    
    tsaer = aerdata['Julian_Day']
    alpha = aerdata['440-675Angstrom']

    print('**** Reading OMI for Config.in %02i-%02i-%04i' %(day,month,year))     
    aaigol = []    
    aodgol = []
    aodstdgol = []
    latgol = []
    longol = [] 
    szagol = []
    vzagol = [] 
    saagol = []
    vaagol = []
    Hsgol = []
    Psgol = []
    As1gol = [] 
    As2gol = [] 
    ssagol = [] 
    
    maskgol = []   
    
    path = '/nobackup/users/sunj/OMI/OMAERO/%4i/%02i/%02i/' %(year, month, day)
    filelist = glob.glob( path + '*.he5')
    
    for io in filelist[:]: 

        orbdata = netCDF4.Dataset(io,'r')
   
        data = orbdata.groups['HDFEOS'].groups['SWATHS'].groups['ColumnAmountAerosol'].groups['Data Fields']
        geo = orbdata.groups['HDFEOS'].groups['SWATHS'].groups['ColumnAmountAerosol'].groups['Geolocation Fields']
        add = orbdata.groups['HDFEOS'].groups['ADDITIONAL'].groups['FILE_ATTRIBUTES']

        lat = geo.variables['Latitude'][:]
        lon = geo.variables['Longitude'][:]       
        meatime = geo.variables['Time'][:] 
        sza = geo.variables['SolarZenithAngle'][:]
        vza = geo.variables['ViewingZenithAngle'][:]
        saa = geo.variables['SolarAzimuthAngle'][:]
        vaa = geo.variables['ViewingAzimuthAngle'][:]
        Hs = geo.variables['TerrainHeight'][:]
        Ps = data.variables['TerrainPressure'][:]
        As1 = data.variables['TerrainReflectivity'][:,:,1]*1e-3
        As2 = data.variables['TerrainReflectivity'][:,:,2]*1e-3
#        As = np.ccolumn_stack([As1, As2])
        cf = data.variables['EffectiveCloudFraction'][:]*1e-2
        ssa = data.variables['SingleScatteringAlbedoPassedThresholdMean'][:,:,0]*1e-3  
        gpqf = geo.variables['GroundPixelQualityFlags'][:] 
        dt = add.TAI93At0zOfGranule
        
        
#        secday = meatime - dt + (-3) * 3600.                             # - time correction + jetlat second of the day         
#        tstemp = day + secday / 24. / 3600. 
#        print (tstemp)
#        alpha1 = np.interp(tstemp, tsaer, alpha)
#        alpha1 = np.array([alpha1] * lat.shape[1]).transpose()  
##        print 'Angstrom exponent OMI:', alpha1

        row = data.variables['XTrackQualityFlags'][:]    

        aai = data.variables['AerosolIndexUV'][:]*1e-2
        aod = data.variables['AerosolOpticalThicknessMW'][:,:,-5]*1e-3           # aod [lat, lon, wvl] 
        aodstd = data.variables['AerosolOpticalThicknessMWPrecision'][:]*1e-3           # aod [lat, lon, wvl] 
#        aod = aod442 * (550./442)** (-alpha1)
        
                
        # print 'Calculating flag'
        szarad = sza/180.*np.pi 
        vzarad = vza/180.*np.pi
        
        raz = np.fabs(vaa - saa)
        raz = raz.data            
        raa = 180 - np.where(raz>180,360-raz,raz)
        raarad = raa/180.*np.pi
        
        scatangle = np.arccos(np.cos(szarad)*np.cos(vzarad) - np.sin(szarad)*np.sin(vzarad)*np.cos(np.pi-raarad))       
        waterflag = np.where(np.bitwise_and(gpqf,15) == 1, False, True)        
        sunglint = np.logical_and(waterflag,np.where(np.rad2deg(scatangle)<30., True, False))
        eclipse = np.where(np.bitwise_and(gpqf,32),True,False) 


        mask = np.logical_or(lat[0:-1,0:-1]>ROI['N'], lat[0:-1,0:-1]<ROI['S'])
        mask = np.logical_or(mask, lon[0:-1,0:-1]<ROI['W'])
        mask = np.logical_or(mask, lon[0:-1,0:-1]>ROI['E'])
#        mask = np.logical_or(mask, sunglint[0:-1,0:-1]) # sunglint pixel, sun eclipse pixel
        mask = np.logical_or(mask, eclipse[0:-1,0:-1])
        mask = np.logical_or(mask, row[0:-1,0:-1]>0)                        # cross track row anomaly 
        mask = np.logical_or(mask, sza[0:-1,0:-1]>75.)
#        mask = np.logical_or(mask, vza[0:-1,0:-1]<35.)
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
        mask = np.logical_or(mask, lat[0:-1,0:-1]>lat[1:,0:-1])             # decreasing oribit pixel 
#        mask = np.logical_or(mask, cf[0:-1,0:-1]>0.35)
#        mask = np.logical_or(mask, As1[0:-1,0:-1]>0.1)
#        mask = np.logical_or(mask, As2[0:-1,0:-1]>0.1)
        

        latgol = latgol + list(np.asarray(lat[0:-1,0:-1]).reshape(-1))
        longol = longol + list(np.asarray(lon[0:-1,0:-1]).reshape(-1))
        aaigol = aaigol + list(np.asarray(aai[0:-1,0:-1]).reshape(-1))
        aodgol = aodgol + list(np.asarray(aod[0:-1,0:-1]).reshape(-1))
        aodstdgol = aodstdgol + list(np.asarray(aodstd[0:-1,0:-1]).reshape(-1))
        szagol = szagol + list(np.asarray(sza[0:-1,0:-1]).reshape(-1))
        saagol = saagol + list(np.asarray(saa[0:-1,0:-1]).reshape(-1))
        vzagol = vzagol + list(np.asarray(vza[0:-1,0:-1]).reshape(-1))
        vaagol = vaagol + list(np.asarray(vaa[0:-1,0:-1]).reshape(-1))
        Hsgol = Hsgol + list(np.asarray(Hs[0:-1,0:-1]).reshape(-1))
        Psgol = Psgol + list(np.asarray(Ps[0:-1,0:-1]).reshape(-1))
        As1gol = As1gol + list(np.asarray(As1[0:-1,0:-1]).reshape(-1))
        As2gol = As2gol + list(np.asarray(As2[0:-1,0:-1]).reshape(-1))
        ssagol = ssagol + list(np.asarray(ssa[0:-1,0:-1]).reshape(-1))
        
        maskgol = maskgol + list(np.asarray(mask).reshape(-1))
        
    
    latarr = np.array(latgol)
    lonarr = np.array(longol)    
    aaiarr = np.array(aaigol) 
    aodarr = np.array(aodgol) 
    aodstdarr = np.array(aodstdgol)
    szaarr = np.array(szagol)
    saaarr = np.array(saagol)    
    vzaarr = np.array(vzagol) 
    vaaarr = np.array(vaagol)
    Psarr = np.array(Psgol)    
    Hsarr = np.array(Hsgol) 
    As1arr = np.array(As1gol)
    As2arr = np.array(As2gol)
    ssaarr = np.array(ssagol)

    


    maskarr = np.array(maskgol)
#    print 'Applying mask'
    aodarr = np.ma.masked_array(aodarr,maskarr)
    
    latvld = np.array(latarr[aodarr.mask == 0]) 
    lonvld = np.array(lonarr[aodarr.mask == 0]) 
    aaivld = np.array(aaiarr[aodarr.mask == 0])
    aodvld = np.array(aodarr[aodarr.mask == 0])
    aodstdvld = np.array(aodstdarr[aodarr.mask == 0])
    szavld = np.array(szaarr[aodarr.mask == 0]) 
    saavld = np.array(saaarr[aodarr.mask == 0]) 
    vzavld = np.array(vzaarr[aodarr.mask == 0])
    vaavld = np.array(vaaarr[aodarr.mask == 0]) 
    Psvld = np.array(Psarr[aodarr.mask == 0]) 
    Hsvld = np.array(Hsarr[aodarr.mask == 0]) 
    As1vld = np.array(As1arr[aodarr.mask == 0]) 
    As2vld = np.array(As2arr[aodarr.mask == 0]) 
    ssavld = np.array(ssaarr[aodarr.mask == 0]) 
    
   

   
#    print 'Grid over ROI'
#    latnew = np.arange(ROI['S'], ROI['N'], res)
#    lonnew = np.arange(ROI['W'], ROI['E'], res)
    latnew = np.arange(ROI['N'], ROI['S'], -res)
    lonnew = np.arange(ROI['W'], ROI['E'], res)     
     
    x,y = np.meshgrid(lonnew,latnew)
    aainew = griddata((lonvld, latvld), aaivld, (x, y), method = 'linear')
    aodnew = griddata((lonvld, latvld), aodvld, (x, y), method = 'linear')
    aodstdnew = griddata((lonvld, latvld), aodstdvld, (x, y), method = 'linear')
    szanew = griddata((lonvld, latvld), szavld, (x, y), method = 'linear')
    saanew = griddata((lonvld, latvld), saavld, (x, y), method = 'linear')
    vzanew = griddata((lonvld, latvld), vzavld, (x, y), method = 'linear')
    vaanew = griddata((lonvld, latvld), vaavld, (x, y), method = 'linear')
    Psnew = griddata((lonvld, latvld), Psvld, (x, y), method = 'linear')
    Hsnew = griddata((lonvld, latvld), Hsvld, (x, y), method = 'linear')
    As1new = griddata((lonvld, latvld), As1vld, (x, y), method = 'linear')
    As2new = griddata((lonvld, latvld), As2vld, (x, y), method = 'linear')
    ssanew = griddata((lonvld, latvld), ssavld, (x, y), method = 'linear')
    
    
    # set NaN to negative vales
    aainew[np.isnan(aainew)] = -32767.
    aodnew[np.isnan(aodnew)] = -32767.
    plumemsk = np.logical_or(plumemsk, aodnew < 0.)
    plumemsk = np.logical_or(plumemsk, aainew < crival)
#    Psnew[np.isnan(szanew)] = -32767.
#    Psnew[np.isnan(vzanew)] = -32767.
#    Psnew[np.isnan(saanew)] = -32767.
#    Psnew[np.isnan(vaanew)] = -32767.
#    Psnew[np.isnan(Psnew)] = -32767.
#    Psnew[np.isnan(As1new)] = -32767.
#    Psnew[np.isnan(As2new)] = -32767.
#    plumemsk = np.logical_or(plumemsk, Psnew < 0.)


    latm = np.ma.masked_array(y,plumemsk)  
    lonm = np.ma.masked_array(x,plumemsk) 
    aaim = np.ma.masked_array(aainew,plumemsk)    
    aodm = np.ma.masked_array(aodnew,plumemsk)    
    aodstdm = np.ma.masked_array(aodstdnew,plumemsk)    
    szam = np.ma.masked_array(szanew,plumemsk)    
    saam = np.ma.masked_array(saanew,plumemsk)    
    vzam = np.ma.masked_array(vzanew,plumemsk)    
    vaam = np.ma.masked_array(vaanew,plumemsk)    
    Psm = np.ma.masked_array(Psnew,plumemsk)    
    Hsm = np.ma.masked_array(Hsnew,plumemsk)    
    As1m = np.ma.masked_array(As1new,plumemsk)   
    As2m = np.ma.masked_array(As2new,plumemsk)  
    ssam = np.ma.masked_array(ssanew,plumemsk)       


    latrm = np.array(latm[latm.mask == 0])    
    lonrm = np.array(lonm[lonm.mask == 0])  
    aairm = np.array(aaim[aaim.mask == 0])
    aodrm = np.array(aodm[aodm.mask == 0])
    aodstdrm = np.array(aodstdm[aodstdm.mask == 0])
    szarm = np.array(szam[szam.mask == 0])
    saarm = np.array(saam[saam.mask == 0])
    vzarm = np.array(vzam[vzam.mask == 0])
    vaarm = np.array(vaam[vaam.mask == 0])
    Psrm = np.array(Psm[Psm.mask == 0])
    Hsrm = np.array(Hsm[Hsm.mask == 0])
    As1rm = np.array(As1m[As1m.mask == 0])
    As2rm = np.array(As2m[As2m.mask == 0])
    ssarm = np.array(ssam[ssam.mask == 0])
    
    
    Asrm = np.column_stack([As1rm, As2rm])
    
    return latrm, lonrm, aairm, aodrm, aodstdrm, szarm, saarm, vzarm, vaarm, ssarm, Psrm, plumemsk, Asrm
    





def readOMAEROgrid(startdate, enddate, ROI, res): 
    dateid = dateList(startdate, enddate)

    tlen = enddate['year'] - startdate['year'] + 1
    mlen = 12
    dlen = 31
    
    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(-179.5, 180, 1)
    XX, YY = np.meshgrid(lon , lat)
    
    AI = np.ones([tlen, mlen, dlen, XX.shape[0], XX.shape[1]]) * np.nan

    print('reading OMAERO grid ... ')
    for i, idate in enumerate(dateid): 
        yy, mm, dd = DOY2date(idate[0], idate[1])
        
        nobackupdir = '/nobackup/users/sunj/OMI/OMAERO_grid/%4i/' % (yy)
        datafile = 'OMAERO_grid_%4i-%02i-%02i.h5' % (yy, mm, dd)  
        try: 
            AIdata = h5py.File(nobackupdir + datafile, 'r')
            
            AItemp = AIdata['AAI'][:]
            mask =  AIdata['mask'][:]
            
            AIdata.close()
            
            mask = np.logical_or(mask, YY < ROI['S'])
            mask = np.logical_or(mask, YY > ROI['N'])
            mask = np.logical_or(mask, XX < ROI['W'])
            mask = np.logical_or(mask, XX > ROI['E'])
            
            AItempm = np.ma.masked_array(AItemp, mask)    
            AItempm[AItempm.mask == 1] = np.nan
            AI[yy - startdate['year'], mm - 1, dd - 1, : , :] = AItempm
            
            
        except:
            pass
     
    
    return AI



def readTEMISOMI(startdate, enddate, ROI, threshold): 
# =============================================================================
# dimension
# =============================================================================
    dates = pd.date_range(startdate, enddate)

    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(-179.5, 180, 1)
    XX, YY = np.meshgrid(lon , lat)
#    coords = pd.DataFrame(np.c_[YY.reshape(-1), XX.reshape(-1)], columns = ['lat', 'lon'])
#    output = pd.DataFrame(np.ones([np.size(XX), len(dates)]) * np.nan, columns = dates)
    output = pd.DataFrame()
    for i, idate in enumerate(dates): 
        sys.stdout.write('\r # %s' % (idate))
        yy, mm, dd = idate.year, idate.month, idate.day
        
        path = '/nobackup/users/sunj/OMI/TEMISOMI/%4i/' %(yy)
        filelist = glob.glob(path + '*.nc')
        
        for io in filelist:
            if io.find('%4i%02i%02i' % (yy, mm, dd)) != -1:
                np.warnings.filterwarnings('ignore')
                data = netCDF4.Dataset(io, 'r')
                lat = data.variables['latitude'][:]
                lon = data.variables['longitude'][:]
                AI = data.variables['absorbing_aerosol_index'][:]
                num = data.variables['number_of_observations'][:]
                sza = data.variables['solar_zenith_angle'][:]
                
                XX, YY = np.meshgrid(lon , lat)
                temp = pd.DataFrame(np.c_[YY.reshape(-1), XX.reshape(-1), AI.reshape(-1), num.reshape(-1), sza.reshape(-1)], 
                                          columns = ['lat', 'lon', 'AI388', 'num', 'sza'])
                temp['date'] = [idate] * len(temp) 
                data.close()
                output = output.append(temp)
# =============================================================================
#  apply mask
# =============================================================================
    mask = ((output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E'])) 
    output = output[mask]
    output = output.reset_index(drop = True)
    output.loc[(output['AI388'] < threshold), 'AI388'] = np.nan
    output.loc[(output['AI388'] > 1) & (abs(output['lat']) > 75), 'AI388'] = np.nan
    output.loc[(output['num'] < 100), 'AI388'] = np.nan

    return output



def readOMLER(month, wvl, ROI): 
# =============================================================================
# dimension
# =============================================================================
    path = '/nobackup/users/sunj/OMI/OMLER/'
    filelist = glob.glob(path + '*.he5')
    
    data = netCDF4.Dataset(filelist[0], 'r')
    lat = data['HDFEOS']['GRIDS']['EarthSurfaceReflectanceClimatology']['Data Fields']['Latitude'][:]
    lon = data['HDFEOS']['GRIDS']['EarthSurfaceReflectanceClimatology']['Data Fields']['Longitude'][:]
    wvls = data['HDFEOS']['GRIDS']['EarthSurfaceReflectanceClimatology']['Data Fields']['Wavelength'][:]
    idx = np.argmin(abs(wvls - wvl))
    temp = data['HDFEOS']['GRIDS']['EarthSurfaceReflectanceClimatology']['Data Fields']['MonthlySurfaceReflectance'][month - 1, idx, :, :] * 1e-3
    
    data.close()
    lon, lat = np.meshgrid(lon, lat)
    mask = (lat >= ROI['S']) & (lat <= ROI['N']) & (lon >= ROI['W']) & (lon <= ROI['E'])
    temp = temp[mask] 
    lat = lat[mask]
    lon = lon[mask]
# =============================================================================
#  apply mask
# =============================================================================
    return lat, lon, temp


def readOMAERUV(dates, ROI, threshold, grid): 
    """
    Read OMAERO raw data orbit by orbit. Pre-processing contains  
    
    Input:  date and region
    Return: selected raw data orbit by orbit
    """
#==============================================================================
# Reading data    
#==============================================================================
    try:
        startdate, enddate = dates[0], dates[1]
    except:
        startdate = enddate = dates[0]
    
    dates = pd.date_range(startdate, enddate)
    
    output = pd.DataFrame()
    for idate in dates:
        sys.stdout.write('\r Reading OMAERUV %02i-%02i-%04i' %(idate.year, idate.month, idate.day))
        path = '/nobackup_1/users/sunj/OMI/OMAERUV/%4i/%02i/%02i/' %(idate.year, idate.month, idate.day)
        orbit = glob.glob(path + '*.he5')[:]
# =============================================================================
# read data    
# =============================================================================
        if len(orbit) == 0:
            print('Warning: No data available on %02i-%02i-%04i' %(idate.year, idate.month, idate.day))
        else: 
            for iorbit in orbit: 
                idx = iorbit.find('-o')
                orbitnum = iorbit[idx+2 : idx + 7]
                sys.stdout.write('\r Reading OMAERUV # %04i-%02i-%02i %s' % (idate.year, idate.month, idate.day, orbitnum))
                try: 
                    _data = h5py.File(iorbit,'r')
               
                    data = _data['/HDFEOS/SWATHS/Aerosol NearUV Swath/Data Fields']
                    geo  = _data['/HDFEOS/SWATHS/Aerosol NearUV Swath/Geolocation Fields']
                    
                    AI = data['UVAerosolIndex'][:]
                    residue = data['Residue'][:]
                    AAOD = data['FinalAerosolAbsOpticalDepth'][:]
                    AOD = data['FinalAerosolOpticalDepth'][:]
                    SSA = data['FinalAerosolSingleScattAlb'][:]
                    ALH = data['FinalAerosolLayerHeight'][:]
                    wvls = data['Wavelength'][:]
                    normR = data['NormRadiance'][:]
                    CF = data['CloudFraction'][:]
                    As = data['SurfaceAlbedo'][:]
                    AI_qf = data['AlgorithmFlags_AerosolIndex'][:]
#                    qf = data['PixelQualityFlags'][:]
                    algorithm_qf = data['FinalAlgorithmFlags'][:]
                    HF = data['HeightFlags'][:]
                    Type = data['AerosolType'][:]
                    
                    lat = geo['Latitude'][:]
                    lon = geo['Longitude'][:]
                    latc = geo['FoV75CornerLatitude'][:]
                    lonc = geo['FoV75CornerLongitude'][:]
                    sza = geo['SolarZenithAngle'][:]
                    vza = geo['ViewingZenithAngle'][:]
                    raa = geo['RelativeAzimuthAngle'][:]
                    row = geo['XTrackQualityFlags'][:]
                    gpqf = geo['GroundPixelQualityFlags'][:]
                    
                    
                    AOD[AOD < 0] = -1
                    AAOD[AAOD < 0] = -1
                    AI[AI < -1e2] = -1e2
                    residue[residue < -1e2] = -1e2
                    AOD500std = NNstd(AOD[:, :, -1])
                    AAOD500std = NNstd(AAOD[:,:, -1])
                    AIstd = NNstd(AI)
                    residuestd = NNstd(residue)
                    
                    glint_angle = np.rad2deg(np.arccos(np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza)) + \
                                                       np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(np.deg2rad(raa))))
                    waterflag = np.where(np.bitwise_and(gpqf,15) == 1, False, True)
                    sun_glint = (waterflag) & (glint_angle <= 20)
                    
                    Ps = geo['TerrainPressure'][:]
                    deltaTime = geo['Time'][:]
                    refTime = datetime.datetime(1993, 1, 1, 0, 0, 0)
                    deltaTime = np.ones(lat.shape) * deltaTime.reshape(len(deltaTime), 1) 
                    
                                    
                    _data.close()
                    
                    setdata = {'lat': lat, 'lon': lon, 'latb1': latc[:, :, 0], 'lonb1': lonc[:, :, 0], 'latb2': latc[:, :, 1], 'lonb2': lonc[:, :, 1],\
                               'latb3': latc[:, :, 2], 'lonb3': lonc[:, :, 2], 'latb4': latc[:, :, 3], 'lonb4': lonc[:, :, 3], \
                               'row': row, 'sun_glint': sun_glint, 'QF': algorithm_qf, 'HF': HF, 'type': Type, \
                               'sza': sza, 'vza': vza, 'raa': raa, 'AI388': AI, 'AI388std': AIstd, 'R354obs': normR[:,:,0] * np.pi / np.cos(np.deg2rad(sza)), 'R388obs': normR[:,:,1] * np.pi / np.cos(np.deg2rad(sza)), \
                               'CF': CF, 'ALH': ALH, 'Ps': Ps, 'As': As[:,:,1], 'deltaTime': deltaTime, 'orbit': np.ones(lat.shape) * int(orbitnum), \
                               'AAOD354': AAOD[:,:,0], 'AOD354': AOD[:,:,0], 'SSA354': SSA[:,:,0], \
                               'AAOD388': AAOD[:,:,1], 'AOD388': AOD[:,:,1], 'SSA388': SSA[:,:,1], \
                               'AAOD500': AAOD[:,:,-1], 'AOD500': AOD[:,:,-1], 'SSA500': SSA[:,:,-1], \
                               'AOD500std': AOD500std, 'AAOD500std': AAOD500std, \
                               'residue': residue, 'residuestd': residuestd, 'AIQF': AI_qf}
                    del AI, normR, AAOD, AOD, SSA, wvls, lat, lon
                    
                    
                    for ikey in setdata.keys():
                        setdata[ikey] = setdata[ikey].reshape(-1)
                    setdata = pd.DataFrame.from_dict(setdata)
    
# =============================================================================
# combine data
# =============================================================================
    #                setdata = pd.DataFrame.from_dict(setdata)
                    output = output.append(setdata)
                except: 
                    pass

# =============================================================================
# criteria 
# =============================================================================
    try:
        mask = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E']) & \
                    (output['row'] == 0) & (output['sza'] < 70)  & \
                    (output['AI388'] > threshold) & (output['residue'] > threshold) & \
                    (output['CF'] <= 0.3) & (~output['sun_glint']) & (output['QF'] < 8)
    #                    & (output['ALH'] > 0) & \
    #                    (output['R354obs'] > 0) & \
    #                    (output['AOD388'] >= 0) & \
    #                    (output['SSA388'] >= 0) 
                    
        output = output[mask].reset_index(drop = True)
        
        
        def meaTime(dTime): 
            mTime = pd.datetime(1993, 1, 1) + pd.Timedelta(seconds = dTime)
            return mTime
        output['dateTime'] = list(map(meaTime, output['deltaTime']))
        output['timeStamp'] = output['dateTime'].values.astype(np.int64) // 10 ** 9
        del output['deltaTime']
    except:
        pass
    return output


def readOMCLDO2(dates, ROI): 
    """
    Read OMCLDO2 raw data orbit by orbit. Pre-processing contains  
    
    Input:  date and region
    Return: selected raw data orbit by orbit
    """
#==============================================================================
# Reading data    
#==============================================================================
    try:
        startdate, enddate = dates[0], dates[1]
    except:
        startdate = enddate = dates[0]
    
    dates = pd.date_range(startdate, enddate)
    
    
    output = pd.DataFrame()
    for idate in dates:
        sys.stdout.write('\r Reading OMAERUV %02i-%02i-%04i' %(idate.year, idate.month, idate.day))
        path = '/nobackup_1/users/sunj/OMI/OMCLDO2/%4i/%02i/%02i/' %(idate.year, idate.month, idate.day)
        orbit = glob.glob(path + '*.he5')[:]

# =============================================================================
# read data    
# =============================================================================
        if len(orbit) == 0:
            print('Warning: No data available on %02i-%02i-%04i' %(idate.year, idate.month, idate.day))
        else: 
            for iorbit in orbit: 
                idx = iorbit.find('-o')
                orbitnum = iorbit[idx+2 : idx + 7]
                sys.stdout.write('\r Reading OMAERUV # %04i-%02i-%02i %s' % (idate.year, idate.month, idate.day, orbitnum))
                try: 
                    _data = h5py.File(iorbit,'r')
               
                    data = _data['/HDFEOS/SWATHS/CloudFractionAndPressure/Data Fields']
                    geo  = _data['/HDFEOS/SWATHS/CloudFractionAndPressure/Geolocation Fields']
                    
                    CF = data['CloudFraction'][:]
                    CP = data['CloudPressure'][:]
                    SP = data['ScenePressure'][:]
                    Ps = data['TerrainPressure'][:]
                    As = data['TerrainReflectivity'][:]
                    algorithm_qf = data['FinalAlgorithmFlags'][:]
                    row = data['XTrackQualityFlags'][:]
                    QA = data['MeasurementQualityFlags'][:]
                    PQ = data['ProcessingQualityFlags'][:]
                    SCDO2 = data['SlantColumnAmountO2O2'][:]
                    SCDO3 = data['SlantColumnAmountO3'][:]
                    SCDNO2 = data['SlantColumnAmountNO2'][:]
                     
                    lat = geo['Latitude'][:]
                    lon = geo['Longitude'][:]
                    latc = geo['FoV75CornerLatitude'][:]
                    lonc = geo['FoV75CornerLongitude'][:]
                    sza = geo['SolarZenithAngle'][:]
                    vza = geo['ViewingZenithAngle'][:]
                    raa = geo['RelativeAzimuthAngle'][:]
                    gpqf = geo['GroundPixelQualityFlags'][:]
                    
                    glint_angle = np.rad2deg(np.arccos(np.cos(np.deg2rad(sza)) * np.cos(np.deg2rad(vza)) + \
                                                       np.sin(np.deg2rad(sza)) * np.sin(np.deg2rad(vza)) * np.cos(np.deg2rad(raa))))
                    waterflag = np.where(np.bitwise_and(gpqf,15) == 1, False, True)
                    sun_glint = (waterflag) & (glint_angle <= 20)
                    
                    deltaTime = geo['Time'][:]
                    refTime = datetime.datetime(1993, 1, 1, 0, 0, 0)
                    deltaTime = np.ones(lat.shape) * deltaTime.reshape(len(deltaTime), 1) 
                    
                                    
                    _data.close()
                    
                    setdata = {'lat': lat, 'lon': lon, 'latb1': latc[:, :, 0], 'lonb1': lonc[:, :, 0], 'latb2': latc[:, :, 1], 'lonb2': lonc[:, :, 1],\
                               'latb3': latc[:, :, 2], 'lonb3': lonc[:, :, 2], 'latb4': latc[:, :, 3], 'lonb4': lonc[:, :, 3], \
                               'row': row, 'sun_glint': sun_glint, 'QF': algorithm_qf, \
                               'sza': sza, 'vza': vza, 'raa': raa, \
                               'CF': CF, 'CP': CP, 'SP': SP, 'As': As, 'Ps': Ps, \
                               'SCDO2': SCDO2, 'SCDO3': SCDO3, 'SCDNO2': SCDNO2}
                    del AI, normR, AAOD, AOD, SSA, wvls, lat, lon
                    
                    
                    for ikey in setdata.keys():
                        setdata[ikey] = setdata[ikey].reshape(-1)
                    setdata = pd.DataFrame.from_dict(setdata)
    
# =============================================================================
# combine data
# =============================================================================
    #                setdata = pd.DataFrame.from_dict(setdata)
                    output = output.append(setdata)
                except: 
                    pass

# =============================================================================
# criteria 
# =============================================================================
    try:
        mask = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E']) & \
                    (output['row'] == 0) & (output['sza'] < 70)  & \
                    (output['AI388'] > threshold) & \
                    (output['CF'] <= 0.3) & (output['CF'] >= 0) & (~output['sun_glint']) & (output['QF'] < 8)
    #                    & (output['ALH'] > 0) & \
    #                    (output['R354obs'] > 0) & \
    #                    (output['AOD388'] >= 0) & \
    #                    (output['SSA388'] >= 0) 
                    
        output = output[mask].reset_index(drop = True)
# =============================================================================
# AAH        
# =============================================================================
        output['AAH'] = output['CP'].copy()
        
        output.loc['AAH', (output['CF'] > 0.25) & ( output['CF'] < 0.75)] = max(output['CP'], output['SP'])
        
        def meaTime(dTime): 
            mTime = pd.datetime(1993, 1, 1) + pd.Timedelta(seconds = dTime)
            return mTime
        output['dateTime'] = list(map(meaTime, output['deltaTime']))
        output['timeStamp'] = output['dateTime'].values.astype(np.int64) // 10 ** 9
        del output['deltaTime']
    except:
        pass
    return output

# =============================================================================
# 
# =============================================================================
def readOMAERUVd(startdate, enddate, ROI, threshold):
    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(-179.5, 180, 1)
    XX, YY = np.meshgrid(lon , lat)
# =============================================================================
# daily data
# =============================================================================
    dates = pd.date_range(startdate, enddate, freq = 'd', closed = 'left')
    dates = pd.DatetimeIndex(dates)
    
    output = pd.DataFrame()
#        output = pd.DataFrame(np.ones([np.size(XX), len(dates)]) * np.nan, columns = dates)
    for i, idate in enumerate(dates): 
        sys.stdout.write('\r # %s' % (idate))
        yy, mm, dd = idate.year, idate.month, idate.day
        path = '/nobackup/users/sunj/OMI/OMAERUVd/%4i/' %(yy)
        filelist = glob.glob( path + '*.he5')
        for io in filelist:
            if io.find('%4im%02i%02i' % (yy, mm, dd)) != -1: 
                data = netCDF4.Dataset(io)
                AI388 = data['HDFEOS']['GRIDS']['Aerosol NearUV Grid']['Data Fields']['UVAerosolIndex'][:]
                AAOD500 = data['HDFEOS']['GRIDS']['Aerosol NearUV Grid']['Data Fields']['FinalAerosolAbsOpticalDepth500'][:]
                AOD500 = data['HDFEOS']['GRIDS']['Aerosol NearUV Grid']['Data Fields']['FinalAerosolOpticalDepth500'][:]
                SSA500 = data['HDFEOS']['GRIDS']['Aerosol NearUV Grid']['Data Fields']['FinalAerosolSingleScattAlb500'][:]
                CF = data['HDFEOS']['GRIDS']['Aerosol NearUV Grid']['Data Fields']['CloudFraction'][:]
                COD = data['HDFEOS']['GRIDS']['Aerosol NearUV Grid']['Data Fields']['CloudOpticalDepth'][:]
                data.close()
                
                AI388[AI388 < -10] = np.nan
                AAOD500[AAOD500 < 0] = np.nan
                AOD500[AOD500 < 0] = np.nan
                SSA500[SSA500 < 0] = np.nan
                CF[CF < 0] = np.nan
                COD[COD < 0] = np.nan
                temp = pd.DataFrame(np.c_[YY.reshape(-1), XX.reshape(-1), AI388.reshape(-1), AAOD500.reshape(-1), AOD500.reshape(-1), SSA500.reshape(-1), CF.reshape(-1), COD.reshape(-1)],
                                          columns = ['lat', 'lon', 'AI388', 'AAOD500', 'AOD500', 'SSA500', 'CF', 'COD'])                
                output = output.append(temp)
                
    mask = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E'])
    output = output[mask]
    output = output.reset_index(drop = True)
    output.loc[(output['CF'] > 0.3) | (output['COD'] > 1e4) | ((abs(output['lat']) > 75) & (output['AI388'] >= 1)), 'AI388'] = np.nan
    output.loc[(output['AI388'] > 10) | (output['AI388'] < threshold), 'AI388'] = np.nan
    
#    output.loc[np.isnan(output['AOD500']), 'AI388'] = np.nan
    return output



def readOMTO3d(startdate, enddate, ROI, threshold):
    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(-179.5, 180, 1)
    XX, YY = np.meshgrid(lon , lat)
# =============================================================================
# daily data
# =============================================================================
    dates = pd.date_range(startdate, enddate, freq = 'd', closed = 'left')
    dates = pd.DatetimeIndex(dates)
    
    output = pd.DataFrame()
#        output = pd.DataFrame(np.ones([np.size(XX), len(dates)]) * np.nan, columns = dates)
    for i, idate in enumerate(dates): 
        sys.stdout.write('\r # %s' % (idate))
        yy, mm, dd = idate.year, idate.month, idate.day
        path = '/nobackup/users/sunj/OMI/OMTO3d/%4i/' %(yy)
        filelist = glob.glob( path + '*.he5')
        for io in filelist:
            if io.find('%4im%02i%02i' % (yy, mm, dd)) != -1: 
                data = netCDF4.Dataset(io)
                AI388 = data['HDFEOS']['GRIDS']['OMI Column Amount O3']['Data Fields']['UVAerosolIndex'][:]
                sza = data['HDFEOS']['GRIDS']['OMI Column Amount O3']['Data Fields']['SolarZenithAngle'][:]
                data.close()
                
                AI388[AI388 < -10] = np.nan
                sza[sza < 0] = np.nan
                temp = pd.DataFrame(np.c_[YY.reshape(-1), XX.reshape(-1), AI388.reshape(-1), sza.reshape(-1)],
                                          columns = ['lat', 'lon', 'AI388', 'sza'])
                temp['date'] =  [idate] * XX.size
                output = output.append(temp)
    mask = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E'])
    output = output[mask]
    output = output.reset_index(drop = True)
    output.loc[(output['AI388'] > 10) | (output['AI388'] < 0), 'AI388'] = np.nan
    
    return output

def readOMAEROe(startdate, enddate, ROI, threshold):
# =============================================================================
# daily data
# =============================================================================
    dates = pd.date_range(startdate, enddate, freq = 'd', closed = 'left')
    dates = pd.DatetimeIndex(dates)
    
    output = pd.DataFrame()
#        output = pd.DataFrame(np.ones([np.size(XX), len(dates)]) * np.nan, columns = dates)
    for i, idate in enumerate(dates): 
        sys.stdout.write('\r # %s' % (idate))
        yy, mm, dd = idate.year, idate.month, idate.day
        path = '/nobackup/users/sunj/OMI/OMAEROe/%4i/' %(yy)
        filelist = glob.glob( path + '*.he5')
        for io in filelist:
            if io.find('%4im%02i%02i' % (yy, mm, dd)) != -1: 
                data = netCDF4.Dataset(io)
                lat = data['HDFEOS']['GRIDS']['ColumnAmountAerosol']['Data Fields']['Latitude'][:]
                lon = data['HDFEOS']['GRIDS']['ColumnAmountAerosol']['Data Fields']['Longitude'][:]
                AI388 = data['HDFEOS']['GRIDS']['ColumnAmountAerosol']['Data Fields']['UVAerosolIndex'][:]
                sza = data['HDFEOS']['GRIDS']['ColumnAmountAerosol']['Data Fields']['SolarZenithAngle'][:]
                data.close()
                
                AI388[AI388 < -10] = np.nan
                sza[sza < 0] = np.nan
                temp = pd.DataFrame(np.c_[lat.reshape(-1), lon.reshape(-1), AI388.reshape(-1), sza.reshape(-1)],
                                          columns = ['lat', 'lon', 'AI388', 'sza'])
                temp['date'] =  [idate] * lat.size
                output = output.append(temp)
    mask = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E'])
    output = output[mask]
    output = output.reset_index(drop = True)
    output.loc[(output['AI388'] > 10) | (output['AI388'] < 0), 'AI388'] = np.nan
    
    return output

# =============================================================================
# 
# =============================================================================
import cv2
def NNstd(data):
    h = np.ones((3,3))
    n = h.sum()
    n1 = n - 1
    c1 = cv2.filter2D(data ** 2, -1, h / n1, borderType = cv2.BORDER_REFLECT)
    c2 = cv2.filter2D(data, -1, h, borderType = cv2.BORDER_REFLECT) ** 2 / (n * n1)
    d = c1 - c2
    d[d < 0] = 0
    J = np.sqrt(d)
    return J

def readOMIAerNN(dates, ROI, SSA):
    try:
        startdate, enddate = dates[0], dates[1]
    except:
        startdate = enddate = dates[0]
    
    dates = pd.date_range(startdate, enddate)
    
    output = pd.DataFrame()
    
    directory = '/nobackup_1/users/sunj/reprocess_global_OMIAerosolRetrieval_NN/OMIMODISCloudMergedCF0.2_SZA65.0_DIST15.0km/'    
    for i, idate in enumerate(dates): 
        path = directory + 'SigmoidLayer_StrictOptimization400Epochs_PybrainV20151203_AerSSA%s/BestOptimizedNN/%4i/%02i/' % (str(SSA), idate.year, idate.month)
        filelist = glob.glob( path + '*.nc')
        
        sublist = []
        for ifile in sorted(filelist):
            idx = ifile.find('AerNN_')
            if '%4im%02i%02i' % (idate.year, idate.month, idate.day) == ifile[idx + 6: idx + 15]:
                sublist.append(ifile)
        if len(sublist) == 0:
            print('Warning: no data on %4i-%02i-%02i' % (idate.year, idate.month, idate.day))
        else:
            temp_day = pd.DataFrame()
            for ifile in sorted(sublist): 
                
                idx = ifile.find('-o')
                orbit = ifile[idx + 2: idx + 7]
                sys.stdout.write('\r # Reading OMIAerNN %4i-%02i-%02i %s' % (idate.year, idate.month, idate.day, orbit))
                                 
                temp = pd.DataFrame()
                data = netCDF4.Dataset(ifile)
                temp['lat'] = data['Latitude'][:]
                temp['lon'] = data['Longitude'][:]
                temp['ALP_MDT'] = data['ALP_MODISAOT550DT'][:]
                temp['ALP_MDTDB'] = data['ALP_MODISAOT550DTDB'][:]
                temp['ALP_OMI'] = data['ALP_OMIAOT'][:]
                temp['AOT'] = data['AOT'][:]
                temp['AOD_MODIS'] = data['MODIS_AOT_550_nm_DTDBland&ocean'][:]
                temp['CF'] = data['MODIS_Cloud_Fraction_Final'][:]
                temp['OMMYDCLD'] = data['MODISCloudFractionOMMYDCLD'][:]
                temp['As'] = data['TerrainReflectivity'][:]
                temp['timeStamp'] = data['Time'][:]
                temp['sza'] = data['SolarZenithAngle'][:]
                temp['saa'] = data['SolarAzimuthAngle'][:]
                temp['vza'] = data['ViewingZenithAngle'][:]
                temp['vaa'] = data['ViewingAzimuthAngle'][:]
                temp['H0'] = data['TerrainHeight'][:]
                temp['SCDO2'] = data['SlantColumnAmountO2O2'][:]
                temp['ALP_MDTDB'][temp['ALP_MDTDB'] > 100]
                data.close()
                
                def timeStamp2dateTime(ts):
                    mTime = datetime.datetime(1993, 1, 1, 0, 0) + datetime.timedelta(seconds = ts)
                    return pd.to_datetime(mTime)
                
                dateTime = list(map(timeStamp2dateTime, temp['timeStamp']))
                temp['dateTime'] = dateTime
                    
                temp_day = temp_day.append(temp)
            output = output.append(temp_day)
        if len(output) > 0:
            mask = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E'])
            output = output[mask]
            output = output.reset_index(drop = True)
# =============================================================================
#     convert from pressure to km
# =============================================================================
            for ikey in ['MDT', 'MDTDB', 'OMI']:
                output['ALH_%s' % ikey] = -8 * np.log(output['ALP_%s' % ikey] / 1013.) - output.H0
        else:
            pass
    return output        

# =============================================================================
#         
# =============================================================================

def main():                    
    from mpl_toolkits.basemap import Basemap
    ROI = {'S':-75, 'N': 75, 'W': -180, 'E': 180}
    #ROI = {'S': -30, 'N': 0, 'W': -20, 'E': 40}
    
    startdate = '%4i-%02i-%02i' % (2016, 1, 1)
    enddate   = '%4i-%02i-%02i' % (2016, 12, 31)
    dates = pd.date_range(startdate, enddate)
    AIplume = 0.
    
    t0 = time.time()
    dAI = readTEMISOMI(startdate, enddate, ROI,  AIplume)
    
    t1 = time.time()
    t2 = time.time()
    coords = np.array(dAI.columns.values.tolist())
    print('Time for read MS data: % 1.4f' % (t1 - t0))
    print('Time for processing MS data into time series: % 1.4f' % (t2 - t1))
    
    
    plt.figure(figsize = (8, 5))
    map = Basemap(llcrnrlon= ROI['W'],llcrnrlat=ROI['S'],urcrnrlon=ROI['E'],urcrnrlat=ROI['N'], lat_0 = 0, lon_0 = 0, projection='cyl',resolution='l')
    map.drawcoastlines(color='black',linewidth=0.6)
    map.drawcountries(color = 'black',linewidth=0.4)
    map.drawmeridians(np.arange(ROI['W'], ROI['E'], 30),labels = [0,1,1,0])
    map.drawparallels(np.arange(ROI['S'], ROI['N'], 30), labels = [0,1,1,0])
    plt.scatter(coords[:, 1], coords[:, 0], c = dAI.loc[dates[40]], s = 4, marker = 's', cmap = 'rainbow', vmin = -2, vmax = 2) 
    plt.colorbar()


if __name__ == '__main__':
    dAI = main()



    
    
