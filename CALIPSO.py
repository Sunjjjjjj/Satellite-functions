# -*- coding: utf-8 -*-
"""
Read CALIPSO L2 aerosol profile data 
- proCALIPSO: return AOD profile, extinction coefficient profile and pressure level of the nearest point from (lat0, lon0)

@author: sunj
"""


######### load python packages
import sys, os
import shutil
import time
import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 
import glob
from scipy import ndimage
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import ListedColormap
from scipy.optimize import leastsq
from math import sqrt
from scipy import spatial 
import matplotlib.colors as colors



    
def proCALIOP(year, month, day, ROI, lat0, lon0, nlay): 
    
    latcal = []
    loncal = []
    bs532cal = []
    bs1064cal = [] 
    ext532cal = []
    ext1064cal = [] 
    prescal = []   
    tempcal = []
    z0cal = []     
    
    
#    print '**** Reading CALIOP %02i-%02i-%04i' %(day,month,year)  
    
    path = '/nobackup/users/sunj/CALIPSO/L2-05kmAPro_Prov-V3-40/%4i/%02i/%02i/' %(year, month, day)
#    path = '/nobackup/users/sunj/CALIPSO/CAL_LID_L1-Standard-V4-10/%4i/%02i/%02i/' %(year, month, day)
    filelist = glob.glob( path + '*.hdf')
    
    
    for io in filelist[0:1]: 
        # reading file
        prodata = netCDF4.Dataset(io,'r')
       
        lat = prodata.variables['Latitude'][:,1]
        lon = prodata.variables['Longitude'][:,1]
        
        bs532 = prodata.variables['Total_Backscatter_Coefficient_532'][:]
        bs1064 = prodata.variables['Backscatter_Coefficient_1064'][:]
        ext532 = prodata.variables['Extinction_Coefficient_532'][:]
        ext1064 = prodata.variables['Extinction_Coefficient_1064'][:]

        
        pres = prodata.variables['Pressure'][:]                                 # unit: hPa
        temp = prodata.variables['Temperature'][:]+273.15                       # unit: K
        z0 = prodata.variables['Surface_Elevation_Statistics'][:,2]*1e3         # idx =2 mean surface elevation, unit: m 
        
        prodata.close() 
        
        # select region of interest 
        idx = np.where((lat > ROI[0]) & (lat < ROI[1]) & (lon > ROI[2]) & (lon < ROI[3]))[0]
    
        latcal = latcal + list(lat[idx])
        loncal = loncal + list(lon[idx])       
        bs532cal = bs532cal + list(bs532[idx,:])
        bs1064cal = bs1064cal + list(bs1064[idx,:])        
        ext532cal = ext532cal + list(ext532[idx,:])
        ext1064cal = ext1064cal + list(ext1064[idx,:])        
        prescal = prescal + list(pres[idx,:])
        tempcal = tempcal + list(temp[idx,:])
        z0cal = z0cal + list(z0[idx])
        
    
    latarr = np.asarray(np.array(latcal)).reshape(-1)
    lonarr = np.asarray(np.array(loncal)).reshape(-1)   
    bs532arr = np.array(bs532cal)
    bs1064arr = np.array(bs1064cal)
    ext532arr = np.array(ext532cal)
    ext1064arr = np.array(ext1064cal)
    presarr = np.array(prescal)
    temparr = np.array(tempcal)
    z0arr = np.array(z0cal)
    
    # convert extinction to AOD 
    L = -0.0065         # lapse rate, unit: K/m
    R = 8.31432         # gas constant 
    g0 = 9.80665        # gravitational accelaration constant
    M = 0.0289644       # air molar mass, unit: kg/mol
    T0 = temparr[:,-1]     # surface temperature, unit: K 
    P0 = presarr[:,-1]     # surface pressure, unit: hPa
    
    # pressure to altitude 
    z = z0arr.reshape((len(z0arr),1)) + T0.reshape((len(T0),1))/L *((presarr/P0.reshape((len(P0),1)))**(-R*L/g0/M)-1)  # unit: m 
    
    # calculate mid-level of each layer    
    z1 = np.zeros(z.shape)
    z1[:,1:] = z[:,:-1]
    z1[:,0] = z[:,0]
    Z = (z + z1)/2
    dz = z1 - z     
    
    # calculate mid-level extinction (mean)
    ext532arr1 = ext532arr
    ext532arr1[:,1:] = ext532arr[:,:-1]
    ext532 = (ext532arr + ext532arr1)/2/1e3         # unit: m^-1

    ext1064arr1 = ext1064arr
    ext1064arr1[:,1:] = ext1064arr[:,:-1]
    ext1064 = (ext1064arr + ext1064arr1)/2/1e3         # unit: m^-1
    
    # convert extinction to AOD 
    AOD = ext532 * dz
    AOD1064 = ext1064 * dz
    AOD[AOD<0] = 0.
    bs532arr[bs532arr<0] = 0.
    ext532arr[ext532arr<0] = 0.
    bs1064arr[bs1064arr<0] = 0.
    ext1064arr[ext1064arr<0] = 0.
    
    # altitude to pressure
    pres_mid = P0.reshape((len(P0),1)) * (1+L/T0.reshape((len(T0),1)) * (Z-z0arr.reshape((len(z0arr),1)))) ** (-g0*M/R/L)    # unit: hPa
    
    # vertical gridding 
#    pres_grid = np.logspace(0,np.log10(pres_mid.max()),100)    # logarithm interpolation
#    pres_grid = np.linspace(0,1013.0,100)               # linear interpolation  
    pres_grid = np.arange(0,1013.0,1)               # linear interpolation  
#    pres_grid = np.array([1,  100, 300, 400, 500, 600, 650, 700, 750, 800, 850, 900, 950, 1000])         # CAMS pressure level 
    z_grid = z0arr.reshape((len(z0arr),1)) + T0.reshape((len(T0),1))/L *((pres_grid/P0.reshape((len(P0),1)))**(-R*L/g0/M)-1)  # unit: m 
    
    AOD_grid = np.zeros([len(latarr),len(pres_grid)])
    bs532_grid = np.zeros([len(latarr),len(pres_grid)])
    ext532_grid = np.zeros([len(latarr),len(pres_grid)])
    bs1064_grid = np.zeros([len(latarr),len(pres_grid)])
    ext1064_grid = np.zeros([len(latarr),len(pres_grid)])
    
    for i in range(len(latarr)): 
        AOD_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], AOD[i,:]) 
        bs532_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], bs532arr[i,:]) 
        ext532_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], ext532arr[i,:])
        bs1064_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], bs1064arr[i,:]) 
        ext1064_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], ext1064arr[i,:])
        
    
    ext532arr[ext532arr<0] = 0.
    bs532arr[bs532arr<0] = 0.
    ext1064arr[ext1064arr<0] = 0.
    bs1064arr[bs1064arr<0] = 0.
    S = ext532arr / (bs532arr + 1.0e-10)    # in case divided by zero
    
    # slab averaged profile
    DSMaodprom = AOD_grid.mean(axis = 0)
    
    # find the nearest profile to (lat0, lon0)
    tree = spatial.KDTree(list(zip(latarr, lonarr)))
    locs = tree.query([lat0, lon0], k = 2)
    locs = np.array(locs)
    # sort from min to max 
    locs = np.sort(locs, axis = 0)    
    
    count = 0
    j = int(locs[1, count])
    
    # if the nearest point is zero profile, then find the next nearest point 
    while(np.where(AOD[j,:] > 0)[0].shape[0] < nlay): 
        count += 1
        locs = tree.query([lat0, lon0], k = count+1)
        locs = np.sort(locs, axis = 0)
        locs = np.array(locs)
        j = int(locs[1, count])
    
    # find the layers with largest nlay AOD 
    idx = np.where(AOD[j,:]>0)[0]
    if len(idx) >= nlay:
        idx = AOD[j,:].argsort()[-nlay:] 
        idx.sort()
        
    idxm = np.where(DSMaodprom>0)[0]
    if len(idxm) >= nlay:
        idxm = DSMaodprom.argsort()[-nlay:] 
        idxm.sort()

    
    DSMaerpro = np.zeros(presarr[j,idx].shape)
    DSMaerpro = AOD[j,idx]
    
    
    z1 = z[j,idx]
    z2 = np.zeros(z1.shape)
    z2[:-1] = z1[1:]
    DSMdz = z1 - z2

    DSMnormpro = DSMaerpro / DSMaerpro.sum()     
    DSMnormprom = DSMaodprom[idxm] / DSMaodprom[idxm].sum()     
#    DSMnormpro = DSMaerpro / DSMdz / DSMaerpro.sum() 

    
##    # verification    
#    plt.figure(figsize=(12,4))
#    plt.subplot(1,4,1)
#    plt.plot(ext532arr[j,:],presarr[j], 'k')
#    plt.gca().invert_yaxis()
#    plt.xlabel('Extinction [m^-1]')
#    plt.ylabel('Pressure [hPa]')
#
#    plt.subplot(1,4,2)
#    plt.plot(bs532arr[j,:],presarr[j],'k')
#    plt.gca().invert_yaxis()
#    plt.xlabel('Backscatter [-]')
#
#    plt.subplot(1,4,3)
#    plt.plot(S[j,:],presarr[j],'k')
#    plt.ylim(0,1200)
#    plt.gca().invert_yaxis()
#    plt.xlabel('Lidar ratio [m^-1]')
#
#    plt.subplot(1,4,4)   
#    plt.plot(AOD[j,:],presarr[j,:],'k-', label = '@ 532 nm')
#    plt.plot(AOD1064[j,:],presarr[j,:],'b-', label = '@ 1064 nm')
#    plt.plot(AOD_grid[j,:],pres_grid,'g.', label = 'gridded')
#    plt.plot(DSMaerpro,presarr[j,idx],'r.', label = 'DSM')
#    plt.xlim([0,.025])
#    plt.xlabel('AOD') 
#    plt.ylabel('Pressure [hPa]')
#    plt.gca().invert_yaxis()
#    plt.legend(frameon = False) 
##
#    plt.figure(figsize = (2.5,4))
#    plt.plot(DSMnormpro,presarr[j,idx],'r.-', label = 'norm')
#    plt.plot(DSMnormprom,pres_grid[idxm],'k.-', label = 'mean norm')
#    plt.xlim([0,1])
#    plt.ylim([0,1013])
#    plt.xlabel('AOD') 
#    plt.gca().invert_yaxis()
#    plt.legend(frameon = False)     
##    
#    plt.figure(figsize = (3,3))
#    plt.scatter(lon0,lat0,s = 8, c = 'b', marker = 'o',linewidth = 0)
#    plt.scatter(lonarr,latarr,s = 8, c = 'darkgray', marker = 'o',linewidth = 0)
#    plt.scatter(lonarr[j],latarr[j],s = 12, c = 'r', marker = 'o',linewidth = 0)

    return latarr, lonarr, bs532_grid.transpose(), ext532_grid.transpose(),  z_grid[0,:].transpose(), presarr[j,idx][::-1], DSMnormpro[::-1], pres_grid[idxm][::-1], DSMnormprom[::-1]



def readCALIOP(year, month, day, ROI, data): 
    lattot = []
    lontot = []
    bs532tot = []
    bs1064tot = []
    ztot = [] 
    output = {}
# =============================================================================
#     
# =============================================================================
    print('**** Reading CALIOP '+data+' %02i-%02i-%04i' %(day,month,year))  
    if data == 'L2': 
        path = '/nobackup/users/sunj/CALIPSO/LID_L2_05kmAPro-Standard-V4-20/%4i/%02i/%02i/' %(year, month, day)
        filelist = glob.glob( path + '*.hdf')
        
        
        for inum, io in enumerate(filelist[:]): 
            temp = {}
            latcal = []
            loncal = []
            bs532cal = []
            bs1064cal = [] 
            ext532cal = []
            ext1064cal = [] 
            prescal = []   
            tempcal = []
            z0cal = []     
    
            prodata = netCDF4.Dataset(io,'r')
            lat = prodata.variables['Latitude'][:,1]
            lon = prodata.variables['Longitude'][:,1]
            bs532 = prodata.variables['Total_Backscatter_Coefficient_532'][:]
            bs1064 = prodata.variables['Backscatter_Coefficient_1064'][:]
            ext532 = prodata.variables['Extinction_Coefficient_532'][:]
            ext1064 = prodata.variables['Extinction_Coefficient_1064'][:]
            pres = prodata.variables['Pressure'][:]                                 # unit: hPa
            T = prodata.variables['Temperature'][:]+273.15                       # unit: K
#            z0 = prodata.variables['Surface_Elevation_Statistics'][:,2]*1e3         # idx =2 mean surface elevation, unit: m 
            P0 = np.nanmax(pres, axis = 1)
            T0 = np.nanmax(T, axis = 1)
            z0 = np.nanmean(prodata.variables['Surface_Elevation_Statistics'], axis = 1) * 1e3
            

            T[T < 0] = np.nan
            pres[pres < 0] = np.nan
            
            prodata.close() 
            
            
            # select region of interest 
            idx = np.where((lat > ROI['S']) & (lat < ROI['N']) & (lon > ROI['W']) & (lon < ROI['E']))[0]
            
            if idx.size != 0: 
#        
#                latcal = latcal + list(lat[idx])
#                loncal = loncal + list(lon[idx])       
#                bs532cal = bs532cal + list(bs532[idx,:])
#                bs1064cal = bs1064cal + list(bs1064[idx,:])        
#                ext532cal = ext532cal + list(ext532[idx,:])
#                ext1064cal = ext1064cal + list(ext1064[idx,:])        
#                prescal = prescal + list(pres[idx,:])
#                tempcal = tempcal + list(T[idx,:])
#                z0cal = z0cal + list(z0[idx])
#                
#            
#                latarr = np.asarray(np.array(latcal)).reshape(-1)
#                lonarr = np.asarray(np.array(loncal)).reshape(-1)   
#                bs532arr = np.array(bs532cal)
#                bs1064arr = np.array(bs1064cal)
#                ext532arr = np.array(ext532cal)
#                ext1064arr = np.array(ext1064cal)
#                presarr = np.array(prescal)
#                temparr = np.array(tempcal)
#                z0arr = np.array(z0cal)
#                
                # convert extinction to AOD 
                L = -0.0065         # lapse rate, unit: K/m
                R = 8.31432         # gas constant 
                g0 = 9.80665        # gravitational accelaration constant
                M = 0.0289644       # air molar mass, unit: kg/mol
                    # surface temperature, unit: K 
                     # surface pressure, unit: hPa
                
                
                
#                # pressure to altitude 
                z = z0.reshape((len(z0),1)) + T0.reshape((len(T0),1))/L *((pres/P0.reshape((len(P0),1)))**(-R*L/g0/M)-1)  # unit: m 
#                
#                # calculate mid-level of each layer    
#                z1 = np.zeros(z.shape)
#                z1[:,1:] = z[:,:-1]
#                z1[:,0] = z[:,0]
#                Z = (z + z1)/2
#                dz = z1 - z     
#                
#                # calculate mid-level extinction (mean)
#                ext532arr1 = ext532arr
#                ext532arr1[:,1:] = ext532arr[:,:-1]
#                ext532 = (ext532arr + ext532arr1)/2/1e3         # unit: m^-1
#            
#                ext1064arr1 = ext1064arr
#                ext1064arr1[:,1:] = ext1064arr[:,:-1]
#                ext1064 = (ext1064arr + ext1064arr1)/2/1e3         # unit: m^-1
#                
#                # convert extinction to AOD 
#                AOD = ext532 * dz
#                AOD1064 = ext1064 * dz
#                AOD[AOD<0] = 0.
#                bs532arr[bs532arr<0] = 0.
#                ext532arr[ext532arr<0] = 0.
#                bs1064arr[bs1064arr<0] = 0.
#                ext1064arr[ext1064arr<0] = 0.
#                
#                # altitude to pressure
#                pres_mid = P0.reshape((len(P0),1)) * (1+L/T0.reshape((len(T0),1)) * (Z-z0arr.reshape((len(z0arr),1)))) ** (-g0*M/R/L)    # unit: hPa
#                
#                # vertical gridding 
#            #    pres_grid = np.logspace(0,np.log10(pres_mid.max()),100)    # logarithm interpolation
#            #    pres_grid = np.linspace(0,1013.0,100)               # linear interpolation  
#                pres_grid = np.arange(0,1013.0,1)               # linear interpolation  
#            #    pres_grid = np.array([1,  100, 300, 400, 500, 600, 650, 700, 750, 800, 850, 900, 950, 1000])         # CAMS pressure level 
#                z_grid = z0arr.reshape((len(z0arr),1)) + T0.reshape((len(T0),1))/L *((pres_grid/P0.reshape((len(P0),1)))**(-R*L/g0/M)-1)  # unit: m 
#                
#                AOD_grid = np.zeros([len(latarr),len(pres_grid)])
#                bs532_grid = np.zeros([len(latarr),len(pres_grid)])
#                ext532_grid = np.zeros([len(latarr),len(pres_grid)])
#                bs1064_grid = np.zeros([len(latarr),len(pres_grid)])
#                ext1064_grid = np.zeros([len(latarr),len(pres_grid)])
#                
#                for i in range(len(latarr)): 
#                    AOD_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], AOD[i,:]) 
#                    bs532_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], bs532arr[i,:]) 
#                    ext532_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], ext532arr[i,:])
#                    bs1064_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], bs1064arr[i,:]) 
#                    ext1064_grid[i,:] = np.interp(pres_grid, pres_mid[i,:], ext1064arr[i,:])
                    
                ext532[ext532<0] = np.nan
                bs532[bs532<0] = np.nan
                
#                lattot.append(latarr)
#                lontot.append(lonarr)
#                bs532tot.append(bs532_grid.transpose())
#                ztot.append(z_grid[0,:].transpose()/1e3)
                
                temp['lat'] = lat[idx]
                temp['lon'] = lon[idx]
                temp['z'] = z[idx] / 1e3
                temp['bs532'] = bs532[idx]
                temp['ext532'] = ext532[idx]
            output['track%02i' % (inum + 1)] = temp
    
    if data == 'L1': 
        path = '/nobackup/users/sunj/CALIPSO/CAL_LID_L1-Standard-V4-10/%4i/%02i/%02i/' %(year, month, day)
        filelist = glob.glob( path + '*.hdf')
        
        
        for inum, io in enumerate(filelist[:]): 
            # reading file
        

            temp = {}
            latcal = []
            loncal = []
            bs532cal = []
            bs1064cal = [] 
   


            prodata = netCDF4.Dataset(io,'r')
            
            
            lat = prodata.variables['Latitude'][:]
            lon = prodata.variables['Longitude'][:]
            bs532 = prodata.variables['Total_Attenuated_Backscatter_532'][:]
            bs1064 = prodata.variables['Attenuated_Backscatter_1064'][:]
    
            pres = prodata.variables['Pressure'][:]                                 # unit: hPa
            Temp = prodata.variables['Temperature'][:]+273.15                       # unit: K
            z0 = prodata.variables['Surface_Elevation'][:]*1e3         # unit: m 
            prodata.close() 
    
            
            # select region of interest 
            idx = np.where((lat > ROI['S']) & (lat < ROI['N']) & (lon > ROI['W']) & (lon < ROI['E']))[0]
            
            if idx.size != 0: 
        
                latcal = latcal + list(lat[idx])
                loncal = loncal + list(lon[idx])       
                bs532cal = bs532cal + list(bs532[idx,:])
                bs1064cal = bs1064cal + list(bs1064[idx,:])
                
            
                latarr = np.asarray(np.array(latcal)).reshape(-1)
                lonarr = np.asarray(np.array(loncal)).reshape(-1)   
                bs532arr = np.array(bs532cal)
                bs1064arr = np.array(bs1064cal)
                
                # pressure to altitude 
                z1 = np.ones(33) * 300.
                z1 = 30.1 + z1.cumsum() * 1e-3
                
                z2 = np.ones(55) * 180.
                z2 = 20.2 + z2.cumsum() * 1e-3
                
                z3 = np.ones(200) * 59.5
                z3 = 8.3 + z3.cumsum() * 1e-3
                
                z4 = np.ones(290) * 30.
                z4 = -0.5 + z4.cumsum() * 1e-3             
                
                z5 = np.ones(5) * 300.
                z5 = -2 + z5.cumsum() * 1e-3
                
                z = list(z5) + list(z4) + list(z3) + list(z2) + list(z1) 
                z = np.array(z[::-1]) 
                
                bs532arr[bs532arr<0] = np.nan
                bs1064arr[bs1064arr<0] = np.nan
                
#                lattot.append(latarr)
#                lontot.append(lonarr)
#                bs532tot.append(bs532arr.transpose())
#                bs1064tot.append(bs1064arr.transpose())
                temp['lat'] = latarr
                temp['lon'] = lonarr
                temp['z'] = z
                temp['BS532'] = bs532arr
                temp['BS1064'] = bs1064arr
            output['track%02i' % (inum + 1)] = temp
    return output







def main(): 
    year = 2017
    month = 12
    day = 12
    ROI = {'S':25, 'N': 50, 'W': -130, 'E': -110}
    dataCLP = readCALIOP(year, month, day, ROI, 'L1')
if __name__ == '__main__':
    main()