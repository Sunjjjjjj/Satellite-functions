# -*- coding: utf-8 -*-
"""
Read Metop A&B GOME2 data 
- rawGOME2: return Metop A&B data, organized by list
- gridGOME2: return gridded data over ROI, organized by np.2darray

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
#from matplotlib.mlab import griddata
from scipy.interpolate import griddata
from otherFunctions import date2DOY, DOY2date, dateList
import h5py

def rawGOME2(year, month, day): 

    print('**** Reading GOME2 %02i-%02i-%04i' %(day,month,year)) 
    
    pathA = '/nobackup/users/sunj/GOME-2/metopa/%4i/%02i/%02i/' %(year, month, day)
    filelistA = glob.glob( pathA + '*.hdf5')
    pathB = '/nobackup/users/sunj/GOME-2/metopb/%4i/%02i/%02i/' %(year, month, day)
    filelistB = glob.glob( pathB + '*.hdf5')


    latA = []
    lonA = []
    aaiA = []     
    for io in filelistA[:]:  

        dataA = netCDF4.Dataset(io,'r')    
        data = dataA.groups['DATA']
        geo = dataA.groups['GEOLOCATION']

        lat = geo.variables['LatitudeCenter'][:]
        lon = geo.variables['LongitudeCenter'][:]
        aai = data.variables['AAI'][:]
        sza = geo.variables['SolarZenithAngle'][:]
        sunglint = data.variables['SunGlintFlag'][:]
        
        
        mask = np.logical_or(sza[0:-1,0:-1] < 0. , sza[0:-1,0:-1] > 75.)  
        mask = np.logical_or(mask, sunglint[0:-1,0:-1] < 0.)                # sunglint pixel
        mask = np.logical_or(mask, aai[0:-1,0:-1] < 0.)                     # scattering aerosol pixel  
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel 
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
        
        
        if mask.all() != True: 
            lon = np.ma.masked_array(lon[0:-1,0:-1],mask)
            lat = np.ma.masked_array(lat[0:-1,0:-1],mask)
            aai = np.ma.masked_array(aai[0:-1,0:-1],mask)  
            
            latA.append(lat)  
            lonA.append(lon)
            aaiA.append(aai)
            

    latB = []
    lonB = [] 
    aaiB = [] 
    for io in filelistB[:]: 

        dataB = netCDF4.Dataset(io,'r')
        data = dataB.groups['DATA']
        geo = dataB.groups['GEOLOCATION']


        lat = geo.variables['LatitudeCenter'][:]
        lon = geo.variables['LongitudeCenter'][:]    
        aai = data.variables['AAI'][:]
        sza = geo.variables['SolarZenithAngle'][:]
        sunglint = data.variables['SunGlintFlag'][:]
        
        
        mask = np.logical_or(sza[0:-1,0:-1] < 0. , sza[0:-1,0:-1] > 75.)  
        mask = np.logical_or(mask, sunglint[0:-1,0:-1] < 0.)                # sunglint pixel
        mask = np.logical_or(mask, aai[0:-1,0:-1] < 0.)                     # scattering aerosol pixel  
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel 
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
        
        
        if mask.all() != True: 
            lon = np.ma.masked_array(lon[0:-1,0:-1],mask)
            lat = np.ma.masked_array(lat[0:-1,0:-1],mask)
            aai = np.ma.masked_array(aai[0:-1,0:-1],mask)  
            
            latB.append(lat)
            lonB.append(lon)
            aaiB.append(aai)
            
    return latA, lonA, aaiA, latB, lonB, aaiB 
            
            








#    # GOME
def gridGOME2(year, month, day, ROI, res):
    print('**** Reading GOME2 %02i-%02i-%04i' %(day,month,year)) 
    
    pathA = '/nobackup/users/sunj/GOME-2/metopa/%4i/%02i/%02i/' %(year, month, day)
    filelistA = glob.glob( pathA + '*.hdf5')
    pathB = '/nobackup/users/sunj/GOME-2/metopb/%4i/%02i/%02i/' %(year, month, day)
    filelistB = glob.glob( pathB + '*.hdf5')
  
    
 
    aaiA = []
    latA = []
    lonA = [] 
    maskA = []   
    raaA = [] 
    scaA = []
    
    for io in filelistA[:]:  

        dataA = netCDF4.Dataset(io,'r')    
        data = dataA.groups['DATA']
        geo = dataA.groups['GEOLOCATION']

        lat = geo.variables['LatitudeCenter'][:]
        lon = geo.variables['LongitudeCenter'][:]
        aai = data.variables['AAI'][:]
        sza = geo.variables['SolarZenithAngle'][:]
        sunglint = data.variables['SunGlintFlag'][:]  
        saa = geo.variables['SolarAzimuthAngle'][:]
        vaa = geo.variables['LineOfSightAzimuthAngle'][:]
        sca = geo.variables['ScatteringAngle'][:]
        
        raa = saa + 180.- vaa 
        
        mask = np.logical_or(sza[0:-1,0:-1] < 0. , sza[0:-1,0:-1] > 75.)  
        mask = np.logical_or(mask, lat[0:-1,0:-1]<ROI[0])
        mask = np.logical_or(mask, lat[0:-1,0:-1]>ROI[1])
        mask = np.logical_or(mask, lon[0:-1,0:-1]<ROI[2])
        mask = np.logical_or(mask, lon[0:-1,0:-1]>ROI[3])
        mask = np.logical_or(mask, sunglint[0:-1,0:-1] < 0.)                # sunglint pixel
        mask = np.logical_or(mask, aai[0:-1,0:-1] < 0.)                     # scattering aerosol pixel  
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel 
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
#        mask = np.logical_or(mask, aai[0:-1,0:-1]<5.)
        
        latA = latA + list(np.asarray(lat[0:-1,0:-1]).reshape(-1))
        lonA = lonA + list(np.asarray(lon[0:-1,0:-1]).reshape(-1))
        aaiA = aaiA + list(np.asarray(aai[0:-1,0:-1]).reshape(-1))
        raaA = raaA + list(np.asarray(raa[0:-1,0:-1]).reshape(-1))
        scaA = scaA + list(np.asarray(sca[0:-1,0:-1]).reshape(-1))
        maskA = maskA + list(np.asarray(mask).reshape(-1))
        
        dataA.close()

    latarrA = np.array(latA)
    lonarrA = np.array(lonA)    
    aaiarrA = np.array(aaiA) 
    raaarrA = np.array(raaA)
    scaarrA = np.array(scaA)
    maskarrA = np.array(maskA)       
    
    aaiarrA = np.ma.masked_array(aaiarrA,maskarrA)

    latvldA = latarrA[aaiarrA.mask == 0] 
    lonvldA = lonarrA[aaiarrA.mask == 0] 
    aaivldA = aaiarrA[aaiarrA.mask == 0]     
    raavldA = raaarrA[aaiarrA.mask == 0]     
    scavldA = scaarrA[aaiarrA.mask == 0]     


 
    aaiB = []
    latB = []
    lonB = [] 
    maskB = []
    raaB = [] 
    scaB = [] 
    
    for io in filelistB[:]:  

        dataB = netCDF4.Dataset(io,'r')    
        data = dataB.groups['DATA']
        geo = dataB.groups['GEOLOCATION']

        lat = geo.variables['LatitudeCenter'][:]
        lon = geo.variables['LongitudeCenter'][:]
        aai = data.variables['AAI'][:]
        sza = geo.variables['SolarZenithAngle'][:]
        sunglint = data.variables['SunGlintFlag'][:]  
        saa = geo.variables['SolarAzimuthAngle'][:]
        vaa = geo.variables['LineOfSightAzimuthAngle'][:]
        sca = geo.variables['ScatteringAngle'][:]
        
        raa = saa + 180.- vaa 
        
        mask = np.logical_or(sza[0:-1,0:-1] < 0. , sza[0:-1,0:-1] > 75.)  
        mask = np.logical_or(mask, lat[0:-1,0:-1]<ROI[0])
        mask = np.logical_or(mask, lat[0:-1,0:-1]>ROI[1])
        mask = np.logical_or(mask, lon[0:-1,0:-1]<ROI[2])
        mask = np.logical_or(mask, lon[0:-1,0:-1]>ROI[3])
        mask = np.logical_or(mask, sunglint[0:-1,0:-1] < 0.)                # sunglint pixel
        mask = np.logical_or(mask, aai[0:-1,0:-1] < 0.)                     # scattering aerosol pixel  
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel 
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
#        mask = np.logical_or(mask, aai[0:-1,0:-1]<5.)
        
        latB = latB + list(np.asarray(lat[0:-1,0:-1]).reshape(-1))
        lonB = lonB + list(np.asarray(lon[0:-1,0:-1]).reshape(-1))
        aaiB = aaiB + list(np.asarray(aai[0:-1,0:-1]).reshape(-1))
        raaB = raaB + list(np.asarray(raa[0:-1,0:-1]).reshape(-1))
        scaB = scaB + list(np.asarray(sca[0:-1,0:-1]).reshape(-1))
        maskB = maskB + list(np.asarray(mask).reshape(-1))

    latarrB = np.array(latB)
    lonarrB = np.array(lonB)    
    aaiarrB = np.array(aaiB) 
    raaarrB = np.array(raaB) 
    scaarrB = np.array(scaB) 
    maskarrB = np.array(maskB)       
    
    aaiarrB = np.ma.masked_array(aaiarrB,maskarrB)

    latvldB = latarrB[aaiarrB.mask == 0] 
    lonvldB = lonarrB[aaiarrB.mask == 0] 
    aaivldB = aaiarrB[aaiarrB.mask == 0]        
    raavldB = raaarrB[aaiarrB.mask == 0]        
    scavldB = scaarrB[aaiarrB.mask == 0]
    
    latvld = np.concatenate((latvldA, latvldB))
    lonvld = np.concatenate((lonvldA, lonvldB))
    aaivld = np.concatenate((aaivldA, aaivldB))
    raavld = np.concatenate((raavldA, raavldB))
    scavld = np.concatenate((scavldA, scavldB))
    
    latnew = np.arange(ROI[1], ROI[0], -res)
    lonnew = np.arange(ROI[2], ROI[3], res)
    
    x,y = np.meshgrid(lonnew,latnew)
    aainew = griddata((lonvld, latvld), aaivld, (x, y), method = 'linear')
    aainew[np.isnan(aainew)] = -32767.
    
#    raavld[raavld < 0] = 360 + raavld[raavld<0]
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.scatter(raavldA, aaivldA, c ='k')
#    plt.scatter(raavldB, aaivldB, c ='r')
#    plt.subplot(1,2,2)
#    plt.scatter(scavldA, aaivldA, c = 'k')    
#    plt.scatter(scavldB, aaivldB, c = 'r')    
    
    return latnew, lonnew, aainew 





def GOME24cfg(year, month, day, ROI, jetlag, plumemsk, crival, res):     
    
    print('**** Reading GOME2 %02i-%02i-%04i' %(day,month,year)) 
    
    pathA = '/nobackup/users/sunj/GOME-2/metopa/%4i/%02i/%02i/' %(year, month, day)
    filelistA = glob.glob( pathA + '*.hdf5')
    pathB = '/nobackup/users/sunj/GOME-2/metopb/%4i/%02i/%02i/' %(year, month, day)
    filelistB = glob.glob( pathB + '*.hdf5')
  
    
 
    aaiA = []
    latA = []
    lonA = [] 
    maskA = []   
    raaA = [] 
    scaA = []
    
    for io in filelistA[:]:  

        dataA = netCDF4.Dataset(io,'r')    
        data = dataA.groups['DATA']
        geo = dataA.groups['GEOLOCATION']

        lat = geo.variables['LatitudeCenter'][:]
        lon = geo.variables['LongitudeCenter'][:]
        aai = data.variables['AAI'][:]
        sunglint = data.variables['SunGlintFlag'][:]  
        
        saa = geo.variables['SolarAzimuthAngle'][:]
        sza = geo.variables['SolarZenithAngle'][:]
        vaa = geo.variables['LineOfSightAzimuthAngle'][:]
        vza = geo.variables['LineOfSightZenithAngle'][:]
        sca = geo.variables['ScatteringAngle'][:]
        raa = geo.variables['RelAzimuthAngle'][:]
        raa1 = saa + 180.- vaa 
        
        print((raa - raa1).mean())          
        
        mask = np.logical_or(sza[0:-1,0:-1] < 0. , sza[0:-1,0:-1] > 75.)  
        mask = np.logical_or(mask, lat[0:-1,0:-1]<ROI[0])
        mask = np.logical_or(mask, lat[0:-1,0:-1]>ROI[1])
        mask = np.logical_or(mask, lon[0:-1,0:-1]<ROI[2])
        mask = np.logical_or(mask, lon[0:-1,0:-1]>ROI[3])
        mask = np.logical_or(mask, sunglint[0:-1,0:-1] < 0.)                # sunglint pixel
        mask = np.logical_or(mask, aai[0:-1,0:-1] < 0.)                     # scattering aerosol pixel  
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel 
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
#        mask = np.logical_or(mask, aai[0:-1,0:-1]<5.)
        
        latA = latA + list(np.asarray(lat[0:-1,0:-1]).reshape(-1))
        lonA = lonA + list(np.asarray(lon[0:-1,0:-1]).reshape(-1))
        aaiA = aaiA + list(np.asarray(aai[0:-1,0:-1]).reshape(-1))
        raaA = raaA + list(np.asarray(raa[0:-1,0:-1]).reshape(-1))
        scaA = scaA + list(np.asarray(sca[0:-1,0:-1]).reshape(-1))
        maskA = maskA + list(np.asarray(mask).reshape(-1))
        
        dataA.close()

    latarrA = np.array(latA)
    lonarrA = np.array(lonA)    
    aaiarrA = np.array(aaiA) 
    raaarrA = np.array(raaA)
    scaarrA = np.array(scaA)
    maskarrA = np.array(maskA)       
    
    aaiarrA = np.ma.masked_array(aaiarrA,maskarrA)

    latvldA = latarrA[aaiarrA.mask == 0] 
    lonvldA = lonarrA[aaiarrA.mask == 0] 
    aaivldA = aaiarrA[aaiarrA.mask == 0]     
    raavldA = raaarrA[aaiarrA.mask == 0]     
    scavldA = scaarrA[aaiarrA.mask == 0]     


 
    aaiB = []
    latB = []
    lonB = [] 
    maskB = []
    raaB = [] 
    scaB = [] 
    
    for io in filelistB[:]:  

        dataB = netCDF4.Dataset(io,'r')    
        data = dataB.groups['DATA']
        geo = dataB.groups['GEOLOCATION']

        lat = geo.variables['LatitudeCenter'][:]
        lon = geo.variables['LongitudeCenter'][:]
        aai = data.variables['AAI'][:]
        sza = geo.variables['SolarZenithAngle'][:]
        sunglint = data.variables['SunGlintFlag'][:]  
        saa = geo.variables['SolarAzimuthAngle'][:]
        vaa = geo.variables['LineOfSightAzimuthAngle'][:]
        sca = geo.variables['ScatteringAngle'][:]
        
        raa = saa + 180.- vaa 
        
        mask = np.logical_or(sza[0:-1,0:-1] < 0. , sza[0:-1,0:-1] > 75.)  
        mask = np.logical_or(mask, lat[0:-1,0:-1]<ROI[0])
        mask = np.logical_or(mask, lat[0:-1,0:-1]>ROI[1])
        mask = np.logical_or(mask, lon[0:-1,0:-1]<ROI[2])
        mask = np.logical_or(mask, lon[0:-1,0:-1]>ROI[3])
        mask = np.logical_or(mask, sunglint[0:-1,0:-1] < 0.)                # sunglint pixel
        mask = np.logical_or(mask, aai[0:-1,0:-1] < 0.)                     # scattering aerosol pixel  
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[1:,0:-1])<-100)      # dateline pixel 
        mask = np.logical_or(mask, (lon[0:-1,0:-1]*lon[0:-1,1:])<-100)
#        mask = np.logical_or(mask, aai[0:-1,0:-1]<5.)
        
        latB = latB + list(np.asarray(lat[0:-1,0:-1]).reshape(-1))
        lonB = lonB + list(np.asarray(lon[0:-1,0:-1]).reshape(-1))
        aaiB = aaiB + list(np.asarray(aai[0:-1,0:-1]).reshape(-1))
        raaB = raaB + list(np.asarray(raa[0:-1,0:-1]).reshape(-1))
        scaB = scaB + list(np.asarray(sca[0:-1,0:-1]).reshape(-1))
        maskB = maskB + list(np.asarray(mask).reshape(-1))

    latarrB = np.array(latB)
    lonarrB = np.array(lonB)    
    aaiarrB = np.array(aaiB) 
    raaarrB = np.array(raaB) 
    scaarrB = np.array(scaB) 
    maskarrB = np.array(maskB)       
    
    aaiarrB = np.ma.masked_array(aaiarrB,maskarrB)

    latvldB = latarrB[aaiarrB.mask == 0] 
    lonvldB = lonarrB[aaiarrB.mask == 0] 
    aaivldB = aaiarrB[aaiarrB.mask == 0]        
    raavldB = raaarrB[aaiarrB.mask == 0]        
    scavldB = scaarrB[aaiarrB.mask == 0]
    
    latvld = np.concatenate((latvldA, latvldB))
    lonvld = np.concatenate((lonvldA, lonvldB))
    aaivld = np.concatenate((aaivldA, aaivldB))
    raavld = np.concatenate((raavldA, raavldB))
    scavld = np.concatenate((scavldA, scavldB))
    
    latnew = np.arange(ROI[0], ROI[1], res)
    lonnew = np.arange(ROI[2], ROI[3], res)
    
    x,y = np.meshgrid(lonnew,latnew)
    aainew = griddata((lonvld, latvld), aaivld, (x, y), method = 'linear')
    aainew[np.isnan(aainew)] = -32767.
    
#    raavld[raavld < 0] = 360 + raavld[raavld<0]
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.scatter(raavldA, aaivldA, c ='k')
#    plt.scatter(raavldB, aaivldB, c ='r')
#    plt.subplot(1,2,2)
#    plt.scatter(scavldA, aaivldA, c = 'k')    
#    plt.scatter(scavldB, aaivldB, c = 'r')    
    
    return latnew, lonnew, aainew 

#year = 2017
#month = 10
#day = 17
#
## region of interest 
#ROI = [45,65,30,60]
#
#res = 0.5
#
#GOME24cfg(year, month, day, ROI, jetlag, plumemsk, 2, res)
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

def readGOME2(dates, ROI, dataset):
#==============================================================================
# Reading data    
#==============================================================================
    try:
        startdate, enddate = dates[0], dates[1]
    except:
        startdate = enddate = dates[0]
    
    dates = pd.date_range(startdate, enddate)
    
# =============================================================================
# read data
# =============================================================================
    output = pd.DataFrame()
    for i, idate in enumerate(dates):
        directory = '/net/bhw428/nobackup_1/users/tilstra/DATA/G2M%s_AAI/' % (dataset)
        filelist = glob.glob(directory + '%4i/%02i/%02i/*.hdf5' % (idate.year, idate.month, idate.day))
        if len(filelist) == 0:
            print('Warning: No data available on GOME2%s %02i-%02i-%04i' %(dataset, idate.year, idate.month, idate.day))
        else: 
            for ifile in filelist: 
                try:
                    temp = {}
                    sys.stdout.write('\r Reading GOME2%s # %04i-%02i-%02i' % (dataset, idate.year, idate.month, idate.day))
                    _data = h5py.File(ifile,'r')
                    data = _data['DATA']
                    geo  = _data['GEOLOCATION']
                    
                    AAH = data['AAH_AbsorbingAerosolHeight'][:]
                    AAH_std = data['AAH_AbsorbingAerosolHeightError'][:]
                    AAP = data['AAH_AbsorbingAerosolPressure'][:]
                    AAP_std = data['AAH_AbsorbingAerosolPressureError'][:]
                    AAI = data['AAI'][:]
                    CA = data['FRESCO_CloudAlbedo'][:]
                    CF = data['FRESCO_CloudFraction'][:]
                    CH = data['FRESCO_CloudHeight'][:]
                    SH = data['FRESCO_FSI_SceneHeight'][:]
                    SA = data['SceneAlbedo'][:]
                    PMD_CF = data['PMD_CloudFraction'][:]
                    H0 = data['SurfaceHeight'][:]
                    Ps = data['SurfacePressure'][:]
                    AAHstd = NNstd(AAH)
                    AAIstd = NNstd(AAI)
                    AAH_flag = data['AAH_ErrorFlag'][:]
    #                QP = data['AAH_QualityProcessing'][:]
    #                QI = data['QualityInput'][:]
                    R340 = data['Reflectance_A'][:]
                    R380 = data['Reflectance_B'][:]
                    sun_glint = data['SunGlintFlag'][:]
                    
                    lat = geo['LatitudeCenter'][:]
                    lon = geo['LongitudeCenter'][:]
                    latc = geo['LatitudeCorner'][:]
                    lonc = geo['LongitudeCorner'][:]
                    saa = geo['SolarAzimuthAngle'][:]
                    sza = geo['SolarZenithAngle'][:]
                    vaa = geo['LineOfSightAzimuthAngle'][:]
                    vza = geo['LineOfSightZenithAngle'][:]
                    dateTime =   geo['Time'][:]
                    
                    _data.close()
                    
                    temp = {'lat': lat, 'lon': lon, 'latb1': latc[0, :, :], 'lonb1': lonc[0, :, :], 'latb2': latc[1,:, :], 'lonb2': lonc[1, :, :],\
                           'latb3': latc[2, :, :], 'lonb3': lonc[2, :, :], 'latb4': latc[3, :, :], 'lonb4': lonc[3, :, :], \
                           'sun_glint': sun_glint, 'Ps': Ps, 'SA': SA, \
                           'sza': sza, 'vza': vza, 'saa': saa, 'vaa': vaa, 'AI380': AAI, 'NNstdAAH': AAHstd, 'NNstdAAI': AAIstd, \
                           'R340': R340, 'R380': R380, \
                           'CF': CF, 'CH': CH, 'CA': CA, 'SH': SH, 'PMD_CF': PMD_CF, 'H0': H0, \
                           'dateTime': dateTime, 'AAH': AAH, 'AAH_std': AAH_std, 'AAP': AAP, 'AAP_std': AAP_std, 'AAH_flag': AAH_flag}
                    for ikey in temp.keys():
                        temp[ikey] = temp[ikey].reshape(-1)
                    temp = pd.DataFrame.from_dict(temp)
                    output = output.append(temp)
                except:
                    pass
# =============================================================================
#  apply mask
# =============================================================================
    try:
        mask = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E']) & \
                (output['sun_glint'] == 0) & (output['AAH_flag'] >= 0) & (output['AAH'] >-1) & (output['CF'] >= 0) & (output['PMD_CF'] >= 0)
        output = output[mask].reset_index(drop = True)
        
        def func(dateTime):
            temp = dateTime.decode()
            temp = pd.to_datetime(temp)
            return temp
        output['dateTime'] = list(map(func, output['dateTime']))
        output['timeStamp'] = output['dateTime'].values.astype(np.int64) // 10 ** 9

        
    except:
        pass
    
    return output



def readGOME2L3(startdate, enddate, ROI, threshold): 
# =============================================================================
# dimension
# =============================================================================
    dates = pd.date_range(startdate, enddate)

    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(-179.5, 180, 1)
    XX, YY = np.meshgrid(lon , lat)
    coords = pd.DataFrame(np.c_[YY.reshape(-1), XX.reshape(-1)], columns = ['lat', 'lon'])
    output = pd.DataFrame(np.ones([np.size(XX), len(dates)]) * np.nan, columns = dates)
    
# =============================================================================
# read data
# =============================================================================
    for i, idate in enumerate(dates): 
        yy, mm, dd = idate.year, idate.month, idate.day
        
        temp = np.ones([2, len(lat), len(lon)]) * np.nan
        
        for ip, ipf in enumerate(['metopa', 'metopb']):
            path = '/nobackup/users/sunj/GOME2-L3/%s/%4i/' %(ipf, yy)
            filelist = glob.glob( path + '*.nc')
            for io in filelist:
                if io.find('%4i%02i%02i' % (yy, mm, dd)) != -1:
                    data = netCDF4.Dataset(io, 'r')
                    temp[ip, :, :] = data.variables['absorbing_aerosol_index'][:]
                    data.close()
        
        temp = np.nanmean(temp, axis= 0)
        temp[temp < threshold] = np.NAN
        output[dates[i]] = temp.reshape(-1)

# =============================================================================
#  apply mask
# =============================================================================
    output = output.join(coords)
    mask = (output['lat'] < ROI['S']) & (output['lat'] > ROI['N']) & (output['lon'] < ROI['W']) & (output['lon'] > ROI['E'])
    output[mask == True] = np.NaN
    output = output.set_index(['lat', 'lon']).T
    
    return output

# =============================================================================
# test code
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
    dAI = readGOME2L3(startdate, enddate, ROI,  AIplume)
    
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

