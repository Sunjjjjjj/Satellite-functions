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
import pandas as pd
from pandas import Series, DataFrame, Panel
from scipy import spatial
import datetime
from scipy.interpolate import griddata
from otherFunctions import *
import h5py


def readOMPSL2(year, month, day, ROI, threshold):
# =============================================================================
# dimension
# =============================================================================
    
    output = pd.DataFrame()
    
    
    path = '/nobackup/users/sunj/OMPS/NMTO3-L2/%4i/%02i/%02i/' % (year, month, day)
    
    output = pd.DataFrame()
    filelist = glob.glob( path + '*.h5')
   
    for io in filelist:
        temp = {}
        if io.find('%4im%02i%02i' % (year, month, day)) > 0:
            idx = io.find('_o')
            orbitnum = int(io[idx+2 : idx + 7])
            sys.stdout.write('\r Reading OMAERUV # %04i-%02i-%02i %s' % (year, month, day, orbitnum))
# =============================================================================
# read data
# =============================================================================
            data = h5py.File(io,'r')
            temp['lat'] = data['GeolocationData']['Latitude'][:].reshape(-1)
            temp['lon'] = data['GeolocationData']['Longitude'][:].reshape(-1)
            for i in range(4):
                temp['latb%i' % (i + 1)] = data['GeolocationData']['LatitudeCorner'][:, :, i].reshape(-1)
                temp['lonb%i' % (i + 1)] = data['GeolocationData']['LongitudeCorner'][:, :, i].reshape(-1)
            
            temp['sza'] = data['GeolocationData']['SolarZenithAngle'][:].reshape(-1)
            temp['vza'] = data['GeolocationData']['ViewingZenithAngle'][:].reshape(-1)
            temp['saa'] = data['GeolocationData']['SolarAzimuthAngle'][:].reshape(-1)
            temp['vaa'] = data['GeolocationData']['ViewingAzimuthAngle'][:].reshape(-1)
            deltaTime = data['GeolocationData']['Time'][:].reshape(-1)
            refTime = datetime.datetime(1993, 1, 1, 0, 0, 0)
            temp['deltaTime'] = (np.ones(data['GeolocationData']['Latitude'][:].shape) * deltaTime.reshape(len(deltaTime), 1) ).reshape(-1)
            temp['orbit'] =  (np.ones(data['GeolocationData']['Latitude'][:].shape) * orbitnum).reshape(-1)
            
            AI = data['ScienceData']['UVAerosolIndex'][:].reshape(-1)
            AI[AI < -1e2] = np.nan
            temp['AI360'] = AI
            
            data.close()
            output = output.append(pd.DataFrame(temp))
            
# =============================================================================
#  apply mask
# =============================================================================
    criteria = (output['lat'] >= ROI['S']) & (output['lat'] <= ROI['N']) & (output['lon'] >= ROI['W']) & (output['lon'] <= ROI['E']) 
    output = output[criteria].reset_index()

    
    def meaTime(dTime): 
        mTime = datetime.datetime(1993, 1, 1, 0, 0) + datetime.timedelta(seconds = dTime)
        return mTime
    mTime = list(map(meaTime, output['deltaTime']))
    
    def timeStamp(mTime): 
        return time.mktime(mTime.timetuple())
    timeStp = list(map(timeStamp, mTime))
    
    output['dateTime'] = mTime
    output['timeStamp'] = timeStp
    output.index = output['dateTime']
    
    
    return output                

#OMPS = readOMPSL2(year, month, day, ROI, -10)



def readOMPSL3(startdate, enddate, ROI, threshold):
# =============================================================================
# dimension
# =============================================================================
    dates = pd.date_range(startdate, enddate)

    lat = np.arange(-89.5, 90, 1)
    lon = np.arange(-179.5, 180, 1)
    XX, YY = np.meshgrid(lon , lat)
    coords = pd.DataFrame(np.c_[YY.reshape(-1), XX.reshape(-1)], columns = ['lat', 'lon'])
    output = pd.DataFrame(np.ones([np.size(XX), len(dates)]) * np.nan, columns = dates)
    
    for i, idate in enumerate(dates): 
        yy, mm, dd = idate.year, idate.month, idate.day
        
        path = '/nobackup/users/sunj/OMPS/NMTO3-L3/%4i/' % (yy)
        filelist = glob.glob( path + '*.h5')
        for io in filelist:
            idx = io.find('OMPS-NPP_NMTO3-L3-DAILY_v2.1_') + len('OMPS-NPP_NMTO3-L3-DAILY_v2.1_')
            if io[idx: idx + 9] == '%4im%02i%02i' % (yy, mm, dd): 
# =============================================================================
# read data
# =============================================================================
                data = h5py.File(io,'r')
                temp = data['UVAerosolIndex'][:]
                temp[temp < threshold] = np.NAN
                data.close()
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
    dAI = readOMPS(startdate, enddate, ROI,  AIplume)
    
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
