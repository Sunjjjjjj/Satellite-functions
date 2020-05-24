#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 09:20:37 2018

@author: sunj
"""
import numpy as np
import datetime


#==============================================================================
# normal date to DOY
#==============================================================================
def date2DOY(year, month, day): 
    """
    Convert normal date to day-of-year. 
    
    Input:  year, mont, day 
    Return: year, day-of-year 
    """
    
    doy = datetime.date(year, month, day).timetuple().tm_yday 

    return year, doy


# call test code
def main(): 

    if date2DOY(2020, 2, 29) == (2020, 60): 
        pass 
    else:
        print ('Exception: wrong calculation ! ')
    
if __name__ == '__main__':
    main()


#==============================================================================
# DOY to normal date 
#==============================================================================
def DOY2date(year, doy): 
    """
    Convert day-of-year to normal date.
    
    Input:  year, day-of-year 
    Return: year, month, day
    """
    
    date = datetime.date(year, 1, 1) + datetime.timedelta(doy - 1)
    year  = date.year
    month = date.month 
    day   = date.day
    
    return year, month, day


# call test code
def main(): 
    if DOY2date(2017, 1) == (2017, 1, 1): 
        pass
    else:
        print ('Exception: wrong calculation ! ')
    
if __name__ == '__main__':
    main()

# =============================================================================
# Date list for daily loop
# =============================================================================
def dateList(startdate, enddate): 
    
    yearstart, doystart = date2DOY(startdate['year'], startdate['month'], startdate['day'])
    yearend,   doyend   = date2DOY(enddate['year'], enddate['month'], enddate['day'])

    dayspermon = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]) 
    
    dateid = []
    # Date span within 1 year        
    if startdate['year'] == enddate['year']: 
        for doy in range(doystart, doyend + 1):
            dateid.append([startdate['year'], doy])
    
    # Date spans multiple years     
    else: 
        stratyear, lastday  = date2DOY(startdate['year'], 12, 31)
        for doy in range(doystart, lastday + 1):
            dateid.append([startdate['year'], doy])
            
        for iyear in range(startdate['year'] + 1, enddate['year']): 
            if iyear % 4 == 0:
                dayspermon[1] = 29
            else:
                dayspermon[1] = 28
            for imon in range(0, 12): 
                for iday in range(1, dayspermon[imon] + 1):
                    iyear, doy = date2DOY(iyear, imon + 1, iday)
                    dateid.append([iyear, doy])
                    
                            
        endyear, firstday = date2DOY(enddate['year'], 1, 1)    
        for doy in range(firstday, doyend + 1): 
            dateid.append([enddate['year'], doy])
            
    return dateid

# =============================================================================
# time zone 
# =============================================================================
def timezone(lon): 
    if abs(lon)%15 < 7.5:
        jetlag = abs(lon)//15
    else:
        jetlag = abs(lon)//15+1
    if lon<0:
        jetlag = - jetlag 
#    print('Time zone:', jetlag) 
    return jetlag 

# =============================================================================
# scattering angle
# =============================================================================

def Theta(sza, saa, vza, vaa):
    szarad = np.deg2rad(sza)
    vzarad = np.deg2rad(vza)
    raa = (vaa + 180. - saa)
    raarad = np.deg2rad(raa)
    theta = np.arccos(np.cos(szarad) * np.cos(vzarad) - np.sin(szarad) * np.sin(vzarad) * np.cos(np.pi - raarad))    
    return np.rad2deg(theta)

# =============================================================================
# distance 
# =============================================================================
def geoDistance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

#    print("Result:", distance)
#    print("Should be:", 278.546, "km")    

# =============================================================================
# rmse
# =============================================================================
def RMSE(data1, data2): 
    return np.sqrt(np.sum((data1 - data2) ** 2) / data1.size)

# =============================================================================
# MAE
# =============================================================================
def MAE(data1, data2): 
    return np.mean(abs(data1 - data2))

# =============================================================================
# score
# =============================================================================
def Score(data1, data2):
    u = np.sum((data1 - data2) ** 2)
    v = np.sum((data1 - data1.mean()) ** 2)     
    return 1 - u / v
# =============================================================================
# Angstrom exponent
# =============================================================================
def Angstorm(wvl1, AOD1, wvl2, AOD2):
    AE = - (np.log(AOD1 / AOD2)) / (np.log(wvl1 / wvl2))
    return AE

def wvldepAOD(wvl1, AOD1, wvl2, AE): 
    AOD2 = AOD1 * (wvl2 / wvl1) ** (-AE)
    return AOD2


# =============================================================================
#  time zone
# =============================================================================
def timeZone(lon):
    return round(lon / 15) 



# =============================================================================
# pressure to height
# =============================================================================
def pressureHeight(Tb, Pb, h0, P):
    # convert extinction to AOD 
    L = -0.0065         # lapse rate, unit: K/m
    R = 8.31432         # gas constant 
    g = 9.80665        # gravitational accelaration constant
    M = 0.0289644       # air molar mass, unit: kg/mol
    
    # pressure to altitude 
    h =  h0 + (Tb / L)  * ((P / Pb) ** (-L * R / g / M) - 1)
    return h
        
# =============================================================================
# space efficiency factor
# =============================================================================
from scipy.stats import variation, zscore
def SPAEF(data1, data2, bins):
    A = np.corrcoef(data1, data2)[0, 1]
    B = variation(data1) / variation(data2)
    data1, data2 = zscore(data1), zscore(data2)
    h1, _ = np.histogram(data1, bins)
    h2, _ = np.histogram(data2, bins)
    h1, h2 = np.float64(h1), np.float64(h2)
    minima = np.minimum(h1, h2)
    
    C = np.sum(minima) / np.sum(h1)
    return 1 - np.sqrt((1 - A)**2 + (1 - B)**2 + (1 - C)**2)

# =============================================================================
# 
# =============================================================================
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap

# =============================================================================
# nearest neighbour std
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
    
    
    
