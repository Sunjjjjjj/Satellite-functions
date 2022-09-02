#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:03:41 2022

@author: kanonyui
"""
import sys
sys.path.insert(0, '/home/sunji/Scripts/')
import os, glob
import netCDF4
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
from mpl_toolkits.basemap import Basemap
import seaborn as sns
from utilities import *


#%% read ARP product
def readARP(inputfile, ROI, qa = 10):
    data = netCDF4.Dataset(inputfile, 'r')
    temp_2d = {}
    temp = pd.DataFrame()
    
    if inputfile.find('L2ARP030') >= 0:
        sys.stdout.write('\r Reading L2 10-min %s ...' % (inputfile))
        
        lon, lat = np.meshgrid(data['longitude'][:], data['latitude'][:])
        mask = ROImask(lat, lon, ROI)
        qa_mask = data['AOT_uncertainty'][:] <= qa
        
        temp_2d['longitude'], temp_2d['latitude'] = lon, lat
        temp['longitude'], temp['latitude'] = lon.reshape(-1), lat.reshape(-1)
        parameters = ['AOT', 'QA_flag', 'AE', 'SSA', 'RF', 'AOT_uncertainty']
        for ikey in parameters:
            temp_2d[ikey] = np.ma.masked_array(data[ikey][:]*1.0, ~(mask & qa_mask)).filled(np.nan)
            temp[ikey] = temp_2d[ikey].reshape(-1)
        temp = temp[~np.isnan(temp.AOT)]
        temp.reset_index(drop = True, inplace = True)
        data.close()
        
    # read L3 hourly product
    if inputfile.find('1HARP031_FLDK') >= 0:
        sys.stdout.write('\r Reading L3 %s ...' % (inputfile))
        
        lon, lat = np.meshgrid(data['longitude'][:], data['latitude'][:])
        mask = ROImask(lat, lon, ROI)
        qa_mask = data['AOT_Merged_uncertainty'][:] <= qa

        temp_2d['longitude'], temp_2d['latitude'] = lon, lat
        temp['longitude'], temp['latitude'] = lon.reshape(-1), lat.reshape(-1)
        parameters = ['AOT_L2_Mean', 'AE_L2_Mean', 'AOT_Merged', 'AOT_Merged_uncertainty', 'AE_Merged']
        for ikey in parameters:
            temp_2d[ikey] = np.ma.masked_array(data[ikey][:]*1.0, ~(mask & qa_mask)).filled(np.nan)
            temp[ikey] = temp_2d[ikey].reshape(-1)
        
        temp = temp[~np.isnan(temp.AOT_Merged)]
        temp.reset_index(drop = True, inplace = True)
        data.close()
        
     # read L3 daily and monthly product
    if (inputfile.find('1DARP031_FLDK') >= 0) | (inputfile.find('1MARP031_FLDK') >= 0):
        sys.stdout.write('\r Reading L3 %s ...' % (inputfile))
        
        lon, lat = np.meshgrid(data['longitude'][:], data['latitude'][:])
        mask = ROImask(lat, lon, ROI)

        temp_2d['longitude'], temp_2d['latitude'] = lon, lat
        temp['longitude'], temp['latitude'] = lon.reshape(-1), lat.reshape(-1)
        parameters = ['AOT_L2_Mean', 'AOT_L3_Merged_Mean', 'AE_L2_Mean', 'AE_L3_Merged_Mean']
        for ikey in parameters:
            temp_2d[ikey] = np.ma.masked_array(data[ikey][:]*1.0, ~(mask)).filled(np.nan)
            temp[ikey] = temp_2d[ikey].reshape(-1)
        
        temp = temp[~np.isnan(temp.AOT_L3_Merged_Mean)]
        temp.reset_index(drop = True, inplace = True)
        data.close()
           
    return temp, temp_2d
    
# inputfile = '/home/sunji/Data/Himawari-8/L3ARP/daily/H08_20220621_0000_1DARP031_FLDK.02401_02401.nc'
# ROI = {'S': 32.2, 'N': 35, 'W': 119, 'E': 121.5}
# data, data_2d = readARP(inputfile, ROI, qa = 10)


#%% read WLF product
def readWLF(inputfile, ROI, qa = 2.5):
    # read L2 10-min product
    if inputfile.find('L2WLF') >= 0:
        sys.stdout.write('\r Reading L2 10-min %s ...' % (inputfile))
        output = pd.read_csv(inputfile, header = 1)
        # ROI mask
        mask = ROImask(output['Lat'], output['Lon'], ROI)
        qa_mask = output['QF'] <= qa
        output = output[mask & qa_mask].reset_index(inplace = False, drop = True)
        output.rename(columns={'Lat': 'latitude', 'Lon': 'longitude'}, inplace = True)

    else:
        # read L3 hourly product
        if inputfile.find('L3WLF') >= 0:
            sys.stdout.write('\r Reading L3 hourly %s ...' % (inputfile))
            output = pd.read_csv(inputfile, header = 1)
        # read L3 daily product
        if inputfile.find('1DWLF') >= 0:
            sys.stdout.write('\r Reading L3 daily %s ...' % (inputfile))
            output = pd.read_csv(inputfile, header = 1)
        # read L3 monthly product
        if inputfile.find('1MWLF') >= 0:
            sys.stdout.write('\r Reading L3 monthly %s ...' % (inputfile))
            output = pd.read_csv(inputfile, header = 1)
        # ROI mask
        mask = ROImask(output['lat'], output['lon'], ROI)
        # qa_mask = output['ave(confidence)'] >= qa
        output = output[mask].reset_index(inplace = False, drop = True)
        output.rename(columns={'lat': 'latitude', 'lon': 'longitude'}, inplace = True)
        
    return output


# inputfile = '/home/sunji/Data/Himawari-8/L2WLF/H08_20220621_2320_L2WLF010_FLDK.06001_06001.csv'
# ROI = {'S':30.5, 'N': 35.5, 'W': 116, 'E': 122.5}
# output = readWLF(inputfile, ROI, qa = 2.5)
    

#%% H08 download by ftp
import ftplib
import schedule
import datetime
def ftpH08WLF_nrt(idate = datetime.datetime.now(), level = 'L3'):
    # convert local time to UTC (8 hour difference), and L2 data time delay is around 1 hour
     # - pd.to_timedelta(40, unit = 'minute')
    # ftp login information
    host = 'ftp.ptree.jaxa.jp'
    user = 'Jiyunting.sun_gmail.com'
    passwd = 'SP+wari8'
    # connect ftp
    ftp = ftplib.FTP(host)
    ftp.login(user, passwd)
    
    localpath = '/home/sunji/Data/Himawari-8/%sWLF/' % (level)
    filename = []
    if level == 'L2':
        # idate = idate - pd.to_timedelta(9, unit = 'H')
        print(idate)
        filelist = ftp.nlst('pub/himawari/L2/WLF/010/%04i%02i/%02i/%02i' % (idate.year, idate.month, idate.day, idate.hour))
        for f in filelist:
            if f.find('_%02i%02i_' % (idate.hour, np.floor(idate.minute / 10) * 10)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
    if level == 'L3':
        # idate = idate - pd.to_timedelta(10, unit = 'H')
        filelist = ftp.nlst('pub/himawari/L3/WLF/010/%04i%02i/%02i' % (idate.year, idate.month, idate.day))
        for f in filelist:
            if f.find('_%02i00_' % (idate.hour)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
                
    ftp.quit()
    return filename



def ftpH08WLF_L3(idate, freq = 'daily'):
    # convert local time to UTC (8 hour difference), and L2 data time delay is around 1 hour
     # - pd.to_timedelta(40, unit = 'minute')
    # ftp login information
    host = 'ftp.ptree.jaxa.jp'
    user = 'Jiyunting.sun_gmail.com'
    passwd = 'SP+wari8'
    # connect ftp
    ftp = ftplib.FTP(host)
    ftp.login(user, passwd)
    
    localpath = '/home/sunji/Data/Himawari-8/L3WLF/'
    filename = []
    if freq == 'daily':
        filelist = ftp.nlst('pub/himawari/L3/WLF/010/%04i%02i/daily/' % (idate.year, idate.month))
        for f in filelist:
            # L3 data time delay is around 2 hours
            if f.find('_%04i%02i%02i_' % (idate.year, idate.month, idate.day)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + '%s/' % freq + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
    elif freq == 'monthly':
        filelist = ftp.nlst('pub/himawari/L3/WLF/010/%04i%02i/monthly/' % (idate.year, idate.month))
        for f in filelist:
            # L3 data time delay is around 2 hours
            if f.find('_%04i%02i%02i_' % (idate.year, idate.month, idate.day)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + '%s/' % freq + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
    ftp.quit()
    return filename
# fn = ftpH08WLFL3(pd.to_datetime('2022-06-21'), freq = 'daily')



def ftpH08ARP_nrt(idate = datetime.datetime.now(), level = 'L3'):
    # convert local time to UTC (8 hour difference), and L2 data time delay is around 2 hours
    # ftp login information
    host = 'ftp.ptree.jaxa.jp'
    user = 'Jiyunting.sun_gmail.com'
    passwd = 'SP+wari8'
    # connect ftp
    ftp = ftplib.FTP(host)
    ftp.login(user, passwd)
    
    localpath = '/home/sunji/Data/Himawari-8/%sARP/' % (level)
    filename = []
    if level == 'L2':
        # idate = idate - pd.to_timedelta(10, unit = 'H')
        filelist = ftp.nlst('pub/himawari/L2/ARP/030/%04i%02i/%02i/%02i' % (idate.year, idate.month, idate.day, idate.hour))
        for f in filelist:
            if f.find('_%02i%02i_' % (idate.hour, np.floor(idate.minute / 10) * 10)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
                
    if level == 'L3':
        # idate = idate - pd.to_timedelta(14, unit = 'H')
        filelist = ftp.nlst('pub/himawari/L3/ARP/031/%04i%02i/%02i' % (idate.year, idate.month, idate.day))
        for f in filelist:
            # L3 data time delay around 6 hours
            if f.find('_%02i00_' % (idate.hour)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
                
    ftp.quit()
    return filename



# filename = ftpH08ARP(idate = datetime.datetime.now(), level = 'L2')

def ftpH08ARP_L3(idate, freq = 'daily'):
    # convert local time to UTC (8 hour difference), and L2 data time delay is around 1 hour
     # - pd.to_timedelta(40, unit = 'minute')
    # ftp login information
    host = 'ftp.ptree.jaxa.jp'
    user = 'Jiyunting.sun_gmail.com'
    passwd = 'SP+wari8'
    # connect ftp
    ftp = ftplib.FTP(host)
    ftp.login(user, passwd)
    
    localpath = '/home/sunji/Data/Himawari-8/L3ARP/'
    filename = []
    if freq == 'daily':
        filelist = ftp.nlst('pub/himawari/L3/ARP/031/%04i%02i/daily/' % (idate.year, idate.month))
        for f in filelist:
            # L3 data time delay is around 2 hours
            if f.find('_%04i%02i%02i_' % (idate.year, idate.month, idate.day)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + '%s/' % freq + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
    elif freq == 'monthly':
        filelist = ftp.nlst('pub/himawari/L3/ARP/031/%04i%02i/monthly/' % (idate.year, idate.month))
        for f in filelist:
            # L3 data time delay is around 2 hours
            if f.find('_%04i%02i%02i_' % (idate.year, idate.month, idate.day)) >= 0:
                idx = f.find('H08')
                fp = open(localpath + '%s/' % freq + f[idx:], 'wb')
                ftp.retrbinary('RETR ' + f, fp.write)
                filename = f[idx:]
                sys.stdout.write('Downloading %s' % f)
    ftp.quit()
    return filename
# fn = ftpH08ARP_L3(pd.to_datetime('2022-06-21'), freq = 'daily')


#%%
# dates = pd.period_range('2022-07-30', '2022-08-15', freq = 'D')
# for idate in dates:
#     print(idate)
#     filename = ftpH08ARP_L3(idate, freq = 'daily')
#     print(filename)