#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 13:41:15 2022

@author: sunji
"""

import sys, os, glob
import numpy as np
import pandas as pd
import imageio
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import shapefile as shp
import geopandas
from shapely.geometry.polygon import LinearRing, Polygon
from matplotlib.colors import LogNorm

#%%
def makegif(input_files, output_name, duration = 0.1):
    images = []
    for i, f in enumerate(input_files):
        images.append(imageio.imread(f))
        
    imageio.mimsave(output_name, images, duration = duration)
        
#%%
def df2grid(data, ROI, res = 0.05):
    # add columns: latitude_g and longitude_g
    data['latitude_g'] = (round(data['latitude'] / res) * res).astype(np.float64).round(2)
    data['longitude_g'] = (round(data['longitude'] / res) * res).astype(np.float64).round(2)
    data_m = data.groupby(['latitude_g', 'longitude_g']).mean()
    data_m['num'] = data.groupby(['latitude_g', 'longitude_g']).count()['latitude'].values
    data_m.reset_index(inplace = True)
    
    # grid 
    xx, yy = np.meshgrid(np.arange(ROI['W'], ROI['E'], res), np.arange(ROI['S'], ROI['N'], res))
    # xx, yy = np.meshgrid(np.arange(data['longitude'].min(), data['longitude'].max(), res), np.arange(data['latitude'].min(), data['latitude'].max(), res))
    grid = pd.DataFrame()
    grid['latitude_g'] = yy.reshape(-1).astype(np.float64).round(2)
    grid['longitude_g'] = xx.reshape(-1).astype(np.float64).round(2)
    # dimension
    ydim, xdim = xx.shape
    # merge two dataframe
    temp = grid.merge(data_m, how = 'left')
    return temp, (ydim, xdim)

# temp, (ydim, xdim) = df2grid(no2, ROI, res = 0.05)

# data = {'longitude': temp['longitude_g'].values.reshape(ydim, xdim), 'latitude': temp['latitude_g'].values.reshape(ydim, xdim), 
#         'parameter': temp['nitrogendioxide_tropospheric_column'].values.reshape(ydim, xdim)}
# parameters['titles'] = [r'$NO_2$柱浓度分布图(%04i-%02i-%02i-%04i-%02i-%02i)' % (ip[0].year, ip[0].month, ip[0].day, ip[-1].year, ip[-1].month, ip[-1].day)]
# parameters['cmaps'] = [palette]
# parameters['vmin'] = [0e15]
# parameters['vmax'] = [0.4e16]
# parameters['cticks'] = [np.arange(parameters['vmin'][0], parameters['vmax'][0] * 1.1, 2e15)]
# parameters['clabels'] = [r'$[molecules/cm^2]$']
# _, axes = plotmap(parameters).pcolormap(data)
  

#%%
# def ROImask(lat, lon, ROI):
#     mask = (lat >= ROI['S']) & (lat <= ROI['N']) & (lon >= ROI['W']) & (lon <= ROI['E'])
#     lat_m = np.ma.masked_array(lat, ~mask)
#     lon_m = np.ma.masked_array(lon, ~mask)
#     return mask, lat_m, lon_m

#%%
class plotmap:
    def __init__(self, parameters):
        # self.data = data
        # self.lat = data['lat']
        # self.lon = data['lon']
        # self.paras = list(set(data.keys()) - set(['lat', 'lon']))
        self.parameters = parameters
        self.ROI = parameters['ROI']
        
        self.bmgrid = parameters['bmgrid']
        self.figsize = parameters['figsize']
        
        self.axes = parameters['axes']
        self.caxes = parameters['caxes']
        self.titles = parameters['titles']
        self.cticks = parameters['cticks']
        self.clabels = parameters['clabels']
        self.corientation = parameters['corientation']
        self.cmaps = parameters['cmaps']
        

            
        
        self.vmin = parameters['vmin']
        self.vmax = parameters['vmax']
        
    def scattermap(self, data, s = 16):
        lat = data['latitude']
        lon = data['longitude']
        paras = list(set(data.keys()) - set(['latitude', 'longitude']))
        
        Axes = []
        fig = plt.figure(figsize = self.figsize)
        for i in range(len(paras)):
            ax = fig.add_axes(self.axes[i])
            cax = self.caxes[i]
            
            temp = data[paras[i]]
            bm = Basemap(llcrnrlon=self.ROI[i]['W'], llcrnrlat=self.ROI[i]['S'], urcrnrlon=self.ROI[i]['E'], urcrnrlat=self.ROI[i]['N'], \
                        lat_0 = 0, lon_0 = 180, projection='cyl',resolution='l')
            bm.drawparallels(np.arange(np.floor(self.ROI[i]['S']), np.ceil(self.ROI[i]['N']), self.bmgrid[i]), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
            bm.drawmeridians(np.arange(np.floor(self.ROI[i]['W']), np.ceil(self.ROI[i]['E']), self.bmgrid[i]), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
            cb = plt.scatter(lon, lat, c = temp, s = s, marker = 's',
                             cmap = self.cmaps[i], vmin = self.vmin[i], vmax = self.vmax[i], visible = True)
            plt.title(self.titles[i], x = 0.45, y = 0.99, fontsize = 12, fontname = 'SimHei')
            
            cax = fig.add_axes(self.caxes[i])
            cbar = plt.colorbar(cax = cax, ticks = self.cticks[i], label = self.clabels[i],
                                orientation = 'vertical', extend = 'both')
            Axes.append([ax, cb, cax, cbar])
        return fig, Axes

    def scatterpoint(self, data, paras, s = 4, marker = 'o', edgecolor = 'k'):
        lat = data['latitude']
        lon = data['longitude']
        
        Axes = []
        fig = plt.figure(figsize = self.figsize)
        for i in range(len(paras)):
            ax = fig.add_axes(self.axes[i])
            cax = self.caxes[i]
            
            temp = data[paras[i]]
            bm = Basemap(llcrnrlon=self.ROI[i]['W'], llcrnrlat=self.ROI[i]['S'], urcrnrlon=self.ROI[i]['E'], urcrnrlat=self.ROI[i]['N'], \
                        lat_0 = 0, lon_0 = 180, projection='cyl',resolution='l')
            bm.drawparallels(np.arange(np.floor(self.ROI[i]['S']), np.ceil(self.ROI[i]['N']), self.bmgrid[i]), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
            bm.drawmeridians(np.arange(np.floor(self.ROI[i]['W']), np.ceil(self.ROI[i]['E']), self.bmgrid[i]), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
            cb = plt.scatter(lon, lat, c = temp, s = s, marker = marker, edgecolor = edgecolor, 
                             cmap = self.cmaps[i], vmin = self.vmin[i], vmax = self.vmax[i], visible = True)
            plt.title(self.titles[i], x = 0.45, y = 0.99, fontsize = 12, fontname = 'SimHei')
            
            cax = fig.add_axes(self.caxes[i])
            cbar = plt.colorbar(cax = cax, ticks = self.cticks[i], label = self.clabels[i],
                                orientation = 'vertical', extend = 'both')
            Axes.append([ax, cb, cax, cbar])
        return fig, Axes
    
    def pixelmap(self, data, paras):
        lat = data['latitude']
        lon = data['longitude']
        
        Axes = []
        fig = plt.figure(figsize = self.figsize)
        for i in range(len(paras)):
            ax = fig.add_axes(self.axes[i])
            cax = self.caxes[i]
            
            temp = data[paras[i]]
            bm = Basemap(llcrnrlon=self.ROI[i]['W'], llcrnrlat=self.ROI[i]['S'], urcrnrlon=self.ROI[i]['E'], urcrnrlat=self.ROI[i]['N'], \
                        lat_0 = 0, lon_0 = 180, projection='cyl',resolution='l')
            bm.drawparallels(np.arange(np.floor(self.ROI[i]['S']), np.ceil(self.ROI[i]['N']), self.bmgrid[i]), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
            bm.drawmeridians(np.arange(np.floor(self.ROI[i]['W']), np.ceil(self.ROI[i]['E']), self.bmgrid[i]), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
            cb = plt.scatter(lon, lat, c = temp, s = 4,
                             cmap = self.cmaps[i], vmin = self.vmin[i], vmax = self.vmax[i], visible = False)
            plt.title(self.titles[i], x = 0.45, y = 0.99, fontsize = 12, fontname = 'SimHei')
            
            cax = fig.add_axes(self.caxes[i])
            cbar = plt.colorbar(cax = cax, ticks = self.cticks[i], label = self.clabels[i],
                                orientation = 'vertical', extend = 'both')
            for ip in range(len(temp)):
                poly = Polygon([(data.longitude_bounds1.iloc[ip], data.latitude_bounds1.iloc[ip]),
                                (data.longitude_bounds2.iloc[ip], data.latitude_bounds2.iloc[ip]),
                                (data.longitude_bounds3.iloc[ip], data.latitude_bounds3.iloc[ip]),
                                (data.longitude_bounds4.iloc[ip], data.latitude_bounds4.iloc[ip])])
                x, y = poly.exterior.xy
                ax.fill(x, y, c = cb.to_rgba(temp.iloc[ip]), linewidth = 1)
            Axes.append([ax, cb, cax, cbar])
        return fig, Axes    


    def pcolormap(self, data):
        # data is a dictionary includes latitude, longitude, and parameters to be plotted
        lat = data['latitude']
        lon = data['longitude']
        paras = list(set(data.keys()) - set(['latitude', 'longitude']))
        
        Axes = []
        fig = plt.figure(figsize = self.figsize)
        for i in range(len(paras)):
            ax = fig.add_axes(self.axes[i])
            cax = self.caxes[i]
            
            temp = data[paras[i]]
            bm = Basemap(llcrnrlon=self.ROI[i]['W'], llcrnrlat=self.ROI[i]['S'], urcrnrlon=self.ROI[i]['E'], urcrnrlat=self.ROI[i]['N'], \
                        lat_0 = 0, lon_0 = 180, projection='cyl',resolution='l')
            bm.drawparallels(np.arange(np.floor(self.ROI[i]['S']), np.ceil(self.ROI[i]['N']), self.bmgrid[i]), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
            bm.drawmeridians(np.arange(np.floor(self.ROI[i]['W']), np.ceil(self.ROI[i]['E']), self.bmgrid[i]), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
            
            cb = plt.pcolor(lon, lat, temp,
                             cmap = self.cmaps[i], vmin = self.vmin[i], vmax = self.vmax[i], visible = True)
            
            plt.title(self.titles[i], x = 0.45, y = 0.99, fontsize = 12, fontname = 'SimHei')
            
            cax = fig.add_axes(self.caxes[i])
            cbar = plt.colorbar(cax = cax, ticks = self.cticks[i], label = self.clabels[i],
                                orientation = 'vertical', extend = 'both')
            Axes.append([ax, cb, cax, cbar])
        return fig, Axes
    
    def contourfmap(self, data,levels):
        # data is a dictionary includes latitude, longitude, and parameters to be plotted
        lat = data['latitude']
        lon = data['longitude']
        paras = list(set(data.keys()) - set(['latitude', 'longitude']))
        
        Axes = []
        fig = plt.figure(figsize = self.figsize)
        for i in range(len(paras)):
            ax = fig.add_axes(self.axes[i])
            cax = self.caxes[i]
            
            temp = data[paras[i]]
            bm = Basemap(llcrnrlon=self.ROI[i]['W'], llcrnrlat=self.ROI[i]['S'], urcrnrlon=self.ROI[i]['E'], urcrnrlat=self.ROI[i]['N'], \
                        lat_0 = 0, lon_0 = 180, projection='cyl',resolution='l')
            bm.drawparallels(np.arange(np.floor(self.ROI[i]['S']), np.ceil(self.ROI[i]['N']), self.bmgrid[i]), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
            bm.drawmeridians(np.arange(np.floor(self.ROI[i]['W']), np.ceil(self.ROI[i]['E']), self.bmgrid[i]), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
            cb = plt.contourf(lon, lat, temp, levels = levels, extend = 'both',
                             cmap = self.cmaps[i], visible = True)
            plt.title(self.titles[i], x = 0.45, y = 0.99, fontsize = 12, fontname = 'SimHei')
            
            cax = fig.add_axes(self.caxes[i])
            cbar = plt.colorbar(cax = cax, ticks = self.cticks[i], label = self.clabels[i],
                                orientation = 'vertical', extend = 'both')
            Axes.append([ax, cb, cax, cbar])
        return fig, Axes

#%%
# def plotmap(data, parameters = {'figsize': (6, 4), 'axes': [[0.1, 0.1, 0.7, 0.8]], 'caxes': [[0.8, 0.2, 0.05, 0.6]],
#                                 'ROI': {'S':30.75, 'N': 33, 'W': 117.75, 'E': 120}, 'bmgrid': 1, 
#                                 'titles': ['-'],
#                                 'cmaps':['rainbow'],
#                                 'vmin': [0], 'vmax': [1],
#                                 'cticks': [np.arange(2e15, 1.1e16, 2e15)],
#                                 'clabels': [r'$[molecules/cm^2]$'],
#                                 'corientation': 'vertical'
#                                 }):
#     # get parameters
#     figsize = parameters['figsize']
#     axes = parameters['axes']
#     caxes = parameters['caxes']
#     ROI = parameters['ROI']
#     bmgrid = parameters['bmgrid']
#     titles = parameters['titles']
#     cmaps = parameters['cmaps']
#     cticks = parameters['cticks']
#     clabels = parameters['clabels']
#     vmin, vmax = parameters['vmin'], parameters['vmax']
    
#     lon, lat, paras = data['lon'], data['lat'], data['para']
    
#     fig = plt.figure(figsize = figsize)
#     Axes = []
#     for i in range(len(axes)):
#         ax = fig.add_axes(axes[i])
#         cax = caxes[i]
        
#         bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
#                     lat_0 = 0, lon_0 = 180, projection='cyl',resolution='l')
#         bm.drawparallels(np.arange(np.floor(ROI['S']), np.ceil(ROI['N']), bmgrid), labels=[True,False,False,False], linewidth = 0, fontsize = 8)
#         bm.drawmeridians(np.arange(np.floor(ROI['W']), np.ceil(ROI['E']), bmgrid), labels=[False,False,False,True], linewidth = 0, fontsize = 8)
#         cb = plt.scatter(lon, lat, c = paras[i], cmap = cmaps[i], vmin = vmin[i], vmax = vmax[i], visible = True)
#         plt.title(titles[i], x = 0.45, y = 0.99, fontsize = 12, fontname = 'SimHei')
        
#         cax = fig.add_axes(caxes[i])
#         cbar = plt.colorbar(cax = cax, ticks = cticks[i], label = clabels[i],
#                             orientation = 'vertical', extend = 'both')
#         # df, p = plotshp(china, ax, color = 'k', linewidth = 0.2, ROI = ['江苏省'], ROI_linewidth = 0.2)
#         # df, p = plotshp(jiangsu, ax, color = 'k', linewidth = 0.2, ROI = ['南京市'], ROI_linewidth = 0.6)
#         Axes.append(ax)
#     return Axes
#%%
def plotshp(inputfile, ax, color = 'gray', linewidth = 1, ROI = [], ROI_linewidth = 2):
    sf =shp.Reader(inputfile)
    
    def read_shapefile(sf):
        #fetching the headings from the shape file
        fields = [x[0] for x in sf.fields][1:]
    #fetching the records from the shape file
        records = [list(i) for i in sf.records()]
        shps = [s.points for s in sf.shapes()]
    #converting shapefile data into pandas dataframe
        df = pd.DataFrame(columns=fields, data=records)
    #assigning the coordinates
        df = df.assign(coords=shps)
        return df
    df = read_shapefile(sf)
    
    # plt.figure()
    shapes = sf.shapes()
    for i in range(len(df)):
        
        if df.name.iloc[i] in ROI:
            # print(df.name.iloc[i])
            lw = ROI_linewidth
        else:
            lw = linewidth 
        coords = np.array(df.coords[i])
        parts = list(shapes[i].parts)
        parts += [len(coords)]
        
        for j in range(len(parts) -1):
            idx1, idx2 = parts[j], parts[j+1]
            # print(idx1, idx2)
            p = ax.plot(coords[:, 0][idx1: idx2], coords[:, 1][idx1: idx2], color = color, linewidth = lw)
    
    
    return df, p 

# jiangsu = '/home/sunji/Data/map/jiangsu.shp'
# fig = plt.figure()
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# plotshp(jiangsu, ax, color = 'gray', linewidth = 1, ROI = ['南京市','南通市'], ROI_linewidth = 2)
#%%
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors
import seaborn as sns

def mycmap(name = 'tropomi'):
    palette = {}
# =============================================================================
#     # air quality
# =============================================================================
    clist = ['whitesmoke', 'lavenderblush','plum', 'cornflowerblue', 
             'cyan','springgreen','yellow', 'orange', 'red']
    palette['tropomi'] = LinearSegmentedColormap.from_list('tropomi',clist)
    
# =============================================================================
#     # vegetation index
# =============================================================================
    c1 = sns.color_palette('BrBG', 36)
    c2 = sns.color_palette('YlGn', 21)
    c = []
    for ic in c1[::-1][-18:-9]:
        c.append(ic)
    palette['vi'] = matplotlib.colors.ListedColormap(c + c2)
# =============================================================================
#     # vegetation index
# =============================================================================
    c1 = sns.color_palette('Greens', 21)
    c2 = sns.color_palette('Blues', 21)
    c = []
    for ic in c1[:11][::-1]:
        c.append(ic)
    palette['wi'] = matplotlib.colors.ListedColormap(c + c2[:11])
# =============================================================================
#     # bare soil index
# =============================================================================
    c1 = sns.color_palette('BrBG', 21)
    c2 = sns.color_palette('BrBG_r', 36)
    c = []
    for ic in c1[:11][::-1]:
        c.append(ic)
    palette['si'] = matplotlib.colors.ListedColormap(c2[5:31])
# =============================================================================
#     # o3 sensitivity
# =============================================================================
    c = sns.color_palette('RdYlBu_r', 20)
    palette['o3sens'] = matplotlib.colors.ListedColormap([c[4], c[10], c[14]])
    
# =============================================================================
#   # aqi
# =============================================================================
    c1 = sns.color_palette("Set2")
    c2 = sns.color_palette("Paired")
    palette['aqi'] = [c1[0], c1[-3], c2[-5], c2[5], c2[-3], c2[-1]]
    
    return palette[name]

# p = mycmap(name = 'aqi')

#%% 
# =============================================================================
# region mask
# =============================================================================
def ROImask(lat, lon, ROI): 
    return (lat >= ROI['S']) & (lat <= ROI['N']) & (lon >= ROI['W']) & (lon <= ROI['E']) 

def ROImask_2d_orb(lat, lon, ROI):
    y, x = np.where(lat >= ROI['S'])
    idx_s = np.where(lat == lat[y, x].min())[0][0]
    
    y, x = np.where(lon >= ROI['W'])
    idx_w = np.where(lon == lon[y, x].min())[1][0]

    y, x = np.where(lat <= ROI['N'])
    idx_n = np.where(lat == lat[y, x].max())[0][0]
    
    y, x = np.where(lon <= ROI['E'])
    idx_e = np.where(lon == lon[y, x].max())[1][0]
    return idx_s, idx_n, idx_w, idx_e

def ROImask_2d(lat, lon, ROI): 
    # gridded coordinates
    lat = np.array(sorted(list(set(lat.reshape(-1)))))
    lon = np.array(sorted(list(set(lon.reshape(-1)))))
    idx_s = abs(lat - ROI['S']).argmin()
    idx_n = abs(lat - ROI['N']).argmin()
    idx_w = abs(lon - ROI['W']).argmin()
    idx_e = abs(lon - ROI['E']).argmin()

    return idx_s, idx_n, idx_w, idx_e

def clipshp(shp, coords, crs = "EPSG:4326"):
    lon, lat = coords['lon'], coords['lat']
    # mask with shapefile
    vector = geopandas.read_file(shp)
    geometry = geopandas.points_from_xy(lon ,lat, crs="EPSG:4326")
    geometry = geopandas.GeoDataFrame(geometry)
    geometry.rename(columns = {0: 'geometry'}, inplace = True)
    geometry_c = geometry.clip(vector)
    # index where data should be masked
    idx = sorted(list(set(geometry.index) - set(geometry_c.index)))
    return idx

# t1 = time.time()
# XX, YY = np.meshgrid(np.arange(65., 128, 0.1), np.arange(15., 55, 0.1))

# coords = {'lat': YY.reshape(-1), 'lon': XX.reshape(-1)}
# idx = clipshp(china, coords, crs = "EPSG:4326")
# t2 = time.time()
# print('Time used: %1.2f s' % (t2 - t1))

#%%
from osgeo import gdal, osr
from affine import Affine
from pyproj import Proj, transform
import fiona
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from skimage import exposure
from rasterio.merge import merge

def geoDistance(lat1, lon1, lat2, lon2):
    R = 6371.0 # unit: km
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c
#%%
# =============================================================================
# satellite image process: mosaic, clip, bandmath
# =============================================================================
def tiffLatLon(filename):
    # coordinates
    temp = rasterio.open(filename)
    # x,y index
    y, x = temp.read(1).shape
    cols, rows = np.meshgrid(np.arange(x), np.arange(y))
    # coordinate system
    p1 = temp.crs
    # transform ciefficient
    transform_coeff = temp.transform
    # Get affine transform for pixel centres
    T1 = transform_coeff * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1
    # All eastings and northings (there is probably a faster way to do this)
    northings, eastings  = np.vectorize(rc2en, otypes=[float, float])(rows, cols)

    # Project all longitudes, latitudes
    p2 = Proj(proj='latlong',datum='WGS84')
    lon, lat = transform(p1, p2, eastings, northings)
    return lat, lon

def RGB(band1, band2, band3, outputname):
    profile = rasterio.open(band1).profile
    profile.update({'count': 3})
    
    band1 = rasterio.open(band1).read(1).astype(np.float)
    band2 = rasterio.open(band2).read(1).astype(np.float)
    band3 = rasterio.open(band3).read(1).astype(np.float)

    with rasterio.open(outputname, 'w', **profile) as dest:
    # I rearanged the band order writting to 2→3→4 instead of 4→3→2
        dest.write(band1,1)
        dest.write(band2,2)
        dest.write(band3,3)
    # Rescale the image (divide by 10000 to convert to [0:1] reflectance
    img = rasterio.open(outputname)
    image = np.array([img.read(1), img.read(2), img.read(3)]).transpose(1,2,0)
    p2, p98 = np.percentile(image, (2,98))
    image = exposure.rescale_intensity(image, in_range=(p2, p98))
    return image

#%% mosaic and clip satellite image
def tiffileList2filename(tiffileList):
    filename = []
    prefix = []
    for ifile in tiffileList:
        file0 = ifile.split("\\")[-1]
        prefix.append(os.path.join(ifile, file0))
        filename.append(os.path.join(ifile, file0))
    return filename, prefix

def get_extent(tiffileList):
    filename, prefix = tiffileList2filename(tiffileList)
    rioData = rasterio.open(filename[0])
    left = rioData.bounds[0]
    bottom = rioData.bounds[1]
    right = rioData.bounds[2]
    top = rioData.bounds[3]
    for ifile in filename[1:]:
        rioData = rasterio.open(ifile)
        left = min(left, rioData.bounds[0])
        bottom = min(bottom, rioData.bounds[1])
        right = max(right, rioData.bounds[2])
        top = max(top, rioData.bounds[3])
    return left, bottom, right, top, filename, prefix

def getRowCol(left, bottom, right, top):
    # spatial resolution 30m
    cols = int((right - left) / 30.0)
    rows = int((top - bottom) / 30.0)
    return cols, rows

def mosaic(filelist, outputname):
    # have problems in mosaic if using the brightest pixel
    left, bottom, right, top, filename, prefix = get_extent(filelist)
    cols, rows= getRowCol(left, bottom, right, top)
    arr = np.zeros((1, rows, cols), dtype=np.float)
    # 打开一个tif文件
    in_ds = gdal.Open(filename[0])
    # 新建一个tif文件
    driver = gdal.GetDriverByName('gtiff')
    out_ds = driver.Create(outputname, cols, rows)
    # 设置tif文件的投影
    out_ds.SetProjection(in_ds.GetProjection())
    out_band = out_ds.GetRasterBand(1)
    # 设置新tif文件的地理变换
    gt = list(in_ds.GetGeoTransform())
    gt[0], gt[3] = left, top
    out_ds.SetGeoTransform(gt)
    # 对要拼接的影像进行循环读取
    for ifile in prefix:
        in_ds = gdal.Open(ifile)
        # 计算新建的tif文件及本次打开的tif文件之间的坐标漂移
        trans = gdal.Transformer(in_ds, out_ds, [])
        # 得到偏移起始点
        success, xyz = trans.TransformPoint(False, 0, 0)
        x, y, z = map(int, xyz)
        # 读取波段信息
        fnBand = in_ds.GetRasterBand(1)
        data = fnBand.ReadAsArray()
        # 写入tif文件之前，最大值设置为255，这一步很关键
        data = data / 65535 * 255
        data[np.where(data == 255)] = 0
        # 影像重合部分处理，重合部分取最大值
        xSize = fnBand.XSize
        ySize = fnBand.YSize
        outData = out_band.ReadAsArray(x, y, xSize, ySize)
        data = np.maximum(data, outData)
        # data = (data + outData) / 2
        out_band.WriteArray(data, x, y)
    del out_band, out_ds
    return outputname
    
# clip with shapefile in WGS84
def clip(input_imagery_file, input_shapefile, outputname):
    dst_crs = 'EPSG:4326'
    
    # input_imagery_file = figpath + iband + 'mosaic.tif'
    transformed_imagery_file  = 'trans.tif'
    
    with rasterio.open(input_imagery_file) as imagery:
        transform, width, height = calculate_default_transform(imagery.crs, dst_crs, imagery.width, imagery.height, *imagery.bounds)
        kwargs = imagery.meta.copy()
        kwargs.update({'crs': dst_crs, 'transform': transform, 'width': width, 'height': height})
        with rasterio.open(transformed_imagery_file, 'w', **kwargs) as dst:
            for i in range(1, imagery.count + 1):
                reproject(
                    source=rasterio.band(imagery, i),
                    destination=rasterio.band(dst, i),
                    src_transform=imagery.transform,
                    src_crs=imagery.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
    
    with fiona.open(input_shapefile, 'r') as shapefile:
        shp =[feature['geometry'] for feature in shapefile]
    
    with rasterio.open(transformed_imagery_file) as src:
        out_image, out_transform = rasterio.mask.mask(src, shp, crop = True)
        out_meta = src.meta
    # Save clipped imagery
    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})
    with rasterio.open(outputname, "w", **out_meta) as dest:
        dest.write(out_image)
        
    return outputname

    # uniform projection
    
    # merge images
    # clip with shapefile
    
    
    
# def Lonlat2Xy(SourceGcs, TargetPcs, lon, lat):
#     '''
#     :param SourceRef: 源地理坐标系统
#     :param TargetRef: 目标投影
#     :param lon: 待转换点的longitude值
#     :param lat: 待转换点的latitude值
#     :return:
#     '''
#     #创建目标空间参考
#     spatialref_target=osr.SpatialReference()
#     spatialref_target.ImportFromEPSG(TargetPcs) #2331为目标空间参考的ESPG编号，西安80 高斯可吕格投影
#     #创建原始空间参考
#     spatialref_source=osr.SpatialReference()
#     spatialref_source.ImportFromEPSG(SourceGcs)  #4326 为原始空间参考的ESPG编号，WGS84
#         #构建坐标转换对象，用以转换不同空间参考下的坐标
#     trans=osr.CoordinateTransformation(spatialref_source,spatialref_target)
#     # coordinate_after_trans 是一个Tuple类型的变量包含3个元素， [0]为y方向值，[1]为x方向值，[2]为高度
#     coordinate_after_trans=trans.TransformPoint(lat,lon)
#     # print(coordinate_after_trans)
# 	 #以下为转换多个点（要使用list）
#     coordinate_trans_points=trans.TransformPoints([(40,117),(36,120)])
#     print(coordinate_trans_points)
#     return coordinate_after_trans

# def Xy2Lonlat(SourcePcs, x, y):
#     '''
#         将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
#         :param SourcePcs:源投影坐标系
#         :param x: 投影坐标x
#         :param y: 投影坐标y
#         :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
#     '''
#     prosrs = osr.SpatialReference()
#     # prosrs投影参考系
#     prosrs.ImportFromEPSG(SourcePcs)#投影坐标的EPSG编号
#     #geosrs地理参考系
#     geosrs = osr.SpatialReference()
#     geosrs.ImportFromEPSG(4326)            #WGS84
#     ct = osr.CoordinateTransformation(prosrs, geosrs)
#     coords = ct.TransformPoint(y, x)
#     return coords[:2]   #(lat,lon)

# #tiff是投影坐标的情况未实测
# def Rowcol2Lonlat(filename, Xpixel,Ypixel):
#     dataset = gdal.Open(filename)  # 打开文件
#     GeoTransform = dataset.GetGeoTransform()
#     XGeo = GeoTransform[0]+GeoTransform[1]*Xpixel+Ypixel*GeoTransform[2];
#     YGeo = GeoTransform[3]+GeoTransform[4]*Xpixel+Ypixel*GeoTransform[5];
    
#     pcs = osr.SpatialReference()
#     pcs.ImportFromWkt(dataset.GetProjection())   
#     print(pcs)
#     if pcs.IsGeographic():    # 是地理坐标
#         return XGeo,YGeo
#     elif pcs.IsProjected():   #是投影坐标
#         coords = Xy2Lonlat(pcs, XGeo, YGeo)  
#         return coords[:2]

# #tiff是投影坐标的情况未实测
# def Lonlat2Rowcol(filename, lon,lat):
#     dataset = gdal.Open(filename)  # 打开文件
#     tiff_geotrans = dataset.GetGeoTransform()
#     pcs = osr.SpatialReference()
#     pcs.ImportFromWkt(dataset.GetProjection())
#     if pcs.IsGeographic():  # 是地理坐标
#         XGeo, YGeo = lon , lat
#     elif pcs.IsProjected():  # 是投影坐标
#         YGeo, XGeo = Lonlat2Xy(4326, pcs, lon, lat)
#     A = [[tiff_geotrans[1], tiff_geotrans[2]],  # 根据公式Xgeo=tiff_geotrans[0]+Xpixel*tiff_geotrans[1]+Yline*tiff_geotrans[2]
#          [tiff_geotrans[4], tiff_geotrans[5]]]  # Ygeo=tiff_geotrans[3]+Xpixel*tiff_geotrans[4]+Yline*tiff_geotrans[5]
#     s = [[XGeo - tiff_geotrans[0]],  # 运用矩阵解二元一次方程组求得行列号
#          [YGeo - tiff_geotrans[3]]]
#     r = np.linalg.solve(A, s)
#     # Xpixel, Ypixel = r[0], r[1]
#     Xpixel,Ypixel = int(r[0]),int(r[1])
#     return Xpixel,Ypixel


