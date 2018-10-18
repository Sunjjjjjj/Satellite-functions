#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
load python packages
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot
from shapely.geometry.polygon import LinearRing, Polygon

"""
Colormaps
"""
rainbow = mpl.cm.rainbow
rainbow.set_gamma(1)
rainbow.set_over(rainbow(0.99))
rainbow.set_under(rainbow(0))

rainbow_r = mpl.cm.rainbow_r
rainbow_r.set_gamma(1)
rainbow_r.set_over(rainbow_r(0.99))
rainbow_r.set_under(rainbow_r(0))

bwr = mpl.cm.bwr
bwr.set_gamma(1)
bwr.set_over(bwr(0.99))
bwr.set_under(bwr(0))

cm = {'rainbow': rainbow,
      'rainbow_r': rainbow_r,
      'bwr': bwr}


def plotSpatial(datasets, ROI, cornercoords = False): 
    """
    Function to plot spatial map on 'cyl' projection. 
    Multiple subplots is applicable.
    Number of subplots = length of datasets = N. Subplot(1, N, x).
    Colormaps are self-defined.
    
    -datasets: dictionary includes all datasets to be plot. 
    datasets = [{'data': data, 'parameter': 'para', 'label': 'label', 'bounds': (x1, x2), 'cmap': cmap}, 
                {'data': data, 'parameter': 'para', 'label': 'label', 'bounds': (x1, x2), 'cmap': cmap}, 
                ...]
    
    -cornercoords: plot use center lat/lon (fast) or corner lat/lon (slow). 
    
    -ROI: region of interest.
    
    Return: handle of fig, basemap and axes for each subplot. 
    
    @author: Sunji
    Last updated date: 2018-10-18
    """
    
    """
    Initialization
    """
    N = len(datasets)
    width, height, wspace = 4, 4, 0.1 
    fig = plt.figure(figsize = (width * N, height))
    fig.set_size_inches(width * N, height)
    fig.tight_layout()
    fig.subplots_adjust(wspace = wspace)
    
    axes = []
    
    """
    Loop over datasets
    """
    for idata in range(N): 
        sys.stdout.write('\r Ploting %i/%i datasets ' % (idata + 1, N))
        data = datasets[idata]['data']
        para = datasets[idata]['parameter']
        label = datasets[idata]['label']
        bounds = datasets[idata]['bounds']
        cmap = datasets[idata]['cmap']
        """
        Layer: basemap
        """
        ax = plt.subplot(1, N, idata + 1)
        bm = Basemap(llcrnrlon=ROI['W'], llcrnrlat=ROI['S'], urcrnrlon=ROI['E'], urcrnrlat=ROI['N'], \
                    lat_0 = 0, lon_0 = 0, projection='cyl',resolution='l')
        bm.drawcoastlines(color='black',linewidth=0.6)
#        m.drawcountries(color = 'black',linewidth=0.4)
        dlat = round((ROI['N'] - ROI['S']) * 0.5)
        dlon = round((ROI['E'] - ROI['W']) * 0.5)
        bm.drawmeridians(np.linspace(ROI['W'] - dlon, ROI['E'] + dlon, 3), labels = [1,0,1,0], linewidth = 0)
        bm.drawparallels(np.linspace(ROI['S'] - dlat, ROI['N'] + dlat, 3), labels = [1,0,1,0], linewidth = 0)
        """
        Layer: data
        """
        if not cornercoords: 
            visible = True
        else: 
            visible = False
        cb = plt.scatter(data.lon ,data.lat, c = data[para], s = 6, visible = visible, cmap=cm[cmap], \
                         marker = 's', alpha = 1, vmin = bounds[0], vmax = bounds[1], edgecolors = 'none')  
        cbar = plt.colorbar(cb, extend = 'both', fraction=0.15, pad= 0.1, shrink = 0.8, aspect = 15, \
                            orientation = 'horizontal', ticks = np.linspace(bounds[0], bounds[1], 5))
        cbar.set_label(label, rotation = 0, labelpad = -45)
        if cornercoords: 
            try:
                for i in range(len(data)): 
                    sys.stdout.write('\r pixel %6i/%6i' % (i + 1, len(data)))
                    poly = Polygon([(data.lonb1[i], data.latb1[i]), 
                                    (data.lonb2[i], data.latb2[i]),
                                    (data.lonb3[i], data.latb3[i]),
                                    (data.lonb4[i], data.latb4[i])])
                    x, y = poly.exterior.xy     
                    plt.fill(x, y, c = cb.to_rgba(data[para].iloc[i]), linewidth = 1)          
            except NameError:
                print('Error: data does not have corner coordinates!')
        axes.append(ax)
    return fig, bm, axes



def main(): 
    caseName = 'CA201712_OMIAs'
    casedir = '/nobackup/users/sunj/data_saved_for_cases/%s' %(caseName)
    data = pd.read_pickle(casedir + '/dataTROPOMI_%4i-%02i-%02i' % (2017, 12, 12)) 
    ROI = {'S':30, 'N': 42.5, 'W': -130, 'E': -117.5}

    
    datasets = [{'data': data, 'parameter': 'AI380', 'label': 'AI380', 'bounds': (0, 8), 'cmap': rainbow},
                {'data': data, 'parameter': 'sza', 'label': 'SZA', 'bounds': (60, 70), 'cmap': rainbow}]
    fig, bm, axes = plotSpatial(datasets, ROI, cornercoords = False)
    return fig, bm, axes
    
if __name__ == '__main__':
    fig, bm, axes = main()
    
    
    
    