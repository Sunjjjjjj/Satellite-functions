#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 10:43:37 2022

@author: sunji
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 12:12:55 2022

@author: sunji
"""

import os, re, sys, glob
sys.path.insert(0, '/home/sunji/Scripts/')
import numpy as np
# from osgeo import gdal
import rasterio
import rasterio.mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
# from sh import ssh
# import paramiko

# =============================================================================
# initialization
# =============================================================================
dst_crs = 'EPSG:4490'
name = '全国各省DEM裁剪_最终版'
path = '/home/sunji/Data/福建土壤生态项目/%s/' % name

city = os.listdir('/home/sunji/Data/福建土壤生态项目/%s/' % name)

for icity in city:
    path = '/home/sunji/Data/福建土壤生态项目/%s/%s/' % (name, icity)
    figpath = '/home/sunji/Data/福建土壤生态项目/%s_reprojected/%s_reprojected/' % (name, icity)
    if not os.path.exists('/home/sunji/Data/福建土壤生态项目/%s_reprojected/' % (name)):
        os.mkdir('/home/sunji/Data/福建土壤生态项目/%s_reprojected/' % (name))

    if not os.path.exists(figpath):
        os.mkdir(figpath)

    filelist = glob.glob(path + '*.tif')
#%%
# =============================================================================
# reproject
# =============================================================================
    for i, f in enumerate(sorted(filelist)):
        sys.stdout.write('\r processing %s' % (f))
           
        # convert projection
        with rasterio.open(f) as src:
            transform, width, height = calculate_default_transform(
                                        src.crs, dst_crs, src.width, src.height, *src.bounds)
            data = src.read()
            
            # data[np.isnan(data)] = 0
            meta = src.meta.copy()
            meta.update({
                'driver': 'GTiff',
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'dtype': data.dtype})
            
            output_reproj = figpath + '%s_epsg4490.tif' % (f[len(path):-4])
            with rasterio.open(output_reproj, 'w', **meta) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=data,
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

