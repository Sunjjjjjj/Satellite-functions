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
    plots = []
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
        dlat = round((ROI['N'] - ROI['S']))
        dlon = round((ROI['E'] - ROI['W']))
        if idata == 0:
            bm.drawparallels(np.arange(ROI['S'], ROI['N'], 5), labels = [1,0,1,0], linewidth = 0)
        bm.drawmeridians(np.arange(ROI['W'], ROI['E'], 5), labels = [1,0,1,0], linewidth = 0)
        
        XX, YY = np.meshgrid(np.arange(-180, 181, 1), np.arange(-90,91,1)) 
        plt.scatter(XX, YY, s = 1000, marker = 's', c = [0.95, 0.95, 0.95])

        """
        Layer: data
        """
        if not cornercoords: 
            visible = True
        else: 
            visible = False
        cb = plt.scatter(data.lon ,data.lat, c = data[para], visible = visible, cmap=cmap, \
                         marker = 's', alpha = 1, vmin = bounds[0], vmax = bounds[1], edgecolors = 'none', label = '')  
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
        plots.append(cb)
    return fig, bm, axes


from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


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
    
    
    
    