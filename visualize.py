# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 12:23:39 2019

@author: Ronny Förster, Stefan Weidlich, Mona Nissen
"""



"""
collection of plotting functions for visualization


******************************************************************************
Importing neccessary libraries
"""

import numpy as np # library for array-manipulation
import pandas as pd # library for DataFrame handling
import trackpy as tp # trackpy offers all tools needed for the analysis of diffusing particles
import seaborn as sns # allows boxplots to be put in one graph
#sns.reset_orig()
import math # offering some maths functions
import matplotlib
#matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['text.latex.unicode'] = False
import matplotlib.pyplot as plt # libraries for plotting
from matplotlib import animation # allows to create animated plots and videos from them
import json
import sys
import datetime
from pdb import set_trace as bp #debugger
import scipy
import os.path

import NanoObjectDetection as nd

## unused for the moment:
#import time
#from matplotlib.animation import FuncAnimation, PillowWriter
#from matplotlib.animation import FuncAnimation


def SetXYLim(x_min_max = None, y_min_max = None):
    """ set axes limits to fixed values """
    
    if (x_min_max is None) == False:
        plt.xlim([x_min_max[0], x_min_max[1]])

    if (y_min_max is None) == False:
        plt.ylim([y_min_max[0], y_min_max[1]])



def Plot1DPlot(plot_np,title = None, xlabel = None, ylabel = None, settings = None):
    """ plot 1D-data in standardized format as line plot """
    
    import NanoObjectDetection as nd
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    plt.figure()
    plt.plot(plot_np)
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    
    
    if settings != None:
        nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter_Histogramm", settings, data = plot_np)



def Plot2DPlot(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1, mymarker = 'x', mylinestyle  = ':', x_lim = None, y_lim = None, y_ticks = None, semilogx = False, FillArea = False, Color = None):
    """ plot 2D-data in standardized format as line plot """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font, params
#    sns.reset_orig()
    
    plt.style.use(params)
    
    plt.figure()
    if semilogx == False:
        if Color == None:
            plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
        else:
            plt.plot(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha, color = Color)
            
        if FillArea == True:
            plt.fill_between(x_np,y_np, y2 = 0, color = Color, alpha=0.5)
    else:
        plt.semilogx(x_np,y_np, marker = mymarker, linestyle  = mylinestyle, alpha = myalpha)
        import matplotlib.ticker

        ax = plt.gca()
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.get_xaxis().set_minor_formatter(matplotlib.ticker.ScalarFormatter())
        
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

#    matplotlib.rcParams.update(params)

    

    if x_lim != None:
        plt.xlim(x_lim)
        
    if y_lim != None:
        plt.ylim(y_lim)
                
    if y_ticks != None:
        frame1 = plt.gca() 
        frame1.axes.yaxis.set_ticks(y_ticks)


    nd.PlotProperties
    plt.style.use(params)
    
    ax = plt.gca()
    return ax



def Plot2DScatter(x_np,y_np,title = None, xlabel = None, ylabel = None, myalpha = 1,
                  x_min_max = None, y_min_max = None):
    """ plot 2D-data in standardized format as individual, scattered points """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    plt.figure()
    plt.scatter(x_np,y_np, alpha = myalpha, linewidths  = 0)
    plt.grid("on")
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)

    SetXYLim(x_min_max, y_min_max)



def Plot2DImage(array_np,title = None, xlabel = None, ylabel = None, ShowColorBar = True):
    """ plot image in standardized format """
    
    from NanoObjectDetection.PlotProperties import axis_font, title_font, params

    
    plt.figure()
    plt.imshow(array_np, cmap='gray')
    
    if ShowColorBar == True:
        plt.colorbar()
    plt.title(title, **title_font)
    plt.xlabel(xlabel, **axis_font)
    plt.ylabel(ylabel, **axis_font)
    
    

def PlotDiameters(ParameterJsonFile, sizes_df_lin, any_successful_check, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None):
    """ plot and/or save diameter histogram (or other statistics) for analyzed particles 
    (main function)
    """
    import NanoObjectDetection as nd
    
    # check for NaN values in DataFrame
    if pd.isna(sizes_df_lin).sum().sum() > 0:
        # drop rows with missing values (e.g. from invalid diameter calculations)
        sizes_df_lin = sizes_df_lin.dropna()
        print('ATTENTION: NaN values were detected in sizes_df_lin. Please check it carefully.\n'
              'In the following plots, rows with NaN values are ignored.\n\n')

    if any_successful_check == False:
        print("NO PARTICLE WAS MEASURED LONG ENOUGH FOR A GOOD STATISTIC !")
    else:
        settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
        show_hist, save_hist = settings["Plot"]["Histogramm_Show"], settings["Plot"]["Histogramm_Save"]
        if show_hist or save_hist:
            DiameterHistogramm(ParameterJsonFile, sizes_df_lin)


        show_pdf, save_pdf = settings["Plot"]["DiameterPDF_Show"], settings["Plot"]["DiameterPDF_Save"]
        if show_pdf or save_pdf:
            DiameterPDF(ParameterJsonFile, sizes_df_lin)

        
        show_diam_traj, save_diam_traj = settings["Plot"]["DiamOverTraj_Show"], settings["Plot"]["DiamOverTraj_Save"]
        if show_diam_traj or save_diam_traj:
            DiameterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_diam_traj, save_diam_traj)


        show_hist_time, save_hist_time = settings["Plot"]["Histogramm_time_Show"], settings["Plot"]["Histogramm_time_Save"]
        if show_hist_time or save_hist_time:
            DiameterHistogrammTime(ParameterJsonFile, sizes_df_lin)
#            PlotDiameter2DHistogramm(ParameterJsonFile, sizes_df_lin, show_hist_time, save_hist_time)


        show_correl, save_correl = settings["Plot"]["Correlation_Show"], settings["Plot"]["Correlation_Save"]
        if show_correl or save_correl:
            Correlation(ParameterJsonFile, sizes_df_lin, show_correl, save_correl)
            
            
        show_pearson, save_pearson = settings["Plot"]["Pearson_Show"], settings["Plot"]["Pearson_Save"]
        if show_pearson or save_pearson:
            Pearson(ParameterJsonFile, sizes_df_lin, show_pearson, save_pearson)



def corrfunc(x, y, **kws):
    """ auxiliary function for applying Pearson statistics on (diameter) data """
    
    pearson, _ = scipy.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("p = {:.2f}".format(pearson),
                xy=(.1, .9), xycoords=ax.transAxes)
    


def Correlation(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    """ produces grid plot for the investigation of particle data correlation
    
    for more information, see
    https://stackoverflow.com/questions/30942577/seaborn-correlation-coefficient-on-pairgrid
    """
    import NanoObjectDetection as nd
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set(style="white")
#    mygraph = sns.pairplot(sizes_df_lin[["diameter", "mass","signal"]])
    
    mycol = list(sizes_df_lin.columns)
    #remove diamter std because it is note done yet
#    mycol = [x for x in mycol if x!= "diameter std"]

    mygraph = sns.pairplot(sizes_df_lin[mycol])
    
    mygraph.map_lower(corrfunc)
    mygraph.map_upper(corrfunc)

    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Correlation",
                                       settings, data = sizes_df_lin)
        
    plt.show()



def Pearson(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    import NanoObjectDetection as nd
    from NanoObjectDetection.PlotProperties import axis_font
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    sns.set_context("paper", rc={"font.size":int(axis_font["size"]),
                                 "axes.titlesize":8,
                                 "axes.labelsize":100})
                            
    mycorr = sizes_df_lin.corr()
    
    plt.figure()
    sns.set(font_scale=0.6)
#    mygraph = sns.heatmap(np.abs(mycorr), annot=True, vmin=0, vmax=1)
    mygraph = sns.heatmap(mycorr, annot=True, vmin=-1, vmax=1, cmap="bwr", cbar_kws={'label': 'Pearson Coefficent'})
    
#    sns.set(font_scale=1.2)
#    mygraph = sns.clustermap(np.abs(mycorr), annot=True)
    

    if save_plot == True:
        print("Saving the Pearson heat map take a while. No idea why =/")
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "PearsonCoefficent",
                                       settings, data = sizes_df_lin)

    plt.show()
    


def ShowAllCorrel(sizes_df_lin,min_correl = 0.4):
    mycorr = sizes_df_lin.corr()
    mycorr_np = np.abs(mycorr.values)
    mycorr_high = (mycorr_np > min_correl) & (mycorr_np < 1)  
    
    # get position of high correlation
    index_corr_high = np.argwhere(mycorr_high)

    # removed mirror side
    index_corr_high = index_corr_high[index_corr_high[:,0] > index_corr_high[:,1]]
    
    mycol = sizes_df_lin.columns
    mycol_corr = mycol[index_corr_high]
    
    for num_loop, (col1, col2) in enumerate(zip(mycol_corr[:,0],mycol_corr[:,1])):
        print(col1,col2)
        
        Plot2DScatter(sizes_df_lin[col1],sizes_df_lin[col2],
                      title = "Correlated Parameters",
                      xlabel = col1, ylabel = col2, myalpha = 1)
#        plt.plot(,sizes_df_lin[col2],'x')
        
    
    plt.show()



def DiameterOverTrajLenght(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None):
    """ plot (and save) calculated particle diameters vs. the number of frames
    where the individual particle is visible (in standardized format) """
        
    import NanoObjectDetection as nd

    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    
    if Histogramm_min_max_auto == 1:
        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
    else:
        histogramm_min = None
        histogramm_max = None 

#    bp()
    
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    my_title = "Particle size over tracking time"
    my_ylabel = "Diameter [nm]"
    my_xlabel = "Trajectory length [frames]"
    
    plot_diameter = sizes_df_lin["diameter"]
    plot_traj_length = sizes_df_lin["traj length"]
    
    x_min_max = nd.handle_data.Get_min_max_round(plot_traj_length, 2)
    x_min_max[0] = 0
    
    y_min_max = nd.handle_data.Get_min_max_round(plot_diameter,1)
    
    Plot2DScatter(plot_traj_length, plot_diameter, title = my_title, 
                  xlabel = my_xlabel, ylabel = my_ylabel,
                  myalpha = 0.6, x_min_max = x_min_max, y_min_max = y_min_max)
 
    if save_plot == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "DiameterOverTrajLength",
                                       settings, data = sizes_df_lin, ShowPlot = show_plot)



def DiameterHistogrammTime(ParameterJsonFile, sizes_df_lin, show_plot = None, save_plot = None, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None):
    """ plot (and save) temporal evolution of the diameter histogram """
    
    #import NanoObjectDetection as nd
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
        
    Histogramm_Show = settings["Plot"]['Histogramm_time_Show']
    Histogramm_Save = settings["Plot"]['Histogramm_time_Save']
    
    if settings["Plot"]["Histogramm_Bins_Auto"] == 1:
        settings["Plot"]["Histogramm_Bins"] = NumberOfBinsAuto(sizes_df_lin)
    binning = settings["Plot"]["Histogramm_Bins"]
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    if Histogramm_min_max_auto == 1:

        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
        
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")
    
    xlabel = 'Diameter [nm]'
    ylabel = 'Absolute occurrence' # better: "Absolute number of occurrences"?
    title = 'Amount of particles analyzed = %r' % len(sizes_df_lin)

    nd.visualize.PlotDiameter2DHistogramm(sizes_df_lin, binning, histogramm_min,
                                          histogramm_max, title, xlabel, ylabel)
  
    if Histogramm_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter_Histogramm_Time", settings, data = sizes_df_lin)
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 



def PlotDiameterHistogramm(sizes, binning, histogramm_min = 0, histogramm_max = 10000, title = '', xlabel = '', ylabel = ''):
    """ plot a histogram of particles sizes

    Parameters
    ----------
    sizes : pandas.Series
    binning : 

    Returns
    -------
    values_hist : 
    ax : AxesSubplot object

    """
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    # import NanoObjectDetection as nd

#    plt.figure()
    fig, ax = plt.subplots()
    diameters = sizes
    show_diameters = diameters[(diameters >= histogramm_min) & (diameters <= histogramm_max)]
    values_hist, positions_hist = np.histogram(sizes, bins = binning)
    # histogram of sizes, only taking into account 
    sns.distplot(show_diameters, bins=binning, rug=True, kde=False) 
    #those that are below threshold size as defined in the initial parameters
#    plt.rc('text', usetex=True)
    plt.rc('text', usetex=False)
    plt.title(title, **title_font)
    #   plt.ylabel(r'absolute occurrence')
    plt.ylabel(ylabel, **axis_font)
    plt.xlabel(xlabel, **axis_font)
    plt.grid(True)
 
    ax.set_xlim(histogramm_min, histogramm_max)

    return values_hist, ax


def PlotDiameter2DHistogramm(sizes_df_lin, binning, histogramm_min = 0, histogramm_max = 10000, title = '', xlabel = '', ylabel = ''):
    """ plot histogram evolution over time
    
    The figure consists of 3 subplots: 
        - a color matrix indicating the number of particles measured with a
          certain size in each frame ('2D histogram')
        - a size histogram on top
        - a 'particles analyzed per frame' plot on the right
    """
    sns.set(style="darkgrid")
    
    time_points = int(sizes_df_lin["traj length"].sum()) # ... for all trajectories
    plot_data = np.zeros([time_points,2]) # create result matrix
    
    # get rid of these strange array-type index values
    sizes_df_lin = sizes_df_lin.set_index("true_particle")
    
    first_element = True # ... for 'plot_data' array initialization only
    
    # iterate over results' DataFrame 
    # NB: particle_data is pd.Series type (with previous column names as index)
    for particle_index, particle_data in sizes_df_lin.iterrows():
        ii_first_frame = int(particle_data["first frame"])
        ii_traj_length = int(particle_data["traj length"])
        ii_last_frame = ii_first_frame + ii_traj_length - 1
        ii_diameter = particle_data["diameter"]
        
        add_data_plot = np.zeros([ii_traj_length,2])
        
        # fill matrix with frames of appearance (col0) 
        # and diameter values (col1; const.)
        add_data_plot[:,0] = np.linspace(ii_first_frame,ii_last_frame,ii_traj_length)
        add_data_plot[:,1] = ii_diameter
        
        if first_element == True: # initialize results' array
            plot_data = add_data_plot
            first_element = False
        else: 
            # add data points to a large array
            # => matrix with frames of occurence and repeated diameter entries 
            #    (cf. 'weighting' option for the 1D histogram)
            plot_data = np.concatenate((plot_data,add_data_plot), axis = 0)

    from matplotlib.ticker import NullFormatter
    
    # the random data (MN2101: what do you want to say with this??)
    diam = plot_data[:,1] # x: repeated diameter values to count into the histogram
    time = plot_data[:,0] # y: frame numbers where analyzed particles appear
    
    # define plot limits and properties
    lim_diam_start = np.min(diam)
    lim_diam_end = np.max(diam)
    lim_time_start = np.min(time)
    lim_time_end = np.max(time)+1
    
    if binning == None:
        diam_bins_nm = 5
        diam_bins = int((lim_diam_end - lim_diam_start)/diam_bins_nm)
    else:
        diam_bins = binning
    
    time_bins = int(lim_time_end - lim_time_start + 0)
    
    nullfmt = NullFormatter()         # no labels
    
    # define the dimensions of different image sections/subplots
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02
    
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]
    
    # start with a quadratic Figure
    plt.figure(figsize=(8, 8))
    # define specific axis handles for the 3 image sections
    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)
    
    # delete ticks between the image sections
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
    
    # plot the 2D histogram/'scatter plot' into the major image section
    axScatter.hist2d(diam, time, bins = [diam_bins, time_bins], cmap = "jet")
    axScatter.set_xlabel("Diameter [nm]")
    axScatter.set_ylabel("Time [frames]")
    axScatter.set_xlim((lim_diam_start, lim_diam_end))
    axScatter.set_ylim((lim_time_start, lim_time_end))
    
    # plot the 1D histograms alongside into the minor image sections (top+right)
    axHistx.hist(diam, bins=diam_bins)
    axHistx.set_ylabel("Occurrence [counts]")
    time_hist = axHisty.hist(time, bins=time_bins, orientation='horizontal')
    axHisty.set_xlabel("Analyzed particles")
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())
    
    # plt.show()      

    

def NumberOfBinsAuto(mydata, average_height = 4):
    number_of_points = len(mydata)
    
    bins = int(np.ceil(number_of_points / average_height))
    
    return bins



def DiameterHistogramm(ParameterJsonFile, sizes_df_lin, histogramm_min = None, histogramm_max = None, Histogramm_min_max_auto = None, binning = None, weighting=False, num_dist_max=2, showInfobox=True, fitNdist=False, showICplot=False):
    """ wrapper for plotting a histogram of particles sizes, 
    optionally with statistical information and distribution fit
    """
    
    title = 'Amount of trajectories analyzed = %r' % len(sizes_df_lin)
    
    if weighting==True:
        # repeat each size value as often as the particle appears in the data
        # e.g. sizes=[5,4,7] with tracklengths=[2,3,2]
        #      => sizes_weighted=[5,5,4,4,4,7,7]
        sizes = sizes_df_lin.diameter.repeat(np.array(sizes_df_lin['valid frames'],dtype='int'))
        title = title + ', weighted by track length'
    else:
        if type(sizes_df_lin) is pd.DataFrame:
            sizes = sizes_df_lin.diameter
        else:
            sizes = sizes_df_lin # should be np.array or pd.Series type here
        title = title + ', each track counted once'     
        
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    # read in plotting parameters
    Histogramm_Show = settings["Plot"]['Histogramm_Show']
    Histogramm_Save = settings["Plot"]['Histogramm_Save']
    
    if settings["Plot"]["Histogramm_Bins_Auto"] == 1:
        settings["Plot"]["Histogramm_Bins"] = NumberOfBinsAuto(sizes_df_lin)
    if binning is None:
        #binning = np.arange(settings["Plot"]["Histogramm_Bins"])
        binning = settings["Plot"]["Histogramm_Bins"]
    if Histogramm_min_max_auto is None:
        Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    if Histogramm_min_max_auto == 1:
        histogramm_min = np.round(np.min(sizes) - 5, -1)
        histogramm_max = np.round(np.max(sizes) + 5, -1)
        
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")

    xlabel = 'Diameter [nm]'
    ylabel = 'Absolute occurrence'

    values_hist, ax = nd.visualize.PlotDiameterHistogramm(sizes, binning, histogramm_min, histogramm_max, title, xlabel, ylabel)
    
    if (showInfobox==True) and (fitNdist==False):
        nd.visualize.PlotInfobox1N(ax, sizes)
    
    if settings["Plot"]["Histogramm_Fit_1_Particle"] == 1:
        diam_grid = np.linspace(histogramm_min,histogramm_max,1000)
        max_hist = values_hist.max()
        
        if fitNdist == False: 
            # consider only one contributing particle size
            nd.visualize.PlotReciprGauss1Size(ax, diam_grid, max_hist, sizes)
            
        else:
            # consider between 1 and num_dist_max contributing particle sizes
            diam_means, diam_CI68s, weights = \
                nd.visualize.PlotReciprGaussNSizes(ax, diam_grid, max_hist, sizes,
                                                   num_dist_max=num_dist_max,
                                                   showICplot=showICplot)
                # weights should be the same, no matter if inverse or not (?!)
            
            if showInfobox==True:
                nd.visualize.PlotInfoboxMN(ax, diam_means, diam_CI68s, weights)
            
    if Histogramm_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Diameter_Histogramm", settings, data = sizes_df_lin, ShowPlot = Histogramm_Show)
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings) 
    
    return ax



def DiameterPDF(ParameterJsonFile, sizes_df_lin, PDF_min_max_auto = None, 
                fitNdist=False, num_dist_max=2, useAIC=True,
                statsInfo=True, showInfobox=True, showICplot=False, 
                fillplot=True, mycolor='C2', plotStats=False, useCRLB=True):
    """ calculate and plot the diameter probability density function of 
    a particle ensemble as the sum of individual PDFs
    NB: each trajectory is considered individually, the tracklength determines
        the PDF widths
    
    assumption: 
        relative error = std/mean either given by CRLB (if useCRLB==True) or by 
        sqrt( 2*N_tmax/(3*N_f - N_tmax) ) 
        with N_tmax : number of considered lagtimes
             N_f : number of frames of the trajectory (=tracklength)

    Parameters
    ----------
    fitNdist : boolean, optional
        If True, a GMM is fitted to the data. The default is False.
    num_dist_max : integer, optional
        Maximum number of Gaussians to fit. Default is 2.

    Returns
    -------
    prob_inv_diam
    diam_grid
    ax
    """
    # import NanoObjectDetection as nd
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DiameterPDF_Show = settings["Plot"]['DiameterPDF_Show']
    DiameterPDF_Save = settings["Plot"]['DiameterPDF_Save']
    
    
    # calculate mean and std of the ensemble (inverse!) and the grid for plotting
    # THE INVERSE DIAMETER IS USED BECAUSE IT HAS A GAUSSIAN DISTRIBUTED ERROR; 
    # WHILE THE DIAMETER ITSELF DOES NOT HAVE SUCH AN EASY PROBABILTY DENSITY FUNCTION (PDF)
    # CALCULATE IT IN THE INVERSE REGIME FIRST; AND CALCULATE BACK TO DIAMETER SPACE
    diam_inv_mean, diam_inv_std = nd.statistics.StatisticOneParticle(sizes_df_lin)
    
    diam_grid = np.linspace(0.1,1000,10000) # prepare grid for plotting (equidistant)
    # diam_grid = np.linspace(1,1000,20000) # prepare grid for plotting (equidistant)
    diam_grid_stepsize = diam_grid[1] - diam_grid[0]
    # diam_grid_inv = 1/diam_grid
    
    # array that save the PDF
    prob_diam = np.zeros_like(diam_grid)
    
    # get mean and std of each inverse diameter
    inv_diam, inv_diam_std = nd.statistics.InvDiameter(sizes_df_lin, settings, useCRLB) # 1/um
    inv_diam = 0.001* inv_diam # 1/nm
    inv_diam_std = 0.001*inv_diam_std # 1/nm
    
    
    for index, (loop_mean, loop_std) in enumerate(zip(inv_diam,inv_diam_std)):
        # law of inverse fcts
        # Y = 1/X --> PDF_Y(Y)=1/(Y^2) * PDF_X(1/Y)
        my_pdf = scipy.stats.norm(loop_mean,loop_std).pdf(1/diam_grid)
        my_pdf = 1/(diam_grid**2) * my_pdf
        
        # normalization to 1
        my_pdf = my_pdf / np.sum(my_pdf) 
        
        # sum it all up
        prob_diam = prob_diam + my_pdf    
    
    # normalize the summed pdf to integral=1
    prob_diam = prob_diam / (np.sum(prob_diam) * diam_grid_stepsize)
    
    # get statistical quantities
    diam_median = np.median(nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0))
    # diam_mean = np.average(diam_grid, weights = prob_diam) # MN: this seems only justified on equidistant grids
    diam_mean = sum(prob_diam*diam_grid)*diam_grid_stepsize
    lim_68CI = nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.68) 
    lim_95CI = nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.95)
    num_trajectories = len(sizes_df_lin) 
        
    
    # set the x limits in the plot
    PDF_min_max_auto = settings["Plot"]["DiameterPDF_min_max_auto"]
    if PDF_min_max_auto == 1:
        PDF_min, PDF_max = nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.999)
        # PDF_min, PDF_max = nd.statistics.GetCI_Interval(prob_inv_diam, diam_grid, 0.999)
        PDF_min = int(np.floor(PDF_min))
        PDF_max = int(np.ceil(PDF_max))
    else:
        PDF_min = settings["Plot"]['PDF_min'] 
        PDF_max = settings["Plot"]['PDF_max'] 
        
        
    # info box for the plot
    if statsInfo==True:    
        title = str("{:.0f} trajectories; mean: {:.2f}, median: {:.1f}, \n CI68: [{:.1f} : {:.1f}],  CI95: [{:.1f} : {:.1f}] nm".format(num_trajectories, diam_mean, diam_median, lim_68CI[0], lim_68CI[1],lim_95CI[0], lim_95CI[1]))
    else:
        title = str("Trajectories: {:.0f}".format(num_trajectories))
        
    xlabel = "Diameter [nm]"
    ylabel = "Probability [a.u.]"
    x_lim = [PDF_min, PDF_max]
    # y_lim = [0, 1.1*np.max(prob_inv_diam)]
    y_lim = [0, 1.1*np.max(prob_diam)]
#    sns.reset_orig()

    # plot it!
    ax = Plot2DPlot(diam_grid, prob_diam, title, xlabel, ylabel, mylinestyle = "-",  
                    mymarker = "", x_lim = x_lim, y_lim = y_lim, y_ticks = [0,0.01], 
                    semilogx = False, FillArea = fillplot, Color = mycolor)
    
    if plotStats: # plot lines for mean, median and CI68-interval
        ax.vlines([diam_mean, diam_median], 0, 1,
                  transform=ax.get_xaxis_transform(), 
                  colors=mycolor, alpha=0.8)
        ax.vlines([lim_68CI[0], lim_68CI[1]], 0, 1, 
                  transform=ax.get_xaxis_transform(), 
                  colors=mycolor, alpha=0.6)
    
    
    mean_diameter = np.round(np.mean(nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.001)),1)
    CI_interval = np.round(nd.statistics.GetCI_Interval(prob_diam, diam_grid, 0.68),1)
    
    nd.logger.info("mean diameter: %s", mean_diameter)
    nd.logger.info("68CI interval: %s", CI_interval)

    
    if settings["Plot"]["Histogramm_Fit_1_Particle"] == 1:
        max_PDF = prob_diam.max()
        
        if fitNdist == False: 
            # consider only one contributing particle size   
            nd.visualize.PlotReciprGauss1Size(ax, diam_grid, max_PDF, sizes_df_lin)
            
        else: 
            # consider multiple contributing particles sizes
            
            # MN: THE FOLLOWING IS A BIT EXPERIMENTAL... PROBABLY NEEDS REVIEW
            # reduce the grid from 10000 to 1000 values
            diamsR = diam_grid[::10]
            probsR = prob_diam[::10] * 6E5 # scale up to be able to use the values as integers
            probsN = probsR.round()
            # create (large!) array with repeated diameter values, cf. weighting scheme
            sizesPDF = diamsR.repeat(np.array(probsN,dtype='int'))
            
            diam_means, diam_stds, weights = \
                nd.visualize.PlotReciprGaussNSizes(ax, diamsR, max_PDF, sizesPDF,
                                                   showICplot=showICplot, useAIC=useAIC,
                                                   num_dist_max=num_dist_max)
            
            if showInfobox==True:
                nd.visualize.PlotInfoboxMN(ax, diam_means, diam_stds, weights)


    if DiameterPDF_Save == True:
        save_folder_name = settings["Plot"]["SaveFolder"]
        
        settings = nd.visualize.export(save_folder_name, "Diameter_Probability", settings,
                                       data = sizes_df_lin, ShowPlot = DiameterPDF_Show)
        
        data = np.transpose(np.asarray([diam_grid, prob_diam]))
        nd.visualize.save_plot_points(data, save_folder_name, 'Diameter_Probability_Data')
        
    nd.handle_data.WriteJson(ParameterJsonFile, settings)
    
    return prob_diam, diam_grid, ax
    
    
    
def myGauss(d,mean,std):
    return 1/((2*np.pi)**0.5*std)*np.exp(-(d-mean)**2/(2.*std**2))



def PlotReciprGauss1Size(ax, diam_grid, max_y, sizes):
    """ add reciprocal Gaussian function to a sizes histogram (or PDF)
    
    NB: No fit is computed here, only an equivalent distribution with the same
        mean and std as the original "sizes" values.
    
    With mean and std of the inversed values, a Gaussian curve in the 
    reciprocal size space is computed, back-transformed and
    normalized to the maximum histogram/PDF value.
    """
    diam_inv_mean, diam_inv_std, diam_inv_median, diam_inv_CI68 = \
        nd.statistics.StatisticOneParticle(sizes)
        
    diam_grid_inv = 1000/diam_grid # 1/um
    
    prob_diam_inv_1size = \
        scipy.stats.norm(diam_inv_mean,diam_inv_std).pdf(diam_grid_inv)
    
    # law of inverse fcts:
    prob_diam_inv_1size = 1/(diam_grid**2) * prob_diam_inv_1size
    # normalization:    
    prob_diam_inv_1size = prob_diam_inv_1size / prob_diam_inv_1size.max() * max_y
    
    # print('median={}, CI68=[{},{}]'.format(1000/diam_inv_median,1000/diam_inv_CI68[0],1000/diam_inv_CI68[1]))
    ax.plot(diam_grid,prob_diam_inv_1size)
    return diam_inv_mean, diam_inv_median, diam_inv_CI68
    
    
    
def PlotReciprGaussNSizes(ax, diam_grid, max_y, sizes, num_dist_max=2, useAIC=False, showICplot=False):
    """ plot the reciprocal fcts of Gaussian fits on top of a histogram/PDF
    
    to do:
        - build a fct that includes both inv. and non-inv. fitting (?)
        - make it working for PDF fitting as well
        
    diam_grid : np.ndarray; plotting grid [nm]
    num_dist_max : int; max. number of size components to consider
    useAIC : boolean; if True, use AIC, if False, use BIC
    """
    inv_diam = 1000/sizes # 1/um
    diam_grid_inv = 1000/diam_grid # 1/um
    
    # use Gaussian mixture model (GMM) fitting to get best parameters
    # diam_means, diam_stds, weights = \
    #     nd.statistics.StatisticDistribution(sizes, weighting=weighting,
    #                                         num_dist_max=num_dist_max,
    #                                         showICplot=showICplot, useAIC=useAIC)
    inv_diam_means, inv_diam_stds, inv_weights = \
        nd.statistics.StatisticDistribution(inv_diam, 
                                            num_dist_max=num_dist_max,
                                            showICplot=showICplot, useAIC=useAIC)
    
    # compute individual Gaussian functions from GMM fitted parameters (inv.space!)
    # dist = np.array([weights[n]*myGauss(diam_grid,diam_means[n],diam_stds[n]) 
    #                  for n in range(weights.size)])
    dist = np.array([inv_weights[n]*myGauss(diam_grid_inv,inv_diam_means[n],
                                            inv_diam_stds[n]) 
                      for n in range(inv_weights.size)])
    
    # law of inverse fcts: convert back to sizes space
    dist = ((diam_grid**2)/1000**2)**(-1) * dist
    
    # convert from 1/um to nm
    diam_means = 1000/inv_diam_means
    diam_CI68s = np.zeros([inv_weights.size,2])
    for n,d in enumerate(dist.transpose()):
        diam_CI68s[n] = nd.statistics.GetCI_Interval(d, diam_grid, 0.68) # STILL A BUG IN HERE
    # weights should stay the same (!?)
    
    # calculate sum of all distributions
    dsum = dist.sum(axis=0)
    # normalize dsum to histogram/PDF max. value...
    normFactor = max_y / dsum.max()
    dsum = normFactor * dsum
    dist = normFactor * dist # and the individual distributions accordingly
    
    ax.plot(diam_grid,dist.transpose(),ls='--')
    ax.plot(diam_grid,dist.sum(axis=0),color='k')
    
    return diam_means, diam_CI68s, inv_weights



def PlotGaussNSizes(ax, diam_grid, max_y, sizes, num_dist_max=2, weighting=False, useAIC=False, showICplot=False):
    """ plot Gaussian fits on top of a histogram/PDF
    
    probably a bit redundant... but at least it's working reliably
    """    
    # use Gaussian mixture model (GMM) fitting to get best parameters
    diam_means, diam_stds, weights = \
        nd.statistics.StatisticDistribution(sizes, weighting=weighting,
                                            num_dist_max=num_dist_max,
                                            showICplot=showICplot, useAIC=useAIC)
    
    # compute individual Gaussian functions from GMM fitted parameters (inv.space!)
    dist = np.array([weights[n]*myGauss(diam_grid,diam_means[n],diam_stds[n]) 
                     for n in range(weights.size)])
    
    # calculate sum of all distributions
    dsum = dist.sum(axis=0)
    # normalize dsum to histogram/PDF max. value...
    normFactor = max_y / dsum.max()
    dsum = normFactor * dsum
    dist = normFactor * dist # and the individual distributions accordingly
    
    ax.plot(diam_grid,dist.transpose(),ls='--')
    ax.plot(diam_grid,dist.sum(axis=0),color='k')
    
    return diam_means, diam_stds, weights




def PlotInfobox1N(ax, sizes):
    """ add textbox to a plot containing statistical information 
    on the size distribution, assuming one single contributing particle size
    
    NB: Quantiles are used here in order to give an idea of the sigma-intervals
        independent of any Gaussian fit.
    """
    from NanoObjectDetection.PlotProperties import axis_font
    
    _, _, my_median = nd.statistics.GetMeanStdMedian(sizes)
    
    diam_inv_mean, diam_inv_std = nd.statistics.StatisticOneParticle(sizes)
    my_mean = 1/diam_inv_mean
    
    # percentage of values that is within 1 or 2 sigma (for the normal distr.):
    sigma1 = 0.682689492
    sigma2 = 0.954499736
    
    if type(sizes) is pd.DataFrame:
        sizes = sizes.diameter
    elif not(type(sizes) is pd.Series):
        sizes = pd.Series(sizes) # assume np.array
        
    # use quantiles to describe the sigma-intervals
    # (this accounts for the skewness of the distribution)
    diam_68 = [sizes.quantile(0.5 - sigma1/2), sizes.quantile(0.5 + sigma1/2)]
    diam_95 = [sizes.quantile(0.5 - sigma2/2), sizes.quantile(0.5 + sigma2/2)]
# #   old version (uses the assumption of a Gaussian fct):
#     diam_68 = [1/(diam_inv_mean + 1*diam_inv_std), 1/(diam_inv_mean - 1*diam_inv_std)]
#     diam_95 = [1/(diam_inv_mean + 2*diam_inv_std), 1/(diam_inv_mean - 2*diam_inv_std)]
# #    diam_99 = [1/(diam_inv_mean + 3*diam_inv_std), 1/(diam_inv_mean - 3*diam_inv_std)]
    
    textstr = '\n'.join([
    r'$\mathrm{median}=  %.1f$ nm' % (my_median),
    r'$\mu_{\mathrm{inv}} = %.1f$ nm' % (my_mean),
    r'$1 \sigma_{\mathrm{q}} = [%.1f; %.1f]$ nm' %(diam_68[0], diam_68[1]), 
    r'$2 \sigma_{\mathrm{q}} = [%.1f; %.1f]$ nm' %(diam_95[0], diam_95[1]), ])
    
    props = dict(boxstyle='round', facecolor='honeydew', alpha=0.7)
    
#    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
#        , bbox=props)
    ax.text(0.55, 0.95, textstr, transform=ax.transAxes, 
            **axis_font, verticalalignment='top', bbox=props)
    
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    ax.text(0.05, 0.95, title, transform=ax.transAxes, fontsize=14,
#            verticalalignment='top', bbox=props) 



def PlotInfoboxMN(ax, diam_means, diam_CI68, weights):
    """ add textbox to a plot containing statistical information 
    on the size distribution, assuming several contributing particle sizes
    """
    # from NanoObjectDetection.PlotProperties import axis_font
    
    mtext = ''
    stext = ''
    wtext = ''
    for n in range(weights.size):
        mtext += '{:.1f},'.format(diam_means[n])
        stext += '{:.1f},'.format(diam_CI68[n])
        wtext += '{:.1f},'.format(100*weights[n]) # [%]
    
    textstr = '\n'.join([
        r'$\mu$ = ['+mtext[:-1]+'] nm', # '[:-1]' removes the last ','
        r'CI68 = ['+stext[:-1]+'] nm', 
        r'$\phi$ = ['+wtext[:-1]+'] %', ])

    props = dict(boxstyle='round', facecolor='honeydew', alpha=0.7)
                
    x_text = 0.7 - (0.05*weights.size)
    ax.text(x_text, 0.95, textstr, transform=ax.transAxes, 
            **{'fontname':'Arial', 'size':'15'}, #**axis_font, 
            verticalalignment='top', bbox=props)
    


def update_progress(job_title, progress):
    """ display a progress bar for an ongoing task
    """
    length = 50 # modify this to change the length
    block = int(round(length*progress))
    msg = "\r{0}: [{1}] {2}%".format(job_title, "#"*block + "-"*(length-block), round(progress*100))
    if progress >= 1: msg += " DONE\r\n"
    sys.stdout.write(msg)
    sys.stdout.flush()



def GetPlotParameters(settings):
    
    with open(settings["Plot"]['SaveProperties']) as json_file:
        catch_all = json.load(json_file)
        
    bp() 
    params = catch_all["params"]
    title_font = catch_all["title_font"]
    axis_font = catch_all["axis_font"]
    
    return params, title_font, axis_font
    


def CreateFileAndFolderName(folder_name, file_name, d_type = 'png'):
    '''
    make a folder of the current date (if required)
    and file name (including data) to save stuff in there 
    '''
    
    dir_name = '%s\\{date:%y%m%d}\\'.format( date=datetime.datetime.now()) %folder_name

    if len(dir_name) > 259:
        print("\n directory name: \n", dir_name)
        raise ValueError ("directory name to long for windows!")

    try:
        os.stat(dir_name)
    except:
        os.mkdir(dir_name) 
    
    time_string = '{date:%H_%M_%S}'.format( date=datetime.datetime.now())
    
#    file_name_image = '%s_%s.png'.format( date=datetime.datetime.now()) %(time_string, file_name)
    file_name_image = '%s_%s'.format( date=datetime.datetime.now()) %(time_string, file_name)
    file_name_image = file_name_image + '.' + d_type
    
    entire_path = dir_name +  file_name_image

    if len(entire_path) > 259:
        print("\n entire path: \n", entire_path)
        raise ValueError ("full path to long name to long for windows!")

    return dir_name, entire_path, time_string


def save_plot_points(data, save_folder_name, save_csv_name):
    dir_name, entire_path, time_string = CreateFileAndFolderName(save_folder_name, save_csv_name, d_type = 'csv')
    
    np.savetxt(entire_path, data, delimiter = ',')



def export(save_folder_name, save_image_name, settings = None, use_dpi = None, data = None, data_header = None, ShowPlot = -1):    
    import NanoObjectDetection as nd
    if settings is None:
        if use_dpi is None:
            sys.exit("need settings or dpi!")
    else:
        use_dpi, settings = nd.handle_data.SpecificValueOrSettings(use_dpi,settings, "Plot", "dpi")
        save_json         = settings["Plot"]["save_json"]
        save_data2csv     = settings["Plot"]["save_data2csv"]
        
    use_dpi = int(use_dpi)
    
    my_dir_name, entire_path_image, time_string = CreateFileAndFolderName(save_folder_name, save_image_name)

    
    plt.savefig(entire_path_image, dpi= use_dpi, bbox_inches='tight')
    nd.logger.info('Figure saved at: {}'.format(my_dir_name))

#    Image.open(entire_path_image).show()

    #here comes the json parameter file 
    if save_json == 1:
        file_name_json = '%s_%s.json'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
        entire_path_json = my_dir_name +  file_name_json
        
        if settings is None:
            settings = 0
        else:
            settings["Plot"]["SaveProperties"] = entire_path_json
        
            with open(entire_path_json, 'w') as outfile:
                json.dump(settings, outfile, sort_keys=True, indent=4)

    #here comes the csv data file 
             
    if save_data2csv == 1:
        if (data is None) == False:
                #here comes the data file 
                data_name_csv = '%s_%s.csv'.format( date=datetime.datetime.now()) %(time_string, save_image_name)
                entire_path_csv = my_dir_name +  data_name_csv
                if type(data) == np.ndarray:
                    np.savetxt(entire_path_csv, data, delimiter="," , fmt='%.10e', header = data_header)
                else:
                    data.to_csv(entire_path_csv, index = False)
         
     # if plot is only displayed in order save it, close it now
    if ShowPlot == False:
        nd.logger.info("Close plot since noone wants it")
        plt.close(plt.gcf())   
    elif ShowPlot == -1:
        nd.logger.info("You can prevent the plot to be shown if you just wanna save it in the export function")
        
         
    return settings



def IntMedianFit(t1_gapless, my_particle = -1):
    """ plot comparison of the median filter smoothed particle intensity ("mass") with original values """
    
    if my_particle == -1:
        # get the first occuring particle
        # (can be different from 0 or 1 because they might have been thrown away due to being stumps)
        my_particle = t1_gapless.iloc[0].particle
        
#    show_curve = t1_gapless[t1_gapless['particle'] == my_particle][['mass','mass_smooth','rel_step']]
    show_curve = t1_gapless[t1_gapless['particle'] == my_particle][['mass','mass_smooth']]
    plt.figure()
    plt.plot(show_curve.mass, 'x:')
    plt.plot(show_curve.mass_smooth, '-')
    plt.title("Intensity Analysis of each Particle")
    plt.legend(['Mass', 'Median Filter'])
    plt.xlabel("Frame")
    plt.ylabel("Mass")
    plt.ylim(0)
    
    
    
def MaxIntFluctuationPerBead(t1_gapless): 
    """ show the maximum relative step for each particle """

    beads_property = pd.DataFrame(columns=['max_step']);
    
    beads_property['max_step'] = t1_gapless.groupby('particle')['rel_step'].max()
    
    # show the maximum relative step for each particle
    plt.figure()
    plt.plot(beads_property,'x')
    plt.title("Intensity Analysis of each Particle")
    plt.xlabel("particle")
    plt.ylabel("Maximum relative step")


    
def CutTrajectorieAtStep(t1_gapless, particle_to_split, max_rel_median_intensity_step):  
    """ display the intensity ("mass") vs. frame# of a particle trajectory that needs to be split """
    
    show_with_fit = t1_gapless[t1_gapless.particle == particle_to_split]
    show_no_fit= t1_gapless[t1_gapless.particle == particle_to_split]
    
    # do not show fitted data on existing sampling point
    show_with_fit.loc[show_no_fit.index, 'mass'] = math.nan
    
    plt.figure()
    #top plot with intensity curve
    ax1 = plt.subplot(2, 1, 1)
    
    # four curves in one plot
    plt.plot(show_no_fit.mass,'.',color='black')
    plt.plot(show_with_fit.mass, '.',color='orange')
    plt.plot(show_with_fit.mass_smooth, '-',color='black')
    #plt.plot(show_no_fit.mass_smooth,':',color='blue')
    
    plt.title("Intensity of Particle No. %i" %particle_to_split)
    #plt.legend(['Fitted Mass', 'Fitted Median Filter','Mass','Median Filter'])
    plt.legend(['Mass','Interpolated Mass', 'Fitted Median Filter'])
    plt.xlabel("Frame")
    plt.ylabel("Mass")
    plt.ylim(0)
    plt.xlim(show_with_fit.index[0],show_with_fit.index[-1])
    ax1.xaxis.grid()

    #bottom plot with slope curve
    ax2 = plt.subplot(2, 1, 2)
    plt.plot(show_with_fit.rel_step, ':.',color='black')
    
    # show threshold
    my_xlim = ax2.get_xlim()
    ax2.hlines(y=max_rel_median_intensity_step, xmin=my_xlim[0], xmax=my_xlim[1], linewidth=2, color='red')
        
    plt.title("Slope of Intensity of Particle No. %i" %particle_to_split)
    plt.legend(['Slope Fitted Mass','Threshold'])
    plt.xlabel("Frame")
    plt.ylabel("Mass")
    plt.ylim(0,2*max_rel_median_intensity_step)
    plt.xlim(show_with_fit.index[0],show_with_fit.index[-1])
    ax2.xaxis.grid()



def DriftAvgSpeed(drift):
    # show average speed
    #ax = plt.figure()
    drift.plot( marker='.')
    plt.title('Drift in laminar flow ')
    plt.ylabel('Drift [px]')
    plt.xlabel('Lateral position in fibre [px]')
    plt.legend(['y', 'x (flow-direction)'])
        
     
        
def DriftTimeDevelopment(drift, number_blocks):
    # SW 180701: There is some issue with the plot in the block beneath!
    # show time developement
    fig, ax = plt.subplots()
    drift.groupby(['y_range'])[['x']].plot(ax=ax)
    ax.legend(np.linspace(1,number_blocks,number_blocks,dtype='int'))
    plt.title('Drift in laminar flow ')
    plt.ylabel('Drift [px]')
    plt.xlabel('Time frame')
    
    
    
def DriftFalseColorMapFlow(calc_drift, number_blocks, y_range):
    
    # show false colour map from flow
    # https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html
    fig, ax = plt.subplots()
    disp_calc_drift = pd.DataFrame()
    for x in range(0,number_blocks):
        disp_calc_drift[str(x)] = calc_drift[calc_drift.y_range == y_range[x]].x
    #heatmap = ax.pcolormesh(y_range,disp_calc_drift.index.values,disp_calc_drift, cmap=plt.cm.hsv, alpha=1)
    #heatmap = ax.contourf(y_range,disp_calc_drift.index.values,disp_calc_drift, 50, cmap="jet", alpha=1, origin='lower')
    heatmap = ax.contourf(disp_calc_drift.index.values,y_range,disp_calc_drift.transpose(), 50, cmap="jet", alpha=1, origin='lower')
    plt.title('y-dependend flow')
    plt.xlabel('Frame')
    plt.ylabel('Lateral position in fibre [px]')
    plt.colorbar(mappable=heatmap, orientation='vertical', label='Drift along fiber direction')
     
    
    
def DriftVectors(calc_drift, number_blocks, y_range):
        
    # DRIFT VECTORS
    # https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html
    fig, ax = plt.subplots()
    disp_calc_drift = pd.DataFrame()
    for x in range(0,number_blocks):
        disp_calc_drift[str(x)] = calc_drift[calc_drift.y_range == y_range[x]].x
    
    start_frame = min(disp_calc_drift.index)
    end_frame = max(disp_calc_drift.index)
    #step = 100;
    
    sampling_grid_y_sub = 1;
    sampling_grid_frame = 100;
    
    my_frames = np.arange(start_frame,end_frame, sampling_grid_frame);
    my_ysub = np.arange(1,number_blocks, sampling_grid_y_sub);
    
    drift_grid = disp_calc_drift.loc[my_frames];
    drift_vec = drift_grid.diff()
    drift_vec = drift_vec.iloc[1:]
    my_frames = my_frames[1:]
    drift_vec_np = np.array(drift_vec)
    
    arrow_x = -drift_vec_np[:,my_ysub]
    arrow_y = arrow_x.copy()
    arrow_y[:,:] = 0
    
    q = plt.quiver(my_frames, y_range[my_ysub], np.matrix.transpose(arrow_x), arrow_y)
          
    qk = plt.quiverkey(q, 0.3, 0.9, 2, '', labelpos='E', coordinates='figure')
    
    plt.title('Horizontal drift between frame')
    plt.xlabel('frame')
    plt.ylabel('Lateral position in core [px]')
    
    
#        heatmap = ax.contourf(disp_calc_drift.index.values,y_range,disp_calc_drift.transpose(), 50, cmap="jet", alpha=1, origin='lower')
#        plt.title('y-dependend flow')
#        plt.xlabel('Frame')
#        plt.ylabel('Lateral position in fibre [px]')
#        plt.colorbar(mappable=heatmap, orientation='vertical', label='Drift along fiber direction')
    #    
        

def DriftFalseColorMapSpeed(calc_drift, number_blocks, y_range):
    # show false colour map from SPEED
    # https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html
    fig, ax = plt.subplots()
    disp_calc_drift = pd.DataFrame()
    for x in range(0,number_blocks):
        disp_calc_drift[str(x)] = calc_drift[calc_drift.y_range == y_range[x]].velocity_x
    #heatmap = ax.pcolormesh(y_range,disp_calc_drift.index.values,disp_calc_drift, cmap=plt.cm.hsv, alpha=1)
    #heatmap = ax.contourf(y_range,disp_calc_drift.index.values,disp_calc_drift, 50, cmap="jet", alpha=1, origin='lower')
    heatmap = ax.contourf(disp_calc_drift.index.values,y_range,disp_calc_drift.transpose(), 50, cmap="Greys", alpha=1, origin='lower')
    plt.title('Speed in fibre')
    plt.xlabel('Frame')
    plt.ylabel('Lateral position in fibre [px]')
    plt.colorbar(mappable=heatmap, orientation='vertical', label='Speed')


def DriftCorrectedTraj(tm_sub):
    # Show drift-corrected trajectories
    fig, ax = plt.subplots()
    tp.plot_traj(tm_sub)
    plt.title('Trajectories: Drift-Correction depending on laterial-position (y)')
    plt.xlabel('x-Position [px]')
    plt.ylabel('Lateral position in fibre (y) [px]')
    
    
def PlotGlobalDrift(d,settings,save=True):
    my_font = 18
    plt.figure(figsize=(6,6))
    plt.plot(d)
    plt.title("accumulated drift", fontsize = my_font+2)
    plt.xlabel("frames", fontsize = my_font)
    plt.ylabel("drift [px]", fontsize = my_font)
    plt.legend(("transversal","along fiber"), fontsize = my_font-2)
    
    if save==True:
        save_folder_name = settings["Plot"]["SaveFolder"]
        settings = nd.visualize.export(save_folder_name, "Global_Drift", settings)
    
   
    
def MsdOverLagtime(lagt_direct, mean_displ_direct, mean_displ_fit_direct_lin, collect_data = None, alpha_values = 0.5, alpha_fit = 0.3):
    from NanoObjectDetection.PlotProperties import axis_font, title_font

    t_ms = lagt_direct * 1000

    # plotting msd-lag-time-tracks for all particles
    plt.plot(t_ms[:-1], mean_displ_direct,'k.', alpha = alpha_values) 
    
    # plotting the lin fits
    plt.plot(t_ms, mean_displ_fit_direct_lin, 'r-', alpha = alpha_fit) 
    
    #ax.annotate(particleid, xy=(lagt_direct.max(), mean_displ_fit_direct_lin.max()))
    plt.title("MSD fit", **title_font)
    plt.ylabel(r"MSD $[\mu m^2]$", **axis_font)
    plt.xlabel("Lagtime [ms]", **axis_font)

    
    
# def VarDiameterOverTracklength(sizes_df_lin, save_folder_name = r'Z:\Datenauswertung\19_ARHCF', save_image_name = 'errorpropagation_diameter_vs_trajectorie_length'):
#     import NanoObjectDetection as nd
#     from nd.PlotProperties import axis_font, title_font
#     # NO WORKING AT THE MOMENT BECAUSE ERROR IS NOT RIGHT YET
    
#     import ronnys_tools as rt
# #    from ronnys_tools.mpl_style import axis_font, title_font
    
#     ax = plt.subplot(111)
#     fig = plt.errorbar(sizes_df_lin['frames'],sizes_df_lin['diameter'], yerr = sizes_df_lin['diameter_std'],fmt='.k', alpha = 0.4, ecolor='orange')
    
#     ax.set_yscale("log")
#     ax.set_ylim(10,300)
    
#     ax.set_yticks([10, 30, 100, 300])
#     ax.set_yticklabels(['10', '30', '100', '300'])
    
#     ax.set_ylabel('Diameter [nm]', **axis_font)
    
#     ax.set_xscale("log")
#     ax.set_xlim(30,1000)
    
#     ax.set_xticks([30, 100, 300, 1000])
#     ax.set_xticklabels(['30', '100', '300', '1000'])
    
#     ax.set_xlabel('Trajectorie duration [frames]', **axis_font)
    
#     ax.grid(which='major', linestyle='-')
#     ax.grid(which='minor', linestyle=':')
    
    
#     #rt.mpl_style.export(save_folder_name, save_image_name)
    
    
def VarDiameterOverMinTracklength(settings, sizes_df_lin, obj_all, num_bins_traj_length = 100, min_traj_length = 0, max_traj_length = 1000):
    """ [insert description here] """
    import NanoObjectDetection as nd
    from nd.PlotProperties import axis_font, title_font
          
    diameter_by_min_trajectorie = pd.DataFrame(np.zeros((max_traj_length, 6)),columns=['min_trajectorie_length', 'particles_min_length', 'tot_traj_length', 'rel_traj_length', 'mean_diam_min', 'std_diam_min'])
    
    
    #    ideal_tot_traj_length = len(f)
    ideal_tot_traj_length = len(obj_all)
    
    min_tracking_frames = np.int(settings["Processing"]["min_tracking_frames"])
    
    for loop_min_traj_length in range(min_tracking_frames,max_traj_length):       
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['min_trajectorie_length'] = loop_min_traj_length       
    
        
        diameter_with_min_trajectorie = sizes_df_lin[(sizes_df_lin['frames'] >= loop_min_traj_length)]
        
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['mean_diam_min'] = np.mean(diameter_with_min_trajectorie['diameter'])
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['std_diam_min'] = np.std(diameter_with_min_trajectorie['diameter'])
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['particles_min_length'] = len(diameter_with_min_trajectorie['diameter'])
    
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['tot_traj_length'] = np.sum(diameter_with_min_trajectorie["frames"])
        diameter_by_min_trajectorie.loc[loop_min_traj_length]['rel_traj_length'] = diameter_by_min_trajectorie.loc[loop_min_traj_length]['tot_traj_length'] / ideal_tot_traj_length
    
    
    # remove empty rows before min tracking frames value
    diameter_by_min_trajectorie = diameter_by_min_trajectorie.loc[min_tracking_frames:]
    
    
    # remove empty rows where no particle is anymore because none is so long as required
    diameter_by_min_trajectorie = diameter_by_min_trajectorie[np.isnan(diameter_by_min_trajectorie['std_diam_min']) == False]
    
    mean_plus_sigma = diameter_by_min_trajectorie['mean_diam_min'] + diameter_by_min_trajectorie['std_diam_min']
    mean_minus_sigma = diameter_by_min_trajectorie['mean_diam_min'] - diameter_by_min_trajectorie['std_diam_min']
    
     
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex = True)
    
    
    ax1.loglog(diameter_by_min_trajectorie['min_trajectorie_length'], diameter_by_min_trajectorie['mean_diam_min'], label='')
    ax1.set_ylabel('Diameter of particles [nm]', **axis_font)
    
    ax1.fill_between(diameter_by_min_trajectorie['min_trajectorie_length'], mean_plus_sigma, mean_minus_sigma, alpha=.25, label='1-sigma interval')
    ax1.set_ylim([10, 100])
    ax1.set_xlim([10,1000])
    
    ax1.tick_params(axis = 'y')
    
    ax1.grid(which='major', linestyle='-')
    ax1.grid(which='minor', linestyle=':')
    
    ax1.legend()
    
    
    
    ax2.loglog(diameter_by_min_trajectorie['min_trajectorie_length'], diameter_by_min_trajectorie['std_diam_min'], label='')
    ax2.set_ylabel('Std of diameter [nm]', **axis_font)
    
    ax2.set_ylim([1, 300])
    
    ax2.tick_params(axis = 'y')
    
    ax2.grid(which='major', linestyle='-')
    ax2.grid(which='minor', linestyle=':')
    
    
    
    ax3.loglog(diameter_by_min_trajectorie['min_trajectorie_length'], diameter_by_min_trajectorie['rel_traj_length'],'-r', label='Ratio of analyzed particles')
    ax3.set_ylabel('Tracked Particles', **axis_font)
    ax3.tick_params(axis = 'y')
    
    ax3.set_xlabel('Minimum Trajectorie Time [frames]', **axis_font)
    
    ax3.grid(which='major', linestyle='-')
    ax3.grid(which='minor', linestyle=':')
    
    ax3.set_ylim([0.005,1,])
    

    
    
    save_folder_name = r'Z:\Datenauswertung\19_ARHCF'
    save_image_name = 'statistic_diameter_vs_trajectorie_length'
    
    nd.mpl_style.export(save_folder_name, save_image_name)
    
    
    
def HistogrammDiameterOverTime(sizes_df_lin,ParameterJsonFile):
    """ ?? obsolete ?? """
    import matplotlib.cm as cm
    from NanoObjectDetection.PlotProperties import axis_font, title_font
    
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
 
    binning = settings["Plot"]["Histogramm_Bins"]
    binning = 25
    
    Histogramm_min_max_auto = settings["Plot"]["Histogramm_min_max_auto"]
    if Histogramm_min_max_auto == 1:

        histogramm_min = np.round(np.min(sizes_df_lin.diameter) - 5, -1)
        histogramm_max = np.round(np.max(sizes_df_lin.diameter) + 5, -1)
        
    histogramm_min, settings = nd.handle_data.SpecificValueOrSettings(histogramm_min, settings, "Plot", "Histogramm_min")
    histogramm_max, settings = nd.handle_data.SpecificValueOrSettings(histogramm_max, settings, "Plot", "Histogramm_max")

    effective_fps = settings["MSD"]["effective_fps"]   
    
    test = sizes_df_lin[["diameter","first_frame","traj_length"]]
    diameter = np.asarray(test["diameter"])
    first_frame = np.asarray(test["first_frame"])
    traj_length = np.asarray(test["traj_length"])
    last_frame = first_frame + traj_length - 1
    
    diameter_time = np.repeat(diameter,traj_length.astype(int))
    my_frame = np.empty(diameter_time.shape)
    
    new_row = 0;
    for loop, value in enumerate(diameter):
        loop_first_frame = new_row
        loop_traj_length = np.int(traj_length[loop])
        loop_last_frame = loop_first_frame + loop_traj_length - 1
        
        frames_with_this_particle = np.linspace(first_frame[loop], last_frame[loop], traj_length[loop])
        
        my_frame[loop_first_frame : loop_last_frame + 1] = frames_with_this_particle
        
        new_row = loop_last_frame + 1
    
    my_time = my_frame / effective_fps
    

#    from NanoObjectDetection.PlotProperties import axis_font, title_font
#    import matplotlib
#    matplotlib.rcParams['text.usetex'] = True
#    matplotlib.rcParams['text.latex.unicode'] = True
    f = plt.figure(figsize=(10,10))
     

    
    range_frame_min = my_time.min()
    range_frame_max = my_time.max()
    
    range_diam_min = histogramm_min
    range_diam_max = histogramm_max
    
    
    ax3 = f.add_subplot(223)
    
    binning_diameter = binning
    binning_time = np.int((range_frame_max - range_frame_min)*effective_fps + 1)
    
    myhist = ax3.hist2d(diameter_time, my_time, cmap=cm.Greys, bins = [binning_diameter, binning_time], 
               range = [[range_diam_min, range_diam_max], [range_frame_min, range_frame_max]])
    ax3.set_xlabel("Diameter [nm]", **axis_font)
    ax3.set_ylabel("Time [s]", **axis_font)
#    ax3.grid("on")


    ax1 = f.add_subplot(221) 
    ax1.hist(diameter_time, bins = binning_diameter,
             range = [range_diam_min, range_diam_max])
    ax1.set_xlim([range_diam_min, range_diam_max])
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position("top")
    ax1.set_xlabel("Diameter [nm]", **axis_font)
    ax1.set_ylabel("counts [a.u.]", **axis_font)
#    ax1.grid("on")


    ax4 = f.add_subplot(224)
    ax4.hist(my_time, orientation="horizontal", histtype ='step', bins = binning_time,
             range = [range_frame_min, range_frame_max])
    ax4.set_ylim([range_frame_min, range_frame_max])
    ax4.yaxis.tick_right()
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel("Time [s]", **axis_font)
    ax4.set_xlabel('Number of Particles', **axis_font)
#    ax4.grid("on")
    
#    ax2 = f.add_subplot(222)
    

    
#    cbar = f.colorbar(ax1, cax=ax2)
#    ax2.colorbar(myhist[3], ax1=ax1)
    
#        # infobox
#    my_mean, my_std, my_median = nd.statistics.GetMeanStdMedian(sizes_df_lin.diameter)
#    
#    textstr = '\n'.join((
#    r'$\mu=%.1f$ nm' % (my_mean, ),
#    r'$\sigma=%.1f$ nm' % (my_std, ),
#    r'$\mathrm{median}=%.1f$ nm' % (my_median, )))
#
#    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#    
#    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, **axis_font, verticalalignment='top', bbox=props)


#=============================================================================

def AnimateDiameterAndRawData_Big2(rawframes, static_background, rawframes_pre, 
                                   sizes_df_lin, traj, ParameterJsonFile):     
    sys.exit("AnimateDiameterAndRawData_Big2 has moved to animate.py! Please change if you see this")

def AnimateDiameterAndRawData_temporal(rawframes_rot, sizes_df_lin_rolling):
    sys.exit("AnimateDiameterAndRawData_temporal has moved to animate.py! Please change if you see this")    

def AnimateStationaryParticles(rawframes_np):
        sys.exit("AnimateStationaryParticles has moved to animate.py! Please change if you see this")
        
def AnimateProcessedRawData(ParameterJsonFile, rawframes_rot, t4_cutted, t6_final, sizes_df_lin, sizes_df_lin_rolling):
    sys.exit("AnimateProcessedRawData has moved to animate.py! Please change if you see this")
    
def AnimateDiameterAndRawData(rawframes_rot, sizes_df_lin, t6_final, settings, DoScatter=True, DoText=False): 
    sys.exit("AnimateDiameterAndRawData has moved to animate.py! Please change if you see this")

def AnimateDiameterAndRawData_Big(rawframes_rot, sizes_df_lin, traj, settings): 
    sys.exit("AnimateDiameterAndRawData_Big has moved to animate.py! Please change if you see this")

