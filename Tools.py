# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 08:54:11 2019

@author: foersterronny
"""


import numpy as np
import pandas as pd
import matplotlib
# False since deprecated from version 3.0
#matplotlib.rcParams['text.usetex'] = False
#matplotlib.rcParams['text.latex.unicode'] = False
import logging

#import matplotlib.pyplot as plt # Libraries for plotting
import NanoObjectDetection_Mfork as nd

from pdb import set_trace as bp #debugger





def LoggerSetLevel(level, TryLogger = False):
    #set the level when a logging message is plotted
       
    if level == "debug":
        nd.logger.setLevel(logging.DEBUG)
    elif level == "info":
        nd.logger.setLevel(logging.INFO)
    elif level == "warning":
        nd.logger.setLevel(logging.WARNING)
    elif level == "error":
        nd.logger.setLevel(logging.ERROR)
    elif level == "critical":
        nd.logger.setLevel(logging.CRITICAL)
    else:
        nd.logger.error("Level unknown. Choose debug, info, warning, error or critical in the json file.")

    if TryLogger == True:
        nd.logger.debug("TEST LOGGER MODE: <debug> ON")
        nd.logger.info("TEST LOGGER MODE: <info> ON")
        nd.logger.warning("TEST LOGGER MODE: <warning> ON")
        nd.logger.error("TEST LOGGER MODE: <error> ON")
        nd.logger.critical("TEST LOGGER MODE: <critical> ON")



def CameraCalibration(folder_dark,folder_bright,subsampling = 0):
    print("THIS IS UNDER CONSTRUCTION. USE CAL_READNOISE FROM RAINER INSTEAD")
    """
    Calculates the camera properties gain and readout noise
    Plots the variance over the intensity
    Requires: 
        1 - bright image: 
            wide histogramm
            not saturated
            unfocussed (no strutures in the image)
            stable light source
            around 100 frames of the same scene
        2 - dark image:
            no light reaches the detector
            around 100 frames of the same scene
            
    Result:
        1 - camera gain
        2 - readout noise
    """
    
    
    # read dark image
#    folder_dark = r"Z:\Data\Shiqi\gain of camera\0,00007s no light 12 bit\100frames"
    dark_orig = nd.handle_data.ReadTiffSeries2Numpy(folder_dark)
    
    
    # read bright image
#    folder_bright = r"Z:\Data\Shiqi\gain of camera\0,00007s lamp 12bit\100frames"
    bright_orig = nd.handle_data.ReadTiffSeries2Numpy(folder_bright)
    
    
    # subsampling
    if subsampling != 0:
        dark_sub    = dark_orig[:,::subsampling,::subsampling]
        bright_sub  = bright_orig[:,::subsampling,::subsampling]
    else:
        dark_sub    = dark_orig
        bright_sub  = bright_orig
    
    
    # process dark
    dark_mean = np.mean(dark_sub,(2,3))        
#    dark_var  = np.var(bright_sub,(2,3))            
    readout_noise = np.std(dark_sub,(2,3))   
    
    
    bright_mean = np.mean(bright_sub,(2,3))        
    bright_var  = np.var(bright_sub,(2,3))        
