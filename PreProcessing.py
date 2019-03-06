# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 11:09:16 2019

@author: Ronny Förster und Stefan Weidlich
"""

# In[]
import NanoObjectDetection as nd
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace as bp #debugger

# In[]
def Main(rawframes_np, ParameterJsonFile):
    settings = nd.handle_data.ReadJson(ParameterJsonFile)
    
    DoSimulation = settings["Simulation"]["SimulateData"]
    
    if DoSimulation == 1:
        print("No data. Do a simulation later on")
        rawframes_np = 0
                
    else:
        # check if constant background shall be removed
        if settings["PreProcessing"]["Remove_CameraOffset"] == 1:
            print('Constant camera background: removed')
            rawframes_np = nd.PreProcessing.SubtractCameraOffset(rawframes_np, settings)
        else:
            print('Constant camera background: not removed')
        
        if settings["PreProcessing"]["Remove_Laserfluctuation"] == 1:
            print('Laser fluctuations: removed')
            rawframes_np = nd.PreProcessing.RemoveLaserfluctuation(rawframes_np, settings)    
            # WARNING - this needs a roughly constant amount of particles in the object!
        else:
            print('Laser fluctuations: not removed')
        
        if settings["PreProcessing"]["Remove_StaticBackground"] == 1:
            print('Static background: removed')
            rawframes_np, static_background = nd.PreProcessing.Remove_StaticBackground(rawframes_np, settings)
        else:
            print('Static background: not removed')
            
        if settings["PreProcessing"]["RollingPercentilFilter"] == 1:
            print('Rolling percentil filter: applied')
            rawframes_np = nd.PreProcessing.RollingPercentilFilter(rawframes_np, settings, settings)
        else:
            print('Rolling percentil filter: not applied')
        
        if settings["PreProcessing"]["ClipNegativeValue"] == 1:
            print('Negative values: removed')
            print("Ronny does not love clipping.")
            rawframes_np[rawframes_np < 0] = 0
        else:
            print('Negative values: kept')
            
         
        nd.handle_data.WriteJson(ParameterJsonFile, settings)
        
    return rawframes_np



def SubtractCameraOffset(rawframes_np, settings):
    #That generates one image that holds the minimum-vaues for each pixel of all times
    rawframes_pixelCountOffsetArray = nd.handle_data.min_rawframes(rawframes_np)
        
    # calculates the minimum of all pixel counts. Assumption:
    # this is the total offset
    offsetCount=np.min(rawframes_pixelCountOffsetArray) 
    
    # I'm now subtracting the offset (only the offset, not the full background) from the complete data. Assumption:
    # Whenever there is a change in intensity, e.g. by fluctuations in incoupling,the lightsource etc., this affects mututally background and signal
    rawframes_np=rawframes_np-offsetCount
    
    
    return rawframes_np


def RemoveLaserfluctuation(rawframes_np, settings):
    Laserfluctuation_Show = settings["Plot"]['Laserfluctuation_Show']
    Laserfluctuation_Save = settings["Plot"]['Laserfluctuation_Save']
    
    if Laserfluctuation_Save == True:
        Laserfluctuation_Show = True
    
    
    # Mean-counts of a given frame
    tot_intensity, rel_intensity = nd.handle_data.total_intensity(rawframes_np, Laserfluctuation_Show)
    
    rawframes_np = rawframes_np / rel_intensity[:, None, None]


    if Laserfluctuation_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "Intensity Fluctuations", \
                                       settings, data = rel_intensity, data_header = "Intensity Fluctuations")
        
    return rawframes_np


def Remove_StaticBackground(rawframes_np, settings, Background_Show = False, Background_Save = False):
    Background_Show = settings["Plot"]['Background_Show']
    Background_Save = settings["Plot"]['Background_Save']
    
    if Background_Save == True:
        Background_Show = True
    '''
    Subtracting back-ground and take out points that are constantly bright
    '''

    static_background = nd.handle_data.min_rawframes(rawframes_np,  display = Background_Show)
    
    rawframes_np = rawframes_np - static_background # Now, I'm subtracting a background, in case there shall be anything left
    
    if Background_Save == True:
        settings = nd.visualize.export(settings["Plot"]["SaveFolder"], "CameraBackground", settings)
    
    
    return rawframes_np, static_background
    

def RollingPercentilFilter(rawframes_np, settings):
    rolling_length = ["PreProcessing"]["RollingPercentilFilter_rolling_length"]
    rolling_step = ["PreProcessing"]["RollingPercentilFilter_rolling_step"]
    percentile_filter = ["PreProcessing"]["RollingPercentilFilter_percentile_filter"]   
    
    for i in range(0,len(rawframes_np)-rolling_length,rolling_step):
        my_percentil_value = np.percentile(rawframes_np[i:i+rolling_length], percentile_filter, axis=0)
        rawframes_np[i:i+rolling_step] = rawframes_np[i:i+rolling_step] - my_percentil_value

    return rawframes_np


