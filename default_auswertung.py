# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:39:56 2019

@author: foersterronny
"""

# Standard Libraries
from __future__ import division, unicode_literals, print_function # For compatibility of Python 2 and 3
# from importlib import reload # only used for debugging --> reload(package_name)

# for easy debugging
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# own Library
import NanoObjectDetection_Mfork as nd


#%% path of parameter file
# this must be replaced by any json file
ParameterJsonFile = '\\\\mars\\user\\nissenmona\\4 Nanoparticle detection+tracking\\Au_OlympusSetup\\...\\DataAnalysis\\...\\parameter.json'


#%% check if the python version and the library are good
nd.CheckSystem.CheckAll(ParameterJsonFile)


#%% read in the raw data to numpy
rawframes_np = nd.handle_data.ReadData2Numpy(ParameterJsonFile)


#%% choose ROI if wanted
# ROI (includes a help how to find it)
settings = nd.handle_data.ReadJson(ParameterJsonFile)

rawframes_super = nd.handle_data.RoiAndSuperSampling(settings, ParameterJsonFile, rawframes_np)


#%% standard image preprocessing
rawframes_pre, static_background = nd.PreProcessing.Main(rawframes_super, ParameterJsonFile)


#%% help with the parameters for finding objects 
settings = nd.handle_data.ReadJson(ParameterJsonFile)

nd.AdjustSettings.Main(rawframes_super, rawframes_pre, ParameterJsonFile)
    

#%% find the objects
obj_all = nd.get_trajectorie.FindSpots(rawframes_pre, ParameterJsonFile)
objects_per_frame = nd.particleStats.ParticleCount(obj_all, rawframes_pre.shape[0])
print(objects_per_frame.describe())


#%% identify static objects
# find trajectories of very slow diffusing (maybe stationary) objects
t1_orig_slow_diff = nd.get_trajectorie.link_df(obj_all, ParameterJsonFile, SearchFixedParticles = True)

# delete trajectories which are not long enough. Stationary objects have long trajcetories and survive the test   
t2_stationary = nd.get_trajectorie.filter_stubs(t1_orig_slow_diff, ParameterJsonFile, FixedParticles = True, BeforeDriftCorrection = True)


#%% cut trajectories if a moving particle comes too close to a stationary object
obj_moving = nd.get_trajectorie.RemoveSpotsInNoGoAreas(obj_all, t2_stationary, ParameterJsonFile)


#%% remove overexposed objects
obj_moving = nd.get_trajectorie.RemoveOverexposedObjects(ParameterJsonFile, obj_moving, rawframes_super)
  

#%% form trajectories of valid particle positions
t1_orig = nd.get_trajectorie.link_df(obj_moving, ParameterJsonFile, SearchFixedParticles = False) 


#%% remove too short trajectories
t2_long = nd.get_trajectorie.filter_stubs(t1_orig, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = True)
   

#%% identify and close gaps in the trajectories
t3_gapless = nd.get_trajectorie.close_gaps(t2_long)


#%% calculate intensity fluctuations as a sign of wrong assignment
t3_gapless = nd.get_trajectorie.calc_intensity_fluctuations(t3_gapless, ParameterJsonFile)


#%% split trajectories if necessary (e.g. too large intensity jumps)
t4_cutted, t4_cutted_no_gaps = nd.get_trajectorie.split_traj(t2_long, t3_gapless, ParameterJsonFile)


#%% drift correction
t5_no_drift = nd.Drift.Main(t4_cutted, ParameterJsonFile, PlotGlobalDrift = True)


#%% only long trajectories are used in the MSD plot in order to get a good fit
t6_final = nd.get_trajectorie.filter_stubs(t5_no_drift, ParameterJsonFile, FixedParticles = False, BeforeDriftCorrection = False, PlotErrorIfTestFails = False)

# #%% export/import t6_final
# t6_final.to_csv('\\\\mars\\user\\nissenmona\\4 Nanoparticle detection+tracking\\Au_OlympusSetup\\20210208_P100+125mix\\DataAnalysis\\v6_620fps_220usET_longer\\t6_final.csv',sep='\t',decimal=',')
# t6_final = pd.read_csv('\\\\mars\\user\\nissenmona\\4 Nanoparticle detection+tracking\\Au_OlympusSetup\\20210208_P100+125mix\\DataAnalysis\\v6_620fps_220usET_longer\\t6_final.csv',sep='\t',decimal=',')


#%% calculate the MSD and process to diffusion and diameter
sizes_df_lin, any_successful_check = nd.CalcDiameter.Main2(t6_final, ParameterJsonFile, 
                                                           MSD_fit_Show = True,t_beforeDrift = t4_cutted)

#sizes_df_lin, any_successful_check = nd.CalcDiameter.OptimizeTrajLenght(t6_final, ParameterJsonFile, obj_all, MSD_fit_Show = True, t_beforeDrift = t4_cutted)


# #%% re-import sizes_df
# sizes_df = pd.read_csv('\\\\mars\\user\\nissenmona\\4 Nanoparticle detection+tracking\\Au_OlympusSetup\\20210208_P100+125mix\\DataAnalysis\\v6_620fps_220usET_longer\\210302\\12_23_32_sizes_df_lin.csv')

#%% visualize results
nd.visualize.PlotDiameters(ParameterJsonFile, sizes_df_lin, any_successful_check)

#%% further plotting
Nfmin = 1000
nd.visualize.DiameterHistogramm(ParameterJsonFile,
                                sizes_df_lin[sizes_df_lin['valid frames']>=Nfmin],
                                weighting=True, showInfobox=True, fitNdist=True,
                                showICplot=False,
                                num_dist_max=2, fitInvSizes=False, showInvHist=False)



