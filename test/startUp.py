#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code sets the directories for running the test codes
Created on Sat Dec 19 12:51:22 2020
@author: Swarnav Banik
sbanik1@umd.edu
"""
# %% Import all ###############################################################
import sys
import matplotlib.pyplot as plt

# %% Add necessary paths ######################################################
sys.path.insert(1, '/Users/swarnav/Google Drive/Work/Projects/Imaging/src')
# %% Define the output directory ##############################################
saveDir = '/Users/swarnav/Google Drive/Work/Projects/Imaging/test/out'
# %% Set some default values ##################################################

params = {
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': True,
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'font.size': 8, 
    'legend.fontsize': 6, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'font.family': 'serif',
}
plt.rcParams.update(params)