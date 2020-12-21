# -*- coding: utf-8 -*-
"""
This module contains all functions for gaussian beam propagation.
Created on Wed May 22 12:11:02 2019
@author: Swarnav Banik
sbanik1@umd.edu
"""
import numpy as np
# %% Common Functions##########################################################
# The following functions take inputs
# Wave Vector k in units um
# Minimum Waist w0 in units um
# Position r,z in units um

#Basic Gaussian Beam Functions#################################################
def RayleighLength(k,w0):
    return k*w0**2/2 #[um]
def GouyPhase(z,zR):
    if (zR == 0 and z>0): raise Exception("GaussianBeam::GuoyPhase:: Rayleigh length can't be 0.")   
    return np.arctan(z/zR)
def waist(z, zR, w0):
    return w0*np.sqrt(1+(z/zR)**2) #[um]
def ROC(z, zR):
    if z!=0: return z*(1+(zR/z)**2) #[um]
    else: return np.inf
def qParam(k,w0,z):
    zR = RayleighLength(k,w0)
    w = waist(z,zR,w0)
    R = ROC(z,zR)
    qinv = (1/R) -1j*(2/(k*w**2))
    q = qinv**-1
    return q
def ExtractWaistFromQ(k,q_in):
    waist = np.sqrt(-2/(k*np.imag(q_in**-1)))
    return waist
    
#Field Patterns################################################################
def PlaneWaveField(E0,r,z,w0,k):
    Phase = k*z
    Amplitude = E0*np.exp(-(r/w0)**2)
    return Amplitude*np.exp(-1j*Phase)
PlaneWaveFieldVec = np.vectorize(PlaneWaveField)

def PointSource(E0,r,z,w0,k):
    Phase = k*z
    Amplitude = r
    if r>w0:
        Amplitude = 0
    else:
        Amplitude = 1
    return Amplitude*np.exp(-1j*Phase)
PointSourceVec = np.vectorize(PointSource)

def GaussBeamField(E0,r,z,w0,k):
    zR = RayleighLength(k,w0)
    if z!=0:
        R = ROC(z,zR)
        Phase = k*z + k*(r**2/(2*R))-GouyPhase(z,zR)
    else: Phase = 0
    Amplitude = E0*(w0/waist(z,zR,w0))*np.exp(-(r/waist(z,zR,w0))**2)
    return Amplitude*np.exp(-1j*Phase)
GaussBeamFieldVec = np.vectorize(GaussBeamField)

#Intensity Evaluation##########################################################
def BeamInt(E,I0):
    I = I0*np.square(np.abs(E))/(np.sum(np.square(np.abs(E))))
    return I

#Beam Parameters Fit function#################################################
def GaussBeamFit(xdata,I0,beamCenterX,beamCenterY,beamWaistX,beamWaistY):
    # Inputs:
    #   I0: peak intensity
    #   xdata: X and Y values of the grid   
    #   beamCenterX and Y: Position of center
    #   beamWaistX and Y: Waist size
    #   xdata, beamCenterX and Y, and beamWaistX and Y are all in the same units.
    I0 = np.float(I0)
    x0 = np.float(beamCenterX); y0 = np.float(beamCenterY);   
    wx = np.float(beamWaistX); wy = np.float(beamWaistY);
    x = xdata[0].astype(float); y = xdata[1].astype(float);
    I =  I0*np.exp(-2*(((x-x0)/wx)**2+((y-y0)/wy)**2))
    return I.ravel()

