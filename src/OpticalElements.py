# -*- coding: utf-8 -*-
"""
This module contains all functions for response of optical elements.
Created on Wed May 22 12:15:23 2019
@author: Swarnav Banik
sbanik1@umd.edu
"""

import numpy as np
import numpy.fft as fourier
import scipy as scp
from PIL import Image
# %% Common Functions #########################################################
# The following functions take inputs
# Wave Vector k in units um
# Minimum Waist w0 in units um
# Position r,z in units um

# Lens Action ###################################################################
def SphLensAction(E,X,Y,k,f,**kwargs):
    # Evaluates the response of a spherical lens at its front focal plane
    # Inputs: E - 2D Field pattern
    #         X,Y - 2D grid representing co-ordinates
    #         k - Wave vector [um^-1]
    #         f - focal length [mm]
    #         FocussedAxis - Along what axis is the beam focused at the back 
    #                        focal plane
    if (E.shape != X.shape or X.shape != Y.shape): 
        raise Exception('OpticalElements::SphLensAction::E,X and Y should have same dimensions.')
    for key, value in kwargs.items():
        if key == 'FocussedAxis': FocAxis = value
    f = f*10**3
    Transform = fourier.fft2(E)
    if FocAxis == 'X':
        Transform = fourier.fftshift(Transform, axes = 0)
    elif FocAxis == 'Y':
        Transform = fourier.fftshift(Transform, axes = 1)
    elif FocAxis == 'NONE':
        Transform = fourier.fftshift(Transform)       
    dx = X[0,1]-X[0,0]
    Xfrq = (2*np.pi*f/k)*fourier.fftshift(fourier.fftfreq(X.shape[1], d=dx))
    dy = dx = Y[1,0]-Y[0,0]
    Yfrq = (2*np.pi*f/k)*fourier.fftshift(fourier.fftfreq(Y.shape[0], d=dy))
    [X, Y] = np.meshgrid(Xfrq,Yfrq)
    return [Transform, X, Y]

def CylLensAction(E,X,Y,k,f,**kwargs):
    # Evaluates the response of a cylindrical lens at its front focal plane
    # Inputs: E - 2D Field pattern
    #         X,Y - 2D grid representing co-ordinates
    #         k - Wave vector [um^-1]
    #         f - focal length [mm]
    #         FocussedAxis - Along what axis is the beam focused at the back 
    #                        focal plane
    #         FocusingAxis - Along what axis does the  lens focus
    if (E.shape != X.shape or X.shape != Y.shape): 
        raise Exception('OpticalElements::CylLensAction::E,X and Y should have same dimensions.')
    for key, value in kwargs.items():
        if key == 'FocusingAxis': FocAxis = value
    f = f*10**3
    if FocAxis == 'X': 
        Transform = fourier.fft(E, axis = 1)
        Transform = fourier.fftshift(Transform, axes = 1)
        dx = X[0,1]-X[0,0]
        Xfrq = (2*np.pi*f/k)*fourier.fftshift(fourier.fftfreq(X.shape[1], d=dx))
        Yfrq = Y[:,0]
    elif FocAxis == 'Y': 
        Transform = fourier.fft(E, axis = 0)
        Transform = fourier.fftshift(Transform, axes = 0)
        dy = dx = Y[1,0]-Y[0,0]
        Yfrq = (2*np.pi*f/k)*fourier.fftshift(fourier.fftfreq(Y.shape[0], d=dy))
        Xfrq = X[0,:]
    else: raise Exception('OpticalElements::CylLensAction::Focussing xxis needs to be specified.')
    [X, Y] = np.meshgrid(Xfrq,Yfrq)
    return [Transform, X, Y]

def PiPlateAction(E,X,Y,y_offset,tilt):
    # Evaluates the response of an imaging system via the PSF
    # Inputs: 
    #         X,Y - 2D grid representing co-ordinates at the plane of pi plate
    #         E: The light field at the plane of pi plate
    #         y_offset, titlt: Offset and tilt of the pi plate
    # Outputs:
    #         The light field after passing through the pi plate
    if (E.shape != X.shape or X.shape != Y.shape): 
        raise Exception('OpticalElements::PiPlateAction::E, X and Y should have same dimensions.')
    Phase = np.angle(E)
    for ii in range(Y.shape[0]):
        for jj in range(Y.shape[1]):
            if Y[ii,jj]>(np.tan(tilt)*X[ii,jj]+y_offset):
                Phase[ii,jj] = Phase[ii,jj]+np.pi
    return np.abs(E)*np.exp(1j*Phase)

def MatrixFreeProp(q_in,d):
    A = 1
    B = d
    C = 0
    D = 1
    q_out = (A*q_in+B)/(C*q_in+D)
    return q_out

def MatrixLens(q_in,f):
    A = 1
    B = 0
    C = -1/f
    D = 1
    q_out = (A*q_in+B)/(C*q_in+D)
    return q_out

# Imaging #####################################################################
def ImageViaPSF(X_o, Y_o, E_o, ASF, **kwargs):
    # Evaluates the response of an imaging system via the PSF
    # Inputs: 
    #         X_o,Y_o - 2D grid representing co-ordinates in object plane
    #         E_o: The light field at the object plane
    #         ASF: Amplitude Spread Function = sqrt(PSF)
    #         norm (optional): Normalize the ASF by some factor
    # Outputs:
    #         I_i: The light field at the image plane

    for key, value in kwargs.items(): 
     if key == 'norm':
         ASF = ASF*value 
    E_ft = fourier.fftshift(fourier.fft2(E_o))
    ASF_ft = fourier.fftshift(fourier.fft2(ASF))    
    E_i = fourier.ifftshift(fourier.ifft2(E_ft*ASF_ft))
    I_i = np.abs(E_i)**2
    return I_i

def ASF(X_o,Y_o,R_airy,**kwargs):
    # Evaluates the Amplitude Spread Function of an imaging system
    # Inputs: 
    #         X_o,Y_o - 2D grid representing co-ordinates in object plane
    #         R_airy: Radial extent of the PSF/ ASF
    #         kind (optional): Kind of ASF, default is airy
    # Outputs:
    #         ASF: The ASF = sqrt(PSF)
    kind = 'airy'
    for key, value in kwargs.items(): 
     if key == 'kind':
         kind = value  
    R = np.sqrt(X_o**2+Y_o**2)
    if kind == 'airy':        
        ASF = scp.special.jv(1,3.8317*R/R_airy)/(3.8317*R/R_airy)
        ASF[R==0] = 0.5
    if kind == 'gaussian':
        R_airy = R_airy*2.672/3.8317; 
        ASF = np.exp(-(X_o**2+Y_o**2)/R_airy**2)
    ASF = ASF/np.sum(np.abs(ASF)**2)
    return ASF

def PixelizeImage(I_org,X_org,Y_org,PixSize_cam):
    # Pixelize the image
    # Inputs: 
    #         X_org,Y_org - 2D grid representing co-ordinates in object plane
    #         I_org: The image
    #         PixSize_cam: The pixel size of the camera
    # Outputs:
    #         X_cam,Y_cam - 2D grid representing co-ordinates in object plane on camera
    #         I_cam: The pixelated image
    #         PixSize_cam: The pixel size on the camera
    if (I_org.shape != X_org.shape or X_org.shape != Y_org.shape): 
        raise Exception('OpticalElements::PixelizeImage::I_org,X_org and Y_org should have same dimensions.')
    if (X_org[0,0]-X_org[0,1] != Y_org[0,0]-Y_org[1,0]): 
        raise Exception('OpticalElements::PixelizeImage::Pixel size in X and Y are not same')
        
    nptsx = int(round(X_org[0,-1]-X_org[0,0]/PixSize_cam))
    nptsy = int(round(Y_org[-1,0]-Y_org[0,0]/PixSize_cam))
    PixSize_cam = [(X_org[0,0]-X_org[0,-1])/nptsx, (Y_org[0,0]-Y_org[-1,0])/nptsy]
    x = np.linspace(X_org[0,0],X_org[0,-1],nptsx)
    y = np.linspace(Y_org[0,0],Y_org[-1,0],nptsy)
    [X_cam,Y_cam] = np.meshgrid(x,y)
    I_org_img = Image.fromarray(I_org)
    I_cam_img = I_org_img.resize((nptsy,nptsx),resample=Image.BILINEAR)
    I_cam = np.asarray(I_cam_img)
    return [X_cam,Y_cam,I_cam, PixSize_cam]
       
