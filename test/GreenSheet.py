# -*- coding: utf-8 -*-
"""
This code evaluates the Green Sheet TEM01 pattern imprinted onto the atoms.
Created on Thu May 23 16:05:00 2019
@author: Swarnav Banik
sbanik1@umd.edu
"""
from __future__ import division
import startUp
import GaussianBeam as GB
import OpticalElements as OE
import numpy as np
import os as os
from scipy import optimize
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

# %% Add path and initialize ##################################################
startUp
if (os.path.isdir(startUp.saveDir) == False):
    os.mkdir(startUp.saveDir)

# %% Common Functions ##############################################

def plotGaussianBeam(num,X,Y,I,I_fit,params,alpha_X,alpha_Y):
    # Plot a gaussian beam
    # Inputs:
    #   num: figure number
    #   X,Y: X and Y as 2D arrays of plane of the beam [mm]
    #   I: Intensity profile of the beam as 2D array 
    #   I_fit: Fitted gaussian beam
    #   params: Fitted gaussian beam params 
    #   alpha_X and alpha_Y: Dimensionless constants that help define an ROI
    if (I.shape != X.shape or X.shape != Y.shape or Y.shape != I_fit.shape): 
        raise Exception('GreenSheet::plotGaussianBeam::X, Y, I and I_fit should have same dimensions.')

    [Ny,Nx] = X.shape
    #Plot Data
    fig = plt.figure(num)
    gs=GridSpec(3,3)
    fig.clf()
    ax1=fig.add_subplot(gs[0:2,0:2]) 
    c = ax1.pcolor(X[0,:]*10**-3, Y[:,0]*10**-3,I, cmap='Greens', vmin=0, vmax=np.max(I))
    ax1.set_title('Total Power = {0} mW'.format(round(np.sum(I),2)))
    fig.colorbar(c, ax=ax1, label = 'Beam Intensity')
    ax1.axis([-alpha_X*1.5, alpha_X*1.5, -alpha_Y*1.5, alpha_Y*1.5])
    ax=fig.add_subplot(gs[2,0:2])
    ax.plot(X[np.int(Ny/2),:]*10**-3,I[np.int(Ny/2),:],'.',X[np.int(Ny/2),:]*10**-3,I_fit[np.int(Ny/2),:],'--')
    ax.set_xlabel('X (mm)')
    ax.set_title('X waist = {0} $\mu$m'.format(round(params[3],2)))
    ax.axis([-alpha_X*1.5, alpha_X*1.5, 0, 1.1*np.max(I_fit)])
    ax=fig.add_subplot(gs[0:2,2])
    ax.plot(I[:,np.int(Nx/2)],Y[:,np.int(Nx/2)]*10**-3,'.',I_fit[:,np.int(Nx/2)],Y[:,np.int(Nx/2)]*10**-3,'--')
    ax.set_ylabel('Y (mm)')
    ax.set_title('Y waist = {0} $\mu$m'.format(round(params[4],2)))
    ax.axis([0, 1.1*np.max(I_fit), -alpha_Y*1.5, alpha_Y*1.5])
    plt.tight_layout()
    plt.show()
    return fig

def plotPhaseShiftedBeam(num,X,Y,I,E,alpha_X,alpha_Y):
    if (I.shape != X.shape or X.shape != Y.shape or Y.shape != E.shape): 
        raise Exception('GreenSheet::plotPhaseShiftedBeam::X, Y, E and I should have same dimensions.')

    [Ny,Nx] = X.shape
    #Plot Data
    fig = plt.figure(num)
    gs=GridSpec(2,2)
    fig.clf()
    ax=fig.add_subplot(gs[0,0]) 
    c = ax.pcolor(X[0,:]*10**-3, Y[:,0]*10**-3,I, cmap='Greens', vmin=0, vmax=np.max(I))
    ax.set_title('Input Beam  Intensity\n Total Power = {0} mW'.format(round(np.sum(I),2)))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    fig.colorbar(c, ax=ax, label = 'Beam Intensity')
    ax.axis([-alpha_X*1.5, alpha_X*1.5, -alpha_Y*1.5, alpha_Y*1.5])
    ax=fig.add_subplot(gs[0,1]) 
    c = ax.pcolor(X[0,:]*10**-3, Y[:,0]*10**-3,np.angle(E), cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Input Beam Phase\n Total Power = {0} mW'.format(round(np.sum(I),2)))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    fig.colorbar(c, ax=ax, label = 'Phase')
    ax.axis([-alpha_X*1.5, alpha_X*1.5, -alpha_Y*1.5, alpha_Y*1.5])
    ax=fig.add_subplot(gs[1,0]) 
    c = ax.pcolor(X[0,:]*10**-3, Y[:,0]*10**-3,I, cmap='Greens', vmin=0, vmax=np.max(I))
    ax.set_title('Output ($\pi$ plate) Beam  Intensity\n Total Power = {0} mW'.format(round(np.sum(I),2)))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.axis([-alpha_X*1.5, alpha_X*1.5, -alpha_Y*1.5, alpha_Y*1.5])
    fig.colorbar(c, ax=ax, label = 'Beam Intensity')
    ax=fig.add_subplot(gs[1,1]) 
    c = ax.pcolor(X[0,:]*10**-3, Y[:,0]*10**-3,np.angle(E), cmap='RdBu', vmin=-np.pi, vmax=np.pi)
    ax.set_title('Output ($\pi$ plate) Beam Phase\n Total Power = {0} mW'.format(round(np.sum(I),2)))
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    fig.colorbar(c, ax=ax, label = 'Phase')
    ax.axis([-alpha_X*1.5, alpha_X*1.5, -alpha_Y*1.5, alpha_Y*1.5])
    plt.tight_layout()
    plt.show()
    return fig
    
def plotGreenSheet(num,X,Y,I,f,alphax,alphay):
    if (I.shape != X.shape or X.shape != Y.shape ): 
        raise Exception('GreenSheet::plotGreenSheetBeam::X, Y and I should have same dimensions.')
    [Ny,Nx] = X.shape
    #Plot
    fig = plt.figure(num)
    gs=GridSpec(3,3)
    fig.clf()
    ax=fig.add_subplot(gs[0:2,0:2]) 
    c = ax.pcolor(X[0,:], Y[:,0],I, cmap='Greens', vmin=0, vmax=np.max(I))
    ax.set_title('Beam at focus f3 = {0} mm \n Total Power = {1} mW'.format(f,round(np.sum(I),2)))
    fig.colorbar(c, ax=ax, label = 'Beam Intensity')
    ax.axis([-1.5*alphax, 1.5*alphax, -1.5*alphay, 1.5*alphay])
    ax=fig.add_subplot(gs[2,0:2])
    ax.plot(X[np.int(Ny/2),:],I[np.int(Ny/2),:],'.')
    ax.set_xlabel('X ($\mu$m)')
    ax.axis([-1.5*alphax, 1.5*alphax, 0, np.max(I)])
    ax=fig.add_subplot(gs[0:2,2])
    ax.plot(I[:,np.int(Nx/2)],Y[:,np.int(Nx/2)],'.')
    ax.set_ylabel('Y ($\mu$m)')
    ax.axis([0, 1.1*np.max(I),-1.5*alphay, 1.5*alphay])
    ax.set_title('Phase plate shift:\nDisplacemnt =  {0} mm  \n Tilt = {1}$^0$'.format(PP_offset,PP_tilt))
    plt.tight_layout()
    plt.show()
    plt.savefig('GS_PP_off{0}_tilt{1}.png'.format(PP_offset,PP_tilt))
    return fig
    
def CircularAperture(E,X,Y,r0,x0,y0):
    # Evaluates the response of a circular aperture right after it
    # Inputs: E - 2D Field pattern before the the aperture
    #         X,Y - 2D grid representing co-ordinates
    #         k - Wave vector [um^-1]
    #         [x0, y0] - Relative center of aperure [mm]
    #         r0 - radius of circular aperture [mm]
    # Outputs: E - 2D Field pattern after the the aperture
    if (E.shape != X.shape or X.shape != Y.shape ): 
        raise Exception('GreenSheet::CircularAperture::I, X and Y should have same dimensions.')
    r0 = r0*10**3
    x0 = x0*10**3
    y0 = y0*10**3
    X = X-x0
    Y = Y-y0
    R = np.sqrt(X**2+Y**2)
    E[R>r0] = 0    
    return E


# %% Input Parameters #########################################################
lamda = 532                                     # Wavelength [nm]
w0 = 680                                        # Waist of incoming collimated beam [um]
P = 350                                         # Total power in beam[mW]
PP_offset = 0                                   # Vertical shift of the phase plate[mm]
PP_tilt = 0                                     # Phase plate [degrees]
# Array Parameters ############################################################
Nx = 2**7
Ny = 2**7
Nz = 5

# %% Derived Values ###########################################################
k = 2*np.pi*10**(3)/lamda                       # wave number [um-1]
zmin = -4*GB.RayleighLength(k,w0)
zmax = 4*GB.RayleighLength(k,w0)
x = np.linspace(-10*w0,10*w0,Nx)
y = np.linspace(-10*w0,10*w0,Ny)
z = np.linspace(zmin,zmax,Nz)
[X,Y,Z] = np.meshgrid(x,y,z)
R = np.sqrt(X**2+Y**2)
THETA = np.arctan2(Y,X)

# %% Input Beam ###############################################################
# Generating Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
X1 = X[:,:,0]
Y1 = Y[:,:,0]
E1 = GB.PlaneWaveFieldVec(1,R[:,:,0],0,w0,k)
I1 = GB.BeamInt(E1,P)
# Generating Fit %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I1 = I1.astype(float)
p0 = [np.max(I1),0,0,w0,w0]
[params1,_] = optimize.curve_fit(GB.GaussBeamFit, [X1,Y1], I1.ravel(), p0=p0)
I1_fit = GB.GaussBeamFit([X1,Y1],*params1).reshape(Ny,Nx)
figure = plotGaussianBeam(1,X1,Y1,I1,I1_fit,params1,1,1)
cwd = os.getcwd()
os.chdir(startUp.saveDir)
figure.set_size_inches(10, 7)
figure.savefig('GS_inputBeam.png')
os.chdir(cwd)

# %% Cylindrical Lens f1 = -40 mm #############################################
f1 = 40
# Generating Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[E2, X2, Y2] = OE.CylLensAction(E1,X1,Y1,k,f1,FocusingAxis = 'Y')
I2 = GB.BeamInt(E2,P)
# Generating Fits %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I2 = I2.astype(float)
p0 = [np.max(I2),0,0,w0,10]
[params2,_] = optimize.curve_fit(GB.GaussBeamFit, [X2,Y2], I2.ravel(),p0 = p0)
I2_fit = GB.GaussBeamFit((X2,Y2),*params2).reshape(Ny,Nx)
plotGaussianBeam(2,X2,Y2,I2,I2_fit,params2,0.5,0.5)

# %% Spherical lens f2 = +300 mm ##############################################
# Generating Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f2 = 300                                    # focal length of second lens  [mm]
[E3, X3, Y3] = OE.SphLensAction(E2,X2,Y2,k,f2,FocussedAxis ='Y')
I3 = GB.BeamInt(E3,P)
# Generating Fits %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
I3 = I3.astype(float)
p0 = [np.max(I3),0,0,100,w0]
[params3,_] = optimize.curve_fit(GB.GaussBeamFit, (X3,Y3), I3.ravel(),p0 = p0)
I3_fit = GB.GaussBeamFit((X3,Y3),*params3).reshape(Nx,Ny)
figure = plotGaussianBeam(3,X3,Y3,I3,I3_fit,params3,2,2)
cwd = os.getcwd()
os.chdir(startUp.saveDir)
figure.set_size_inches(10, 7)
figure.savefig('GS_phasePlate.png')
os.chdir(cwd)

# %% Phase Plate ##############################################################
E4 = OE.PiPlateAction(E3,X3,Y3,PP_offset*10**3,PP_tilt*np.pi/180)
X4 = X3
Y4 = Y3
E4 = CircularAperture(E4,X4,Y4,20,0,PP_offset)
I4 = GB.BeamInt(E4,P)
figure = plotPhaseShiftedBeam(4,X4,Y4,I4,E4,float(1/15),float(50/15))
cwd = os.getcwd()
os.chdir(startUp.saveDir)
figure.set_size_inches(10, 7)
figure.savefig('GS_phasePlate.png')
os.chdir(cwd)

# %% Final Lens f3 = 200 mm ###################################################
# Generating Data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f3 = 200                                    # focal length of second lens  [mm]
[E5, X5, Y5] = OE.SphLensAction(E4,X4,Y4,k,f3,FocussedAxis ='XY')
I5 = GB.BeamInt(E5,P)
figure = plotGreenSheet(5,X5,Y5,I5,f3,0.5*10**3,2*10**1)

cwd = os.getcwd()
os.chdir(startUp.saveDir)
figure.set_size_inches(10, 7)
figure.savefig('GS_final.png')
os.chdir(cwd)

