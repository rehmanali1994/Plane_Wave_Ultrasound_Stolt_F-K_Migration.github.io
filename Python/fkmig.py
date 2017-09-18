# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:57:23 2016
@author: rehmanali
"""

import numpy as np
import pdb

def fkmig(SIG, fs, pitch, TXangle = 0, c = 1540, t0 = 0):
    """ fkmig   f-k migration for plane wave imaging
    x, z, migSIG = fkmig(SIG, fs, pitch, TXangle, c, t0) performs an f-k 
    migration of the signals stored in the array SIG. migSIG contains the 
    migrated signals. The x-axis is parallel to the transducer and
    pointing from element #1 to element #N (x = 0 at the CENTER of
    the transducer). The z-axis is PERPENDICULAR to the transducer and
    pointing downward (z = 0 at the level of the transducer).
    
    Remaining Inputs:
    -------------------------------------------------------
    1) fs: sample frequency (in Hz, REQUIRED)
    2) pitch: pitch of the linear transducer (in m, REQUIRED)
    3) TXangle: steering (transmit) angles (in rad, default = 0)
            One must have TXangle.size = SIG.shape[2].
            PARAM['TXangle'] can also be a scalar.
    4) c: longitudinal velocity (in m/s, default = 1540 m/s)
    5) t0: acquisition start time (in s, default = 0)    
    
    Important details on fkmig:
    --------------------------
    1) The signals - typically RF signals - in SIG must be acquired using a
       PLANE WAVE configuration with a linear array as used in ultrafast
       ultrasound imaging. If SIG is 2-D, each column corresponds to a
       single RF signal over time, with the FIRST COLUMN corresponding to
       the FIRST ELEMENT.
    3) The steering angle is positive (TXangle > 0) if the 1st
       element is the first to transmit. The steering angle is negative
       (TXangle < 0) if the last element is the first to transmit.
    
    IMPORTANT NOTE: fkmig does not use the transmit time delays as input
    parameters. The transmit delays are determimed from the specified speed
    of sound (c) and the steering angle (TXangle). 
    
    Reference
    --------- 
    Garcia et al., Stolt's f-k migration for plane wave ultrasound imaging.
    IEEE Trans Ultrason Ferroelectr Freq Control, 2013;60:1853-1867. """

    # Function Defining Linear Interpolation to be Used Later
    def interpLIN(dx,y,xi):
        # -- Linear interpolation along columns
        siz = y.shape;
        yi = np.zeros(siz);
        
        # -- Classical interpolation
        idx = xi/dx;
        I = idx>(siz[0]-2);
        idx[I] = 0; # arbitrary junk index
        idxf = np.int32(np.floor(idx));
        for k in np.arange(siz[1]):
            idxfk = idxf[0:siz[0],k];
            idxk = idx[0:siz[0],k] - idxfk;
            yi[0:siz[0],k] = y[idxfk,k]*(1-idxk) + y[idxfk+1,k]*idxk;
        yi[I] = 0;
        return yi;

    # Get the dimensions of the data
    nt, nx = SIG.shape;   
    
    #-- Zero-padding before FFTs 
    #-- Time direction: extensive zero-padding is required with linear interpolation
    ntshift = int(2*np.ceil(t0*fs/2));
    ntFFT = 4*nt+ntshift; 
    #-- X direction: in order to avoid lateral edge effects
    factor = 1.5;
    nxFFT = int(2*np.ceil(factor*nx/2));
    
    #-- Grid for temporal and spatial frequency
    f0 = np.arange(ntFFT/2 + 1)*fs/ntFFT;
    kx = np.roll(np.arange(-nxFFT/2,nxFFT/2)+1,int(nxFFT/2+1))/pitch/nxFFT;
    Kx, f = np.meshgrid(kx,f0);
    
    #-- Temporal FFT
    SIG = np.fft.fft(SIG,n=ntFFT,axis=0);
    # The signal is real: only the positive temporal frequencies are kept:
    SIG = SIG[0:ntFFT/2+1,:];
    
    sinA = np.sin(TXangle);
    cosA = np.cos(TXangle);
    
    #-- ERM velocity
    v = c/np.sqrt(1+cosA+sinA**2);
    
    #-- Compensate for steering angle and/or depth start
    #-- Assumes that t=0 is when any element first sends its tx
    dt = sinA*((nx-1)*(TXangle<0)-np.arange(nx))*pitch/c; # steering angle
    tTrim, fTrim = np.meshgrid(dt+t0,f0); # depth start
    SIG = SIG*np.exp(-2*1j*np.pi*tTrim*fTrim);
    
    #-- Spatial FFT
    SIG = np.fft.fft(SIG,n=nxFFT,axis=1);
    
    #-- Note: we choose kz = 2*f/c (i.e. z = c*t/2)
    beta = ((1+cosA)**1.5)/(1+cosA+sinA**2);
    fkz = v*np.sqrt(Kx**2+4*((f**2)/(c**2))/(beta**2));
    
    #-- Remove evanescent parts
    SIG[np.abs(f)/(np.abs(Kx)+np.spacing(1)) < c] = 0;
    
    #-- Linear interpolation in the frequency domain: f -> fkz
    SIG = interpLIN(fs/ntFFT,SIG.real,fkz) + 1j*interpLIN(fs/ntFFT,SIG.imag,fkz);
    
    #-- Obliquity factor
    SIG = SIG * f / (fkz + np.spacing(1));
    SIG[0] = 0;
    
    #-- Axial IFFT
    SIGnegf = np.conj(np.fliplr(np.roll(SIG,-1,axis=1)));
    SIG = np.concatenate((SIG, SIGnegf[ntFFT/2-1:0:-1,:]), axis = 0); 
    SIG = np.fft.ifft(SIG,axis=0);
    
    #-- Compensate for steering angle
    gamma = sinA/(2-cosA);
    dx = -gamma*np.arange(ntFFT)/fs*c/2;
    Kx, gamma_z = np.meshgrid(kx, dx); # steering compensation
    SIG = SIG*np.exp(-2*1j*np.pi*Kx*gamma_z);
    
    #-- Spatial IFFT
    #-- Final migrated signal
    migSIG = np.fft.ifft(SIG,axis=1);
    migSIG = migSIG[np.arange(nt)+ntshift,0:nx];
    
    #-- Coordinates of Migrated Signal
    x = (np.arange(nx)-(nx-1)/2)*pitch;
    z = (np.arange(nt))*c/2/fs;
    
    return x, z, migSIG;

