# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:57:23 2016

@author: rehmanali
"""

import numpy as np
from sys import platform
import os, pdb

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
    https://github.com/rehmanali1994/Plane_Wave_Ultrasound_Stolt_F-K_Migration.github.io.git
    Reference
    --------- 
    Garcia et al., Stolt's f-k migration for plane wave ultrasound imaging.
    IEEE Trans Ultrason Ferroelectr Freq Control, 2013;60:1853-1867. """

    # Get the dimensions of the data
    nt, nx = SIG.shape;  
    
    # Save signals to a file
    np.savetxt("SIG.txt", SIG.flatten()); 
    
    # Run (maybe compile too if necessary) CUDA code that does f-k migration
    if (platform == "darwin") or (platform == "linux"):
        if not(os.path.isfile("fkmigCUDA.out")):
            os.system("nvcc fkmigCUDA.cu -o fkmigCUDA.out -I/usr/local/cuda/include -L/usr/local/cuda/lib -lcufft");
        os.system("./fkmigCUDA.out SIG.txt "+str(nt)+" "+str(nx)+" "+str(fs)+" "+str(pitch)+" "+str(TXangle)+" "+str(c)+" "+str(t0)+" migSIG.txt");
    elif "win" in platform.lower():
        os.system("fkmigCUDA.exe SIG.txt "+str(nt)+" "+str(nx)+" "+str(fs)+" "+str(pitch)+" "+str(TXangle)+" "+str(c)+" "+str(t0)+" migSIG.txt");
    
    # Load migrated image from file
    migSIG = np.loadtxt("migSIG.txt").reshape((nt, nx)); 
    
    # Delete the text files created in this process
    if (platform == "darwin") or (platform == "linux"):
        os.system("rm SIG.txt");
        os.system("rm migSIG.txt");
    elif "win" in platform.lower():
        os.system("del SIG.txt");
        os.system("del migSIG.txt");
    
    #-- Coordinates of Migrated Signal
    x = (np.arange(nx)-(nx-1)/2)*pitch;
    z = (np.arange(nt))*c/2/fs;
    
    return x, z, migSIG;
    
