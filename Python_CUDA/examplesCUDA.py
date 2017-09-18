# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 10:58:19 2016

@author: rehmanali
"""

from fkmigCUDA import fkmig
import numpy as np
import scipy.io as sio
from scipy.signal import hilbert
import matplotlib.pyplot as plt

#-- Load the RF data
# The RF data have been converted to 8 bits due to size limit (up to 1.5
# MB) of the zipped files in the Supplementary materials.
RFdata1 = sio.loadmat('../RFdata1.mat');
RFdata2 = sio.loadmat('../RFdata2.mat');
RF1 = RFdata1['RF1']; param1 = RFdata1['param1'][0];
RF2 = RFdata2['RF2']; param2 = RFdata2['param2'][0];
RF1 = np.double(RF1); RF1 = RF1 - np.mean(RF1);
RF2 = np.double(RF2); RF2 = RF2 - np.mean(RF2);

#-- Example #1: Nylon fibers 
migRF1 = np.zeros(RF1[:,:,0].shape, dtype = 'complex128');
for idx in np.arange(7):
    x1, z1, migRF_idx = fkmig(RF1[:,:,idx], np.double(param1['fs']), np.double(param1['pitch']), \
        TXangle = np.double(param1['TXangle'][0][:,idx]), c = np.double(param1['c']));
    migRF1 += migRF_idx/7;
im1_mig = (np.abs(hilbert(np.real(migRF1))))**0.7;
exts = (np.min(x1)-np.mean(np.diff(x1)), np.max(x1)+np.mean(np.diff(x1)), \
    np.min(z1)-np.mean(np.diff(z1)), np.max(z1)-np.mean(np.diff(z1)))
plt.title('F-K Migrated Point Targets\n7 Angles Compounded\n$(-1.5^o : 0.5^o : 1.5^o)$');
plt.imshow(np.flipud(im1_mig), cmap = 'gray', extent = exts, interpolation='none');
plt.xticks(0.01*np.arange(-1,2));
plt.yticks(0.01*np.arange(11)); 
plt.gca().invert_yaxis()
plt.xlabel('Azimuth (m)'); plt.ylabel('Depth (m)');
plt.show();

#-- Example #2: Circular Targets
migRF2 = np.zeros(RF2[:,:,0].shape, dtype = 'complex128');
for idx in np.arange(7):
    x2, z2, migRF_idx = fkmig(RF2[:,:,idx], np.double(param2['fs']), \
        np.double(param2['pitch']), TXangle = np.double(param2['TXangle'][0][:,idx]), \
        c = np.double(param2['c']), t0 = np.double(param2['t0']));
    migRF2 += migRF_idx/7;
im2_mig = (np.abs(hilbert(np.real(migRF2))))**0.5;
exts = (np.min(x2)-np.mean(np.diff(x2)), np.max(x2)+np.mean(np.diff(x2)), \
    np.min(z2)-np.mean(np.diff(z2)), np.max(z2)-np.mean(np.diff(z2)))
plt.title('F-K Migrated Circular Cysts\n7 Angles Compounded\n$(-1.5^o : 0.5^o : 1.5^o)$');
plt.imshow(np.flipud(im2_mig), cmap = 'gray', extent = exts, interpolation='none');
plt.xticks(0.01*np.arange(-1,2));
plt.yticks(0.01*np.arange(6)); 
plt.gca().invert_yaxis()
plt.xlabel('Azimuth (m)'); plt.ylabel('Depth (m)');
plt.show();

#-- Example #3: Nylon fibers (Before and After Migration)
idx = 3; # RF data with angle = 0
x1, z1, migRF1 = fkmig(RF1[:,:,idx], np.double(param1['fs']), np.double(param1['pitch']), \
        TXangle = np.double(param1['TXangle'][0][:,idx]), c = np.double(param1['c']));
im_mig = (np.abs(hilbert(np.real(migRF1))))**0.7;
im = (np.abs(hilbert(np.real(RF1[:,:,idx]))))**0.7;
exts = (np.min(x1)-np.mean(np.diff(x1)), np.max(x1)+np.mean(np.diff(x1)), \
    np.min(z1)-np.mean(np.diff(z1)), np.max(z1)-np.mean(np.diff(z1)))
plt.subplot(121); plt.title('RF Data Before Migration')
plt.imshow(im, cmap = 'gray', extent = exts, interpolation='none');
plt.xticks(0.01*np.arange(-1,2));
plt.yticks(0.01*np.arange(10));
plt.xlabel('Azimuth (m)'); plt.ylabel('Depth (m)');
plt.subplot(122); plt.title('F-K Migrated Image')
plt.imshow(im_mig, cmap = 'gray', extent = exts, interpolation='none');
plt.xticks(0.01*np.arange(-1,2));
plt.yticks(0.01*np.arange(10));
plt.xlabel('Azimuth (m)'); plt.ylabel('Depth (m)');
plt.show();
