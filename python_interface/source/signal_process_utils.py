# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:44:31 2024

@author: JEGEN
"""

import numpy as np

# Do a proper fft on a data set
def ffta(x=None, N=None, Dim=None):
  
    if N is None:
        N = np.max(x.shape)
    if Dim is None:
        Dim = np.argmax(x.shape)
    
    y = np.fft.fftshift(np.fft.fft(x, n=N, axis=Dim))
    
    if N % 2 == 0:
        # even
        f = np.arange(-N/2, N/2) / N
    else:
        # odd
        f = np.arange(-(N-1)/2, (N-1)/2 + 1) / N
    return y, f                    
