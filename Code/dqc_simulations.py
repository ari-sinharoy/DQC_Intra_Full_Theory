# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:25:31 2024

@author: as836
"""

import os
import numpy as np
from dqc_kernel import *
from scipy.interpolate import interp1d 
import matplotlib.pyplot as plt
import seaborn as sns
import time

def scmm(data):    
    return (data - data.min()) / (data.max() - data.min())

def movav(data, n=3):
    _ndat = np.pad(data, (int((n-1)/2), int((n-1)/2)), 'constant', constant_values = (data[0], data[-1]))
    return np.convolve(_ndat, np.ones(n)/n, mode='valid')


# define some constants
_uB = 9.274*1E-24
_h = 6.626*1E-34
_g = 2.0023

# molecular parameters

## set P(r) 
_rrange = np.arange(1.0, 3.0, 0.02)
_mu1, _sm1 = 1.5, 0.03 # in nm
_pr1 = np.exp(-(_rrange - _mu1)**2/(2*_sm1**2))
_pr1 /= _pr1.sum()
_prf = interp1d(_rrange, _pr1)

# a series of pulse amplitudes (MHz) and lengths (ns) 
_wps = [125., 83.4, 62.5, 50., 41.7, 35.72, 31.26]
_tps = [4, 6, 8, 10, 12, 14, 16]

# run simulations
for k in range(len(_wps)):
    _start = time.time()

    # theta values
    _thtab = np.linspace(0, np.pi/2, 100)
    _pth = np.sin(_thtab)
    _pth /= _pth.sum()
    
    # pulse parameters
    _wid = 50  
    _mult = 2
    _omtab = np.linspace(-_mult*_wid, _mult*_wid, 100)
    _pom = np.exp(-_omtab**2/(2*_wid**2))
    _pom /= _pom.sum()
        
    _wp = _wps[k] # pulse field in MHz
    _tp = _tps[k]*1E-3 # pulse length in us
    
    # pulse separations in time-domain
    _tm = 0.6
    _t1 = 0.04
    _dt = 0.008
    #_tmin = 0.3
    _ttab = np.arange(0, _tm, _dt)
    
    # simulate the signal
    _mmax = 4*1E4 ## max number of iterations
    
    _res1 = 0 # DQC coherence transfer pathway
    _res2 = 0 # other coherence transfer pathway
    _res3 = 0 # other coherence transfer pathway

    _m = 1
    while _m <= _mmax:
    
        _om1, _om2 = np.random.choice(_omtab, p = _pom), np.random.choice(_omtab, p = _pom)
        #_om1, _om2 = 0., 0. # for ideal pulses
        
        if _om1==_om2:
            _om2 += 1E-4
    
        _r = np.random.choice(_rrange, p = _pr1)
        _th = np.random.choice(_thtab, p = _pth)
        
        _dip = 2*np.pi*52.04/_r**3*(1-3*np.cos(_th)**2)
    
        _sig1 = sig(_om1, _om2, _dip, (_tm - _ttab)/2, _t1, (_tm + _ttab)/2, _wp, _tp, path = "1")
        _sig2 = sig(_om1, _om2, _dip, (_tm - _ttab)/2, _t1, (_tm + _ttab)/2, _wp, _tp, path = "2")
        _sig3 = sig(_om1, _om2, _dip, (_tm - _ttab)/2, _t1, (_tm + _ttab)/2, _wp, _tp, path = "3")
        
        _res1 += _sig1.real
        _res2 += _sig2.real
        _res3 += _sig3.real
        
        _m += 1
        
    _res1 /= _mmax
    _res2 /= _mmax
    _res3 /= _mmax
    
    _end = time.time() - _start
    
    print("SimuLation with %sk iterations took %s sec." %(int(_mmax*1E-3), np.round(_end,2)))
    
    plt.plot(_ttab, _res1, 'red')
    plt.plot(_ttab, _res2, 'blue')
    plt.plot(_ttab, _res3, 'orange')
    plt.show()
    
    # save the simulation results
    _fname = 'DQC_Birad-I_pi-'+str(int(_tps[k]))+'ns.txt'
    np.savetxt(_fname, np.c_[_ttab, _res1, _res2, _res3])