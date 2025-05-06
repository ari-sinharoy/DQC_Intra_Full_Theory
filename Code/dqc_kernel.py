# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 22:51:25 2024

@author: as836
"""

import numpy as np

def Cs(om1, om2, b12):
    _b12 = b12 / (2*np.pi)
    _omN, _omP = (om1 - om2), (om1 + om2) 
    _bet = np.arctan(_b12/_omN)
    _kappa = np.sqrt(_omN**2 + _b12**2)
    
    _c1, _c2 = 1/2*(_kappa + _omP), 1/2*(-_kappa + _omP)
    
    return _c1, _c2, _bet

def spinrot(om, b1, tau):
    _d11 = (b1**2 + om**2*np.cos(2*np.pi*np.sqrt(b1**2 + om**2)*tau)) / (b1**2 + om**2)
    _d12 = om*np.sin(2*np.pi*np.sqrt(b1**2 + om**2)*tau) / np.sqrt(b1**2 + om**2)
    _d13 = -b1*om*(-1 + np.cos(2*np.pi*np.sqrt(b1**2 + om**2)*tau)) / (b1**2 + om**2)
    _d14 = np.cos(2*np.pi*np.sqrt(b1**2 + om**2)*tau)
    _d15 = b1*np.sin(2*np.pi*np.sqrt(b1**2 + om**2)*tau) / np.sqrt(b1**2 + om**2)
    _d16 = (om**2 + b1**2*np.cos(2*np.pi*np.sqrt(b1**2 + om**2)*tau)) / (b1**2 + om**2)
    
    return _d11, _d12, _d13, _d14, _d15, _d16

r'''
def spinrot(om, b1, tau):
    _d11 = 1
    _d12 = 0
    _d13 = 0
    _d14 = -1
    _d15 = 0
    _d16 = (om**2 + b1**2*np.cos(np.sqrt(b1**2 + om**2)*tau)) / (b1**2 + om**2)
    
    return _d11, _d12, _d13, _d14, _d15, _d16
'''

def c31coef(om1, om2, a12, tp, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c31x1 = 1/2*np.cos(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d11*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d11*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c31y1 = -1/2*np.cos(a12*tp/2)*(
        _d14*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d14*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c31z1 = -1/2*np.cos(a12*tp/2)*(
        _d15*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d13*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d15*np.cos(_c2*tp) - _d13*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c31x2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1-_c2)*tp)*(
        _d22*np.cos(1/2*(_c1+_c2)*tp) + _d21*np.sin(1/2*(_c1+_c2)*tp))*np.sin(_bet)
    
    _c31y2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1-_c2)*tp)*(
        -_d24*np.cos(1/2*(_c1+_c2)*tp) + _d22*np.sin(1/2*(_c1+_c2)*tp))*np.sin(_bet)
    
    _c31z2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1-_c2)*tp)*(
        -_d25*np.cos(1/2*(_c1+_c2)*tp) + _d23*np.sin(1/2*(_c1+_c2)*tp))*np.sin(_bet)
    
    _c31yx = (_d23*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(
            _d22*np.cos(_c1*tp) - _d22*np.cos(_c2*tp) + 
            _d21*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31yy = (-_d25*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(
            -_d24*np.cos(_c1*tp) + _d24*np.cos(_c2*tp) + 
            _d22*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31zy = (-_d25*np.sin(a12*tp/2)*(
        (1 + np.cos(_bet))*(_d13*np.cos(_c1*tp) + _d15*np.sin(_c1*tp)) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) - 
        _d16*np.cos(a12*tp/2)*(
            -_d24*np.cos(_c1*tp) + _d24*np.cos(_c2*tp) + 
            _d22*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31yz = (_d26*np.sin(a12*tp/2)*(
        (1 + np.cos(_bet))*(_d12*np.cos(_c1*tp) + _d14*np.sin(_c1*tp)) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(
            -_d25*np.cos(_c1*tp) + _d25*np.cos(_c2*tp) + 
            _d23*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31zx = (_d23*np.sin(a12*tp/2)*(
        (1 + np.cos(_bet))*(_d13*np.cos(_c1*tp) + _d15*np.sin(_c1*tp)) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(
            -_d22*np.cos(_c1*tp) + _d22*np.cos(_c2*tp) + 
            _d21*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31xy = (_d25*np.sin(a12*tp/2)*(
        -_d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(-_d11*np.cos(_c2*tp) + _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(
            _d24*np.cos(_c1*tp) - _d24*np.cos(_c2*tp) + 
            _d22*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31xz = (_d26*np.sin(a12*tp/2)*(
        _d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(
            _d25*np.cos(_c1*tp) - _d25*np.cos(_c2*tp) + 
            _d23*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31zz = (_d26*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(
            _d25*np.cos(_c1*tp) - _d25*np.cos(_c2*tp) + 
            _d23*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c31xx = (_d23*np.sin(a12*tp/2)*(
        -_d12*(1 + np.cos(_bet))*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(
            _d22*np.cos(_c2*tp) - _d21*np.sin(_c1*tp) + _d21*np.sin(_c2*tp))*np.sin(_bet) + 
        np.cos(_c1*tp)*(_d11*_d23*(1 + np.cos(_bet))*np.sin(a12*tp/2) - 
                        _d13*_d22*np.cos(a12*tp/2)*np.sin(_bet)))
    
    return _c31x1, _c31y1, _c31z1, _c31x2, _c31y2, _c31z2, _c31xx, _c31xy, _c31yx, _c31yy, _c31zz, _c31xz, _c31zx, _c31yz, _c31zy

    
def c32coef(om1, om2, a12, tp, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c32x1 = 1/2*np.cos(a12*tp/2)*(2*np.cos(_bet/2)**2*(
        _d14*np.cos(_c1*tp) - _d12*np.sin(_c1*tp)) + 
        2*(_d14*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c32y1 = 1/2*np.cos(a12*tp/2)*(
        2*_d12*np.cos(_c1*tp)*np.cos(_bet/2)**2 + _d11*(1 + np.cos(_bet))*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d11*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c32z1 = 1/2*np.cos(a12*tp/2)*(
        2*np.cos(_bet/2)**2*(_d15*np.cos(_c1*tp) - _d13*np.sin(_c1*tp)) + 
        2*(_d15*np.cos(_c2*tp) - _d13*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c32x2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1 - _c2)*tp)*(
        _d24*np.cos(1/2*(_c1 + _c2)*tp) - _d22*np.sin(1/2*(_c1 + _c2)*tp))*np.sin(_bet)
    
    _c32y2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1 - _c2)*tp)*(
        _d22*np.cos(1/2*(_c1 + _c2)*tp) + _d21*np.sin(1/2*(_c1 + _c2)*tp))*np.sin(_bet)
    
    _c32z2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1 - _c2)*tp)*(
        _d25*np.cos(1/2*(_c1 + _c2)*tp) - _d23*np.sin(1/2*(_c1 + _c2)*tp))*np.sin(_bet)
    
    _c32yx = (-_d25*np.sin(a12*tp/2)*(
        _d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(_d24*np.cos(_c1*tp) - _d24*np.cos(_c2*tp) + 
                               _d22*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32yy = (-_d23*np.sin(a12*tp/2)*(
        _d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(_d22*np.cos(_c1*tp) - _d22*np.cos(_c2*tp) + 
                               _d21*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32zy = (_d23*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(-_d22*np.cos(_c1*tp) + _d22*np.cos(_c2*tp) + 
                               _d21*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32yz = (_d26*np.sin(a12*tp/2)*(
        _d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(_d25*np.cos(_c1*tp) - _d25*np.cos(_c2*tp) + 
                               _d23*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32zx = (_d25*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(-_d24*np.cos(_c1*tp) + _d24*np.cos(_c2*tp) + 
                               _d22*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32xy = (_d23*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(_d22*np.cos(_c1*tp) - _d22*np.cos(_c2*tp) + 
                               _d21*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32xz = (-_d26*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(_d25*np.cos(_c1*tp) - _d25*np.cos(_c2*tp) + 
                               _d23*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32zz = (-_d26*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(-_d25*np.cos(_c1*tp) + _d25*np.cos(_c2*tp) + 
                               _d23*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c32xx = (_d25*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(_d24*np.cos(_c1*tp) - _d24*np.cos(_c2*tp) + 
                               _d22*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    return _c32x1, _c32y1, _c32z1, _c32x2, _c32y2, _c32z2, _c32xx, _c32xy, _c32yx, _c32yy, _c32zz, _c32xz, _c32zx, _c32yz, _c32zy

def c33coef(om1, om2, a12, tp, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c33x1 = -1/2*np.cos(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d11*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d11*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c33y1 = 1/2*np.cos(a12*tp/2)*(
        _d14*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d14*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c33z1 = -1/2*np.cos(a12*tp/2)*(
        _d15*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d13*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d15*np.cos(_c2*tp) - _d13*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c33x2 = -np.sin(a12*tp/2)*np.sin(1/2*(_c1-_c2)*tp)*(
        _d22*np.cos(1/2*(_c1+_c2)*tp) + _d21*np.sin(1/2*(_c1+_c2)*tp))*np.sin(_bet)
    
    _c33y2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1-_c2)*tp)*(
        _d24*np.cos(1/2*(_c1+_c2)*tp) - _d22*np.sin(1/2*(_c1+_c2)*tp))*np.sin(_bet)
    
    _c33z2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1-_c2)*tp)*(
        -_d25*np.cos(1/2*(_c1+_c2)*tp) + _d23*np.sin(1/2*(_c1+_c2)*tp))*np.sin(_bet)
    
    _c33yx = (_d23*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(
            _d22*np.cos(_c1*tp) - _d22*np.cos(_c2*tp) + 
            _d21*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33yy = (-_d25*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(
            -_d24*np.cos(_c1*tp) + _d24*np.cos(_c2*tp) + 
            _d22*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33zy = (_d25*np.sin(a12*tp/2)*(
        (1 + np.cos(_bet))*(_d13*np.cos(_c1*tp) + _d15*np.sin(_c1*tp)) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(
            -_d24*np.cos(_c1*tp) + _d24*np.cos(_c2*tp) + 
            _d22*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33yz = (-_d26*np.sin(a12*tp/2)*(
        (1 + np.cos(_bet))*(_d12*np.cos(_c1*tp) + _d14*np.sin(_c1*tp)) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) - 
        _d15*np.cos(a12*tp/2)*(
            -_d25*np.cos(_c1*tp) + _d25*np.cos(_c2*tp) + 
            _d23*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33zx = (-_d23*np.sin(a12*tp/2)*(
        (1 + np.cos(_bet))*(_d13*np.cos(_c1*tp) + _d15*np.sin(_c1*tp)) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) - 
        _d16*np.cos(a12*tp/2)*(
            -_d22*np.cos(_c1*tp) + _d22*np.cos(_c2*tp) + 
            _d21*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33xy = (_d25*np.sin(a12*tp/2)*(
        -_d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(-_d11*np.cos(_c2*tp) + _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(
            _d24*np.cos(_c1*tp) - _d24*np.cos(_c2*tp) + 
            _d22*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33xz = (_d26*np.sin(a12*tp/2)*(
        -_d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(-_d11*np.cos(_c2*tp) + _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(
            -_d25*np.cos(_c1*tp) + _d25*np.cos(_c2*tp) + 
            _d23*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33zz = (_d26*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(
            _d25*np.cos(_c1*tp) - _d25*np.cos(_c2*tp) + 
            _d23*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c33xx = (_d23*np.sin(a12*tp/2)*(
        -_d12*(1 + np.cos(_bet))*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(
            _d22*np.cos(_c2*tp) - _d21*np.sin(_c1*tp) + _d21*np.sin(_c2*tp))*np.sin(_bet) + 
        np.cos(_c1*tp)*(_d11*_d23*(1 + np.cos(_bet))*np.sin(a12*tp/2) - 
                        _d13*_d22*np.cos(a12*tp/2)*np.sin(_bet)))    

    return _c33x1, _c33y1, _c33z1, _c33x2, _c33y2, _c33z2, _c33xx, _c33xy, _c33yx, _c33yy, _c33zz, _c33xz, _c33zx, _c33yz, _c33zy

def c34coef(om1, om2, a12, tp, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c34x1 = 1/2*np.cos(a12*tp/2)*(
        -2*_d14*np.cos(_c1*tp)*np.cos(_bet/2)**2 + _d12*(1 + np.cos(_bet))*np.sin(_c1*tp) + 
        2*(-_d14*np.cos(_c2*tp) + _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c34y1 = -1/2*np.cos(a12*tp/2)*(
        2*_d12*np.cos(_c1*tp)*np.cos(_bet/2)**2 + _d11*(1 + np.cos(_bet))*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d11*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c34z1 = 1/2*np.cos(a12*tp/2)*(
        2*np.cos(_bet/2)**2*(_d15*np.cos(_c1*tp) - _d13*np.sin(_c1*tp)) + 
        2*(_d15*np.cos(_c2*tp) - _d13*np.sin(_c2*tp))*np.sin(_bet/2)**2)
    
    _c34x2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1 - _c2)*tp)*(
        -_d24*np.cos(1/2*(_c1 + _c2)*tp) + _d22*np.sin(1/2*(_c1 + _c2)*tp))*np.sin(_bet)
    
    _c34y2 = -np.sin(a12*tp/2)*np.sin(1/2*(_c1 - _c2)*tp)*(
        _d22*np.cos(1/2*(_c1 + _c2)*tp) + _d21*np.sin(1/2*(_c1 + _c2)*tp))*np.sin(_bet)
    
    _c34z2 = np.sin(a12*tp/2)*np.sin(1/2*(_c1 - _c2)*tp)*(
        _d25*np.cos(1/2*(_c1 + _c2)*tp) - _d23*np.sin(1/2*(_c1 + _c2)*tp))*np.sin(_bet)
    
    _c34yx = (-_d25*np.sin(a12*tp/2)*(
        _d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(_d24*np.cos(_c1*tp) - _d24*np.cos(_c2*tp) + 
                               _d22*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34yy = (-_d23*np.sin(a12*tp/2)*(
        _d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(_d22*np.cos(_c1*tp) - _d22*np.cos(_c2*tp) + 
                               _d21*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34zy = (-_d23*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(_d22*np.cos(_c1*tp) - _d22*np.cos(_c2*tp) + 
                               _d21*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34yz = (-_d26*np.sin(a12*tp/2)*(
        _d11*np.cos(_c1*tp)*(1 + np.cos(_bet)) - 2*_d12*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d11*np.cos(_c2*tp) - _d12*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d13*np.cos(a12*tp/2)*(-_d25*np.cos(_c1*tp) + _d25*np.cos(_c2*tp) + 
                               _d23*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34zx = (-_d25*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(_d24*np.cos(_c1*tp) - _d24*np.cos(_c2*tp) + 
                               _d22*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34xy = (_d23*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(_d22*np.cos(_c1*tp) - _d22*np.cos(_c2*tp) + 
                               _d21*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34xz = (_d26*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(-_d25*np.cos(_c1*tp) + _d25*np.cos(_c2*tp) + 
                               _d23*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34zz = (-_d26*np.sin(a12*tp/2)*(
        _d13*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d15*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d13*np.cos(_c2*tp) + _d15*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d16*np.cos(a12*tp/2)*(-_d25*np.cos(_c1*tp) + _d25*np.cos(_c2*tp) + 
                               _d23*(np.sin(_c1*tp) - np.sin(_c2*tp)))*np.sin(_bet))
    
    _c34xx = (_d25*np.sin(a12*tp/2)*(
        _d12*np.cos(_c1*tp)*(1 + np.cos(_bet)) + 2*_d14*np.cos(_bet/2)**2*np.sin(_c1*tp) + 
        2*(_d12*np.cos(_c2*tp) + _d14*np.sin(_c2*tp))*np.sin(_bet/2)**2) + 
        _d15*np.cos(a12*tp/2)*(_d24*np.cos(_c1*tp) - _d24*np.cos(_c2*tp) + 
                               _d22*(-np.sin(_c1*tp) + np.sin(_c2*tp)))*np.sin(_bet))
    
    return _c34x1, _c34y1, _c34z1, _c34x2, _c34y2, _c34z2, _c34xx, _c34xy, _c34yx, _c34yy, _c34zz, _c34xz, _c34zx, _c34yz, _c34zy


def c5coef(om1, om2, a12, tp, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)

    _c31x1, _c31y1, _c31z1, _c31x2, _c31y2, _c31z2, _c31xx, _c31xy, _c31yx, _c31yy, _c31zz, _c31xz, _c31zx, _c31yz, _c31zy = c31coef(om1, om2, a12, tp, b1, tau)    
    _c32x1, _c32y1, _c32z1, _c32x2, _c32y2, _c32z2, _c32xx, _c32xy, _c32yx, _c32yy, _c32zz, _c32xz, _c32zx, _c32yz, _c32zy = c32coef(om1, om2, a12, tp, b1, tau)
    _c33x1, _c33y1, _c33z1, _c33x2, _c33y2, _c33z2, _c33xx, _c33xy, _c33yx, _c33yy, _c33zz, _c33xz, _c33zx, _c33yz, _c33zy = c33coef(om1, om2, a12, tp, b1, tau)    
    _c34x1, _c34y1, _c34z1, _c34x2, _c34y2, _c34z2, _c34xx, _c34xy, _c34yx, _c34yy, _c34zz, _c34xz, _c34zx, _c34yz, _c34zy = c34coef(om1, om2, a12, tp, b1, tau)
    
    _c5y = 1/2*(-_c31xz*np.cos(a12*tp/2)*np.cos(_c1*tp) - _c32zy*np.cos(a12*tp/2)*np.cos(_c1*tp) + 
                _c33xz*np.cos(a12*tp/2)*np.cos(_c1*tp) + _c34zy*np.cos(a12*tp/2)*np.cos(_c1*tp) - 
                _c31xz*np.cos(a12*tp/2)*np.cos(_c2*tp) - _c32zy*np.cos(a12*tp/2)*np.cos(_c2*tp) + 
                _c33xz*np.cos(a12*tp/2)*np.cos(_c2*tp) + _c34zy*np.cos(a12*tp/2)*np.cos(_c2*tp) - 
                _c31xz*np.cos(a12*tp/2)*np.cos(_c1*tp)*np.cos(_bet) + _c32zy*np.cos(a12*tp/2)*np.cos(_c1*tp)*np.cos(_bet) + 
                _c33xz*np.cos(a12*tp/2)*np.cos(_c1*tp)*np.cos(_bet) - _c34zy*np.cos(a12*tp/2)*np.cos(_c1*tp)*np.cos(_bet) + 
                _c31xz*np.cos(a12*tp/2)*np.cos(_c2*tp)*np.cos(_bet) - _c32zy*np.cos(a12*tp/2)*np.cos(_c2*tp)*np.cos(_bet) - 
                _c33xz*np.cos(a12*tp/2)*np.cos(_c2*tp)*np.cos(_bet) + _c34zy*np.cos(a12*tp/2)*np.cos(_c2*tp)*np.cos(_bet) - 
                2*_c32x2*np.cos(_c1*tp)*np.sin(a12*tp/2) + 2*_c34x2*np.cos(_c1*tp)*np.sin(a12*tp/2) - 
                2*_c32x2*np.cos(_c2*tp)*np.sin(a12*tp/2) + 2*_c34x2*np.cos(_c2*tp)*np.sin(a12*tp/2) + 
                4*_c31y1*np.cos(_c1*tp)*np.cos(_bet/2)**2*np.sin(a12*tp/2) - 4*_c33y1*np.cos(_c1*tp)*np.cos(_bet/2)**2*np.sin(a12*tp/2) + 
                2*_c32x2*np.cos(_c1*tp)*np.cos(_bet)*np.sin(a12*tp/2) - 2*_c34x2*np.cos(_c1*tp)*np.cos(_bet)*np.sin(a12*tp/2) - 
                2*_c32x2*np.cos(_c2*tp)*np.cos(_bet)*np.sin(12*tp/2) + 2*_c34x2*np.cos(_c2*tp)*np.cos(_bet)*np.sin(a12*tp/2) + 
                _c31yz*np.cos(a12*tp/2)*np.sin(_c1*tp) - _c32zx*np.cos(a12*tp/2)*np.sin(_c1*tp) - 
                _c33yz*np.cos(a12*tp/2)*np.sin(_c1*tp) + _c34zx*np.cos(a12*tp/2)*np.sin(_c1*tp) + 
                _c31yz*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c1*tp) + _c32zx*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c1*tp) - 
                _c33yz*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c1*tp) - _c34zx*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c1*tp) + 
                4*_c31x1*np.cos(_bet/2)**2*np.sin(a12*tp/2)*np.sin(_c1*tp) - 4*_c33x1*np.cos(_bet/2)**2*np.sin(a12*tp/2)*np.sin(_c1*tp) + 
                _c31yz*np.cos(a12*tp/2)*np.sin(_c2*tp) - _c32zx*np.cos(a12*tp/2)*np.sin(_c2*tp) - 
                _c33yz*np.cos(a12*tp/2)*np.sin(_c2*tp) + _c34zx*np.cos(a12*tp/2)*np.sin(_c2*tp) - 
                _c31yz*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c2*tp) - _c32zx*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c2*tp) + 
                _c33yz*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c2*tp) + _c34zx*np.cos(a12*tp/2)*np.cos(_bet)*np.sin(_c2*tp) + 
                4*_c32y2*np.cos(_bet/2)**2*np.sin(a12*tp/2)*np.sin(_c2*tp) - 4*_c34y2*np.cos(_bet/2)**2*np.sin(a12*tp/2)*np.sin(_c2*tp) + 
                4*_c31y1*np.cos(_c2*tp)*np.sin(a12*tp/2)*np.sin(_bet/2)**2 - 4*_c33y1*np.cos(_c2*tp)*np.sin(a12*tp/2)*np.sin(_bet/2)**2 + 
                4*_c32y2*np.sin(a12*tp/2)*np.sin(_c1*tp)*np.sin(_bet/2)**2 - 4*_c34y2*np.sin(a12*tp/2)*np.sin(_c1*tp)*np.sin(_bet/2)**2 + 
                4*_c31x1*np.sin(a12*tp/2)*np.sin(_c2*tp)*np.sin(_bet/2)**2 - 4*_c33x1*np.sin(a12*tp/2)*np.sin(_c2*tp)*np.sin(_bet/2)**2 + 
                (2*np.cos(a12*tp/2)*(
                    (_c31x2 + _c32y1 - _c33x2 - _c34y1)*(np.cos(_c1*tp) - np.cos(_c2*tp)) - 
                    (_c31y2 - _c32x1 - _c33y2 + _c34x1)*(np.sin(_c1*tp) - np.sin(_c2*tp))) + 
                2*np.sin(a12*tp/2)*(
                    -((_c31zy - _c32xz - _c33zy + _c34xz)*(np.cos(_c1*tp) - np.cos(_c2*tp))) - 
                    (_c31zx + _c32yz - _c33zx - _c34yz)*(np.sin(_c1*tp) - np.sin(_c2*tp))))*np.sin(_bet))
    
    _c5x = 1/4*(_c31xx + _c31yy - 4*_c32zz + _c33xx + _c33yy - 4*_c34zz + 
                (_c31xx + _c31yy + _c33xx + _c33yy)*np.cos((_c1-_c2)*tp) + 
                2*(_c31xx - _c31yy + _c33xx - _c33yy)*np.cos((_c1+_c2)*tp) + 
                2*(_c31xy - _c31yx + _c33xy - _c33yx)*np.cos(_bet)*np.sin((_c1-_c2)*tp) - 
                2*(_c31xy + _c31yx + _c33xy + _c33yx)*np.sin((_c1+_c2)*tp) + 
                2*np.sin(1/2*(_c1-_c2)*tp)**2*(
                    -((_c31xx + _c31yy + _c33xx + _c33yy)*np.cos(2*_bet)) + 
                    2*(_c31z1 - _c31z2 + _c33z1 - _c33z2)*np.sin(2*_bet)))
    
    _c5z = 1/4*(
        _c31xx + _c31yy - _c32xx - _c32yy + _c33xx + _c33yy - _c34xx - _c34yy + 
        (_c31xx + _c31yy - _c32xx - _c32yy + _c33xx + _c33yy - _c34xx - _c34yy)*np.cos((_c1 - _c2)*tp) - 
        2*(_c31xx - _c31yy + _c32xx - _c32yy + _c33xx - _c33yy + _c34xx - _c34yy)*np.cos((_c1 + _c2)*tp) +
        2*(_c31xy - _c31yx - _c32xy + _c32yx + _c33xy - _c33yx - _c34xy + _c34yx)*np.sin((_c1 - _c2)*tp)*np.cos(_bet) +
        2*(_c31xy + _c31yx + _c32xy + _c32yx + _c33xy + _c33yx + _c34xy + _c34yx)*np.sin((_c1 + _c2)*tp) +
        2*np.sin(1/2*(_c1-_c2)*tp)**2*(
            (-_c31xx - _c31yy + _c32xx + _c32yy - _c33xx - _c33yy + _c34xx + _c34yy)*np.cos(2*_bet) + 
            2*(_c31z1 - _c31z2 - _c32z1 + _c32z2 + _c33z1 - _c33z2 - _c34z1 + _c34z2)*np.sin(2*_bet)))
    
    return _c5y, _c5x, _c5z

def c9coefY(om1, om2, a12, tp, t1, t2, b1, tau):

    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c5y, _c5x, _c5z = c5coef(om1, om2, a12, tp, b1, tau)
    
    _c9x1, _c9y1, _c9x2, _c9y2 = 0, 0, 0, 0
    
    _c9xz = 2*_c5y*(
        -((_d11 - _d14)*(_d21 - _d24)) + 
        (-4*_d12*_d22 + (_d11 + _d14)*(_d21 + _d24))*np.cos(2*(_c1 + _c2)*t1) - 
        2*((_d11 + _d14)*_d22 + _d12*(_d21 + _d24))*np.sin(2*(_c1 + _c2)*t1))
    
    _c9yz = 1j*_c9xz
    _c9zx = 1*_c9xz
    _c9zy = 1j*_c9xz
    
    return _c9x1, _c9y1, _c9x2, _c9y2, _c9xz, _c9zx, _c9yz, _c9zy

def c9coefX(om1, om2, a12, tp, t1, t2, b1, tau):

    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c5y, _c5x, _c5z = c5coef(om1, om2, a12, tp, b1, tau)
        
    _c9y1 = _c5x*np.sin(1/2*(_c1 - _c2)*t1)*(
        4*((_d11 + _d14)*_d22 - _d12*(_d21 + _d24))*np.cos(1/2*(_c1 - _c2)*t1)**3*np.sin(_bet) + 
        np.sin(1/2*(_c1 - _c2)*t1)*(
            -((3*_d16 + _d11*_d21 + 2*_d14*_d21 + 6*_d12*_d22 + 2*_d11*_d24 + _d14*_d24 - _d26 + 
               (_d16 + _d11*_d21 + 2*_d14*_d21 + 6*_d12*_d22 + 2*_d11*_d24 + _d14*_d24 + 
                _d26)*np.cos((_c1 - _c2)*t1))*np.sin(_bet)) + 
            2*(-((_d11 + _d14)*_d22) + _d12*(_d21 + _d24))*np.sin((_c1 - _c2)*t1)*np.sin(3*_bet) + 
            (-_d16 + _d11*_d21 + 2*_d12*_d22 + _d14*_d24 - _d26)*np.sin(1/2*(_c1 - _c2)*t1)**2*np.sin(4*_bet)))

    _c9y2 = _c5x*np.sin(1/2*(_c1 - _c2)*t1)*(
        4*(-((_d11 + _d14)*_d22) + _d12*(_d21 + _d24))*np.cos(1/2*(_c1 - _c2)*t1)**3*np.sin(_bet) + 
        np.sin(1/2*(_c1 - _c2)*t1)*(
            (-_d16 + _d11*_d21 + 2*_d14*_d21 + 6*_d12*_d22 + 2*_d11*_d24 + _d14*_d24 + 3*_d26 + 
            (_d16 + _d11*_d21 + 2*_d14*_d21 + 6*_d12*_d22 + 2*_d11*_d24 + _d14*_d24 + _d26)*np.cos((_c1 - _c2)*t1))*np.sin(2*_bet) + 
            2*((_d11 + _d14)*_d22 - _d12*(_d21 + _d24))*np.sin((_c1 - _c2)*t1)*np.sin(3*_bet) + 
            (_d16 - _d11*_d21 - 2*_d12*_d22 - _d14*_d24 + _d26)*np.sin(1/2*(_c1 - _c2)*t1)**2*np.sin(4*_bet)))

    _c9xz = _c5x*(4*((_d11 + _d14)*_d22 - _d12*(_d21 + _d24))*np.cos(2*(_c1 - _c2)*t1)*np.cos(_bet)**2 - 
                  (_d16 - _d11*_d21 - 2*_d12*_d22 - _d14*_d24 + _d26)*(
                      4*np.cos(1/2*(_c1 - _c2)*t1)*np.cos(3*_bet)*np.sin(1/2*(_c1 - _c2)*t1)**3 - 
                      np.cos(_bet)*np.sin((_c1 - _c2)*t1)) + 
                  np.cos((_c1 - _c2)*t1)*(
                      -((_d16 + 3*_d11*_d21 + 4*_d14*_d21 + 14*_d12*_d22 + 
                         4*_d11*_d24 + 3*_d14*_d24 + _d26)*np.cos(_bet)*np.sin((_c1 - _c2)*t1)) + 
                      4*((_d11 + _d14)*_d22 - _d12*(_d21 + _d24))*np.sin(_bet)**2))
    
    _c9x1 = 1j*_c9y1
    _c9x2 = 1j*_c9y2
    
    _c9yz = -1j*_c9xz
    _c9zx = -1*_c9xz
    _c9zy = 1j*_c9xz
    
    return _c9x1, _c9y1, _c9x2, _c9y2, _c9xz, _c9zx, _c9yz, _c9zy

def c9coefZ(om1, om2, a12, tp, t1, t2, b1, tau):

    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c5y, _c5x, _c5z = c5coef(om1, om2, a12, tp, b1, tau)
        
    _c9y1 = 2*_c5z*((_d15*_d23 - _d13*_d25)*np.sin((_c1 - _c2)*t1)*np.sin(_bet) - 
                    (_d13*_d23 + _d15*_d25)*np.sin(1/2*(_c1 - _c2)*t1)**2*np.sin(2*_bet))
    
    _c9xz = 4*_c5z*((_d15*_d23 - _d13*_d25)*np.cos((_c1 - _c2)*t1) - 
                    (_d13*_d23 + _d15*_d25)*np.cos(_bet)*np.sin((_c1 - _c2)*t1))
    
    _c9x1 = 1j*_c9y1
    _c9y2 = -1*_c9y1
    _c9x2 = -1j*_c9y1
    
    _c9yz = -1j*_c9xz
    _c9zx = -1*_c9xz
    _c9zy = 1j*_c9xz
    
    return _c9x1, _c9y1, _c9x2, _c9y2, _c9xz, _c9zx, _c9yz, _c9zy


def c11coefY(om1, om2, a12, tp, t1, t2, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c9x1, _c9y1, _c9x2, _c9y2, _c9xz, _c9zx, _c9yz, _c9zy = c9coefY(om1, om2, a12, tp, t1, t2, b1, tau)
    
    _c11x1 = -1j/2*_c9xz*np.exp(-1j/2*(_c1 + _c2)*t2)*(_d11 - _d14)*(
        np.sin(1/2*(a12 + _c1 - _c2)*t2) + np.sin(1/2*(a12 - _c1 + _c2)*t2) + 
        2*np.sin(1/2*(_c1 - _c2)*t2)*(
            -1j*np.cos(_bet)*np.sin(a12*t2/2) - np.cos(a12*t2/2)*np.sin(_bet)))
    
    _c11y1 = -1j*_c11x1
    
    _c11xz = _c9xz*np.exp(-1j/2*(_c1 + _c2)*t2)*(
        ((_d13 - 1j*_d15)*(_d23 + 1j*_d25) + (_d11 - _d14)*_d26)*(
            np.cos(1/2*(a12 + _c1 - _c2)*t2) + np.cos(1/2*(a12 - _c1 + _c2)*t2)) + 
        2j*((_d13 - 1j*_d15)*(_d23 + 1j*_d25) + (-_d11 + _d14)*_d26)*np.cos(a12*t2/2)*np.cos(_bet)*np.sin(1/2*(_c1 - _c2)*t2) + 
        2*((_d13 - 1j*_d15)*(_d23 + 1j*_d25) + (_d11 - _d14)*_d26)*np.sin(a12*t2/2)*np.sin(1/2*(_c1 - _c2)*t2)*np.sin(_bet))
    
    _c11yz = -1j*_c11xz
    
    _c11x2 = 1/2*_c9xz*np.exp(-1j*(_c1 + _c2)*t2)*(_d21 - _d24)*(
        1j*(np.exp(1j*_c2*t2)*(-1 + np.cos(_bet)) - np.exp(1j*_c1*t2)*(1 + np.cos(_bet))
            )*np.sin(a12*t2/2) + (np.exp(1j*_c1*t2) - np.exp(1j*_c2*t2))*np.cos(a12*t2/2)*np.sin(_bet))
    
    _c11y2 = -1j*_c11x2
    
    _c11zx = _c9xz*np.exp(-1j/2*(_c1 + _c2)*t2)*(
        (_d16*(_d21 - _d24) + (_d13 + 1j*_d15)*(_d23 - 1j*_d25))*(
            np.cos(1/2*(a12 + _c1 - _c2)*t2) + np.cos(1/2*(a12 - _c1 + _c2)*t2)) + 
        2j*(_d16*(_d21 - _d24) - (_d13 + 1j*_d15)*(_d23 - 1j*_d25))*np.cos(a12*t2/2)*np.cos(_bet)*np.sin(1/2*(_c1 - _c2)*t2) + 
        2*(_d16*(_d21 - _d24) + (_d13 + 1j*_d15)*(_d23 - 1j*_d25))*np.sin(a12*t2/2)*np.sin(1/2*(_c1 - _c2)*t2)*np.sin(_bet))
    
    _c11zy = -1j*_c11zx
    
    return _c11x1, _c11y1, _c11x2, _c11y2, _c11xz, _c11yz, _c11zx, _c11zy

def c11coefX(om1, om2, a12, tp, t1, t2, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c9x1, _c9y1, _c9x2, _c9y2, _c9xz, _c9zx, _c9yz, _c9zy = c9coefX(om1, om2, a12, tp, t1, t2, b1, tau)
    
    _c11x1 = 1/4*np.exp(-1j/2*a12*t2)*(_d11 - _d14)*(
        (-_c9xz + 2j*_c9y1 + (_c9xz + 2j*_c9y1)*np.exp(1j*a12*t2))*(
            np.exp(1j*_c1*t2) + np.exp(1j*_c2*t2)) + 
        (np.exp(1j*_c1*t2) - np.exp(1j*_c2*t2))*(
            -_c9xz*np.cos(_bet) + 2j*_c9y1*np.cos(_bet) + _c9xz*np.sin(_bet) + 2j*_c9y2*np.sin(_bet) + 
            np.exp(1j*a12*t2)*(
                (_c9xz + 2j*_c9y1)*np.cos(_bet) + (_c9xz - 2j*_c9y2)*np.sin(_bet))))
    
    _c11y1 = 1j*_c11x1
    
    _c11xz = 2*np.exp(1j/2*(_c1 + _c2)*t2)*(
        np.cos(a12*t2/2)*(
            -_c9xz*((_d13 + 1j*_d15)*(_d23 - 1j*_d25) + (-_d11 + _d14)*_d26)*np.cos(1/2*(_c1 - _c2)*t2) + 
            np.sin(1/2*(_c1 - _c2)*t2)*(1j*_c9xz*((_d13 + 1j*_d15)*(_d23 - 1j*_d25) + (_d11 - _d14)*_d26)*np.cos(_bet) + 
                                        2*(_c9y1*(_d13 + 1j*_d15)*(_d23 - 1j*_d25) + 
                                           _c9y2*(_d11 - _d14)*_d26)*np.sin(_bet))) + 
        np.sin(a12*t2/2)*(
            -2*(_c9y2*(_d13 + 1j*_d15)*(_d23 - 1j*_d25) + _c9y1*(_d11 - _d14)*_d26)*np.cos(1/2*(_c1 - _c2)*t2) + 
            np.sin(1/2*(_c1 - _c2)*t2)*(2j*(_c9y2*(_d13 + 1j*_d15)*(_d23 - 1j*_d25) + 
                                            _c9y1*(-_d11 + _d14)*_d26)*np.cos(_bet) + 
                                        _c9xz*((_d13 + 1j*_d15)*(_d23 - 1j*_d25) + 
                                               (-_d11 + _d14)*_d26)*np.sin(_bet))))
    
    _c11yz = 1j*_c11xz
    
    _c11x2 = 1/2*(_d21 - _d24)*(
        np.cos(a12*t2/2)*(-1j*np.exp(1j*_c1*t2)*(2*_c9y2*(-1 + np.cos(_bet)) - 1j*_c9xz*np.sin(_bet)) + 
                          np.exp(1j*_c2*t2)*(2j*_c9y2*(1 + np.cos(_bet)) + _c9xz*np.sin(_bet))) + 
        np.sin(a12*t2/2)*(-1j*np.exp(1j*_c2*t2)*(_c9xz + _c9xz*np.cos(_bet) - 2j*_c9y1*np.sin(_bet)) + 
                          np.exp(1j*_c1*t2)*(1j*_c9xz*(-1 + np.cos(_bet)) + 2*_c9y1*np.sin(_bet))))
    
    _c11y2 = 1j*_c11x2
    
    _c11zx = -2*np.exp(1j/2*(_c1 + _c2)*2)*(
        np.sin(a12*t2/2)*(
            2*(_c9y2*_d16*(_d21 - _d24) + _c9y1*(_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.cos(1/2*(_c1 - _c2)*t2) + 
            np.sin(1/2*(_c1 - _c2)*t2)*(2j*(_c9y2*_d16*(-_d21 + _d24) + _c9y1*(_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.cos(_bet) + 
                                        _c9xz*(_d16*(-_d21 + _d24) + (_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.sin(_bet))) + 
        np.cos(a12*t2/2)*(
            _c9xz*(_d16*(_d21 - _d24) - (_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.cos(1/2*(_c1 - _c2)*t2) + 
            np.sin(1/2*(_c1 - _c2)*t2)*(-1j*_c9xz*(_d16*(_d21 - _d24) + (_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.cos(_bet) - 
                                        2*(_c9y1*_d16*(_d21 - _d24) + _c9y2*(_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.sin(_bet))))
    
    _c11zy = 1j*_c11zx
    
    return _c11x1, _c11y1, _c11x2, _c11y2, _c11xz, _c11yz, _c11zx, _c11zy

def c11coefZ(om1, om2, a12, tp, t1, t2, b1, tau):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)

    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    _c9x1, _c9y1, _c9x2, _c9y2, _c9xz, _c9zx, _c9yz, _c9zy = c9coefZ(om1, om2, a12, tp, t1, t2, b1, tau)
    
    _c11x1 = 1j*np.exp(1j/2*(_c1 + _c2)*t2)*(_d11 - _d14)*(
        (2*_c9y1*np.cos(a12*t2/2) + _c9xz*np.sin(a12*t2/2))*(
            np.cos(1/2*(_c1 - _c2)*t2) + 1j*np.cos(_bet)*np.sin(1/2*(_c1 - _c2)*t2)) + 
        (_c9xz*np.cos(a12*t2/2) - 2*_c9y1*np.sin(a12*t2/2))*np.sin(1/2*(_c1 - _c2)*t2)*np.sin(_bet))
    
    _c11y1 = 1j*_c11x1
    
    _c11xz = 2*np.exp(1j/2*(_c1 + _c2)*t2)*(
        -(((_d13 + 1j*_d15)*(_d23 - 1j*_d25) + (-_d11 + _d14)*_d26)*np.cos(1/2*(_c1 - _c2)*t2)*(
            _c9xz*np.cos(a12*t2/2) - 2*_c9y1*np.sin(a12*t2/2))) + 
        1j*((_d13 + 1j*_d15)*(_d23 - 1j*_d25) + (_d11 - _d14)*_d26)*np.cos(_bet)*(
            _c9xz*np.cos(a12*t2/2) - 2*_c9y1*np.sin(a12*t2/2))*np.sin(1/2*(_c1 - _c2)*t2) + 
        ((_d13 + 1j*_d15)*(_d23 - 1j*_d25) + (-_d11 + _d14)*_d26)*(
            2*_c9y1*np.cos(a12*t2/2) + _c9xz*np.sin(a12*t2/2))*np.sin(1/2*(_c1 - _c2)*t2)*np.sin(_bet))
    
    _c11yz = 1j*_c11xz
    
    _c11x2 = -1j*np.exp(1j/2*(_c1 + _c2)*t2)*(_d21 - _d24)*(
        (2*_c9y1*np.cos(a12*t2/2) + _c9xz*np.sin(a12*t2/2))*(
            np.cos(1/2*(_c1 - _c2)*t2) - 1j*np.cos(_bet)*np.sin(1/2*(_c1 - _c2)*t2)) + 
        (_c9xz*np.cos(a12*t2/2) - 2*_c9y1*np.sin(a12*t2/2))*np.sin(1/2*(_c1 - _c2)*t2)*np.sin(_bet))
    
    _c11y2 = 1j*_c11x2
    
    _c11zx = 2*np.exp(1j/2*(_c1 + _c2)*t2)*(
        -((_d16*(_d21 - _d24) - (_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.cos(1/2*(_c1 - _c2)*t2)*(
            _c9xz*np.cos(a12*t2/2) - 2*_c9y1*np.sin(a12*t2/2))) + 
        1j*(_d16*(_d21 - _d24) + (_d13 - 1j*_d15)*(_d23 + 1j*_d25))*np.cos(_bet)*(
            _c9xz*np.cos(a12*t2/2) - 2*_c9y1*np.sin(a12*t2/2))*np.sin(1/2*(_c1 - _c2)*t2) + 
        (_d16*(_d21 - _d24) - (_d13 - 1j*_d15)*(_d23 + 1j*_d25))*(
            2*_c9y1*np.cos(a12*t2/2) + _c9xz*np.sin(a12*t2/2))*np.sin(1/2*(_c1 - _c2)*t2)*np.sin(_bet))
    
    _c11zy = 1j*_c11zx
    
    return _c11x1, _c11y1, _c11x2, _c11y2, _c11xz, _c11yz, _c11zx, _c11zy

def sig(om1, om2, a12, tp, t1, t2, b1, tau, path = "1"):
    
    b12 = - a12/2 
    _c1, _c2, _bet = Cs(om1, om2, b12)
    
    _d11, _d12, _d13, _d14, _d15, _d16 = spinrot(om1, b1, tau)
    _d21, _d22, _d23, _d24, _d25, _d26 = spinrot(om2, b1, tau)
    
    if path == "1":
        _c11x1, _c11y1, _c11x2, _c11y2, _c11xz, _c11yz, _c11zx, _c11zy = c11coefY(om1, om2, a12, tp, t1, t2, b1, tau)
    elif path == "2":
        _c11x1, _c11y1, _c11x2, _c11y2, _c11xz, _c11yz, _c11zx, _c11zy = c11coefX(om1, om2, a12, tp, t1, t2, b1, tau)
    elif path == "3":
        _c11x1, _c11y1, _c11x2, _c11y2, _c11xz, _c11yz, _c11zx, _c11zy = c11coefZ(om1, om2, a12, tp, t1, t2, b1, tau)
    
    _sigy = 1/4*(np.sin(a12*t2/2)*(
        np.cos(_c1*t2)*(_c11xz + _c11zx + (_c11xz - _c11zx)*np.cos(_bet) - 2*(_c11x1 + _c11x2)*np.sin(_bet)) + 
        np.cos(_c2*t2)*(_c11xz + _c11zx + (-_c11xz + _c11zx)*np.cos(_bet) + 2*(_c11x1 + _c11x2)*np.sin(_bet)) - 
        np.sin(_c1*t2)*(_c11yz + _c11zy + (_c11yz - _c11zy)*np.cos(_bet) - 2*(_c11y1 + _c11y2)*np.sin(_bet)) - 
        np.sin(_c2*t2)*(_c11yz + _c11zy + (-_c11yz + _c11zy)*np.cos(_bet) + 2*(_c11y1 + _c11y2)*np.sin(_bet))) + 
        np.cos(a12*t2/2)*(
            np.sin(_c1*t2)*(2*(_c11x1 + _c11x2 + (_c11x1 - _c11x2)*np.cos(_bet)) - (_c11xz + _c11zx)*np.sin(_bet)) + 
            np.sin(_c2*t2)*(2*(_c11x1 + _c11x2) + 2*(-_c11x1 + _c11x2)*np.cos(_bet) + (_c11xz + _c11zx)*np.sin(_bet)) + 
            np.cos(_c1*t2)*(2*(_c11y1 + _c11y2 + (_c11y1 - _c11y2)*np.cos(_bet)) - (_c11yz + _c11zy)*np.sin(_bet)) + 
            np.cos(_c2*t2)*(2*(_c11y1 + _c11y2) + 2*(-_c11y1 + _c11y2)*np.cos(_bet) + (_c11yz + _c11zy)*np.sin(_bet))))
    
    return _sigy