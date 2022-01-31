""""
Author: Chandler Kenworthy
Description: A variety of different functions to try and model the Monte-Carlo simulated signal
"""

import numpy as np
from scipy.special import wofz

def gaussian(m, A, mu, sigma):
    """ Standard normal distribution """
    return A*(1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((m-mu)**2)/(2*sigma**2))

def scaled_lorentzian(m, A, m_0, gamma):
    return (A/np.pi)*((0.5*gamma)/((m-m_0)**2 + (0.5*gamma)**2))

def breit_wigner(m, M, w, A):
    """
    The relativistic breit wigner distribution 
    """
    gamma2 = np.sqrt((M**2)*((M**2) + (w**2))) 
    # Array with length len(m)
    k = (2*np.sqrt(2)*M*w*gamma2)/(np.pi*np.sqrt((M**2)*gamma2)) 
    # Array with length len(m)
    return A*k/((m**2 - M**2)**2 + (M*w)**2)

def voigt(m, alpha, gamma, shift):
    """
    The Voigt profile
    """
    sigma = alpha/np.sqrt(2*np.log(2))
    return np.real(wofz(((m-shift) + 1j*gamma)/sigma/np.sqrt(2)))/sigma/np.sqrt(2*np.pi)

def cball(m, A, b, u, loc):
    """
    The scaled crystal ball function
    """
    from scipy.stats import crystalball
    return A*crystalball.pdf(m, b, u, loc, 1)

def double_crystal_ball(x, A, mu, s, a_l, a_R, n_l, n_h):
    t = (x-mu)/s
    low = np.where(t <= a_l)[0]
    mid = np.where(np.logical_and(t > a_l, t < a_R))[0]
    high = np.where(t >= a_R)[0]
    a_l, a_R = np.abs(a_l), np.abs(a_R)
    l = ((n_l/a_l)**n_l)*np.exp(-0.5*(a_l**2))*(((n_l/a_l) - a_l - t[low])**(-n_l))
    m = np.exp(-0.5*(t[mid]**2))
    h = ((n_h/a_R)**n_h)*np.exp(-0.5*(a_R**2))
    ###
    x = a_R - t[high]
    print(n_h)
    h *= ((n_h/a_R) - [i**(-n_h) for i in x])
    ###
    
    return A*np.concatenate((l, m ,h))





