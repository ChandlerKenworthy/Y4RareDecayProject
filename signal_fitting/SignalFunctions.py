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

def double_crystal_ball(M, N, mu, s, a_low, a_high, n_low, n_high):
    """
    The double crystal ball function as defined in slide 16 of
    https://www.physik.uzh.ch/dam/jcr:73d6fc85-f994-4efe-b483-31e385a1609f/Crivelli_2016.pdf
    https://indico.in2p3.fr/event/11794/contributions/6962/attachments/5667/7069/Grevtsov_JJC15_v1.4.pdf
    """
    
    return None





