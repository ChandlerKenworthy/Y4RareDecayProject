""""
Author: Chandler Kenworthy
Description: A variety of different functions to try and model the Monte-Carlo simulated signal
"""

import numpy as np

def gaussian(m, A, mu, sigma):
    """ Standard normal distribution """
    return A*(1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((m-mu)**2)/(2*sigma**2))

def scaled_lorentzian(m, A, m_0, gamma):
    return (A/np.pi)*((0.5*gamma)/((m-m_0)**2 + (0.5*gamma)**2))

# Breit-Wigner, Crystal Ball, Double crystal ball, Lorentzian