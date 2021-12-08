""""
Author: Chandler Kenworthy
Description: A variety of different functions to try and model the Monte-Carlo simulated signal
"""

import numpy as np

def gaussian(m, mu, sigma):
    """ Standard normal distribution """
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((m-mu)**2)/(2*sigma**2))

# Breit-Wigner, Crystal Ball, Double crystal ball, Lorentzian