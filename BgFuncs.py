import numpy as np

def exponential(x, A, k, B, C):
    """
    A simple exponential function with a shift of B and amplitude A
    """
    return (A*np.exp(-((k*x)+B)))+C

def gaussian(x, mu, sigma):
    """
    A standard Gaussian distribution (not normalised)
    """
    return (1/(np.sqrt(2*np.pi)*sigma))*np.exp((x-mu)**2/(2*(sigma**2)))

def power(x, A, B, C, D):
    """
    A simple power law function of the form A(x-B)^C
    """
    return A*(x-B)**(C) + D

def logarithmic(x, A, B, C):
    """
    A logarithm-like function of the form f(x)=-Bln((x-A)) + C
    """
    return -(B * np.log(x-A)) + C