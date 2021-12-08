import numpy as np

def exponential(m, a, b, c):
    """ a*exp[-(bm+c)] """
    return a * np.exp(-(b*m + c))

def expsquare(m, a, b, c):
    """ defm """
    return a*np.exp(-(m-b)**2)+c

def gaussian(m, mu, sigma):
    """ Standard normal distribution """
    return (1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((m-mu)**2)/(2*sigma**2))

def linear(m, a, b):
    """ am + b """
    return a*m + b

def quadratic(m, a, b, c):
    """ am^2 + bm + c """
    return a*(m**2) + (b*m) + c

def cubic(m, a, b, c, d):
    """ Standard cubic """
    return a*(m**3) + b*(m**2) + (c*m) + d
    
def quartic(m, a, b, c, d, f):
    return a*(m**4) + b*(m**3) + c*(m**2) + (d*m) + f
    