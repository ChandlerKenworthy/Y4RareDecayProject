import numpy as np
import pandas as pd
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
from utilities import Data, Consts

# TODO: Merge these features into the Data object of utilities

BINS = 100 # Number of bins to bin the data into
NORMALISED = True # TODO: Add implementation to change this
ROI = [4500, 6500] # The region of interest in mass (MeV)
TYPE = 'real'
FNAME = f'{BINS}bins_{NORMALISED}'

# Setting all constants required

if TYPE == 'real':
    fname, suffix = Consts().get_real_tuple()
else:
    fname, suffix = Consts().get_simulated_tuple()
    
d = Data(fname, suffix)
# Setup the Data object

df = d.fetch_features(['Lb_M'])
# Get the invariant masses for each event
print('UPDATE: Fetched the invariant masses')
masses = df['Lb_M'].to_numpy()
# Get the invariant mass, as calculated by ROOT

bin_space = np.linspace(int(np.floor(np.min(masses))), int(np.ceil(np.max(masses))), BINS+1)
# Create a uniformly distributed array of bin edge positions between the rounded min/max of masses

bin_centres = [(bin_space[i] + bin_space[i+1])/2 for i in range(len(bin_space) - 1)]
bin_width = np.mean([(bin_space[i+1] - bin_space[i]) for i in range(len(bin_space)-1)])
print('UPDATE: Finished calculating the bin-space, bin width and bin centroids')
# Calculate the centres of each bin and the width of each bin
    
frequencies, _ = np.histogram(masses, bin_space, density=False)
errors = np.sqrt(frequencies) 
# Error on the bin is proportional to the sqrt of the counts

if NORMALISED:
    frequencies_normed, _ = np.histogram(masses, bin_space, density=True)
    scaling_factor = np.mean(frequencies_normed[0]/frequencies[0])
    frequencies, errors = frequencies_normed, (errors * scaling_factor)
# Use numpy to bin the data using the bin space we calculated
print('UPDATE: Data binned and errors on bins calculated')

data = np.array([frequencies, errors, bin_centres]).T
# Combine the data into a big array so applying mass cuts is easier

data = data[data[:,2] >= ROI[0], :]
data = data[data[:,2] <= ROI[1], :]
# Remove all bins where the centroid is outside the region of interest

data = data[~np.any(data == 0, axis=1)]
# Remove all the bins where the bin is empty, typically the blinded region

pd.DataFrame(data, columns=['Frequency', 'Error', 'Mass']).to_csv(f'/home/user211/project/data/{FNAME}.txt', index=False)
print(f'COMPLETE: Data with {BINS} bins was output to "/home/user211/project/data/{FNAME}.txt"')