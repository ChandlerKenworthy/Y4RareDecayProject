import numpy as np
import matplotlib.pyplot as plt
from utilities import Data, Plots, Consts
import BgFuncs as bf

from scipy.optimize import curve_fit
# Useful imports and utlities

BINS = 200
NORMALISED = False # TODO: Add implementation to change this
ROI = [4500, 6500]
fits = {'gaussian': [bf.gaussian, [1, 1]], 'exponential': [bf.exponential, [1, 1, 1, 1]]}
# Setting all constants required

RFNAME, RSUFFIX = Consts().get_real_tuple()
rd = Data(RFNAME, RSUFFIX)
# Setup the real Data object

df = rd.fetch_features(['Lb_M'])
print('UPDATE: Fetched the invariant masses')
masses = df['Lb_M'].to_numpy()
# Get the invariant mass, as calculated by ROOT

bin_space = np.linspace(int(np.floor(np.min(masses))), int(np.ceil(np.max(masses))), BINS+1)
# Create a uniformly distributed array of bin edge positions between the rounded min/max of masses

bin_centres = [(bin_space[i] + bin_space[i+1])/2 for i in range(len(bin_space) - 1)]
bin_width = np.mean([(bin_space[i+1] - bin_space[i]) for i in range(len(bin_space)-1)])
print('UPDATE: Finished calculating the bin-space, bin width and bin centroids')
# Calculate the centres of each bin and the width of each bin
    
frequencies, _ = np.histogram(masses, bin_space, density=NORMALISED)
# Use numpy to bin the data using the bin space we calculated, this is not normalised
errors = np.sqrt(frequencies) 
# Error on the bin is proportional to the sqrt of the counts
print('UPDATE: Data binned and errors on bins calculated')

data = np.array([frequencies, errors, bin_centres]).T
# Combine the data into a big array so applying mass cuts is easier

data = data[data[:,2] >= ROI[0], :]
data = data[data[:,2] <= ROI[1], :]
# Remove all bins where the centroid is outside the region of interest

data = data[~np.any(data == 0, axis=1)]
# Remove all the bins where the bin is empty, typically the blinded region

for function_name, function_and_p0 in fits.items():
    popt, pcov = curve_fit(function_and_p0[0], data[:,2], data[:,0], p0=function_and_p0[1], sigma=data[:,1])
    # Use a chi-square fitting routine first, this is the same as a least squares but with errors
    fits[function_name] = [function_and_p0[0], popt]
    # Update the initial guess parameters to the optimised paramters

func_plot_range = np.linspace(int(np.floor(np.min(data[:,2]))), int(np.ceil(np.max(data[:,2]))), 500)
# Define a range to plot the function over

fig, ax = plt.subplots(1, 1, figsize=(11, 9))
ax.bar(data[:,2], data[:,0], yerr=data[:,1], edgecolor='k', width=bin_width, label='Real Data')
for function, params in fits.items():
    ax.plot(func_plot_range, params[0](func_plot_range, *params[1]), label=f'{function.capitalize()}')
ax.set_ylabel('Frequency')
ax.set_xlabel('Mass (MeV)')
plt.legend(frameon=False)
plt.show()
plt.savefig('/home/user211/project/images/optimised_fits.png', dpi=700)