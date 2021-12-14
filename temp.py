import utilities as ut
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = ut.Data(*ut.Consts().get_real_tuple())
df = data.fetch_features(['Lb_M'])
masses = df['Lb_M'].to_numpy()
indices = np.where(np.logical_and(masses > 5199, masses < 5801))[0]
masses = masses[indices]
masses = np.sort(masses)
for mass in masses:
    print(mass)
print(len(masses))
