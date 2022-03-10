import numpy as np
import pandas as pd
import uproot as up

# Utility functions
def add_df_prefix(boolean_mask, prefix):
    preselection_features = []
    begin, end = False, False
    begin_position, end_position = 0, 0
    updated_mask = ''
    
    for i, char in enumerate(boolean_mask):
        updated_mask += char
        if char == ' ' and not begin:
            updated_mask = updated_mask[:-1]
            updated_mask += f"{prefix}['"
            begin_position = i
            begin = True 
        elif char == ' ' and begin:
            updated_mask = updated_mask[:-1]
            updated_mask += "']"
            end_position = i
            end = True
        if begin and end:
            preselection_features.append(boolean_mask[begin_position + 1:end_position])
            begin, end = False, False
    preselection_features = list(dict.fromkeys(preselection_features))
    if len(preselection_features) == 0:
        preselection_features = boolean_mask
    if len(preselection_features) == 1:
        preselection_features = preselection_features[0]
    return updated_mask, preselection_features

# Define the simulation and actual data path names and decay tree name
simulation_path = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MC/2016MD/fullSampleOct2021/job207-CombDVntuple-15314000-MC2016MD_Full-pKmue-MC.root"
actual_path = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/realData/2016MD/halfSampleOct2021/blindedTriggeredL1520Selec-collision-firstHalf2016MD-pKmue_Fullv9.root"
decay_tree_name = ':DTT1520me/DecayTree'
version = '7.0.0'
preselection = True
preselection_path = 'preselection.txt'
random_seed = 0

# Open the file with all the user requested features, some may be expressions
user_features = pd.read_csv('request.txt', index_col=None, header=None, sep=',')
user_features.columns = ['Features']

# Find all the features that are custom expressions
user_features['IsCustom'] = [' ' in f for f in user_features['Features']]

# Find all the features we need to request as custom expressions may contain multiple
user_features['Request'] = [add_df_prefix(f, 'df')[1] for f in user_features['Features']]

# If applying preselection get the preselections from the text file
if preselection: 
    preselections = pd.read_csv(preselection_path, index_col=None, header=None)
    preselections.columns = ['Type', 'Expression']
    preselections.set_index('Type', inplace=True)
    sim_ps = preselections.loc['simulation']['Expression']
    real_ps = preselections.loc['actual']['Expression']
    
# Define what features we will be requesting
request_features = user_features['Request'].to_list()
# Unpack any sub-lists inside this list
single_features = [f for f in request_features if type(f) is not list]
multi_features = [f for f in request_features if type(f) is list]
multi_features_flattened = [item for sublist in multi_features for item in sublist]
# Ensure no features are requested multiple times (remove repeats)
request_features = list(dict.fromkeys(single_features + multi_features_flattened))

# Get the features for the simulated data
sim_request_features = request_features
if preselection:
    # Make the pre-selection expression maleable to eval() and get features needed
    # to perform the pre-selection
    sim_eval_ps, sim_ps_fts = add_df_prefix(sim_ps, 'sdf')
    sim_request_features = list(dict.fromkeys(sim_request_features + sim_ps_fts + ['Lb_BKGCAT']))
    
with up.open(simulation_path + decay_tree_name) as f:
    # Call in all of these data
    sdf = f.arrays(["eventNumber"] + sim_request_features, library='pd')
    sdf.set_index("eventNumber", inplace=True)
    # Randomly shuffle the rows around
    sdf = sdf.sample(frac=1, random_state=random_seed)
    # Remove duplicate events
    sdf = sdf[~sdf.index.duplicated(keep='first')]

if preselection:
    # Evaluate the pre-selection to remove events
    sdf = sdf[eval(sim_eval_ps)]
    
# Add a category column and a column to track simulated events
sdf['category'] = np.where(sdf['Lb_BKGCAT'].isin([10, 50]), 1, 0)
sdf['IsSimulated'] = True

# Get the features for the real data
real_request_features = request_features
if preselection:
    # Make the pre-selection expression maleable to eval() and get features needed
    # to perform the pre-selection
    real_eval_ps, real_ps_fts = add_df_prefix(real_ps, 'rdf')
    real_request_features = list(dict.fromkeys(real_request_features + real_ps_fts))
    
with up.open(actual_path + decay_tree_name) as f:
    # Call in all of these data
    rdf = f.arrays(["eventNumber"] + real_request_features, library='pd')
    rdf.set_index("eventNumber", inplace=True)
    # Randomly shuffle the rows around
    rdf = rdf.sample(frac=1, random_state=random_seed)
    # Remove duplicate events
    rdf = rdf[~rdf.index.duplicated(keep='first')]

if preselection:
    # Evaluate the pre-selection to remove events
    rdf = rdf[eval(real_eval_ps)]     
rdf['IsSimulation'] = False

print(sdf.head())
