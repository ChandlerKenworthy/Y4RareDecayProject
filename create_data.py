import numpy as np
import pandas as pd
import uproot as up
import os
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

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
simulation_path = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MCNorm/2016MD/halfSampleFeb22/job246-CombDVntuple-MCNorm-15144059-S28r2Restrip-firstHalf-2016MD-pKmumu-PF__PreselectedV1.root"
actual_path = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/realDataNorm/2016MD/halfSampleFeb22/job228-CombDVntuple-collision-firstHalf-2016MD-pKmumu-PF__PreselectedV1.root"
# Normalisation mode paths

#simulation_path = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MC/2016MD/fullSampleOct2021/job207-CombDVntuple-15314000-MC2016MD_Full-pKmue-MC.root"
#actual_path = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/realData/2016MD/halfSampleOct2021/blindedTriggeredL1520Selec-collision-firstHalf2016MD-pKmue_Fullv9.root"
# Signal mode paths

decay_tree_name = ':DTT1520mm/DecayTree'
version = '0.0.2'
preselection = False
preselection_path = 'preselection.txt'
random_seed = 0
equalise_event_numbers = False
restrict_mass_sidebands = [[4500, 5200], [5800, 6500]]
train, val, test = 0.6, 0.2, 0.2

# Open the file with all the user requested features, some may be expressions
user_features = pd.read_csv('request.txt', index_col=None, sep=',')
user_features['FeatureName'].fillna(user_features['Features'], inplace=True)

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
request_features = list(dict.fromkeys(single_features + multi_features_flattened + ['Lb_M']))

# Get the features for the simulated data
sim_request_features = list(dict.fromkeys(request_features + ['Lb_BKGCAT']))
if preselection:
    # Make the pre-selection expression maleable to eval() and get features needed
    # to perform the pre-selection
    sim_eval_ps, sim_ps_fts = add_df_prefix(sim_ps, 'sdf')
    sim_request_features = list(dict.fromkeys(sim_request_features + sim_ps_fts))

with up.open(simulation_path + decay_tree_name) as f:
    # Call in all of these data
    sdf = f.arrays(["eventNumber"] + sim_request_features, library='pd')
    sdf.set_index("eventNumber", inplace=True)
    # Randomly shuffle the rows around
    sdf = sdf.sample(frac=1, random_state=random_seed)
    # Remove duplicate events
    sdf = sdf[~sdf.index.duplicated(keep='first')]

if preselection:
    print(f"Evaluating pre-selection for simulated data\nCurrently there are {len(sdf)} events")
    # Evaluate the pre-selection to remove events
    sdf = sdf[eval(sim_eval_ps)]
    print(f"Preselection applied without error\nNow there are {len(sdf)} events\n")
    
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

if restrict_mass_sidebands != None:
    rdf = rdf[np.logical_or(rdf['Lb_M'] < restrict_mass_sidebands[0][1], rdf['Lb_M'] > restrict_mass_sidebands[1][0])]
    rdf = rdf[np.logical_or(rdf['Lb_M'] > restrict_mass_sidebands[0][0], rdf['Lb_M'] < restrict_mass_sidebands[1][1])]

if preselection:
    print(f"Evaluating pre-selection for real data\nCurrently there are {len(rdf)} events")
    # Evaluate the pre-selection to remove events
    rdf = rdf[eval(real_eval_ps)]     
    print(f"Preselection applied without error\nNow there are {len(rdf)} events\n")
rdf['IsSimulated'] = False
rdf['category'] = 0

# Remove the extra column that is in the simulated dataframe
sdf.drop('Lb_BKGCAT', axis=1, inplace=True)

# Join the dataframes together
# TODO: Sort of features now then concat then dropna

df = pd.concat([sdf, rdf], ignore_index=True, sort=False, axis=0)

# Remove events with missing values
#df.dropna(inplace=True, axis=0)
# Above causes issues for real data not sure why...column mismatch?

# Randomly shuffle the new dataframe
df = df.sample(frac=1, random_state=random_seed)

# Only keep the features we needed to fulfill the users request
df = df[list(dict.fromkeys(request_features + ['category', 'Lb_M', 'IsSimulated']))]

# Evaluate the custom expressions
for index, row in user_features.iterrows():
    # Does this row contain a custom feature
    if row['IsCustom']:
        df[row['FeatureName']] = eval(add_df_prefix(row['Features'], 'df')[0])
    else:
        pass
    
# Now we have made the custom features drop anything else not needed
df = df[list(dict.fromkeys(user_features['FeatureName'].to_list() + ['Lb_M', 'IsSimulated', 'category']))]

# Apply an event ratio restriction
if equalise_event_numbers:
    nbg = df['category'].value_counts()[0]
    nsg = df['category'].value_counts()[1]
    n_to_remove = np.abs(nsg - nbg)
    
    if nsg > nbg:
        frac = n_to_remove / nsg
        remove_idx = df.query('category == 1').sample(frac=frac, random_state=random_seed).index.to_list()
        df.drop(remove_idx, axis=0, inplace=True)
    elif nsg < nbg:
        frac = n_to_remove / nbg
        remove_idx = df.query('category == 0').sample(frac=frac, random_state=random_seed).index.to_list()
        df.drop(remove_idx, axis=0, inplace=True)
    else:
        # They are already equal
        pass
    
# Reset the index as well
df.reset_index(drop=True, inplace=True)

# Make a directory for all these data
if not os.path.isdir(f'data_files/{version}'):
    os.mkdir(f'data_files/{version}')
df.to_csv(f'data_files/{version}/all.csv')

# Do the train/val/test split
train_and_test = df[:int(np.floor(len(df)*(train + val)))]
val = df[int(np.floor(len(df)*(train + val))):]

y = train_and_test['category']
# The binary labels (classification problem) are assigned as the y column
x = train_and_test.drop(['category'], axis=1)
# Remove the category column from the training inputs

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=(train/(train+test)), test_size=1-(train/(train+test)), random_state=random_seed)
# Split the training and test data accordingly so that the input fractions are maintained relative
X_val, y_val = val.drop(['category'], axis=1), val['category']

# Do the normalisation using sklearns transformer
cols_to_transform = X_train.columns.to_list()
cols_to_transform = [i for i in cols_to_transform if i not in ['Lb_M', 'IsSimulated', 'category']]

ct = ColumnTransformer([
        ('normaliser', StandardScaler(), cols_to_transform)
    ], remainder='passthrough')

ct.fit(X_train)
X_trains = ct.transform(X_train)
X_vals = ct.transform(X_val)
X_tests = ct.transform(X_test)

X_train = pd.DataFrame(X_trains, index=X_train.index, columns=X_train.columns).fillna(0)
X_val = pd.DataFrame(X_vals, index=X_val.index, columns=X_val.columns).fillna(0)
X_test = pd.DataFrame(X_tests, index=X_test.index, columns=X_test.columns).fillna(0)
# In case you divide by a zero std. dev. fill the NaNs with zeros

# Output all these files
train = X_train.copy()
train['category'] = y_train
val = X_val.copy()
val['category'] = y_val
test = X_test.copy()
test['category'] = y_test

train.to_csv(f'data_files/{version}/train.csv')
val.to_csv(f'data_files/{version}/val.csv')
test.to_csv(f'data_files/{version}/test.csv')

# Output the metadata file as well
print(f"Output to CSV: data_files/{version}/all.csv\nGenerating metadata file...")

f = open(f'outputs/csv_metadata/{version}.txt', 'w')
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
today = date.today()
d1 = today.strftime("%d/%m/%Y")
f.write(f"Version: {version}\nCompile Date: {d1}\nCompile Time: {current_time}\nPreselection Applied: {preselection}\n")
f.write(f"Equal Event Ratio: {equalise_event_numbers}\nFeatures Included:\n")

for i, p in enumerate(user_features['Features'].to_list()):
    string = f"{i}. {p}\n"
    f.write(string)
f.close()

print(f"INFO: Metadata file generated at outputs/csv_metadata/{version}.txt'")