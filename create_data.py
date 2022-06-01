import numpy as np
import pandas as pd
import uproot as up
import os
from datetime import datetime, date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer


def add_df_prefix(boolean_mask, prefix):
    """
    For some set of evaluable conditions with feature names present
    return a string with the appropriate Pythonic dataframe wrapping around it
    so it can be used inside of an eval expression.
    """
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


def create_csv(kwargs):
    """
    Generate a CSV using the arguments passed separate for the training,
    validation and test data samples. Also output an entire non-normalised
    version
    """
    
    special_masses = ['Lb_M01_Subst1_K2mu', 'Lb_M01_Subst1_K2pi', 'Lb_M01_Subst1_K2p', 'Lb_M01_Subst0_p2mu',
                      'Lb_M01_Subst0_p2K', 'Lb_M01_Subst0_p2pi', 'Lb_M01_Subst01_Kp~2pK', 'Lb_M01_Subst01_Kp~2piK',
                      'Lb_M01_Subst01_Kp~2ppi', 'Lb_M01_Subst01_Kp~2pipi', 'Lb_M0123_Subst0_p2pi', 'Lb_M0123_Subst01_Kp~2piK',
                      'Lb_M01']

    # Open the file with all the user requested features, some may be expressions
    user_features = pd.read_csv(kwargs['request'], index_col=None, sep=',')
    user_features['FeatureName'].fillna(user_features['Features'], inplace=True)
    
    # Find all the features that are custom expressions
    user_features['IsCustom'] = [' ' in f for f in user_features['Features']]
    
    # Find all the features we need to request as custom expressions may contain multiple
    user_features['Request'] = [add_df_prefix(f, 'df')[1] for f in user_features['Features']]
    
    mu = 105.6583745 # MeV
    me = 0.5109989461 # MeV
    
    if kwargs['DileptonQ2']:
        expression = "np.sqrt((me**2+mu**2)+2*(np.sqrt((mu**2)+np.power( L1_P ,2))*np.sqrt((me**2)+np.power( L2_P ,2))-(( L1_PX * L2_PX )+( L1_PY * L2_PY )+( L1_PZ * L2_PZ ))))"
        if kwargs['isNormalisation']:
            # Di muon final state
            expression = "np.sqrt((2*mu**2)+2*(np.sqrt((mu**2)+np.power( L1_P ,2))*np.sqrt((mu**2)+np.power( L2_P ,2))-(( L1_PX * L2_PX )+( L1_PY * L2_PY )+( L1_PZ * L2_PZ ))))"
        add_row = {'Features': expression,
                   'FeatureName': 'QSQR',
                   'IsCustom': True,
                   'Request': add_df_prefix(expression, 'df')[1]}
        user_features = user_features.append(add_row, ignore_index=True)
    
    # If applying preselection get the preselections from the text file
    if kwargs['preselect']: 
        preselections = pd.read_csv(kwargs['preselect_path'], index_col=None, header=None)
        preselections.columns = ['Type', 'Expression']
        preselections.set_index('Type', inplace=True)
        sim_ps = preselections.loc['simulation']['Expression']
        real_ps = preselections.loc['actual']['Expression']
        
    # Define what features the user is requesting
    request_features = user_features['Request'].to_list()
    
    # Unpack any sub-lists inside this list
    single_features = [f for f in request_features if type(f) is not list]
    multi_features = [f for f in request_features if type(f) is list]
    multi_features_flattened = [item for sublist in multi_features for item in sublist]
    
    # Ensure no features are requested multiple times (remove repeats)
    request_features = list(dict.fromkeys(single_features + multi_features_flattened + ['Lb_M']))
    
    # Get the features for the simulated data
    sim_request_features = list(dict.fromkeys(request_features + ['Lb_BKGCAT']))
    if kwargs['preselect']:
        # Make the pre-selection expression maleable to eval() and get features needed
        # to perform the pre-selection
        sim_eval_ps, sim_ps_fts = add_df_prefix(sim_ps, 'sdf')
        sim_request_features = list(dict.fromkeys(sim_request_features + sim_ps_fts))
    
    sdf, rdf = pd.DataFrame(), pd.DataFrame()
    
    print(f"INFO: Reading in data from simulated tree")
    with up.open(kwargs['sim_path'] + kwargs['decay_tree_name']) as f:
        # Call in all of these data
        sdf = f.arrays(["eventNumber"] + sim_request_features, library='pd')
        sdf.set_index("eventNumber", inplace=True)
        # Randomly shuffle the rows around
        sdf = sdf.sample(frac=1, random_state=kwargs['random_seed'])
        # Remove duplicate events
        sdf = sdf[~sdf.index.duplicated(keep='first')]
        
    if kwargs['preselect']:
        print(f"INFO: Evaluating pre-selection for simulated data\nINFO: Currently there are {len(sdf)} events")
        # Evaluate the pre-selection to remove events
        sdf = sdf[eval(sim_eval_ps)]
        print(f"INFO: Preselection applied without error\nINFO: Currently there are {len(sdf)} events\n")
        
    # Add a category column and a column to track simulated events
    sdf['category'] = np.where(sdf['Lb_BKGCAT'].isin([10, 50]), 1, 0)
    sdf['IsSimulated'] = True
    print("INFO: Simulated data manipulation complete")
    
    # Get the features for the real data
    real_request_features = request_features
    if kwargs['preselect']:
        # Make the pre-selection expression maleable to eval() and get features needed
        # to perform the pre-selection
        real_eval_ps, real_ps_fts = add_df_prefix(real_ps, 'rdf')
        real_request_features = list(dict.fromkeys(real_request_features + real_ps_fts))
        
    with up.open(kwargs['actual_path'] + kwargs['decay_tree_name']) as f:
        # Call in all of these data
        rdf = f.arrays(["eventNumber"] + real_request_features, library='pd')
        # Set the event number as the indexer
        rdf.set_index("eventNumber", inplace=True)
        # Randomly shuffle the rows around
        rdf = rdf.sample(frac=1, random_state=kwargs['random_seed'])
        # Remove duplicate events
        rdf = rdf[~rdf.index.duplicated(keep='first')]
        
    if kwargs['preselect']:
        print(f"INFO: Evaluating pre-selection for real data\nINFO: Currently there are {len(rdf)} events")
        # Evaluate the pre-selection to remove events
        rdf = rdf[eval(real_eval_ps)]     
        print(f"INFO: Preselection applied without error\nINFO: Currently there are {len(rdf)} events\n")
        
    # Add a category column and a column to track simulated events
    rdf['IsSimulated'] = False
    
    if kwargs['isNormalisation']:
        # For normalisation mode this is only true if sidebands are restricted!
        rdf['category'] = np.where(rdf['Lb_M'].between(5200, 5800), 2, 0)
        # Assign a category of '2' to events that are in the full normalisation
        # sample and probably background but within the signal region
    else:
        rdf['category'] = 0
    
        # Restrict the mass sidebands
    if type(kwargs['restrict_mass']) is tuple:
        print(f"INFO: Restricting mass sidebands\nINFO: Currently there are {len(rdf)} events")
        print(f"INFO: There are {np.count_nonzero(rdf['Lb_M'] < 4500)} events less than the lower mass threshold")
        print(f"INFO: There are {np.count_nonzero(rdf['Lb_M'] > 6500)} events above the higher mass threshold")
        print(f"INFO: There are {np.count_nonzero(np.logical_and((rdf['Lb_M'] > 5200).to_numpy(), (rdf['Lb_M'] < 5800)).to_numpy())} events between threshold values")
        rdf = rdf[np.logical_or(rdf['Lb_M'].between(*kwargs['restrict_mass'][1]).to_numpy(), rdf['Lb_M'].between(*kwargs['restrict_mass'][0]).to_numpy())].copy()
        # this is throwing away signal events you moron
        print(f"INFO: Mass restriction complete\nINFO: Currently there are {len(rdf)} events")
    
    # Remove the extra column that is in the simulated dataframe
    sdf.drop('Lb_BKGCAT', axis=1, inplace=True)
        
    # Evaluate the custom expressions
    for index, row in user_features.iterrows():
        # Does this row contain a custom feature
        if row['IsCustom']:
            sdf[row['FeatureName']] = eval(add_df_prefix(row['Features'], 'sdf')[0])
            rdf[row['FeatureName']] = eval(add_df_prefix(row['Features'], 'rdf')[0])
    
    # Now each of the two dataframes has all the necessary features made in the dataframe
    # we now want to restrict features to those requested plus a few useful others
    # they could have different features due to preselection
    restrict_features = user_features['FeatureName'].to_list() + ['Lb_M', 'IsSimulated', 'category']
    sdf, rdf = sdf[restrict_features], rdf[restrict_features]
    if kwargs['isNormalisation']:
        sdf['category'] == np.where(np.logical_and(sdf['category']==1, sdf['QSQR'].between(3000,3178)),1,0)
    
    # Join the dataframes together
    df = pd.concat([sdf, rdf], ignore_index=True, sort=False, axis=0)
    
    # Remove events with missing values
    if kwargs['dropnan']:
        print(f'INFO: {len(df)} events in combined data\nINFO: Removing events with NaN values')
        print(f'INFO: Columns with NaN present:\n{df.isna().sum()}\n')
        df.dropna(inplace=True, axis=0)
        print(f'INFO: NaN events removed\nINFO: {len(df)} events retained')
        
    # Randomly shuffle the new dataframe
    df = df.sample(frac=1, random_state=kwargs['random_seed'])
        
    # Apply an event ratio restriction
    if kwargs['equalise_event_numbers']:
        nbg = df['category'].value_counts()[0]
        nsg = df['category'].value_counts()[1]
        print(f'INFO: Sample currently includes {nbg} background and {nsg} signal events')
    
        # Need to remove this many events
        n_to_remove = np.abs(nsg - nbg)
        
        if nsg > nbg:
            # Too much signal!
            frac = n_to_remove / nsg
            remove_idx = df.query('category == 1').sample(frac=frac, random_state=kwargs['random_seed']).index.to_list()
            df.drop(remove_idx, axis=0, inplace=True)
        elif nsg < nbg:
            # Too much background
            frac = n_to_remove / nbg
            remove_idx = df.query('category == 0').sample(frac=frac, random_state=kwargs['random_seed']).index.to_list()
            df.drop(remove_idx, axis=0, inplace=True)
        else:
            # They are already equal
            pass
        print(f'INFO: Sample now includes {df["category"].value_counts()[0]} background and {df["category"].value_counts()[1]} signal events')
    
    # Reset the index as well
    df.reset_index(drop=True, inplace=True)
    
    # Make a directory for all these data
    if not os.path.isdir(f'data_files/{kwargs["ver"]}'):
        os.mkdir(f'data_files/{kwargs["ver"]}')
    df.to_csv(f'data_files/{kwargs["ver"]}/all.csv')
    # Note that the 'all' data is not normalised!
    
    # Do the train/val/test split
    train_and_test_idx = int(np.floor(len(df)*(kwargs['train_fraction'] + kwargs['test_fraction'])))
    train_and_test = df[:train_and_test_idx]
    validation = df[train_and_test_idx:]

    y = train_and_test['category']
    # The binary labels (classification problem) are assigned as the y column
    x = train_and_test.drop(['category'], axis=1)
    # Remove the category column from the training inputs
    
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=(kwargs['train_fraction']/(kwargs['train_fraction']+kwargs['test_fraction'])), test_size=1-(kwargs['train_fraction']/(kwargs['train_fraction']+kwargs['test_fraction'])), random_state=kwargs['random_seed'])
    # Split the training and test data accordingly so that the input fractions are maintained relative
    X_val, y_val = validation.drop(['category'], axis=1), validation['category']
    
    # Do the normalisation using sklearns transformer
    cols_to_transform = X_train.columns.to_list()
    cols_to_transform = [i for i in cols_to_transform if i not in ['Lb_M', 'IsSimulated', 'category', 'QSQR', *special_masses]]
    # The columns to apply the transformer to 
    
    ct = ColumnTransformer([('normaliser', StandardScaler(), cols_to_transform)], remainder='passthrough')

    ct.fit(X_train)
    # Get the values for the normaliser from X_train
    
    X_trains = ct.transform(X_train)
    X_vals = ct.transform(X_val)
    X_tests = ct.transform(X_test)
    # Transform the original dataframes to a new numpy copy

    X_train = pd.DataFrame(X_trains, index=X_train.index, columns=X_train.columns).fillna(0)
    X_val = pd.DataFrame(X_vals, index=X_val.index, columns=X_val.columns).fillna(0)
    X_test = pd.DataFrame(X_tests, index=X_test.index, columns=X_test.columns).fillna(0)
    # In case you divide by a zero std. dev. fill the NaNs with zeros
    
    # Output all these files
    train_df = X_train.copy()
    train_df['category'] = y_train
    val_df = X_val.copy()
    val_df['category'] = y_val
    test_df = X_test.copy()
    test_df['category'] = y_test

    train_df.to_csv(f'data_files/{kwargs["ver"]}/train.csv')
    val_df.to_csv(f'data_files/{kwargs["ver"]}/val.csv')
    test_df.to_csv(f'data_files/{kwargs["ver"]}/test.csv')
    
    # Output the metadata file as well
    print(f'Output to CSV: data_files/{kwargs["ver"]}/all.csv\nGenerating metadata file...')

    f = open(f'outputs/csv_metadata/{kwargs["ver"]}.txt', 'w')
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")
    f.write(f"Signal Events: {df['category'].value_counts()[1]}\n")
    f.write(f"Background Events: {df['category'].value_counts()[0]}\n")
    f.write(f"Version: {kwargs['ver']}\nCompile Date: {d1}\nCompile Time: {current_time}\nPreselection Applied: {kwargs['preselect']}\nNaN Dropped: {kwargs['dropnan']}\n")
    f.write(f"Equal Event Ratio: {kwargs['equalise_event_numbers']}\nRestrict Masses: {kwargs['restrict_mass']}\nNormalisation Mode: {kwargs['isNormalisation']}\nFeatures Included:\n")

    for i, p in enumerate(user_features['Features'].to_list()):
        string = f"{i}. {p}\n"
        f.write(string)
    f.close()

    print(f"INFO: Metadata file generated at outputs/csv_metadata/{kwargs['ver']}.txt")