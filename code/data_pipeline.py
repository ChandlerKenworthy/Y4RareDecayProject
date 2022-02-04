from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
from utilities import Data

def normalise_df_features(df, cols=None, params=None):
    """
    Normalise features assuming a Gaussian distribution of said feature
    for the requested dataframe. 

    Parameters
    ----------
    (dataframe) : df
        The dataframe to apply the normalisation on

    (array, bool) : cols
        Either an array of columns over which to normalise or True to
        normalise over all columns in the provided dataframe

    (array, bool) : params
        Either False in which parameters for the normalisation are 
        calculated or a tuple of Series objects with means and
        standard deviations

    Returns
    -------
    (dataframe) : normalised_df
        A copy of the dataframe with the requested features normalised
    """

    df_copy = df.copy()
    # Do not edit the original dataframe
    if cols is None:
        # Apply normalisation to all columns
        if params is None:
            df_copy = df_copy - df_copy.mean()
            norm_df = df_copy / df_copy.std()
        else:
            norm_df = (df_copy - params[0])/params[1]
    else:
        # Only apply normalisation to a set of columns
        if params is None:
            df_copy[cols] = df_copy[cols].apply(lambda x : (x - x.mean())/x.std() , axis=0)
        else:
            df_copy[cols] = (df_copy[cols] - params[0])/params[1]
        norm_df = df_copy

    norm_df = norm_df.fillna(0)
    # If the std of a column was zero before this is now full of NaNs, set back to equal zero

    return norm_df

def prepare_data(features, train_frac=0.6, val_frac=0.2, test_frac=0.2, random_state=0):
    """
    Prepare the data for the neural networks by reading it in, normalising 
    appropriate features, checking for NaNs or infs and replacing accordingly.
    Also randomly shuffle and split the data into training, validation and
    testing samples.

    Can also provide a dataframe in features with all features and categories 
    pre-existing and this will simply carve up into sets for you. It must
    have a column called 'category' using 1 for sim. signal, 0 for real bg
    and 2 for siml background.
    """

    if (train_frac + val_frac + test_frac) != 1.0:
        raise Exception('The data split fractions must sum to 1. Consider changing your frac arguments')

    real = Data("../data/blindedTriggeredL1520Selec-collision-firstHalf2016MD-pKmue_Full.root", ":DTT1520me/DecayTree")
    siml = Data("../data/job185-CombDVntuple-15314000-MC2016MD_100F-pKmue-MC.root", ":DTT1520me/DecayTree")
    # Create data objects for the real and the simulated data
    if type(features) is pd.DataFrame:
        combined = features.copy()
        combined['category'] = np.where(combined['category'].isin([2, 0]), 0, 1)
    else:
        banned_features = ['Lb_M']
        features = [ft for ft in features if ft not in banned_features]
        # Remove any requested features with obvious information leakage

        rf = real.fetch_features(features)
        sf = siml.fetch_features(features + ['Lb_BKGCAT'])
        # Fetch the desired features from each data tuple, including BKGCAT in the simulated

        sf['category'] = np.where(sf['Lb_BKGCAT'].isin([10,50]), 1, 0)
        sf.drop('Lb_BKGCAT', axis=1, inplace=True)
        rf['category'] = 0
        # Add a category column correctly labelling background or signal events

        probnn_features = [ft for ft in features if 'ProbNN' in ft]
        sf[probnn_features] = sf[probnn_features].mask(sf[probnn_features] < 0, 0)
        rf[probnn_features] = rf[probnn_features].mask(rf[probnn_features] < 0, 0)
        # ProbNN features sometimes have -1000 rather than zero probability, fix this

        sf.reset_index(inplace=True)
        rf.reset_index(inplace=True)
        # By default the eventNumber columns are used as an index, change it to a ranged index

        combined = pd.concat([sf, rf], ignore_index=True, sort=False)
        # Combine the simulated and real data into a larger dataframe ignoring index

        combined.drop('eventNumber', inplace=True, axis=1)
        # Remove the eventNumber column since it is now useless

    combined = shuffle(combined, random_state=random_state)
    # Randomly shuffle the data in a row-by-row fashion 

    train_and_test = combined[:int(np.floor(len(combined)*(train_frac + val_frac)))]
    val = combined[int(np.floor(len(combined)*(train_frac + val_frac))):]
    # Split apart the (test+train) and (val) sets using fraction specified

    y = train_and_test['category']
    # The binary labels (classification problem) are assigned as the y column
    X = train_and_test.drop(['category'], axis=1)
    # Remove the category column from the training inputs

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(train_frac/(train_frac+test_frac)), test_size=1-(train_frac/(train_frac+test_frac)), random_state=random_state)
    # Split the training and test data accordingly so that the input fractions are maintained relative
    # to the entire dataset

    X_val, y_val = val.drop(['category'], axis=1), val['category']
    # Create the validation datasets i.e. inputs and labels

    (X_train_means, X_train_stds) = X_train.mean(), X_train.std()
    # Find the means and standard deviations of every feature in the training data

    X_test_normalised = normalise_df_features(X_test, cols=X_train.columns, params=(X_train_means, X_train_stds))
    X_val_normalised = normalise_df_features(X_val, cols=X_train.columns, params=(X_train_means, X_train_stds))
    X_train_normalised = normalise_df_features(X_train)
    # Normalise all the sets using the normalisation parameters found for the training data

    return (X_train_normalised, y_train), (X_val_normalised, y_val), (X_test_normalised, y_test)

def plot_history_curves(history, epochs):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['binary_accuracy']
    val_accuracy = history.history['val_binary_accuracy']
    epoch_range = range(1, epochs+1, 1)    

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ax[0].plot(epoch_range, loss, 'r.', label='Training Loss')
    ax[0].plot(epoch_range, val_loss, 'b.', label='Validation Loss')
    ax[1].plot(epoch_range, accuracy, 'r.', label='Training Accuracy')
    ax[1].plot(epoch_range, val_accuracy, 'b.', label='Validation Accuracy')
    ax[0].legend()
    ax[1].legend()
    plt.legend(frameon=False)
    plt.show()