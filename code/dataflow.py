#
# Package Name: DataFlow
# Author: Chandler Kenworthy
# Version: 1.3
#

class Flow:
    # Description goes here
    
    def __init__(self, features, sim_fname, real_fname, csv_path=None):
        """
        Set up the flow object by pre-requesting all the features that
        will be needed as features when training the neural network.
        
        Parameters
        ----------
        features : array_like
            A list of features that will be used to train the neural network
        """
        
        if (csv_path is not None):
            import pandas as pd
            self.combined = pd.read_csv(csv_path, sep=" ", index_col='Index')
            self.features = list(self.combined.columns)
        else:
            self.features = self.check_common_features(features)
            self.simulated_preselection = None
            self.real_preselection = None
            self.simulated_df_prefix = 'sf'
            self.real_df_prefix = 'rf'
            self.simulated_preselection_features = None
            self.real_preselection_features = None
            self.sf = None
            self.rf = None
            self.sim_fname = sim_fname
            self.real_fname = real_fname
            self.sim_tree = ":DTT1520me/DecayTree"
            self.real_tree = ":DTT1520me/DecayTree"
            self.preselection_applied = False
            self.combined = None
                    
    
    def check_common_features(self, features):
        """
        Check whether the features provided are common to the real and
        simulated datasets. Echo all features that were not common.
        
        Parameters
        ----------
        features : array_like
            A list of features
        
        Returns
        -------
        array_like
            A list of features that are common to both real and simulated
            data tuples
        """
        
        import pandas as pd
        common_features_list = pd.read_csv('common_features.txt', sep=' ', index_col=None)['Feature'].to_list()
        # Read in the common features from the file
        common_features = [feature for feature in features if feature in common_features_list]
        
        for feature in [j for j in features if j not in common_features]:
            print(f'WARN: Requested feature {feature} is not common to simulated and real data!')
            # Warn the user about all the features that had to be removed
        
        return common_features
    
    
    def generate_feature(self, new_feature_name, expression):
        """
        Generate a new feature derived from features parsed at initial time of creation of the
        flow object.
        
        Parameters
        ----------
        new_feature_name : string
            The name of the newly consturcted feature which should be assinged to the 
            column name in the overarching dataframe
        expression : string
            An evaluable string used to generate the required feature. Must be in a Pythonic
            and numpy friendly syntax
        """
        
        import numpy as np
        if self.combined is None:
            print('WARN: No combined dataframe currently exists')
            print('WARN: Automatically combining dataframes')
            self.combine_data()
        expression = self.add_preselection_df_prefix(expression, 'combined')[0]
        print(f'Attempting to evaluate:\n{expression}')
        custom_series = eval(expression)
        self.combined[new_feature_name] = custom_series
        print(f'{new_feature_name} calculated successfully')
    
    
    def to_csv(self, fname):
        """
        Export the combined dataframe object to a CSV file for faster reading
        in and out next time. This will output an index column and headers.
        
        Parameters
        ----------
        fname : String
            The filepath and or filename that are to be used when writing
            out the CSV
        
        TODO: Be able to instantiate a Flow object with a CSV file
        """
        
        if (self.combined is None):
            print('WARN: No data to output to CSV!')
        else:
            self.combined.to_csv(fname, sep=" ", index_label='Index', header=True, index=True)
    
    
    def drop_features(self, features, target='all'):
        """
        Remove the requested features completely from the dataframe. Warning: this action
        is permenant and cannot be undone! 
        
        Parameters
        ----------
        features : array_like
            The list of features to remove from the dataframe.
        target : string
            The frame to target the feature dropping upon. Defaults to "all"
            which removes features from the combined data. Other options are
            "sim" or "real"
        """
        
        if (self.sf is None) and (self.rf is None): 
            print("WARN: Dataframes are empty, unable to drop features")
            print("Try calling combine_data or apply_preselection first to avoid this error")
        else:
            if target == 'sim':
                self.sf.drop(features, axis=1, inplace=True)
                self.features = [f for f in self.features if f not in features]
            elif target == 'real':
                self.rf.drop(features, axis=1, inplace=True)
                self.features = [f for f in self.features if f not in features]
            else:
                self.combined.drop(features, axis=1, inplace=True)

    
    def get_features(self, features, tuple):
        """
        Get the requested features from the given ROOT tuple. Automatically sets the eventNumber
        as the indexing column.
        
        Parameters
        ----------
        features : array_like
            A list of features to get from the ROOT tuples
        tuple : string
            Whether to access these features from the simulated tuple or real tuple. This cannot
            do both simultaneously.
            
        Returns
        -------
        dataframe
            A dataframe with index eventNumber along with all associated features. This is not
            sorted a NaNs or infs are not removed. 
        """
        
        import uproot as up
        if tuple == 'real':
            fName = self.real_fname
            tree = self.real_tree
        else:
            fName = self.sim_fname
            tree = self.real_tree
        
        fts = ["eventNumber"] + features
        if tuple == 'sim':
            fts += ["Lb_BKGCAT"]
        fts = list(dict.fromkeys(fts))
        # Remove any duplicate features
        with up.open(fName + tree) as f:
            # Open the tuple using UpRoot
            df = f.arrays(fts, library="pd")
            df.set_index("eventNumber", inplace=True)
            df.columns = fts[1:]
            # Remove duplicate events
        return df
    
    def get_simulated(self):
        """
        Get the simulated events in their own dataframe, includes eventNumber
        """
        
        return self.sf
    
    
    def get_real(self):
        """
        Get the real events in their own dataframe, includes eventNumber
        """
        
        return self.rf
    
    
    def add_preselection_df_prefix(self, boolean_mask, prefix):
        """
        Add the dataframe wrappers around the relevant parts of the boolean
        mask that performs the preselection. That is wrap feature names in
        quotation and the dataframe identifier
        
        Parameters
        ----------
        boolean_mask : string
            A Pythonic friendly string that is a singular or set of logical
            operators with feature names as they are found in the tuples
        prefix : string
            The identifer for the dataframe that the given boolean mask
            will be applied upon
            
        Returns
        -------
        string
            The same boolean mask provided but with feature names wrapped
            appropriately so that eval can be used
        array_like
            A list of all features that are needed to evaluate the provided
            boolean mask
        """
        
        preselection_features = []
        begin, end = False, False
        begin_position, end_position = 0, 0
        updated_mask = ''
        
        for i, char in enumerate(boolean_mask):
            updated_mask += char
            # Add the character to the current feature name
            if char == ' ' and not begin:
                # The beginning of a feature is when there is a space
                updated_mask = updated_mask[:-1]
                # Remove the space character added to the string
                updated_mask += f"self.{prefix}['"
                # This is the start of a new feature so add the appropriate prefix
                begin_position = i
                begin = True 
                # Set the begin position to the current position and update
                # the begin variable to show we have a beginning position
            elif char == ' ' and begin:
                # If a beginning position already exists this must be an end point
                updated_mask = updated_mask[:-1]
                # Remove the space character from the end of the string
                updated_mask += "']"
                # Add the closing bracket to match the dataframe wrapper
                end_position = i
                end = True
                # Set the end position and update the end variable to be true
            if begin and end:
                # We have both a begin and end point
                preselection_features.append(boolean_mask[begin_position + 1:end_position])
                begin, end = False, False
                # Reset the beginning and ending flags
        
        preselection_features = list(dict.fromkeys(preselection_features))
        # Remove all duplicates
        
        return updated_mask, preselection_features
        
        
    def set_simulated_preselection(self, boolean_mask):
        """
        Set the string to be evaluated when applying the pre-selection to
        the simulated data. This should be of a Pythonically readable form.
        Use feature names as they appear in the tuples without any wrapper.
        
        Parameters
        ----------
        boolean_mask : string
            A string in a Python-friendly logical format. Use feature names
            as they appear in the ROOT tuples
        """
        
        updated_mask, pre_features = self.add_preselection_df_prefix(boolean_mask, self.simulated_df_prefix)
        self.simulated_preselection = updated_mask
        self.simulated_preselection_features = pre_features
        
    
    def set_real_preselection(self, boolean_mask):
        """
        Set the string to be evaluated when applying the pre-selection to
        the real data. This should be of a Pythonically readable form.
        Use feature names as they appear in the tuples without any wrapper.
        
        Parameters
        ----------
        boolean_mask : string
            A string in a Python-friendly logical format. Use feature names
            as they appear in the ROOT tuples
        """
        
        updated_mask, pre_features = self.add_preselection_df_prefix(boolean_mask, self.real_df_prefix)
        self.real_preselection = updated_mask
        self.real_preselection_features = pre_features
        
    
    def apply_preselection(self):
        """
        Apply the current unique preselection criteria to the corresponding datasets.
        This will trigger a call for all features initially specified and those needed
        to perform the preselection.
        """

        import numpy as np
        if (self.simulated_preselection is None) or (self.real_preselection is None):
            print('WARN: No preselection has been specified for either/both the simulated and real data')
        else:
            sim_features = list(dict.fromkeys(self.features + self.simulated_preselection_features))
            real_features = list(dict.fromkeys(self.features + self.real_preselection_features))
            # Find all common features between those used for training and those needed for preselection
            # so we request as few features as possible and avoid duplicates
            
            self.sf = self.get_features(sim_features, 'sim')
            self.rf = self.get_features(real_features, 'real')
            # Get all the features that will be required 
            self.sf = self.sf[eval(self.simulated_preselection)]
            self.rf = self.rf[eval(self.real_preselection)]
            # Evaluate the pre-selection criteria
        
        self.preselection_applied = True
        
    
    def combine_data(self, random_shuffle=True, random_state=0, remove_na=True):
        """
        Combine the simulated and real dataframes into a single larger frame with the
        eventNumber index removed and replaced with a range index. Potentially randomly
        shuffle the data inside the frame as well. The frame is combined so that
        the real data is stacked below the simulated data. 
        
        Parameters
        ----------
        random_shuffle : bool
            Whether to apply a random shuffling of the order of rows in the combined
            dataframe. Shuffles after combining. 
        random_state : int
            The seed for the random shuffle.
        remove_na : bool
            Whether events with any missing values should be removed
        """
        
        import numpy as np
        import pandas as pd
        from sklearn.utils import shuffle
        
        if not self.preselection_applied or self.sf is None or self.rf is None:
            # We have no dataframes and no preselection has happened so data cant be combined
            print('WARN: The dataframes are empty or pre-selection has not been applied')
            print('WARN: Attempting empty preselection and recombination')
            self.sf = self.get_features(self.features, 'sim')
            self.rf = self.get_features(self.features, 'real')
        self.sf['category'] = np.where(self.sf['Lb_BKGCAT'].isin([10,50]), 1, 0)
        self.sf.drop('Lb_BKGCAT', axis=1, inplace=True)
        self.rf['category'] = 0
        
        self.sf = self.sf[self.features + ['category']]
        self.rf = self.rf[self.features + ['category']]
        # Remove any columns that were used for pre-selection
        
        self.sf = self.sf[~self.sf.index.duplicated(keep='first')]
        self.rf = self.rf[~self.rf.index.duplicated(keep='first')]
        # Remove duplicate events
        
        self.combined = pd.concat([self.sf, self.rf], ignore_index=True, sort=False)
        if random_shuffle:
            self.combined = shuffle(self.combined, random_state=random_state)
        if remove_na:
            has_missing = self.combined[self.combined.isna().any(axis=1)]
            cols_with_nans = has_missing.columns[has_missing.isna().any()].to_list()
            if len(cols_with_nans) > 0:
                print(f'INFO: Columns which had one or more NaN values:\n{cols_with_nans}')
            nan_events = self.combined.iloc[pd.isnull(self.combined).any(1).to_numpy().nonzero()[0]]['category'].to_list()
            print(f'INFO: Removing events with missing values...\nINFO: {nan_events.count(0)} background and {nan_events.count(1)} signal events were removed')
            self.combined.dropna(inplace=True)
            
    
    def get_combined_data(self):
        """
        Get the combined dataframe that is ready to be fed into a neural network for 
        training. It will have a category column with the correct labels.
        
        Returns
        -------
        dataframe
            A dataframe of the combined simulated and real event data.
        """
        
        try: 
            if self.combined is None:
                self.combine_data()
        except ValueError:
            # self.combined is a dataframe so boolean value cannot be evaluated
            pass
        return self.combined
    
    
    def set_event_ratio(self, ratio=1.0, random_state=0):
        """
        Trim the data via the category to enforce a particular ratio of 
        background to signal events. The default ratio is 0.1
        
        Parameters
        ----------
        ratio : float
            The fraction of signal to background events to be obtained in 
            these data
        random_state : int
            An integer seed for the random sampling algorithm
        """
        
        curr_bg = self.combined['category'].value_counts()[0]
        curr_sg = self.combined['category'].value_counts()[1]
        
        if (self.combined is None):
            self.combine_data()
        
        frac_bg_to_remove = 1 - ((curr_sg / ratio) / curr_bg)
        
        remove_idx = self.combined.query('category == 0').sample(frac=frac_bg_to_remove, random_state=random_state).index.to_list()
        self.combined.drop(remove_idx, axis=0, inplace=True)
    

    def normalise_features(self, df, params=None):
        """
        Normalise features assuming a Gaussian distribution of said feature
        for the requested dataframe. 

        Parameters
        ----------
        df : dataframe
            The dataframe to apply the normalisation on

        params : array_like
            Either False in which parameters for the normalisation are 
            calculated or a tuple of Series objects with means and
            standard deviations

        Returns
        -------
        dataframe
            A copy of the dataframe with the requested features normalised
        """

        df_copy = df.copy()
        # Do not edit the original dataframe
        
        # Apply normalisation to all columns
        if params is None:
            df_copy = df_copy - df_copy.mean()
            norm_df = df_copy / df_copy.std()
        else:
            norm_df = (df_copy - params[0])/params[1]
        norm_df = norm_df.fillna(0)
        # If the std of a column was zero before this is now full of NaNs, set back to equal zero

        return norm_df


    def get_train_val_test_split(self, train=0.6, val=0.2, test=0.2, random_state=0, normalise=True):
        """
        Split the combined dataframe into a training, validation and test sample with
        the fractions specified. Apply a random shuffle if required. This will also
        normalise all the features.
        
        Parameters
        ----------
        train : float
            The fraction of these data to be reserved for training
        val : float
            The fraction of these data to be reserved for validation
        test : float
            The fraction of these data to be reserved for testing
        random_state : int
            The integer seed to use in the random generator performing
            the shuffling. 
            
        Returns
        -------
        I need to finish writing this bit.
        """
        
        from sklearn.model_selection import train_test_split
        import numpy as np
        
        train_and_test = self.combined[:int(np.floor(len(self.combined)*(train + val)))]
        val = self.combined[int(np.floor(len(self.combined)*(train + val))):]
        # Split apart the (test+train) and (val) sets using fraction specified

        y = train_and_test['category']
        # The binary labels (classification problem) are assigned as the y column
        x = train_and_test.drop(['category'], axis=1)
        # Remove the category column from the training inputs

        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=(train/(train+test)), test_size=1-(train/(train+test)), random_state=random_state)
        # Split the training and test data accordingly so that the input fractions are maintained relative
        # to the entire dataset

        X_val, y_val = val.drop(['category'], axis=1), val['category']
        # Split the validation data up
        
        if normalise:
            (X_train_means, X_train_stds) = X_train.mean(), X_train.std()
            # Find the means and standard deviations of every feature in the training data
            X_test = self.normalise_features(X_test, params=(X_train_means, X_train_stds))
            X_val = self.normalise_features(X_val, params=(X_train_means, X_train_stds))
            X_train = self.normalise_features(X_train)
            # Normalise all the sets using the normalisation parameters found for the training data

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
