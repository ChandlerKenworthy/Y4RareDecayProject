#
# Package Name: DataFlow
# Author: Chandler Kenworthy
# Version: 0.1
#

class Flow:
    # Description goes here
    
    def __init__(self, features):
        """
        Set up the flow object by pre-requesting all the features that
        will be needed as features when training the neural network.
        
        Parameters
        ----------
        features : array_like
            A list of features that will be used to train the neural network
        """
        
        self.features = self.check_common_features(features)
        self.simulated_preselection = None
        self.real_preselection = None
        self.simulated_df_prefix = 'sf'
        self.real_df_prefix = 'rf'
        self.simulated_preselection_features = None
        self.real_preselection_features = None
        self.sf = None
        self.rf = None
        self.sim_fname = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MC/2016MD/100FilesCheck/job185-CombDVntuple-15314000-MC2016MD_100F-pKmue-MC.root"
        self.real_fname = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/realData/2016MD/halfSampleOct2021/blindedTriggeredL1520Selec-collision-firstHalf2016MD-pKmue_Full.root"
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
        common_features_list = pd.read_csv('code/neural_network/common_features.txt', sep=' ', index_col=None)['Feature'].to_list()
        # Read in the common features from the file
        common_features = [feature for feature in features if feature in common_features_list]
        
        for feature in [j for j in features if j not in common_features]:
            print(f'WARN: Requested feature {feature} is not common to simulated and real data!')
            # Warn the user about all the features that had to be removed
        
        return common_features
    
    
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
            df = df[~df.index.duplicated(keep='first')]
            # Remove duplicate events
        return df
    
    
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
            
            self.sf = self.sf[eval(self.simulated_preselection)]
            self.rf = self.rf[eval(self.real_preselection)]
        
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
        
        if not self.preselection_applied or self.sf == None or self.rf == None:
            # We have no dataframes and no preselection has happened so data cant be combined
            print('WARN: The dataframes are empty or pre-selection has not been applied')
            print('WARN: Attempting empty preselection and recombination')
            self.sf = self.get_features(self.features, 'sim')
            self.rf = self.get_features(self.features, 'real')
        self.sf['category'] = np.where(self.sf['Lb_BKGCAT'].isin([10,50]), 1, 0)
        self.sf.drop('Lb_BKGCAT', axis=1, inplace=True)
        self.rf['category'] = 0
        self.combined = pd.concat([self.sf, self.rf], ignore_index=True, sort=False)
        self.combined.drop('eventNumber', inplace=True, axis=1)
        if random_shuffle:
            self.combined = shuffle(self.combined, random_state=random_state)
        if remove_na:
            has_missing = self.combined[self.combined.isna().any(axis=1)]
            cols_with_nans = has_missing.columns[has_missing.isna().any()].to_list()
            print(f'INFO: Columns which had one or more NaN values:\n{cols_with_nans}')
            nan_events = self.combined.iloc[pd.isnull(self.combined).any(1).to_numpy().nonzero()[0]]['category'].to_list()
            print(f'INFO: {nan_events.count(0)} background and {nan_events.count(1)} signal events were removed')
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
        
        if self.combined == None:
            self.combine_data()
        return self.combined
    

    def normalise_features(self, df, params):
        """
        Normalise all the columns of the given dataframe according to the parameters
        passed in params.
        
        Parameters
        ----------
        df : dataframe
            A dataframe on which the normalisation will occur. This will not be
            done inplace
        params : tuple (list)
            A tuple of lists with the first item the mean and second item the
            standard deviation for each list. The length of the tuple must
            match the number of columns in df
        """
        
        return 0


    def get_train_val_test_split(self, train=0.6, val=0.2, test=0.2, random_sate=0, normalise=True):
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
        X = train_and_test.drop(['category'], axis=1)
        # Remove the category column from the training inputs

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=(train/(train+test)), test_size=1-(train/(train+test)), random_state=random_sate)
        # Split the training and test data accordingly so that the input fractions are maintained relative
        # to the entire dataset

        X_val, y_val = val.drop(['category'], axis=1), val['category']
        # Split the validation data up
        
        if normalise:
            (X_train_means, X_train_stds) = X_train.mean(), X_train.std()
            # Find the means and standard deviations of every feature in the training data
            
            X_test = normalise_df_features(X_test, cols=X_train.columns, params=(X_train_means, X_train_stds))
            X_val = normalise_df_features(X_val, cols=X_train.columns, params=(X_train_means, X_train_stds))
            X_train = normalise_df_features(X_train)
            # Normalise all the sets using the normalisation parameters found for the training data

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    

particle_features = ['PX', 'PY', 'PZ', 'PT']
particles = ['L1', 'p']
feats = [particle + "_" + ft for ft in particle_features for particle in particles]
test = Flow(feats)
test.set_simulated_preselection("(((( Lb_M01_Subst0_p2K <1019.461-12)|( Lb_M01_Subst0_p2K >1019.461+12))&((((((243716.98437715+ p_P **2)**0.5+ K_PE + L2_PE )**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2000)&(((((243716.98437715+ p_P **2)**0.5+ K_PE + L1_PE )**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2000))&((((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(0.26112103+ L2_P **2)**0.5)**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2320)&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&(((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(11163.69140675+ L1_P **2)**0.5)**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2320)&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0)))))&(( Lb_M23 >3178.05)|( Lb_M23 <3000))&((((((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)>1865+20)|(((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&((((((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)>3097+35)|(((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)<3097-35))&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))&((((((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)>1865+20)|(((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID >0)&( p_ID >0))|(( L2_ID <0)&( p_ID <0))))|((( L1_ID >0)&( p_ID >0))|(( L1_ID <0)&( p_ID <0))))&((( p_PX * L1_PX + p_PY * L1_PY + p_PZ * L1_PZ )/( p_P * L1_P )<np.cos(1e-3))&(( p_PX * L2_PX + p_PY * L2_PY + p_PZ * L2_PZ )/( p_P * L2_P )<np.cos(1e-3))&(( K_PX * L1_PX + K_PY * L1_PY + K_PZ * L1_PZ )/( K_P * L1_P )<np.cos(1e-3))&(( K_PX * L2_PX + K_PY * L2_PY + K_PZ * L2_PZ )/( K_P * L2_P )<np.cos(1e-3)))&(( p_PX * K_PX + p_PY * K_PY + p_PZ * K_PZ )/( p_P * K_P )<np.cos(1e-3)))&( L1_L0MuonDecision_TOS )&(( Lb_Hlt1TrackMVADecision_TOS )|( Lb_Hlt1TrackMuonDecision_TOS ))&( Lb_Hlt2Topo2BodyDecision_TOS | Lb_Hlt2Topo3BodyDecision_TOS | Lb_Hlt2Topo4BodyDecision_TOS | Lb_Hlt2TopoMu2BodyDecision_TOS | Lb_Hlt2TopoMu3BodyDecision_TOS | Lb_Hlt2TopoMu4BodyDecision_TOS )&(( LStar_M >1448)&( LStar_M <1591))&(( Lb_BKGCAT ==10)|( Lb_BKGCAT ==50)))")
test.set_real_preselection("((( Lb_M01_Subst0_p2K <1019.461-12)|( Lb_M01_Subst0_p2K >1019.461+12))&((((((243716.98437715+ p_P **2)**0.5+ K_PE + L2_PE )**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2000)&(((((243716.98437715+ p_P **2)**0.5+ K_PE + L1_PE )**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2000))&((((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(0.26112103+ L2_P **2)**0.5)**2-( p_PX + K_PX + L2_PX )**2-( p_PY + K_PY + L2_PY )**2-( p_PZ + K_PZ + L2_PZ )**2)**0.5)>2320)&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&(((((((880354.49999197+ p_P **2)**0.5+(243716.98437715+ K_P **2)**0.5+(11163.69140675+ L1_P **2)**0.5)**2-( p_PX + K_PX + L1_PX )**2-( p_PY + K_PY + L1_PY )**2-( p_PZ + K_PZ + L1_PZ )**2)**0.5)>2320)&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0)))))&(( Lb_M23 >3178.05)|( Lb_M23 <3000))&((((((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)>1865+20)|(((( K_PE +(19479.95517577+ L2_P **2)**0.5)**2-( K_PX + L2_PX )**2-( K_PY + L2_PY )**2-( K_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))|((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))&((((((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)>3097+35)|(((((11163.69140675+ K_P **2)**0.5+ L1_PE )**2-( K_PX + L1_PX )**2-( K_PY + L1_PY )**2-( K_PZ + L1_PZ )**2)**0.5)<3097-35))&((( L1_ID <0)&( p_ID >0))|(( L1_ID >0)&( p_ID <0))))|((( L2_ID <0)&( p_ID >0))|(( L2_ID >0)&( p_ID <0))))&((((((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)>1865+20)|(((((243716.98437715+ p_P **2)**0.5+(19479.95517577+ L2_P **2)**0.5)**2-( p_PX + L2_PX )**2-( p_PY + L2_PY )**2-( p_PZ + L2_PZ )**2)**0.5)<1865-20))&((( L2_ID >0)&( p_ID >0))|(( L2_ID <0)&( p_ID <0))))|((( L1_ID >0)&( p_ID >0))|(( L1_ID <0)&( p_ID <0))))&((( p_PX * L1_PX + p_PY * L1_PY + p_PZ * L1_PZ )/( p_P * L1_P )<np.cos(1e-3))&(( p_PX * L2_PX + p_PY * L2_PY + p_PZ * L2_PZ )/( p_P * L2_P )<np.cos(1e-3))&(( K_PX * L1_PX + K_PY * L1_PY + K_PZ * L1_PZ )/( K_P * L1_P )<np.cos(1e-3))&(( K_PX * L2_PX + K_PY * L2_PY + K_PZ * L2_PZ )/( K_P * L2_P )<np.cos(1e-3)))&(( p_PX * K_PX + p_PY * K_PY + p_PZ * K_PZ )/( p_P * K_P )<np.cos(1e-3)))")
test.apply_preselection()