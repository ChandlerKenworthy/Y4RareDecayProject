class Data:
    """
    An object which is used to handle data input, output and updating when handling particle event data. Uses
    a combination of UpRoot to read in ROOT tuples and Pandas dataframes.
    """
    
    def __init__(self, fName, suffix):
        """
        Create the Data object. This will not by default read in any data on instantiation. It will only
        read in data as and when is required. One Data object should persist for each ROOT tuple being
        considered.
        """
        
        self.fName = fName
        self.suffix = suffix
        self.data = None
        self.ncuts = 0 # The number of cuts applied
        self.nevents_lost = [] # Number & % of events removed during each cut [[], [], []]
        self.first_length = 0 # The number of events before anything was applied
        
    def fetch_features(self, features=None, remove_na=True):
        """
        A utility function to return a set of specific features. This is only to be used by the external
        for a quick return of features requested. Features should already be appropriately formatted.
        
        Parameters
        ----------
        features : list
            All the features you want
            
        Returns
        -------
        df : pd.DataFrame
            A dataframe of all the features you care about
        """
        import uproot as up
        fts = ["eventNumber"] + features
        with up.open(self.fName + self.suffix) as f:
            print("OPENED")
            # Open the tuple using UpRoot
            df = f.arrays(fts, library="pd")
            df.set_index("eventNumber", inplace=True)
            df.columns = fts[1:]
            df = df[~df.index.duplicated(keep='first')]
            # Remove duplicate events
        # Combine requested features with the current frame
        self.update_data(df, remove_na=remove_na)
        return df
    
    def get_data(self, particles=None, features=None):
        """
        Get the dataframe which contains all previously requested features with
        the cut applied. If the features are already present in the dataframe it
        returns a subset of the original dataframe with only these features. 
        Otherwise the non-exsistent features are requested, returned and added
        to the overarching dataframe.
        
        Returns
        -------
        dat : pd.DataFrame/pd.Series
            A dataframe or series object of all the features that were requested.
        """

        if self.data is None:
            # The dataframe is currently empty, if features are requested then add them and return that
            if features is None: 
                raise Exception("For an empty dataframe you must specify a feature to read in")
            else:
                # Some features have been specified they might have particles included
                if particles is None:
                    self.get_specific_features(features)
                    self.first_length = len(self.data)
                    return self.data[features]
                else:
                    self.get_particle_data(particles, features)
                    self.first_length = len(self.data)
                    return self.data[[particle + "_" + feature for feature in features for particle in particles]]
        else:
            # The dataframe already has some features contained within it
            if features is None:
                # No features were requested so just return the entire dataframe that currently exists
                return self.data
            else:
                # Specific features have been requested, only bother fetching new features
                # this will significantly speed up processing time than joining duplicate data and
                # then removing it again
                if particles is None:
                    new_features = [feature for feature in features if feature not in self.data.columns]
                else:
                    new_features = [particle + "_" + feature for feature in features for particle in particles if particle + "_" + feature not in self.data.columns]
                if len(new_features) != 0:
                    # Get the new features we need and stick them onto the dataframe
                    self.get_specific_features(new_features)
                    return self.data[new_features]
                    # We have used get_specific_features not get_particle_data since new_features already has
                    # the L1_ etc prefix added
                else:
                    return self.data[[particle + "_" + feature for feature in features for particle in particles]]
    
    def get_keys(self):
        """
        Get all the available keys that are present in the decay tree specified
        by self.fName and self.suffix.

        Returns
        --------
            keys : list
            A list of string type keys which can be used to fetch specific data
            columns of interest for each event. 
        """

        import uproot as up
        with up.open(self.fName + self.suffix) as f:
            keys = f.keys()
        return keys
    
    def get_cut_desc(self):
        """
        Get the total number of cuts made, the number of events lost as an absolute
        and percentage value in each cut and the type of cut and its paramters in 
        the order it was applied.
        
        Returns
        -------
        ncuts : int
            The total number of cuts applied to these data.
            
        nevents_lost : list
            A list of lists with the i-th sub-list containg the number of events lost
            in cut number i. The first element of the sub-list is the absolute number
            of events. The last element is the percentage events removed.
        """
        
        return self.ncuts, self.nevents_lost
    
    def apply_cut(self, indices, verbose=False):
        """
        Apply a cut to the current data set. This will remove all events with an eventNumber
        in the indices array.
        
        Parameters
        ----------
        indicies : array
            An array of eventNumbers (ints) corresponding to unique events that are to be
            cut/removed from the data.
        """
        
        initial_length = len(self.data)
        self.data.drop(indices, axis=0, inplace=True)
        final_length = len(self.data)
        self.nevents_lost.append([initial_length - final_length, (initial_length-final_length)/initial_length])
        self.ncuts += 1
        
        if verbose:
            print('=====================\n')
            print(f'Cut made!\nEvents Removed: {initial_length - final_length}')
            print(f'Fractional Decrease in Events: {(initial_length-final_length)/initial_length}')
            print(f'Percentage of All Events Left: {(final_length/self.first_length)*100:.3f}%')
            print('\n=====================')
    
    def update_data(self, new_data, remove_na=True):
        """
        An internal use function to update the self.data attribute when new features
        are added, calculated or used during calculations. By default this removes all
        rows with missing features such that cuts operate correctly.
        
        Parameters
        ----------
        new_data : pd.DataFrame
            A dataframe with new features to add to existing events. The index must be
            already set to the eventNumber as dataframes are joined along this axis.
            
        remova_na : bool, default=True
            Whether to remove rows with missing event features. Defaults to true to ensure
            cuts operate correctly.
        """
        import pandas as pd
        if self.data is None:
            self.data = new_data
        else:
            if not (len([1 for i in self.data.columns if i in new_data.columns]) == len(self.data.columns)):  
                # All the columns match 
                self.data = new_data
                # If all the columns match then set new data as the current
            else: 
                # None or some of the columns might match
                self.data = pd.concat([self.data, new_data], axis=1)
                # Remove duplicated columns
                self.data = self.data.loc[:,~self.data.columns.duplicated()]                
        if remove_na:
            self.data.dropna(inplace=True)
            # Remove all events where any feature is missing
        if self.first_length == 0:
            self.first_length = len(self.data)
        
    def get_particle_data(self, particles, features, drop_duplicates=True):
        """
        Get the features specified in features from the ROOT tuples. Can be specified
        for multiple particles.

        Parameters
        ----------
        particles : list
            A list of particles for the features to be requested from. These should be
            the names as given in the tuple for example L1 for the muon.
        
        features : list
            A list of features to gather for each of the particles
            specified in particles.

        Returns
        -------
        df : pd.DataFrame
            A pd.DataFrame with columns labelled according to feature names concatenated
            with particle names.
        """

        import uproot as up
        # Generate all the features to fetch from the tuple
        fts = ["eventNumber"] + [particle + "_" + feature for feature in features for particle in particles]
        with up.open(self.fName + self.suffix) as f:
            # Open the tuple using UpRoot
            df = f.arrays(fts, library="pd")
            df.set_index("eventNumber", inplace=True)
            # Always set the index to the unique eventNumber identifier
            df.columns = fts[1:]
            # Relabel the columns to something somewhat relevant
            if drop_duplicates:
                df = df[~df.index.duplicated(keep='first')]
                # Removed "duplicated" events and keep only one 
        self.update_data(df)
    
    def get_specific_features(self, features, drop_duplicates=True):
        """
        Gather specific particle independent features from the dataset. By 
        default this will not remove duplicates.

        Parameters
        ----------
            features : list
            A list of features to retrieve for each event. The columns of the
            returned dataframes will be given corresponding names.

            drop_duplicates: bool, default=False
            Whether to drop duplicate events. This is determined by simply removing
            all duplicate rows. For few features this can cause removal of different
            events. It defaults to False.
        """

        import uproot as up
        
        with up.open(self.fName + self.suffix) as f:
            df = f.arrays(["eventNumber", *features], library="pd")
            df.set_index("eventNumber", inplace=True)
            if drop_duplicates:
                df = df[~df.index.duplicated(keep='first')]
        self.update_data(df)


class Consts:
    """
    A simple class which just stores a load of file paths to the data files
    and some suffix names which specify the particular decay tree. Saves writing
    them out millions of times. 
    """
    
    def __init__(self):
        """
        Make the consts object and set the file paths and suffixes based on some 
        hard-coded values.
        """
        self.real_tuple_fName = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/realData/2016MD/halfSampleOct2021/blindedTriggeredL1520Selec-collision-firstHalf2016MD-pKmue_Full.root"
        self.real_tuple_suffix = ":DTT1520me/DecayTree"
        self.sim_tuple_fName_old = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MC/2016MD/100FilesCheck/job185-CombDVntuple-15314000-MC2016MD_100F-pKmue-MC.root"
        self.sim_tuple_suffix_old = ":DTT1520me/DecayTree"
        self.sim_tuple_fName = "/disk/moose/lhcb/djdt/Lb2L1520mueTuples/MC/2016MD/fullSampleOct2021/job207-CombDVntuple-15314000-MC2016MD_Full-pKmue-MC.root"
        self.sim_tuple_suffix = ":DTT1520me/DecayTree"
    
    

    def get_real_tuple(self):
        """
        Get the file location of the tuple containing real LHCb data as well as the suffix
        relating to the decay tree of interest.
        
        Returns
        -------
        fData : tuple
            A tuple of the file location and suffix (fName, suffix).
        """
        return self.real_tuple_fName, self.real_tuple_suffix

    def get_simulated_tuple(self):
        """
        Get the file location of the tuple containing Monte-Carlo data as well as the suffix
        relating to the decay tree of interest.
        
        Returns
        -------
        fData : tuple
            A tuple of the file location and suffix (fName, suffix).
        """
        return self.sim_tuple_fName, self.sim_tuple_suffix
    

class Cut:
    """
    A generic object that can be used to make cuts on data using some set of selection criteria.
    This will calculate the cut to be performed and then leverage the Data object to perform
    the inplace cut operation.
    """

    def __init__(self, data):
        """
        Create the Cut object. The cuts must act on some data and so a filepath to the dataset
        of interest is required along with a suffix specifying the decay tree. The cut will act
        on the same variables for all particles specified within the particles argument.

        Parameters
        ----------
            data : Data
            An instantiated Data object that points to a specific tuple and decay tree.
        """

        self.d = data

    def nbody(self, n):
        """
        Make a cut on these particles data based on the number of bodies that were detected
        in the event. Returns the eventNumbers to be cut. 

        Parameters
        ----------
        n : int, list
            The number of bodies to make a cut around such that only events with n
            decay bodies are kept. This can be a list of ints to keep. In essence
            for [1, 3] all events with 1 or 3 bodies are retained.
        """

        self.d.get_data(features=["totCandidates"])
        if type(n) is int:
            df = self.d.get_data().loc[self.d.get_data()["totCandidates"] != n]
        else:
            df = self.d.get_data().loc[~self.d.get_data()["totCandidates"].isin(n)]
        self.d.apply_cut(list(df.index))
        
    def probnn_cut(self, particle, p, on, type='lt', verbose=False):
        """
        Make a cut based on a particles ProbNN variables. This will specifically cut
        on particle e.g. K with respect to on e.g. K_ProbNNp. It will cut around
        the value p according to type.
        
        Parameters
        ----------
        particle : string
            The particle whose ProbNN variable is to be cut on e.g. L1
            
        p : float
            A probability value between 0 and 1 to cut around. This will cut around
            the normalised ProbNN probabilities.
            
        on : string
            The particle ProbNN prefix. That is the probability that particle in
            the particle specified by on such that if on="mu" and particle="k" then
            you cut around K_ProbNNmu.
            
        type : string (default='lt')
            The type of cut to do either a lower than (lt) or greater than (gt) 
            cut. That is for a type of lt events where Particle_ProbNNon is
            less than p are removed.
        """
        
        items = ['mu', 'pi', 'p', 'k', 'd', 'e', 'ghost']
        feature_of_interest = particle + "_ProbNN" + on
        fts = ["ProbNN" + item for item in items] # Get all the probability features for this particle
        df = self.d.get_data(particle, fts)
        fts = [particle + "_" + ft for ft in fts]
        df[fts] = df[fts].mask(df[fts].lt(0),0) # Set all -ve probabilities to zero
        df['tProb'] = df[fts].sum(axis=1) # Sum the probabilitiy features
        df.iloc[:,0:-2] = df.iloc[:,0:-2].div(df.tProb, axis=0) # Normalise the probabilities
        if type == "lt":
            kill = df.loc[df[feature_of_interest] <= p]
        else:
            kill = df.loc[df[feature_of_interest] >= p]
        dropEvents = list(kill.index)
        self.d.apply_cut(dropEvents, verbose=verbose)
        
    def not_itself_probnn(self, particle, p):
        """
        Make a cut based on the probabiltiy that a particle is not itself this
        will cut all events based on the P(particle != particle) <= p. 
        """
        items = ['mu', 'pi', 'p', 'k', 'd', 'e', 'ghost']
        ptype = {'L1': 'mu', 'L2': 'e', 'p': 'p', 'K': 'k'}
        foi = particle + "_ProbNN" + ptype[particle] # The probabiltiy the particle is itself
        fts = ["ProbNN" + item for item in items]
        df = self.d.get_data(particle, fts)
        fts = [particle + "_" + ft for ft in fts]
        df[fts] = df[fts].mask(df[fts].lt(0),0) # Set all -ve probabilities to zero
        df['tProb'] = df[fts].sum(axis=1) # Sum the probabilitiy features
        df.iloc[:,0:-2] = df.iloc[:,0:-2].div(df.tProb, axis=0) # Normalise the probabilities
        kill = df.loc[1 - df[foi] <= p]
        self.d.apply_cut(list(kill.index))
        
    def mom_cut(self, particles, max_pT, min_pT=0):
        """
        Remove all events outside the transverse momentum range specified for
        the particles specified.
        """
        
        import numpy as np
        if type(particles) != list:
            particles = [particles]
        self.d.get_data(particles, ['PT'])
        fts = [particle + "_PT" for particle in particles]
        dropEvents = []
        df = self.d.get_data()
        for ft in fts:
            j = df[~df[ft].between(min_pT, max_pT)]
            dropEvents.append(list(j.index))
        dropEvents = [item for sublist in dropEvents for item in sublist]
        # Flatten the dropEvents list so that apply_cut can deal with it
        dropEvents = np.unique(dropEvents)
        self.d.apply_cut(dropEvents)
        
class Model:
    """
    An object which stores a collection of model functions which can easily be called
    at any point
    """
    
    def __init__(self):
        pass
    
    def exponential(m, a, b, c):
        """ a*exp[-(bm+c)] """
        import numpy as np
        return a * np.exp(-(b*m + c))

    def expsquare(m, a, b, c):
        """ defm """
        import numpy as np
        return a*np.exp(-(m-b)**2)+c

    def gaussian(m, mu, sigma):
        """ Standard normal distribution """
        import numpy as np
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
    
    def gaussian(m, A, mu, sigma):
        """ Standard normal distribution """
        import numpy as np
        return A*(1/np.sqrt(2*np.pi*sigma**2))*np.exp(-((m-mu)**2)/(2*sigma**2))

    def scaled_lorentzian(m, A, m_0, gamma):
        import numpy as np
        return (A/np.pi)*((0.5*gamma)/((m-m_0)**2 + (0.5*gamma)**2))

    def breit_wigner(m, M, w, A):
        """
        The relativistic breit wigner distribution 
        """
        import numpy as np
        gamma2 = np.sqrt((M**2)*((M**2) + (w**2))) 
        # Array with length len(m)
        k = (2*np.sqrt(2)*M*w*gamma2)/(np.pi*np.sqrt((M**2)*gamma2)) 
        # Array with length len(m)
        return A*k/((m**2 - M**2)**2 + (M*w)**2)

    def voigt(m, alpha, gamma, shift):
        """
        The Voigt profile
        """
        import numpy as np
        from scipy.special import wofz
        sigma = alpha/np.sqrt(2*np.log(2))
        return np.real(wofz(((m-shift) + 1j*gamma)/sigma/np.sqrt(2)))/sigma/np.sqrt(2*np.pi)

    def cball(m, A, b, u, loc):
        """
        The scaled crystal ball function
        """
        from scipy.stats import crystalball
        return A*crystalball.pdf(m, b, u, loc, 1)

    def dscb(x, mu, sigma, alow, ahigh, nlow, nhigh):
        # Works but needs some serious optimisation
        # See https://arxiv.org/pdf/2011.07560.pdf
        import numpy as np
        z = (x - mu)/sigma
        values = []
        
        # Apply a specific function to the shifted values based on initial values
        for v in z:
            fx = 0
            if v < -alow:
                fx = np.exp(-0.5 * (alow**2)) * (((alow/nlow) * (nlow/alow - alow - v))**(-nlow))
            elif v > ahigh:
                fx = np.exp(-0.5 * (ahigh**2)) * (((ahigh/nhigh) * (nhigh/ahigh - ahigh + v))**(-nhigh))
            else:
                fx = np.exp(-0.5 * (v**2))
            values.append(fx)
        values = np.array(values)
        x_gaps = np.array([x[i+1] - x[i] for i in range(0, len(x)-1)])
        # The widths of each 'bin' that the function is being evaluated over
        mean_values = np.array([np.mean([values[i], values[i+1]]) for i in range(len(values)-1)])
        
        N = 1/np.sum(np.multiply(x_gaps, mean_values))
        #print(f'Scaling factor: {N}\nNormalised: {np.sum(N*values)}')
        return N * values
        