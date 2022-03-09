class ProbabilityDistribution:
    # A class for generating probability distributions from an underlying 
    # model and data sample
    
    def __init__(self, train, val, test, nbins=40):
        """
        Initalize the probability distribution object
        
        Parameters
        ----------
        model_type : string
            Whether the model has been constructed using TensorFlow (TF) or 
            SKlearn (SK). This is imperative to this class working.
            
        train : dataframe
            The training data used to train this model with the class labels
            attatched in a category column. 
            
        val : dataframe
            The validation data used during training this model with the class 
            labels attatched in a category column. 
            
        test : dataframe
            The test data used to evaluate the final performance of the model.
            Class labels must be present in the category column.
        """
        
        import numpy as np      
        self.data = {'train': train, 'val': val, 'test': test}
        self.nbins = np.linspace(0, 1, nbins+1)
        self.colors = {
            "<class 'tensorflow.python.keras.engine.sequential.Sequential'>": 
                {'train': '#3366ff', 'val': '#66cc00', 'test': '#ff9900'},
            "<class 'xgboost.sklearn.XGBClassifier'>":
                {'train': '#0000ff', 'val': '#33cc00', 'test': '#cc0000'},
            "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>":
                {'train': '#6666ff', 'val': '#339900', 'test': '#ff6666'},
            "<class 'sklearn.ensemble._forest.RandomForestClassifier'>":
                {'train': '#99ccff', 'val': '#66cc99', 'test': '#ff6666'}
            }
        self.labels = ['Train', 'Validation', 'Test']
        self.linestyles = {'background': 'solid', 'signal': 'dashed'}
        self.savefig = None
        
        
    def set_savefig(self, savefig):
        self.savefig = savefig
    
    def get_model_predictions(self, model, dataset, keep_only=False):
        """
        Generate a single numpy array of probabilities. That is the probability
        of an event to be a signal event.
        """
        
        data = self.data[dataset].copy()
        if keep_only != False:
            data = data[data['category'] == keep_only]
        data.drop(['category'], axis=1, inplace=True)
        
        if str(type(model)) == "<class 'tensorflow.python.keras.engine.sequential.Sequential'>":
            predictions = model.predict(data)
        elif str(type(model)) in ["<class 'xgboost.sklearn.XGBClassifier'>", 
                                  "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>", 
                                  "<class 'sklearn.ensemble._forest.RandomForestClassifier'>"]:
            predictions = model.predict_proba(data)[:,1]
        else:
            print(f"MODEL TYPE: {type(model)}")
            print("FAULT: Unable to generate predictions from unknown model type")
        return predictions
    
    def plot_singular(self, models, model_names, dataset, split_bg_sig=False, hide_errors=False, colors=None):
        """
        Generate a probability distribution plot for the models predicted values
        for a singular dataset for example the test set.
        
        Parameters
        ----------
        model : model or array_like
            The model the predictions are to be made from. This must already be
            trained. Can be an array of models.
            
        model_name : string or array_like
            The name of the model or models that will be plotted
            
        dataset : string
            The dataset label. Either train, val or test
        """
        
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        import numpy as np
        
        if colors is None:
            colors = self.colors
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        fig.patch.set_facecolor('#FFFFFF')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        args = {
            'bins': self.nbins,
            'histtype': 'step',
            'density': True,
            'linewidth': 2,
        }
        
        fonts = {'fontname': 'Arial'}
        
        for i, model in enumerate(list(models)):
            if not split_bg_sig:
                args['linestyle'] = 'solid'
                predictions = self.get_model_predictions(model, dataset)
                args['label'] = f"{model_names[i]} {dataset.capitalize()}"
                args['color'] = colors[str(type(model))][dataset]
                freqs = ax.hist(predictions, **args)[0]
                if not hide_errors:
                    ax.fill_between(self.nbins[:-1], freqs-np.sqrt(freqs), freqs+np.sqrt(freqs), alpha=0.2, color=colors[str(type(model))][dataset], edgecolor=None, step='post', hatch='//')
            else:
                l = ['background', 'signal']
                for n in [0, 1]:
                    args['linestyle'] = self.linestyles[l[n]]
                    args['label'] = f"{model_names[i]} {dataset} {l[n]}"
                    args['color'] = colors[str(type(model))][dataset]
                    predictions = self.get_model_predictions(model, dataset, keep_only=n)
                    freqs = ax.hist(predictions, **args)[0]
                    if not hide_errors:
                        ax.fill_between(self.nbins[:-1], freqs-np.sqrt(freqs), freqs+np.sqrt(freqs), alpha=0.2, color=colors[str(type(model))][dataset], edgecolor=None, step='post', hatch='//')
                
        ax.set_ylim(bottom=0)
        plt.ylabel('Normalised Frequency', horizontalalignment='right', y=1.0, fontsize=14, **fonts)
        plt.xlabel('Probability', horizontalalignment='right', x=1.0, fontsize=14, **fonts)
        plt.title(f'Prob. Dist. for Model(s) {model_names}')
        plt.legend(loc='upper center', ncol=1, fancybox=False, shadow=True, frameon=False)
        plt.tight_layout()
        if self.savefig != None:
            plt.savefig(f'{self.savefig}/{model_names}_{dataset}_probdist.png')
        plt.show()
        
        
    def plot_multiple(self, model, model_name):
        
        
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        import numpy as np
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        fig.patch.set_facecolor('#FFFFFF')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        datasets = ['train', 'val', 'test']
        fonts = {'fontname': 'Arial'}
        
        for i, j in enumerate(datasets):
        
            args = {
                'bins': self.nbins,
                'histtype': 'step',
                'label': j.capitalize(),
                'density': True,
                'color': self.colors[str(type(model))][j],
                'linewidth': 2,
                'linestyle': 'solid'
            }
        
            predictions = self.get_model_predictions(model, j)
            freqs = ax.hist(predictions, **args)[0]
            ax.fill_between(self.nbins[:-1], freqs-np.sqrt(freqs), freqs+np.sqrt(freqs), alpha=0.1, color=self.colors[str(type(model))][j], edgecolor=None, step='post', hatch='//')
    
        ax.set_ylim(bottom=0)
        plt.ylabel('Normalised Frequency', horizontalalignment='right', y=1.0, fontsize=14, **fonts)
        plt.xlabel('Probability', horizontalalignment='right', x=1.0, fontsize=14, **fonts)
        plt.title(f'Prob. Dist. for Model {model_name}')
        plt.legend(loc='upper center', ncol=1, fancybox=False, shadow=True, frameon=False)
        plt.tight_layout()
        if self.savefig != None:
            plt.savefig(f'{self.savefig}/{model_name}_probdist.png')
        plt.show()


class PunziMetric:
    # Make plots for the Punzi metric 
    
    def __init__(self, train, val, test, probability_space, npoints=500, significance=5, savefig=None):
        import numpy as np
        self.data = {'train': train, 'val': val, 'test': test}
        self.savefig = savefig
        self.significance = significance
        self.probability_space = np.linspace(probability_space[0], probability_space[1], npoints)

        
    def set_savefig(self, savefig):
        self.savefig = savefig
    
    def get_model_predictions(self, model, dataset, keep_only=False):
        """
        Generate a single numpy array of probabilities. That is the probability
        of an event to be a signal event.
        """
        
        data = self.data[dataset].copy()
        if keep_only != False:
            data = data[data['category'] == keep_only]
        data.drop(['category'], axis=1, inplace=True)
        
        if str(type(model)) == "<class 'tensorflow.python.keras.engine.sequential.Sequential'>":
            predictions = model.predict(data)
        elif str(type(model)) in ["<class 'xgboost.sklearn.XGBClassifier'>", "<class 'sklearn.neighbors._classification.KNeighborsClassifier'>"]:
            predictions = model.predict_proba(data)[:,1]
        else:
            print(f"MODEL TYPE: {type(model)}")
            print("FAULT: Unable to generate predictions from unknown model type")
        return predictions
    
    def plot_singular(self, model, model_labels, dataset):
        import numpy as np
        
        labels = self.data[dataset]
        labels = labels['category'].to_numpy()
        punzis = {}
        
        for i, m in enumerate(list(model)):
            nsignal_before = self.data[dataset]['category'].value_counts()[1]
            # How many signal events are in the original data sample
            
            feature_cols = self.data[dataset].columns.to_list()
            feature_cols.remove('category')
            # Get all the columns with features and not the category
            predictions = self.get_model_predictions(m, dataset)
            # Make predictions on the labelled data using the trained MVA model
            
            ps = []
            for p in self.probability_space:
                # Iterate through each point in probability space
                
                # Each event is predicted a probability P. If P < p we say that,
                # it is background, else it is signal. I.e. p -> 1 in limit of high
                # background rejection, high purity sample 
                
                preds = np.squeeze(np.where(predictions < p, 0, 1))
                all = np.array([preds, labels]).T
                
                efficiency = len(np.squeeze(np.where((all == [1, 1]).all(axis=1)))) / nsignal_before
                # Truth matched signal i.e. model said it was signal and it actually was 
                
                background = len(np.squeeze(np.where((all == [1, 0]).all(axis=1))))
                # background you thought was signal, so did not reject i.e. it got through 
                
                #print(f'{p:.4f} {efficiency:.5f} {background}')
                # Background is how much background is left after the cut i.e. how much REAL background is left (incorrectly identified as signal)
                
                punzi = efficiency / (self.significance/2 + np.sqrt(background))
                # Calculate the Punzi as defined by LHCb 
                ps.append(punzi)
            punzis[model_labels[i]] = ps
                
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        import numpy as np
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        fig.patch.set_facecolor('#FFFFFF')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        fonts = {'fontname': 'Arial'}
        print(punzis.keys())
        for i, m in enumerate(model):
            ax.scatter(self.probability_space, punzis[model_labels[i]], label=model_labels[i], s=2)
        
        plt.ylabel('Normalised Frequency', horizontalalignment='right', y=1.0, fontsize=14, **fonts)
        plt.xlabel('Probability', horizontalalignment='right', x=1.0, fontsize=14, **fonts)
        plt.title(f'Punzi Scan for Model {model_labels}/{dataset}')
        plt.legend(loc='upper center', ncol=1, fancybox=False, shadow=True, frameon=False)
        plt.tight_layout()
        if self.savefig != None:
            plt.savefig(f'{self.savefig}/{model_labels[0]}_probdist.png')
        plt.show()