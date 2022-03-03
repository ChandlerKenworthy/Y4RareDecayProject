class ProbabilityDistribution:
    # A class for generating probability distributions from an underlying 
    # model and data sample
    
    def __init__(self, model_type, train, val, test, nbins=40):
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
        self.model_type = model_type        
        self.data = {'train': train, 'val': val, 'test': test}
        self.nbins = np.linspace(0, 1, nbins+1)
        self.colors = {'train': '#3366ff', 'val': '#66cc00', 'test': '#ff0000'}
        self.labels = ['Train', 'Validation', 'Test']
        self.savefig = None
        
        
    def set_savefig(self, savefig):
        self.savefig = savefig
        
        
    def set_model_type(self, type):
        self.model_type = type
        
    
    def get_model_predictions(self, model, dataset, keep_only=False):
        """
        Generate a single numpy array of probabilities. That is the probability
        of an event to be a signal event.
        """
        
        data = self.data[dataset].copy()
        if keep_only != False:
            data = data[data['category'] == keep_only]
        data.drop(['category'], axis=1, inplace=True)
        
        if self.model_type == 'NN':
            predictions = model.predict(data)
        elif self.model_type == 'SK':
            predictions = model.predict_proba(data)[:,1]
        else:
            print("FAULT: Unable to generate predictions from unknown model type")
        return predictions
    
    def plot_singular(self, model, model_name, dataset, split_bg_sig=False):
        """
        Generate a probability distribution plot for the models predicted values
        for a singular dataset for example the test set.
        
        Parameters
        ----------
        model : model
            The model the predictions are to be made from. This must already be
            trained. 
            
        dataset : string
            The dataset label. Either train, val or test
        """
        
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator
        import numpy as np
        
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        fig.patch.set_facecolor('#FFFFFF')
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        
        args = {
            'bins': self.nbins,
            'histtype': 'step',
            'label': dataset.capitalize(),
            'density': True,
            'color': self.colors[dataset],
            'linewidth': 2,
            'linestyle': 'solid'
        }
        
        fonts = {'fontname': 'Arial'}
        
        if not split_bg_sig:
            predictions = self.get_model_predictions(model, dataset)
            freqs = ax.hist(predictions, **args)[0]
            ax.fill_between(self.nbins[:-1], freqs-np.sqrt(freqs), freqs+np.sqrt(freqs), alpha=0.1, color=self.colors[dataset], edgecolor=None, step='post', hatch='//')
        else:
            l = ['background', 'signal']
            linestyles = ['solid', 'dashed']
            for i in [0, 1]:
                args['label'], args['linestyle'] = f"{dataset} {l[i]}", linestyles[i]
                predictions = self.get_model_predictions(model, dataset, keep_only=i)
                freqs = ax.hist(predictions, **args)[0]
                ax.fill_between(self.nbins[:-1], freqs-np.sqrt(freqs), freqs+np.sqrt(freqs), alpha=0.1, color=self.colors[dataset], edgecolor=None, step='post', hatch='//')
            
        ax.set_ylim(bottom=0)
        plt.ylabel('Normalised Frequency', horizontalalignment='right', y=1.0, fontsize=14, **fonts)
        plt.xlabel('Probability', horizontalalignment='right', x=1.0, fontsize=14, **fonts)
        plt.title(f'Prob. Dist. for Model {model_name}')
        plt.legend(loc='upper center', ncol=1, fancybox=False, shadow=True, frameon=False)
        plt.tight_layout()
        if self.savefig != None:
            plt.savefig(f'{self.savefig}/{model_name}_{dataset}_probdist.png')
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
                'color': self.colors[j],
                'linewidth': 2,
                'linestyle': 'solid'
            }
        
            predictions = self.get_model_predictions(model, j)
            freqs = ax.hist(predictions, **args)[0]
            ax.fill_between(self.nbins[:-1], freqs-np.sqrt(freqs), freqs+np.sqrt(freqs), alpha=0.1, color=self.colors[j], edgecolor=None, step='post', hatch='//')
    
        ax.set_ylim(bottom=0)
        plt.ylabel('Normalised Frequency', horizontalalignment='right', y=1.0, fontsize=14, **fonts)
        plt.xlabel('Probability', horizontalalignment='right', x=1.0, fontsize=14, **fonts)
        plt.title(f'Prob. Dist. for Model {model_name}')
        plt.legend(loc='upper center', ncol=1, fancybox=False, shadow=True, frameon=False)
        plt.tight_layout()
        if self.savefig != None:
            plt.savefig(f'{self.savefig}/{model_name}_probdist.png')
        plt.show()
