import pandas as pd
import scikit_raman.preprocessing as preprocessing
import scikit_raman.feature_reduction as f_reduction
import matplotlib.pyplot as plt

class PipelineClass:
    """
    A class used to represent a unique pipeline.
    
    Attributes
    --------
    
    file_type: str
        A string that represent the type of the input file.
    input_file: str
        A string that represent the path to access the input file.
    preprocessing_step: list
        A list of strings with the name of preprocessing function to apply. 
        The names must be taken from scikit_raman preprocessing.
    feature_reduction_steps: list
        A list of strings with the name of feature reduction function to apply. 
        The names must be taken from scikit_raman feature reduction.
    classificator: string
        A string that represent the name of the classificator to use. The name
        must be taken from scikit_raman machine_learning or deep_learning.
    input_function: dict
        A dictionary formatted followed our policy. It contains the input 
        variable for every funciton if the user want to change the default 
        values.  
    """
    
    def __init__(self, file_type, input_file, preprocessing_steps, feature_reduction_steps, classificator, input_function):
        self.file_type = file_type
        self.input_file = input_file
        self.preprocessing_steps = preprocessing_steps
        self.feature_reduction_steps = feature_reduction_steps
        self.classificator = classificator
        self.input_function = input_function
        self.load_file()
        
    def set_preprocessing_steps(self, preprocessing_steps):
        """
        Set the values of preprocessing steps.

        Parameters
        ----------
        preprocessing_steps : list
            A list of strings with the name of preprocessing function to apply. 
            The names must be taken from scikit_raman preprocessing.

        Returns
        -------
        None.

        """
        self.preprocessing_steps = preprocessing_steps
        
    def set_input_file(self, input_file):
        """
        Set the values of feature reduction steps.

        Parameters
        ----------
        input_file : str
            A string that represent the path to access the input file.

        Returns
        -------
        None.

        """
        self.input_file = input_file
        
    def set_file_type(self, file_type):
        """
        Set the values of file type variable.

        Parameters
        ----------
        file_type : str
            A string that represent the type of the input file.

        Returns
        -------
        None.

        """
        self.file_type = file_type
        
    def set_feature_reduction(self, feature_reduction):
        """
        Set the values of feature reduction variable.

        Parameters
        ----------
        feature_reduction : list
            A list of strings with the name of feature reduction function to 
            apply. The names must be taken from scikit_raman feature reduction.

        Returns
        -------
        None.

        """
        self.feature_reduction = feature_reduction
    
    def set_classificator(self, classificator):
        """
        Set the values of classificator variable.

        Parameters
        ----------
        classificator : str
        A string that represent the name of the classificator to use. The name
        must be taken from scikit_raman machine_learning or deep_learning.

        Returns
        -------
        None.

        """
        self.classificator = classificator
    
    def set_input_function(self, input_function):
        """
        Set the values of input function variable

        Parameters
        ----------
        input_function : dict
            A dictionary formatted followed our policy. It contains the input 
            variable for every funciton if the user want to change the default 
            values.

        Returns
        -------
        None.

        """
        self.input_function = input_function
        
    def load_file(self):
        """
        load file for input in df variable

        Returns
        -------
        None.

        """
        if self.file_type == 'csv':
            self.df = pd.read_csv(self.input_file)
        elif self.file_type == 'pkl':
            self.df = pd.read_pickle(self.input_file)
        else:
            print("File type not supported")
            
    def get_df(self):
        """
        Return df variable

        Returns
        -------
        pd.DataFrame
            DataFrame represent input of the pipeline.

        """
        return self.df

    def add_labels(self, dictionary):
        """
        Add labels to the dataframe.

        Parameters
        ----------
        dictionary : dict
            A dictionary that mapped category to label.

        Returns
        -------
        None.

        """
        self.df = preprocessing.category_to_label(self.df, dictionary)
        
    def plot_spectra_matplotlib(self):
        """
        Plot all spectra of the dataframe with matplotlib.

        Returns
        -------
        None.

        """
        for i in range(len(self.df)):
            x = self.df.iloc[i]['x-axis']
            y = self.df.iloc[i]['spectra']
            plt.plot(x,y)
            
    def plot_one_spectra_matplotlib(self, i):
        """
        Plot spectra of one patient with matplotlib.

        Parameters
        ----------
        i : int
            index of the spectra to plot.

        Returns
        -------
        None.

        """
        x = self.df.iloc[i]['x-axis']
        y = self.df.iloc[i]['spectra']
        plt.plot(x,y)
        
    def execute_preprocessing(self):
        """
        Execute all preprocessing steps.

        Returns
        -------
        None.

        """
        for step in self.preprocessing_steps:
            if hasattr(preprocessing, step):
                input_string = '(self.df'
                if step in self.input_function:
                    names = self.input_function[step].keys()
                    values = self.input_function[step].values()
                    for n,v in zip(names, values):
                        input_string = input_string + ',' +str(n)+'='+str(v)
                input_string = input_string + ')'
                step = "preprocessing."+step
                self.df = eval(step+input_string)
            else:
                print("Error %s is not in the preprocessing steps." %(step))
                
    def execute_feature_reduction(self):
        """
        Execute all feature reduction steps.

        Returns
        -------
        None.

        """
        for step in self.feature_reduction_steps:
            if hasattr(f_reduction, step):
                input_string = '(self.df'
                if step in self.input_function:
                    names = self.input_function[step].keys()
                    values = self.input_function[step].values()
                    for n,v in zip(names, values):
                        input_string = input_string + ',' +str(n)+'='+str(v)
                input_string = input_string + ')'
                step = "f_reduction."+step
                self.df = eval(step+input_string)
            else:
                print("Error %s is not in the feature reduction steps." %(step))
        
                