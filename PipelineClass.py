import pandas as pd
import scikit_raman.preprocessing as preprocessing
import scikit_raman.feature_reduction as f_reduction

class PipelineClass:
    
    def __init__(self, file_type, input_file, preprocessing_steps, feature_reduction_steps, classificator, input_function):
        self.file_type = file_type
        self.input_file = input_file
        self.preprocessing_steps = preprocessing_steps
        self.feature_reduction_steps = feature_reduction_steps
        self.classificator = classificator
        self.input_function = input_function
        self.load_file()
        
    def set_preprocessing_steps(self, preprocessing_steps):
        self.preprocessing_steps = preprocessing_steps
        
    def set_input_file(self, input_file):
        self.input_file = input_file
        
    def set_file_type(self, file_type):
        self.file_type = file_type
        
    def set_feature_reduction(self, feature_reduction):
        self.feature_reduction = feature_reduction
    
    def set_classificator(self, classificator):
        self.classificator = classificator
        
    def load_file(self):
        if self.file_type == 'csv':
            self.df = pd.read_csv(self.input_file)
        elif self.file_type == 'pkl':
            self.df = pd.read_pickle(self.input_file)
        else:
            print("File type not supported")
            
    def get_df(self):
        return self.df

    def add_labels(self, dictionary):
        self.df = preprocessing.category_to_label(self.df, dictionary)        
        
    def execute_preprocessing(self):
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
        
                