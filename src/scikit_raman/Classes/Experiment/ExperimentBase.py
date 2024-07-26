import os
import yaml
import scikit_raman.Classes.Dataset as dataset
import pandas as pd

class Experiment:

	def __init__(self, configuration_file):
		_, file_extension = os.path.splitext(configuration_file)
		if file_extension == '.yaml':
			self.configuration_file = configuration_file
			self.configurations = self.load_file_yaml(configuration_file)
		else:
			raise Exception("Configuration file type not supported")

	def load_file_yaml(self, file_path):
		with open(file_path, 'r') as f:
			data = yaml.load(f, Loader=yaml.FullLoader)
		return data

	def load_dataset(self):
		file_name = self.configurations['file_name']
		label_dictionary = self.configurations['label_dictionary']
		_, file_extension = os.path.splitext(file_name)
		file_extension = file_extension.replace('.', '')
		data = dataset.Dataset.load_file(file_extension, file_name, label_dictionary=label_dictionary)
		return data

	def load_drugs(self):
		farmaci_files = self.configurations['farmaci_files']
		farmaci_ds_filename = self.configurations['farmaci_ds_filename']
		df_list = []
		for file_name in farmaci_files:
			df_list.append(pd.read_csv(file_name))
		if len(df_list) == 0:
			raise Exception("No specified file paths")
		elif len(df_list) == 1:
			df_drugs = df_list[0]
		else:
			df_drugs = pd.concat(df_list, ignore_index=True)
		label_dictionary = self.configurations['label_dictionary_drugs']
		_, file_extension = os.path.splitext(farmaci_ds_filename)
		file_extension = file_extension.replace('.', '')
		ds_drugs = dataset.Dataset.load_file(file_extension, farmaci_ds_filename, '',
									 label_dictionary)
		return df_drugs, ds_drugs

	def load_preprocessing(self):
		return self.load_file_yaml(self.configurations['base_path_yaml']+
								   self.configurations['preprocessing_file'])
		#with open(self.configurations['preprocessing_file'], 'r') as f:
		#	data = yaml.load(f, Loader=yaml.FullLoader)
		#return data