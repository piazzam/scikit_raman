import os
import yaml
import scikit_raman.Classes.Dataset as dataset
import pandas as pd
import scikit_raman.Classes.Experiment.utils as utils
import scikit_raman.Classes.Evaluator as ev

class Experiment:

	def __init__(self, configuration_file):
		base_directory, _ = os.path.split(configuration_file)
		_, file_extension = os.path.splitext(configuration_file)
		if file_extension == '.yaml':
			self.configuration_file = configuration_file
			self.base_path = base_directory
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
		return self.load_file_yaml(os.path.join(self.base_path, 
                                          self.configurations['preprocessing_file']))

	def k_fold(self, ds, model):
		k_fold_parameters_file = os.path.join(self.base_path, self.configurations['k_fold_parameter_file'])
		# k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
		k_fold_parameters = self.load_file_yaml(k_fold_parameters_file)
		n_classes = self.configurations['n_classes']
		if k_fold_parameters is None:
			r = model.train_model_cv(ds, n_classes)
		else:
			r = model.train_model_cv(ds, n_classes, **k_fold_parameters)
		return r

	def loocv(self, ds, model):
		loocv_fold_parameters_file = os.path.join(self.base_path, self.configurations['loocv_parameter_file'])
		# k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
		loocv_fold_parameters = self.load_file_yaml(loocv_fold_parameters_file)
		n_classes = self.configurations['n_classes']
		if loocv_fold_parameters is None:
			r = model.train_model_leave_one_patient_out(ds, n_classes)
		else:
			r = model.train_model_leave_one_patient_out(ds, n_classes, **loocv_fold_parameters)
		return r

	def store_results(self, r):
		target_names = self.configurations['target_names']
		path = self.configurations['path_out']
		evaluator_file = os.path.join(self.base_path, self.configurations['evaluator_file'])
		evaluator_data = utils.parse_yaml_file(evaluator_file)
		evaluator_steps = utils.parse_yaml_file(evaluator_file)['evaluator_steps']
		if not (os.path.exists(path)):
			os.mkdir(path)
		classes = self.configurations['classes']
		evaluator = ev.Evaluator(r, classes, target_names)
		for step in evaluator_steps:
			if step in evaluator_data['parameters']:
				func = getattr(evaluator, step)
				func(folder_path=path, **evaluator_data['parameters'][step])
			else:
				func = getattr(evaluator, step)
				func(folder_path=path)