import os
import yaml
import scikit_raman.Classes.Dataset as dataset
import pandas as pd
import scikit_raman.Classes.Experiment.utils as utils
import scikit_raman.Classes.Evaluator as ev

class Experiment:
	"""
	A class used to automatically execute an experimentation.

	...

	Attributes
	----------
	configuration_file : string
		Path for configuration file.
	configurations : dict
		Dictonary obtained from .yaml configuration file.
	base_path : string
		Folder that contains all configuration files.

	Methods
	-------
	"""
	def __init__(self, configuration_file):
		"""
		Constructor for class Experiment

		Parameters
		----------
		configuration_file : string
			Path to .yaml configuration file.
		"""
		base_directory, _ = os.path.split(configuration_file)
		_, file_extension = os.path.splitext(configuration_file)
		if file_extension == '.yaml':
			self.configuration_file = configuration_file
			self.base_path = base_directory
			self.configurations = self._load_file_yaml(configuration_file)
		else:
			raise Exception("Configuration file type not supported")

	def _load_file_yaml(self, file_path):
		"""
		Load a .yaml file

		Parameters
		----------
		file_path : string
			Path to .yaml file

		Returns
		-------
		dict
			Dictionary composed of information contained in .yaml file.
		"""
		with open(file_path, 'r') as f:
			data = yaml.load(f, Loader=yaml.FullLoader)
		return data

	def _load_dataset(self):
		"""
		Read the information related to Dataset in configuration file and load a scikit_raman.Dataset object.

		Returns
		-------
		scikit_raman.Dataset
			Dataset object loaded from information in .yaml file.
		"""
		file_name = self.configurations['file_name']
		label_dictionary = self.configurations['label_dictionary']
		_, file_extension = os.path.splitext(file_name)
		file_extension = file_extension.replace('.', '')
		data = dataset.Dataset.load_file(file_extension, file_name, label_dictionary=label_dictionary)
		return data

	def _load_drugs(self):
		"""
		Read the information related drugs in configuration file and load a scikit_raman.Dataset object and the drug
		pandas.Dataframe.

		Returns
		-------
		pandas.Dataframe
			Association between patient and taken drugs.
		scikit_raman.Dataset
			Dataset composed of spectra of drugs.
		"""
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

	def _load_preprocessing(self):
		"""
		Load preprocessing configuration
		Returns
		-------
		dict
			Dictionary containing preprocessing information.
		"""
		return self._load_file_yaml(os.path.join(self.base_path,
                                          self.configurations['preprocessing_file']))

	def _k_fold(self, ds, model):
		"""
		Execute k-fold function.
		Parameters
		----------
		ds : scikit_raman.Dataset
			Dataset object.
		model : scikit_raman.MLMolde or scikit_raman.Keras.DLModelKeras
			A Machine Learning or Deep Learning model.
		Returns
		-------
		dict
			Results obtained from k-fold procedure.
		"""
		k_fold_parameters_file = os.path.join(self.base_path, self.configurations['k_fold_parameter_file'])
		# k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
		k_fold_parameters = self._load_file_yaml(k_fold_parameters_file)
		n_classes = self.configurations['n_classes']
		if k_fold_parameters is None:
			r = model.train_model_cv(ds, n_classes)
		else:
			r = model.train_model_cv(ds, n_classes, **k_fold_parameters)
		return r

	def _loocv(self, ds, model):
		"""
		Execute leave-one-patient-out cross validation.

		Parameters
		----------
		ds : scikit_raman.Dataset
			Dataset object.
		model : scikit_raman.MLMolde or scikit_raman.Keras.DLModelKeras
			A Machine Learning or Deep Learning model.
		Returns
		-------
		dict
			Results obtained from leave-one-patient-out procedure.
		"""
		loocv_fold_parameters_file = os.path.join(self.base_path, self.configurations['loocv_parameter_file'])
		# k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
		loocv_fold_parameters = self._load_file_yaml(loocv_fold_parameters_file)
		n_classes = self.configurations['n_classes']
		if loocv_fold_parameters is None:
			r = model.train_model_leave_one_patient_out(ds, n_classes)
		else:
			r = model.train_model_leave_one_patient_out(ds, n_classes, **loocv_fold_parameters)
		return r

	def _store_results(self, r):
		"""
		Execute store results part of the pipeline.
		Parameters
		----------
		r : dict
			Results dictionary
		"""
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