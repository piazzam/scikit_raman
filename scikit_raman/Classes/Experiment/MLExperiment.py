import pickle
import scikit_raman.Classes.Experiment.ml_model_sklearn as ml_model_sklearn
import scikit_raman.Classes.Experiment.ExperimentBase as ExperimentBase
import scikit_raman.Classes.Preprocessing.Processor as preprocessing
import scikit_raman.Classes.MLModel as MLModel
import sklearn.decomposition as decomposition
import scikit_raman.Classes.Experiment.utils as utils
import os

class MLExperiment(ExperimentBase.Experiment):
	"""
	A class used to automatically execute a ML experimentation. Children class of ExperimentBase.Experiment.

	Attributes
	----------

	Methods
	-------
	experiment(self)
		Execute the defined experimentation.
	"""

	def __init__(self, configuration_file):
		"""
		Constructor for class MLExperiment

		Parameters
		----------
		configuration_file : string
			Configuration file.
		"""
		super().__init__(configuration_file)
		if self.configurations['experiment_type'] == 'dl':
			raise Exception("Try to perform a DL experiments on a ML object")

	def _load_model(self):
		"""
		Load the model defined in the configuration file.

		Returns
		-------
		scikit-learn
			scikit-learn object corresponding to the desired model.
		"""
		if self.configurations['model_name'] == 'load_model':
			_, file_extension = os.path.splitext(self.configurations['model_path'])
			if file_extension == '.pkl':
				with open(self.configurations['model_path'], 'rb') as f:
					model = pickle.load(f)
					if not isinstance(model, MLModel.MLModel):
						model = MLModel.MLModel(model)
			else:
				raise Exception("File type not supported")
		else:
			model = ml_model_sklearn.name_to_object(self.configurations['model_path'], self.configurations['seed'])
		return model

	def experiment(self):
		"""
		Execute the experiment defined in configuration file.
		"""
		ds = self._load_dataset()
		model = self._load_model()
		experiment_with_preprocessing = self.configurations['with_preprocessing']
		proc = preprocessing.Processor(ds)
		if experiment_with_preprocessing:
			preprocessing_steps = self._load_preprocessing()
			for step in preprocessing_steps['preprocessing_steps']:
				if step in preprocessing_steps['parameters']:
					func = getattr(proc, step)
					func(**preprocessing_steps['parameters'][step])
				else:
					func = getattr(proc, step)
					func()
		experiment_with_drugs = self.configurations['with_drugs']
		if experiment_with_drugs:
			df_drugs, ds_drugs = self._load_drugs()
			drug_names = self.configurations['drug_names']
			ds.load_drugs(df_drugs, drug_names)
			polvere = self.configurations['polvere']
			if polvere:
				drugs = utils.extract_drugs_list_polvere(ds_drugs)
			else:
				drugs = utils.extract_drugs_list_fisio(ds_drugs)
			proc.remove_drugs(drugs, drug_names)
		experiment_with_pca = self.configurations['with_pca']
		if experiment_with_pca:
			n_components = self.configurations['pca_components']
			pca = decomposition.PCA(n_components=n_components)
			result = pca.fit_transform(ds.spectra)
			ds.spectra = result
		evaluation_procedure = self.configurations['evaluation_procedure']
		if evaluation_procedure == 'k_fold':
			r = self._k_fold(ds, model)
			self._store_results(r)
		elif evaluation_procedure == 'loocv':
			r = self._loocv(ds, model)
			self._store_results(r)
		else:
			raise ("Evaluation procedure not defined")