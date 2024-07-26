import pickle
import scikit_raman.Classes.Experiment.ml_model_sklearn as ml_model_sklearn
import scikit_raman.Classes.Experiment.ExperimentBase as ExperimentBase
import scikit_raman.Classes.Preprocessing.Processor as preprocessing
import scikit_raman.Classes.Evaluator as ev
import sklearn.decomposition as decomposition
import scikit_raman.Classes.Experiment.utils as utils
import os

class MLExperiment(ExperimentBase.Experiment):

	def __init__(self, configuration_file):
		super().__init__(configuration_file)
		if self.configurations['experiment_type'] == 'dl':
			raise Exception("Try to perform a DL experiments on a ML object")

	def load_model(self):
		try:
			_, file_extension = self.configurations['model_name']
			if file_extension == '.pkl':
				with open(self.configurations['model_path'], 'rb') as f:
					model = pickle.load(f)
			else:
				raise Exception("File type not supported")
		except Exception as exc:
			model = ml_model_sklearn.name_to_object(self.configurations['model_name'], self.configurations['seed'])
		return model

	def experiment(self):
		ds = self.load_dataset()
		model = self.load_model()
		experiment_with_preprocessing = self.configurations['with_preprocessing']
		proc = preprocessing.Processor(ds)
		if experiment_with_preprocessing:
			preprocessing_steps = self.load_preprocessing()
			for step in preprocessing_steps['preprocessing_steps']:
				if step in preprocessing_steps['parameters']:
					func = getattr(proc, step)
					func(**preprocessing_steps['parameters'][step])
				else:
					func = getattr(proc, step)
					func()
		experiment_with_drugs = self.configurations['with_drugs']
		if experiment_with_drugs:
			df_drugs, ds_drugs = self.load_drugs()
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
			print('ok')
			r = self.k_fold(ds, model)
			self.store_results(r)
		elif evaluation_procedure == 'loocv':
			r = self.loocv(ds, model)
			self.store_results(r)
		else:
			raise ("Evaluation procedure not defined")

	def k_fold(self, ds, model):
		k_fold_parameters_file = self.configurations['base_path_yaml']+self.configurations['k_fold_parameter_file']
		#k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
		k_fold_parameters = self.load_file_yaml(k_fold_parameters_file)
		n_classes = self.configurations['n_classes']
		if k_fold_parameters is None:
			r = model.train_model_cv(ds, n_classes)
		else:
			r = model.train_model_cv(ds, n_classes, **k_fold_parameters)
		return r

	def loocv(self, ds, model):
		loocv_fold_parameters_file = self.configurations['base_path_yaml']+self.configurations['loocv_parameter_file']
		#k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
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
		evaluator_file = self.configurations['base_path_yaml']+self.configurations['evaluator_file']
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