import os
import yaml
import scikit_raman.Classes.Dataset as ds
import scikit_raman.Classes.Keras.DLModel as DLModel
import ml_model_sklearn
import dl_model_keras

class Experiment:

	def __init__(configuration_file):
		_, file_extension = os.path.splitext(configuration_file)
		if file_extension != '.yaml':
			raise Exception("File type not supported")
		self.configuration_file = configuration_file

	def load_file():
		with open(self.configuration_file, 'r') as f:
			data = yaml.load(f, Loader=yaml.SafeLoader)
		return data

	def load_dataset():
		file_name = self.configuration_file['file_name']
		label_dictionary = self.configuration_file['label_dictionary']
		_, file_extension = os.path.splitext(file_name)
		file_extension = file_extension.replace('.', '')
		data = ds.Dataset.load_file(file_extension, file_name, label_dictionary=label_dictionary)

	def load_model():
		experiment_type = self.configuration_file['experiment_type']
		model_name = self.configuration_file['model_name']
		if experiment_type == 'ml':
			model = ml_model_sklearn.name_to_object(model_name, self.configuration_file['seed'])
		elif experiment_type == 'dl':
			if model_name == 'benchmark':
				model = DLModel.load_model_benchmark()
			else:
				model = dl_model_keras.load_model(model_name)

	def load_drugs():
		farmaci_files = self.configuration_file['farmaci_files']
		farmaci_ds_filename = self.configuration_file['farmaci_ds_filename']
		df_list = []
		for file_name in farmaci_files:
			ds_list.append(pd.read_csv(file_name))
		if len(ds_list) == 0:
			raise Exception("No specified file paths")
		elif len(ds_list) == 1:
			df_drugs = ds_list[0]
		else:
			df_drugs = pd.concat([df_asma, df_bpco], ignore_index=True)
		label_dictionary = self.configuration_file['label_dictionary_drugs']
		_, file_extension = os.path.splitext(farmaci_ds_filename)
		file_extension = file_extension.replace('.', '')
		ds_drugs = Dataset.load_file(file_extension, farmaci_ds_filename, '',
                             label_dictionary)
		return df_drugs, ds_drugs

	def experiment():
		ds = load_dataset()
		model = load_model()
		experiment_with_preprocessing = self.configuration_file['with_preprocessing']
		if experiment_with_preprocessing:
			processor = Processor(ds)
			list_preprocessing = self.configuration_file['with_preprocessing']
			for step in list_preprocessing:
				processor.step()
		experiment_with_drugs = self.configuration_file['with_drugs'] 
		if experiment_with_drugs:
			df_drugs, ds_drugs = load_drugs()
			drug_names = self.configuration_file['drug_names']
			ds.load_drugs(df_drugs, drug_names)
			polvere = self.configuration_file['polvere']
			if polvere:
				drugs = extract_drugs_list_polvere(ds_drugs)
			else:
				drugs = extract_drugs_list_fisio(ds_drugs)
			processor.remove_drugs(drugs, drug_names)
		experiment_with_pca = self.configuration_file['with_pca']
		if experiment_with_pca:
			n_components = self.configuration_file['pca_components']
			pca = PCA(n_components=n_components)
        	result = pca.fit_transform(ds.spectra)
        	ds.spectra = result
        evaluation_type = self.configuration_file['evaluation_type']
        if evaluation_type == "k-fold":
        	number_classes = self.configuration_file['number_classes']
        	k = self.configuration_file['k']
        	if 'fold_level' in self.configuration_file:
        		fold_level=self.configuration_file['fold_level']
        	else:
        		fold_level = True
        	if 'get_patient_prediction' in self.configuration_file:
        		get_patient_prediction = self.configuration_file['get_patient_prediction']
        	else:
        		get_patient_prediction = True
        	if self.configuration_file['experiment_type'] == 'dl' and 'return_history' in self.configuration_file: 
        		return_history=self.configuration_file['return_history']
        	else:
        		return_history = True
        	if self.configuration_file['experiment_type'] == 'dl' and 'data_augmentation' in self.configuration_file:
            	data_augmentation = self.configuration_file['data_augmentation']
            	if 'f_name' in self.configuration_file:
            		f_name = self.configuration_file['f_name']
            	else:
            		f_name = 'emsc'
            	if 'f_params' in self.configuration_file:
            		f_params = self.configuration_file['f_params']
            	else:
            		f_params = None
            else:
            	data_augmentation=False
            	emsc = ''
            	f_params = None
            if self.configuration_file['experiment_type'] == 'dl' and 'save_model' in self.configuration_file:
            	save_model = self.configuration_file['save_model']
            	if 'model_path' in self.configuration_file['model_path']:
            		model_path = self.configuration_file['model_path']
            	else:
            		model_path = ''
            else:
            	save_model = False
            	model_path = ''
            if self.configuration_file['experiment_type'] == 'dl' and 'save_weights' in self.configuration_file:
            	save_weights = self.configuration_file['save_weights']
            	if 'weights_path' in self.configuration_file['weights_path']:
            		weights_path = self.configuration_file['weights_path']
            	else:
            		weights_path = ''
            else:
            	save_weights = False
            	weights_path = ''
            random_state=self.configuration_file['seed'] 
            check_users_separated=True
            if check_users_separated:
        	r = model.train_model_leave_one_patient_out(ds, number_classes, k=k, )


















