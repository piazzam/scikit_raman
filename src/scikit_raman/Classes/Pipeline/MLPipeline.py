import scikit_raman.Classes.Pipeline.PipelineBase as PipelineBase
import scikit_raman.Classes.Preprocessing.Processor as preprocessing
import scikit_raman.Classes.Pipeline.ml_model_sklearn as ml_model_sklearn
import scikit_raman.Classes.Experiment.utils as utils
import pickle
import sklearn.decomposition as decomposition

class MLPipeline(PipelineBase.PipelineBase):

    def __init__(self, configuration_file):
        super().__init__(configuration_file)
        if self.configurations['experiment_type'] == 'dl':
            raise Exception("Try to perform a DL pipeline on a ML object")

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

    def pipeline(self):
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
        p = self.predict(ds, model)
        return p