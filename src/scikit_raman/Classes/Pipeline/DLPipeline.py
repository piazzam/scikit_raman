import scikit_raman.Classes.Pipeline.PipelineBase as PipelineBase
import scikit_raman.Classes.Keras.Model.DLModel as DLModel
import scikit_raman.Classes.Preprocessing.Processor as preprocessing
import scikit_raman.Classes.Pipeline.dl_model_keras as dl_model_keras
import scikit_raman.Classes.Experiment.utils as utils
import os

class DLPipeline(PipelineBase.PipelineBase):

    def __init__(self, configuration_file):
        super().__init__(configuration_file)
        if self.configurations['experiment_type'] == 'ml':
             raise Exception("Try to perform a ML pipeline on a DL object")

    def load_model(self):
        if self.configurations['model_name'] == 'benchmark':
            model = DLModel.DLModelKeras.load_model_benchmark(self.configurations['n_dims'],
                                                              number_classes=self.configurations['n_classes'])
        elif self.configurations['model_name'] == 'load_model':
            base_model = dl_model_keras.load_model(self.configurations['model_path']) #Qui credo ci sia un errore. L'opzione da riferisirsi non è model_path?
            model_configuration_file = utils.parse_yaml_file(os.path.join(self.base_path,
                                                                          self.configurations['model_configuration']))
            model = DLModel.DLModelKeras(base_model, **model_configuration_file)
        else:
            raise Exception("model_name not recognized")
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
        p = self.predict(ds, model)
        return p