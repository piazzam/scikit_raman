import scikit_raman.Classes.Experiment.ExperimentBase as ExperimentBase
import scikit_raman.Classes.Keras.Model.DLModel as DLModel
import scikit_raman.Classes.Preprocessing.Processor as preprocessing
import dl_model_keras
import utils
import os
import scikit_raman.Classes.Evaluator as ev

class DLExperiment(ExperimentBase.Experiment):

    def __init__(self, configuration_file):
        super().__init__(configuration_file)
        if self.configurations['experiment_type'] == 'ml':
            raise Exception("Try to perform a ML experiments on a DL object")

    def load_model(self):
        if self.configurations['model_name'] == 'benchmark':
            model = DLModel.DLModelKeras.load_model_benchmark(self.configurations['n_dims'])
        else:
            base_model = dl_model_keras.load_model(self.configurations['model_name'])
            model_configuration_file = utils.parse_yaml_file(self.configurations['base_path_yaml']
                                                             +self.configurations['model_configuration'])
            model = DLModel.DLModelKeras(base_model, **model_configuration_file)
        return model

    def experiment(self):
        ds = self.load_dataset()
        model = self.load_model()
        experiment_with_preprocessing = self.configurations['with_preprocessing']
        proc = preprocessing.Processor(ds)
        if experiment_with_preprocessing:
            preprocessing_steps = self.load_preprocessing()
            for step in preprocessing_steps:
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
        evaluation_procedure = self.configurations['evaluation_procedure']
        if evaluation_procedure == 'k_fold':
            r = self.k_fold(ds, model)
            self.store_results(r)
        elif evaluation_procedure == 'loocv':
            r = self.loocv(ds, model)
            self.store_results(r)
        else:
            raise ("Evaluation procedure not defined")

    def k_fold(self, ds, model):
        k_fold_parameters_file = self.configurations['base_path_yaml']+self.configurations['k_fold_parameter_file']
        k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
        n_classes = self.configurations['n_classes']
        r = model.train_model_cv(ds, n_classes, **k_fold_parameters)
        return r

    def loocv(self, ds, model):
        k_fold_parameters_file = self.configurations['base_path_yaml']+self.configurations['loocv_parameter_file']
        k_fold_parameters = utils.parse_yaml_file(k_fold_parameters_file)
        n_classes = self.configurations['n_classes']
        r = model.train_model_leave_one_patient_out(ds, n_classes, **k_fold_parameters)
        return r

    def store_results(self, r):
        target_names = self.configurations['target_names']
        path = self.configurations['path_out']
        evaluator_file = self.configurations['base_path_yaml']+self.configurations['evaluator_file']
        evaluator_steps = utils.parse_yaml_file(evaluator_file)
        if not (os.path.exists(path)):
            os.mkdir(path)
        classes = self.configurations['classes']
        evaluator = ev.Evaluator(r, classes, target_names)
        for step in evaluator_steps:
            func = getattr(evaluator, step)
            func(folder_path=path)
