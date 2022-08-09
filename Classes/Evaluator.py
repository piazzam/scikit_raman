from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import os


class Evaluator:
    """
        A class that represents an Evaluator object for MlModel and DLModel of scikit_raman.
        ...

        Attributes
        ----------
        results : dict
            dictionary returned by evaluation function of scikit_raman classes.
        classes : list
            list of the classes.
        """

    def __init__(self, results, classes):
        self.results = results
        self.classes = classes

    def total_result_patient(self, show = True, save = True, folder_path = "results/"):
        """
        Creates the total results at patient level.
        :param show: bool
            If true the results are shown.
        :param save: bool
            If true the results are save.
        :param folder_path: str
            If save parameter is true folder path on which save the results.
        :return: list
            Returns the confusion matrix and the report.
        """
        cm_patient_level = confusion_matrix(self.results['patient_level']['pred_list'],
                                            self.results['patient_level']['label_list'])
        report_patient_level = classification_report(self.results['patient_level']['pred_list'],
                                                     self.results['patient_level']['label_list'], output_dict=True,
                                                     labels=self.classes)
        df_patient_level = pd.DataFrame(cm_patient_level, index=self.classes, columns=self.classes)
        df_report = pd.DataFrame(report_patient_level)
        plt.figure(figsize=(10, 7))
        cm_plot = sn.heatmap(df_patient_level, annot=True, fmt='d', annot_kws={"fontsize":24})
        if save:
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
            df_patient_level.to_csv(folder_path + "cm_total_patient.csv")
            df_report.to_csv(folder_path + "report_total_patient.csv")
            cm_plot.figure.savefig(folder_path + "cm_total_patient.png")
        if show:
            cm_plot.figure.show()
            print(classification_report(self.results['patient_level']['pred_list'],
                                                     self.results['patient_level']['label_list'], output_dict=False,
                                                     labels=self.classes))
            print(cm_patient_level)
        return cm_patient_level, report_patient_level

    def total_result(self, show = True, save = True, folder_path = "/result"):
        """
        Creates the total results.
        :param show: bool
            If true the results are shown.
        :param save: bool
            If true the results are save.
        :param folder_path: str
            If save parameter is true folder path on which save the results.
        :return: list
            Returns the confusion matrix and the report.
        """
        cm_total = confusion_matrix(self.results['total_prediction']['pred_list'],
                                            self.results['total_prediction']['label_list'])
        report_total = classification_report(self.results['total_prediction']['pred_list'],
                                                     self.results['total_prediction']['label_list'], output_dict=True,
                                                     labels=self.classes)
        df_total = pd.DataFrame(cm_total, index=self.classes, columns=self.classes)
        df_report = pd.DataFrame(report_total)
        plt.figure(figsize=(10, 7))
        cm_plot = sn.heatmap(df_total, annot=True, fmt='d', annot_kws={"fontsize":24})
        if save:
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
            df_total.to_csv(folder_path + "cm_total.csv")
            df_report.to_csv(folder_path + "report_total.csv")
            cm_plot.figure.savefig(folder_path + "cm_total.png")
        if show:
            cm_plot.figure.show()
            print(classification_report(self.results['total_prediction']['pred_list'],
                                        self.results['total_prediction']['label_list'], output_dict=False,
                                        labels=self.classes))
            print(cm_total)
        return cm_total, report_total

    def results_every_patient(self, show = True, save = True, folder_path = "/result"):
        """
        Create the results for every patient.
        :param show: bool
            If true the results are shown.
        :param save: bool
            If true the results are save.
        :param folder_path: str
            If save parameter is true the results are save in this folder.
        :return: list
            Return the confusion matrix and the reports.
        """
        total_names_list = self.results['total_prediction']['names_list']
        i = 0
        k = 0
        for el in np.unique(total_names_list):
            labels = []
            prediction = []
            for j in range(i, len(total_names_list)):
                if el == total_names_list[j]:
                    labels.append(self.results['total_prediction']['label_list'][j])
                    prediction.append(self.results['total_prediction']['pred_list'][j])
                k = j
            i = k
            cm_model = confusion_matrix(prediction, labels)
            report_model = classification_report(prediction, labels, output_dict=True, labels=self.classes)
            df_cm_model = pd.DataFrame(cm_model, index=self.classes, columns=self.classes)
            df_report_model = pd.DataFrame(report_model)
            plt.figure(figsize=(10, 7))
            cm_plot = sn.heatmap(df_cm_model, annot=True, fmt='d', annot_kws={"fontsize":24})
            if save:
                os.makedirs(os.path.dirname(folder_path), exist_ok=True)
                df_cm_model.to_csv(folder_path + str(el) + ".csv")
                cm_plot.figure.savefig(folder_path + str(el) + ".png")
                df_report_model.to_csv(folder_path + str(el) + ".csv")
            if show:
                print("Result of patient {}" + str(el))
                print(cm_model)
                print(classification_report(prediction, labels, output_dict=False, labels=self.classes))
                cm_plot.figure.show()

    def save_prediction_csv(self, folder_path = "results/"):
        """
        Save every prediction (in any format) to csv file.
        :param folder_path: str
            Path on which save the csv files.
        """
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        for el in self.results:
            df = pd.DataFrame.from_dict(self.results[el])
            df.to_csv(folder_path+str(el)+'.csv')

    def total_result_patient(self, show = True, save = True, folder_path = "results/"):
        """
        Creates the total results at patient level.
        :param show: bool
            If true the results are shown.
        :param save: bool
            If true the results are save.
        :param folder_path: str
            If save parameter is true folder path on which save the results.
        :return: list
            Returns the confusion matrix and the report.
        """
        cm_patient_level = confusion_matrix(self.results['patient_level']['pred_list'],
                                            self.results['patient_level']['label_list'])
        report_patient_level = classification_report(self.results['patient_level']['pred_list'],
                                                     self.results['patient_level']['label_list'], output_dict=True,
                                                     labels=self.classes)
        df_patient_level = pd.DataFrame(cm_patient_level, index=self.classes, columns=self.classes)
        df_report = pd.DataFrame(report_patient_level)
        plt.figure(figsize=(10, 7))
        cm_plot = sn.heatmap(df_patient_level, annot=True, fmt='d', annot_kws={"fontsize":24})
        if save:
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
            df_patient_level.to_csv(folder_path + "cm_total_patient.csv")
            df_report.to_csv(folder_path + "report_total_patient.csv")
            cm_plot.figure.savefig(folder_path + "cm_total_patient.png")
        if show:
            cm_plot.figure.show()
            print(classification_report(self.results['patient_level']['pred_list'],
                                                     self.results['patient_level']['label_list'], output_dict=False,
                                                     labels=self.classes))
            print(cm_patient_level)
        return cm_patient_level, report_patient_level

    def store_model(self, folder_path = "results/models/"):
        """
        Store a list of model in json format.
        :param folder_path: str, optional
            folder path to which store the models.
        :return: list
            list of the saved models.
        """
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        saved_models = self.results['saved_models']['models']
        patient_names = self.results['saved_weoghts']['names_list']
        for i in range(len(saved_models)):
            with open(folder_path + "model_"+patient_names[i]+'.json', 'w') as json_file:
                json_file.write(saved_models[i])
        return saved_models

    def store_weights(self, folder_path = "results/models/"):
        """
        Store the weights of the model.
        :param folder_path: str, optional
            folder path in which store the weights.
        :return: list
            list of the saved weights.
        """
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        saved_weights = self.results['saved_weights']['weights']
        patient_names = self.results['saved_weoghts']['names_list']
        for i in range(len(saved_weights)):
            np.savetxt(folder_path+'weights_'+patient_names[i]+'.csv', saved_weights, fmt='%s', delimiter = ',')
        return saved_weights