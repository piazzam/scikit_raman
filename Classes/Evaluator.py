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
                                            self.results['patient_level']['label_list'],
                                            labels = self.classes)
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

    def total_result(self, show = True, save = True, folder_path = "results/"):
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
                                            self.results['total_prediction']['label_list'], labels = self.classes)
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

    def results_every_patient(self, show = True, save = True, folder_path = "results/"):
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
            cm_model = confusion_matrix(prediction, labels, labels = self.classes)
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

    def total_result_patient_k_fold(self, show=True, save=True, folder_path="results"):
        predicted = []
        labels = []
        for list in self.results['fold_level']['pred_list']:
            predicted.extend(list)
        for list in self.results['fold_level']['label_list']:
            labels.extend(list)
        cm_patient_level = confusion_matrix(predicted,
                                                 labels,
                                                 labels = self.classes)
        report_patient_level = classification_report(predicted,
                                                          labels, output_dict=True,
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
            print(classification_report(predicted,
                                        labels, output_dict=False,
                                        labels=self.classes))
            print(cm_patient_level)
        return cm_patient_level, report_patient_level
    # def total_result_patient(self, show = True, save = True, folder_path = "results/"):
    #     """
    #     Creates the total results at patient level.
    #     :param show: bool
    #         If true the results are shown.
    #     :param save: bool
    #         If true the results are save.
    #     :param folder_path: str
    #         If save parameter is true folder path on which save the results.
    #     :return: list
    #         Returns the confusion matrix and the report.
    #     """
    #     cm_patient_level = confusion_matrix(self.results['patient_level']['pred_list'],
    #                                         self.results['patient_level']['label_list'],
    #                                         labels = self.classes)
    #     report_patient_level = classification_report(self.results['patient_level']['pred_list'],
    #                                                  self.results['patient_level']['label_list'], output_dict=True,
    #                                                  labels=self.classes)
    #     df_patient_level = pd.DataFrame(cm_patient_level, index=self.classes, columns=self.classes)
    #     df_report = pd.DataFrame(report_patient_level)
    #     plt.figure(figsize=(10, 7))
    #     cm_plot = sn.heatmap(df_patient_level, annot=True, fmt='d', annot_kws={"fontsize":24})
    #     if save:
    #         os.makedirs(os.path.dirname(folder_path), exist_ok=True)
    #         df_patient_level.to_csv(folder_path + "cm_total_patient.csv")
    #         df_report.to_csv(folder_path + "report_total_patient.csv")
    #         cm_plot.figure.savefig(folder_path + "cm_total_patient.png")
    #     if show:
    #         cm_plot.figure.show()
    #         print(classification_report(self.results['patient_level']['pred_list'],
    #                                                  self.results['patient_level']['label_list'], output_dict=False,
    #                                                  labels=self.classes))
    #         print(cm_patient_level)
    #     return cm_patient_level, report_patient_level

    def plot_history(self, network_history, patient_name, folder_path = "results/training" , save = True, show = True):
        x_plot = list(range(1,len(network_history.history["loss"])+1))
        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.plot(x_plot, network_history.history['loss'])
        plt.plot(x_plot, network_history.history['val_loss'])
        plt.legend(['Training', 'Validation'])

        if show:
            plt.show()
        if save:
            os.makedirs(os.path.dirname(folder_path+"/loss/"), exist_ok=True)
            plt.savefig(folder_path+"/loss/" + str(patient_name) + ".png")

        plt.figure()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.plot(x_plot, network_history.history['categorical_accuracy'])
        plt.plot(x_plot, network_history.history['val_categorical_accuracy'])
        plt.legend(['Training', 'Validation'], loc='lower right')
        if show:
            plt.show()
        if save:
            os.makedirs(os.path.dirname(folder_path + "/accuracy/"), exist_ok=True)
            plt.savefig(folder_path+"/accuracy/" + str(patient_name) + ".png")
    def result_training_history(self, folder_path = "results/training", save = True, show = True):
        patient_names = self.results['history']['patients']
        histories = self.results['history']['histories']
        if save:
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        for history, patient_name in zip(histories, patient_names):
            self.plot_history(history, patient_name, folder_path, save, show)