from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import RocCurveDisplay, roc_curve, roc_auc_score
from tensorflow.keras.utils import to_categorical



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

    def __init__(self, results, classes, target_names = {0:'COV+', 1:'COV-', 2:'CTRL'}):
        self.results = results
        self.classes = classes
        self.target_names = target_names

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
        specificity = self.calculate_specificity_personalized(self.results['patient_level']['pred_list'],
                                                             self.results['patient_level']['label_list'])
        if len(self.classes) == 2:
            self.create_roc_curve_plot_binary(self.results['patient_level']['pred_list'],
                                                    self.results['patient_level']['label_list'],
                                                    show=show, save=save, filename=folder_path + "roc_curve_total_patient.png")
        for k in specificity.keys():
            report_patient_level[k]['specificity'] = specificity[k]
        report_patient_level['macro avg']['specificity'] = self.calculate_macro_avg(specificity)
        report_patient_level['weighted avg']['specificity'] = self.calculate_weighted_avg(specificity)
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

    def calculate_specificty(self):
        specificity = {}
        for l in self.classes:
            prec, recall, _, _ = precision_recall_fscore_support(
                np.array(self.results['total_prediction']['label_list']) == l,
                np.array(self.results['total_prediction']['pred_list']) == l,
                pos_label=True, average=None)
            specificity[str(l)] = 1-recall[0]
        return specificity

    def calculate_specificity_personalized(self, y_pred, y_true):
        specificity = {}
        for l in self.classes:
            prec, recall, _, _ = precision_recall_fscore_support(
                np.array(y_true) == l,
                np.array(y_pred) == l,
                pos_label=True, average=None)
            specificity[str(l)] = 1 - recall[0]
        return specificity

    def calculate_macro_avg(self, d):
        l = d.values()
        return sum(l) / len(l)

    def calculate_weighted_avg(self, d):
        weights = {}
        count = {}
        for el in self.classes:
            weights[str(el)] = 0
            count[str(el)] = 0
        total_labels = self.results['total_prediction']['label_list']
        total_labels_number = len(total_labels)
        for el in total_labels:
            count[str(el)] += 1
        for el in weights.keys():
            weights[el] = count[el] / total_labels_number
        for el in d.keys():
            d[el] = weights[el] * d[el]
        l = d.values()
        return sum(l)# / len(l)
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
        if len(self.classes) == 2:
            self.create_roc_curve_plot_binary(self.results['total_prediction']['pred_list'],
                                                    self.results['total_prediction']['label_list'],
                                                    show=show, save=save, filename=folder_path + "roc_curve_total.png")
        specificity = self.calculate_specificty()
        for k in specificity.keys():
            report_total[k]['specificity'] = specificity[k]
        report_total['macro avg']['specificity'] = self.calculate_macro_avg(specificity)
        report_total['weighted avg']['specificity'] = self.calculate_weighted_avg(specificity)
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
        for el in np.unique(total_names_list):
            labels = []
            prediction = []
            for j in range(0, len(total_names_list)):
                if el == total_names_list[j]:
                    labels.append(self.results['total_prediction']['label_list'][j])
                    prediction.append(self.results['total_prediction']['pred_list'][j])
            cm_model = confusion_matrix(prediction, labels, labels = self.classes)
            report_model = classification_report(prediction, labels, output_dict=True, labels=self.classes)
            specificity = self.calculate_specificity_personalized(prediction, labels)
            if len(self.classes):
                self.create_roc_curve_plot_binary(prediction,
                                            labels,
                                            show=show, save=save, filename=folder_path + str(el) + '.png')
            for k in specificity.keys():
                report_model[k]['specificity'] = specificity[k]
            report_model['macro avg']['specificity'] = self.calculate_macro_avg(specificity)
            report_model['weighted avg']['specificity'] = self.calculate_weighted_avg(specificity)
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

    def save_prediction_pkl(self, folder_path = "results/"):
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        for el in self.results:
            df = pd.DataFrame.from_dict(self.results[el])
            df.to_pickle(folder_path+str(el)+'.pkl')

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
        specificity = self.calculate_specificity_personalized(predicted, labels)
        if len(self.classes) == 2:
            self.create_roc_curve_plot_binary(predicted,
                                              labels,
                                              show=show, save=save,
                                              filename=folder_path + "roc_curve_total_patient.png")
        for k in specificity.keys():
            report_patient_level[k]['specificity'] = specificity[k]
        report_patient_level['macro avg']['specificity'] = self.calculate_macro_avg(specificity)
        report_patient_level['weighted avg']['specificity'] = self.calculate_weighted_avg(specificity)
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

    def extract_patient_level_from_total(self):
        total_names_list = self.results['total_prediction']['names_list']
        labels_patient_level = []
        predictions_patient_level = []
        for el in np.unique(total_names_list):
            labels = []
            prediction = []
            for j in range(0, len(total_names_list)):
                if el == total_names_list[j]:
                    labels.append(self.results['total_prediction']['label_list'][j])
                    prediction.append(self.results['total_prediction']['pred_list'][j])
            labels_patient_level.append(max(set(labels), key=labels.count))
            predictions_patient_level.append(max(set(prediction), key=prediction.count))
        return labels_patient_level, predictions_patient_level

    def results_patient_level_from_total(self, show=True, save=True, folder_path="results"):
        total_names_list = self.results['total_prediction']['names_list']
        labels_patient_level = []
        predictions_patient_level = []
        for el in np.unique(total_names_list):
            labels = []
            prediction = []
            for j in range(0, len(total_names_list)):
                if el == total_names_list[j]:
                    labels.append(self.results['total_prediction']['label_list'][j])
                    prediction.append(self.results['total_prediction']['pred_list'][j])
            labels_patient_level.append(max(set(labels), key=labels.count))
            predictions_patient_level.append(max(set(prediction), key=prediction.count))
        cm_model = confusion_matrix(predictions_patient_level, labels_patient_level, labels=self.classes)
        report_model = classification_report(predictions_patient_level, labels_patient_level, output_dict=True, labels=self.classes)
        specificity = self.calculate_specificity_personalized(predictions_patient_level, labels_patient_level,)
        if len(self.classes) == 2:
            self.create_roc_curve_plot_binary(predictions_patient_level,
                                              labels_patient_level,
                                              estimator_name='patient_classificator',
                                              show=show, save=save,
                                              filename=folder_path + str(el) + '.png'
                                              )
        for k in specificity.keys():
            report_model[k]['specificity'] = specificity[k]
        report_model['macro avg']['specificity'] = self.calculate_macro_avg(specificity)
        report_model['weighted avg']['specificity'] = self.calculate_weighted_avg(specificity)
        df_cm_model = pd.DataFrame(cm_model, index=self.classes, columns=self.classes)
        df_report_model = pd.DataFrame(report_model)
        plt.figure(figsize=(10, 7))
        cm_plot = sn.heatmap(df_cm_model, annot=True, fmt='d', annot_kws={"fontsize": 24})
        if save:
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
            df_cm_model.to_csv(folder_path + str(el) + ".csv")
            cm_plot.figure.savefig(folder_path + str(el) + ".png")
            df_report_model.to_csv(folder_path + str(el) + ".csv")
        if show:
            print("Result patient level")
            print(cm_model)
            print(classification_report(predictions_patient_level, labels_patient_level, output_dict=False, labels=self.classes))
            cm_plot.figure.show()

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

    def plot_history_binary(self, network_history, patient_name, folder_path = "results/training" , save = True, show = True):
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
        plt.plot(x_plot, network_history.history['accuracy'])
        plt.plot(x_plot, network_history.history['val_accuracy'])
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
        i = 0
        for history, patient_name in zip(histories, patient_names):
            self.plot_history(history, i, folder_path, save, show)
            i += 1

    def result_training_history_binary(self, folder_path = "results/training", save = True, show = True):
        patient_names = self.results['history']['patients']
        histories = self.results['history']['histories']
        if save:
            os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        i = 0
        for history, patient_name in zip(histories, patient_names):
            self.plot_history_binary(history, i, folder_path, save, show)
            i += 1

    def create_roc_curve_plot_binary(self, y_pred, y, filename = "results/roc_auc.png", save=True, show=True):
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred)
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        if show:
            plt.show()
        if save:
            plt.savefig(filename)

    def create_roc_curve_plot_multiclass(self, y_pred, y, filename = "results/roc_auc.png", save=True, show=True):
        y_pred_prob = to_categorical(
            y_pred, num_classes=None, dtype='float32'
        )
        for i in range(len(self.classes)):
            fpr, tpr, thresh = roc_curve(y, y_pred_prob[:, i], pos_label=i)
            plt.plot(fpr, tpr, linestyle='--', label=self.target_names[i] + ' vs Rest')
        plt.plot([0, 1], [0, 1], 'k--', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        if show:
            plt.show()
        if save:
            plt.savefig(filename)