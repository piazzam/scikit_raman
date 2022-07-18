import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

class MLModel:
    """
    A class that represent a MLModel for raman spectra analysis

    ...

    Attributes
    ----------
    model: sklearn model
        The model used for classification - training. This class is compatible with
        sklearn models.
    """

    def __init__(self, model):
        self.model = model

    def train_model_cv(self, dataset, k = 10, fold_level = True, get_patient_prediction = True):
        """
        Train a model with k-fold cross validation. Print the confusion matrix and the performances
        at every fold and after all folds.
        :param dataset: scikit_raman.Dataset
            A dataset object from scikit_raman class
        :param k: int
            Number of folds. The default value is 10.
        """
        folds = dataset.k_fold(k)
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        fold_pred_list = []
        fold_label_list = []
        fold_names_list = []
        names_list = []
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            names_train_cv = dataset.user[train_idx]
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            names_list.append(np.unique(names_test_cv))

            self.model.fit(X_train_cv, y_train_cv)

            y_pred = self.model.predict(X_test_cv)

            if get_patient_prediction:
                tot_pred_list.extend(y_pred)
                tot_label_list.extend(y_test_cv)
                tot_names_list.extend(names_test_cv)
            if fold_level:
                labels = []
                list_pred = []
                for el in np.unique(names_test_cv):
                    l = []
                    l_v = []
                    for i in range(len(names_test_cv)):
                        if names_test_cv[i] == el:
                            l_v.append(y_test_cv[i])
                            l.append(y_pred[i])
                    list_pred.append(l)
                    labels.append(l_v)

                patient_level = []
                for el in list_pred:
                    counts = np.bincount(el)
                    patient_level.append(np.argmax(counts))
                labels_patient_level = []
                for el in labels:
                    counts = np.bincount(el)
                    labels_patient_level.append(np.argmax(counts))
                fold_pred_list.append(patient_level)
                fold_label_list.append(labels_patient_level)
                fold_names_list.append(np.unique(names_test_cv))
        dictionary = {}
        if fold_level:
            nested_dictionary = {'pred_list': fold_pred_list, 'label_list': fold_label_list,
                                 'names_list': fold_names_list}
            dictionary['fold_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        return dictionary

    def train_model_leave_one_patient_out(self, dataset, get_patient_prediction = True, patient_level = True):
        """
        Train the model with leave_one_patient_out cross validation. Print the confusion matrix
        and the performances at every fold and after all folds.
        :param dataset: scikit_raman.Dataset
            A dataset object for traning task.
        """
        folds = dataset.leave_one_patient_cv()
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        pat_pred_list = []
        pat_label_list = []
        pat_names_list = []
        names_list = []
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            names_train_cv = dataset.user[train_idx]
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            names_list.append(names_test_cv)

            self.model.fit(X_train_cv, y_train_cv)

            y_pred = self.model.predict(X_test_cv)

            if get_patient_prediction:
                tot_pred_list.extend(y_pred)
                tot_label_list.extend(y_test_cv)
                tot_names_list.extend(names_test_cv)
            if patient_level:
                counts = np.bincount(y_pred)
                pat_pred_list.extend(np.argmax(counts))
                pat_label_list.extend(y_test_cv[0])
                pat_names_list.extend(np.unique(names_test_cv))
        dictionary = {}
        if patient_level:
            nested_dictionary = {'pred_list': pat_pred_list, 'label_list': pat_label_list, 'names_list': pat_names_list}
            dictionary['patient_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        return dictionary

    def fit_model(self, X_train, y_train):
        """
        Fit the model.
        :param X_train: np.array
            Trainset to use for training
        :param y_train: np.array
            Labels used for the training
        """
        self.model.fit(X_train, y_train)

    def test_model(self, X_test, y_test):
        """
        Test the model. Print the performances and the confusion matrix of the model.
        :param X_test: np.array
            Testset to use for the classification
        :param y_test:np.array
            Labels.
        """
        y_pred = self.model.predict(X_test)
        print(y_pred)
        print(y_test)

        cm_model = confusion_matrix(y_pred, y_test)
        report = classification_report(y_pred, y_test)

        print(cm_model)
        print(report)

    def grid_search_parameters(self, dataset, space, k_fold = True, k = 10, scoring = 'accuracy', n_jobs = 1):
        """
        Apply the grid search strategy to optimize the hyperparameters of the Machine Learning models.
        :param dataset: scikit_raman.Dataset
            Dataset on which apply the search.
        :param space: dict
            For every parameter to optimize the possible value on which apply the search.
        :param k_fold: Bool
            If true a k-fold cross validation strategy is applied. Otherwise a leave one patient out
            cross validation is applied. Default value is True.
        :param k: int
            Number of folds for the k-fold cross validation. Default value is 10.
        :param scoring: str
            Metrics on which optimize the hyperparameters. Default value is 'accuracy'
        :param n_jobs: int
            N° oj jobs to run in parallel. See sklearn docs for further information.
        :return:
            result.best_score: double
                Best score obtained by the model.
            result.best_params: dict
                Best params found by the grid search.
        """
        if k_fold:
            folds = dataset.k_fold(k)
        else:
            folds = dataset.leave_one_patient_cv()
        search = GridSearchCV(self.model, space, scoring=scoring, n_jobs=n_jobs, cv=folds, verbose=3)
        result = search.fit(dataset.spectra, dataset.labels)
        print('Best score: ', result.best_score_)
        print('Best params: ' + str(result.best_params_))
        return result.best_score_, result.best_params_