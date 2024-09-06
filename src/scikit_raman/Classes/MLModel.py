import copy

import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scikit_raman.module.utility as utils
import pickle
import os

class MLModel:
    """
    A class used to represent a Machine Learning model.

    ...

    Attributes
    ----------
    model : sklearn
        A Machine Learning model defined with sklearn library.

    Methods
    -------
    train_model_cv(self, dataset, k = 10, fold_level = True, get_patient_prediction = True, check_users_separated=True,
                       set_seed=True, random_state=42, save_model=False, model_path="model_saved/model/")
        Train a model with k-fold cross validation and stores the results

    train_model_leave_one_patient_out(self, dataset, get_patient_prediction = True, patient_level = True, set_seed=True,
                                          random_state=42, save_model=False, model_path="model_saved/model/")
        Train a model with leave-one-patient-out cross-validation and stores the results

    fit_model(self, X_train, y_train, set_seed=True, random_state=42)
        Given a training set it trains the model

    test_model(self, X_test)
        Given a test set it test the model and return the predictions
    """

    def __init__(self, model):
        """
        Creates an MLModel object

        Parameters
        ----------
        model: sklearn object
            A sklearn object to be trained and experimented.
        """
        self.model = model

    def train_model_cv(self, dataset, k = 10, fold_level = True, get_patient_prediction = True, check_users_separated=True,
                       set_seed=True, random_state=42, save_model=False, model_path="model_saved/model/"):
        """
        It trains a sklearn object with k-fold cross validation and it returns the results as a dictionary.
        Parameters
        ----------
        dataset : scikit_raman.Dataset
            Dataset object on which apply the k-fold
        k : int, optional (default is 10)
            Number of folds
        fold_level : bool, optional (default is True)
            Whether to return results at a fold level
        get_patient_prediction : bool, optional (default is True)
            Whether to return results at a spectra level
        check_users_separated : bool, optional (default is True)
            Whether to check if spectra of the same patients are separated among the folds
        set_seed : bool, optional (default is True)
            Whether to set a seed
        random_state : int, optional (default is 42)
            Seed value
        save_model : bool, optional (default is False)
            Whether to store the trained model
        model_path : string, optional (default is "model_saved/model/")
            Path to which store the trained model

        Returns
        -------
        dict
            a dictionary which contains the results of experimentation divided between the different types of
            results

        """
        if set_seed:
            utils.set_seed(random_state)
        folds = dataset.k_fold(k)
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        fold_pred_list = []
        fold_label_list = []
        fold_names_list = []
        names_list = []
        total_users = np.unique(dataset.user)
        for j, (train_idx, test_idx) in enumerate(folds):
            trained_model = copy.deepcopy(self.model)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            users_train = np.unique(dataset.user[train_idx])
            user_test = np.unique(dataset.user[test_idx])
            if check_users_separated:
                if len(users_train) + len(user_test) > len(total_users):
                    raise Exception("Mixed train and test")
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            names_list.append(np.unique(names_test_cv))
            trained_model.fit(X_train_cv, y_train_cv)
            y_pred = trained_model.predict(X_test_cv)
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
                if save_model:
                    try:
                        with open(model_path + "fold_"+str(j) + ".pkl", 'wb') as outp:
                            pickle.dump(trained_model, outp, pickle.HIGHEST_PROTOCOL)
                    except FileNotFoundError:
                        os.makedirs(model_path, exist_ok=True)
                        with open(model_path + "fold_" + str(j) + ".pkl", 'wb') as outp:
                            pickle.dump(trained_model, outp, pickle.HIGHEST_PROTOCOL)
        dictionary = {}
        if fold_level:
            nested_dictionary = {'pred_list': fold_pred_list, 'label_list': fold_label_list,
                                 'names_list': fold_names_list}
            dictionary['fold_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        return dictionary

    def train_model_leave_one_patient_out(self, dataset, get_patient_prediction = True, patient_level = True, set_seed=True,
                                          random_state=42, save_model=False, model_path="model_saved/model/"):
        """
        It trains a sklearn object with leave-one-patient cross validation and it returns the results as a dictionary.
        Parameters
        ----------
        dataset : scikit_raman.Dataset
            Dataset object on which apply the k-fold
        get_patient_prediction : bool, optional (default is True)
            Whether to return results at a spectra level
        patient_level : bool, optional (default is True)
            Whether to return results at a patient level
        set_seed : bool, optional (default is True)
            Whether to set a seed
        random_state : int, optional (default is 42)
            Seed value
        save_model : bool, optional (default is False)
            Whether to store the trained model
        model_path : string, optional (default is "model_saved/model/")
            Path to which store the trained model

        Returns
        -------
        dict
            a dictionary which contains the results of experimentation divided between the different types of
            results

        """
        if set_seed:
            utils.set_seed(random_state)
        folds = dataset.leave_one_patient_cv()
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        pat_pred_list = []
        pat_label_list = []
        pat_names_list = []
        names_list = []
        for j, (train_idx, test_idx) in enumerate(folds):
            trained_model = copy.deepcopy(self.model)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            names_list.append(names_test_cv)
            trained_model.fit(X_train_cv, y_train_cv)
            y_pred = trained_model.predict(X_test_cv)
            if get_patient_prediction:
                tot_pred_list.extend(y_pred)
                tot_label_list.extend(y_test_cv)
                tot_names_list.extend(names_test_cv)
            if patient_level:
                counts = np.bincount(y_pred)
                pat_pred_list.append(np.argmax(counts))
                pat_label_list.append(y_test_cv[0])
                pat_names_list.append(np.unique(names_test_cv)[0])
            if save_model:
                try:
                    with open(model_path + str(names_test_cv[0]) + ".pkl", 'wb') as outp:
                        pickle.dump(trained_model, outp, pickle.HIGHEST_PROTOCOL)
                except FileNotFoundError:
                    os.makedirs(model_path, exist_ok=True)
                    with open(model_path + str(names_test_cv[0]) + ".pkl", 'wb') as outp:
                        pickle.dump(trained_model, outp, pickle.HIGHEST_PROTOCOL)
        dictionary = {}
        if patient_level:
            nested_dictionary = {'pred_list': pat_pred_list, 'label_list': pat_label_list, 'names_list': pat_names_list}
            dictionary['patient_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        return dictionary

    def fit_model(self, X_train, y_train, set_seed=True, random_state=42):
        """
        Train the model on a given training set.

        Parameters
        ----------
        X_train : np.array
            Array of spectra representing the test set
        y_train : np.array
            Array of numerical labels of the training set
        set_seed : bool, optional (default is True)
            Whether to set a seed value
        random_state : int, optional (default is 42)
            The value of the random seed
        """
        if set_seed:
            utils.set_seed(random_state)
        self.model.fit(X_train, y_train)

    def test_model(self, X_test):
        """
        Test the model on a given test set.

        Parameters
        ----------
        X_test : np.array
            Array of spectra representing the test set

        Returns
        -------
        np.array
            Array of numerical labels predicted by the model
        """
        y_pred = self.model.predict(X_test)
        return y_pred

    def _grid_search_parameters(self, dataset, space, k_fold = True, k = 10, scoring = 'accuracy', n_jobs = 1):
        """
        It apply the grid-search strategy for the optimization of hyperparameters.

        Parameters
        ----------
        dataset : scikit_raman.Dataset
            Dataset on which apply the grid-search
        space : dict
            Dictionary contains str as keys and list of possible parameters. For more information see documentation
            of GridSearchCV function of scikit-learn.
        k_fold : bool, optional (default is True)
            Whether to compute the grid-search in k-fold cross validation or leave-one-patient cross validation
        k : int, optional (default is 10)
            Number of folds for the k-fold cross validation process
        scoring : string, optional (default is 'accuracy')
            Metric to be monitored for the k-fold function
        n_jobs : int, optional (default is 1)
            Number of jobs to run in parallel

        Returns
        -------
        float
            Mean cross-validated score of the best_estimator
        dict
            Parameter setting that gave the best results on the hold out data.

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

    def _random_search_parameters(self, dataset, space, k_fold = True, k = 10, scoring = 'accuracy', n_jobs = 1):
        """
        It apply the random search strategy for the optimization of hyperparameters.

        Parameters
        ----------
        dataset : scikit_raman.Dataset
            Dataset on which apply the grid-search
        space : dict
            Dictionary contains str as keys and list of possible parameters. For more information see documentation
            of GridSearchCV function of scikit-learn.
        k_fold : bool, optional (default is True)
            Whether to compute the grid-search in k-fold cross validation or leave-one-patient cross validation
        k : int, optional (default is 10)
            Number of folds for the k-fold cross validation process
        scoring : string, optional (default is 'accuracy')
            Metric to be monitored for the k-fold function
        n_jobs : int, optional (default is 1)
            Number of jobs to run in parallel

        Returns
        -------
        float
            Mean cross-validated score of the best_estimator
        dict
            Parameter setting that gave the best results on the hold out data.

        """
        if k_fold:
            folds = dataset.k_fold(k)
        else:
            folds = dataset.leave_one_patient_cv()
        search = RandomizedSearchCV(self.model, space, scoring=scoring, n_jobs=n_jobs, cv=folds, verbose=3)
        result = search.fit(dataset.spectra, dataset.labels)
        print('Best score: ', result.best_score_)
        print('Best params: ' + str(result.best_params_))
        return result.best_score_, result.best_params_