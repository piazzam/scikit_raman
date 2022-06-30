import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class MLModel:

    def __init__(self, model):
        self.model = model

    def train_model_cv(self, dataset, k = 10):
        folds = dataset.k_fold(k)
        tot_pred_list = []
        tot_label_list = []
        tot_name_list = []
        dataset.spectra_to_numpy()
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[train_idx]
            y_train_cv = dataset.labels[train_idx]
            names_train_cv = dataset.user[train_idx]
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            tot_name_list.append(names_test_cv)
            print(f"{j}-th fold, current patient: ", np.unique(names_test_cv))
            self.model.fit(X_train_cv, y_train_cv)

            y_pred = self.model.predict(X_test_cv)
            # y_pred = np.argmax(pred, axis =1)
            print(y_pred)
            print(y_test_cv)

            tot_pred_list.extend(y_pred)
            tot_label_list.extend(y_test_cv)
            cm_model = confusion_matrix(y_pred, y_test_cv)
            report = classification_report(y_pred, y_test_cv)

            print(cm_model)
            print(report)
        cm_tot = confusion_matrix(tot_pred_list, tot_label_list)
        print(cm_tot)
        report_tot = classification_report(tot_pred_list, tot_label_list)
        print(report_tot)

    def fit_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        print(y_pred)
        print(y_test)

        cm_model = confusion_matrix(y_pred, y_test)
        report = classification_report(y_pred, y_test)

        print(cm_model)
        print(report)