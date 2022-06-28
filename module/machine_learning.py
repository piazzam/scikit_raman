from scikit_raman.module.utility import spectra_to_numpy
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def cv_model(df, model, folds):
    tot_pred_list = []
    tot_label_list = []
    tot_name_list = []
    spectra = spectra_to_numpy(df)
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train_cv = spectra[train_idx]
        X_test_cv = spectra[train_idx]
        y_train_cv = df.label.values[train_idx]
        names_train_cv = df.user.values[train_idx]
        y_test_cv = df.label.values[test_idx]
        names_test_cv = df.user.values[test_idx]
        tot_name_list.append(names_test_cv)
        print(f"{j}-th fold, current patient: ",np.unique(names_test_cv))
        #model.fit(X_train_cv, y_train_cv_cat)
        
        y_pred = model.predict(X_test_cv)
        #y_pred = np.argmax(pred, axis =1)
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
    
def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    #y_pred = np.argmax(pred, axis =1)
    print(y_pred)
    print(y_test)
    
    cm_model = confusion_matrix(y_pred, y_test)
    report = classification_report(y_pred, y_test)
    
    print(cm_model)
    print(report)