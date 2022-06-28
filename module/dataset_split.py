from sklearn.model_selection import  StratifiedKFold, GroupKFold, LeaveOneGroupOut, GroupShuffleSplit

def k_fold(k, df):
    x_train = df['spectra'].values
    if 'label' not in df.columns:
        raise Exception("This dataframe doesn't have a label column")
    else:
        y_train = df['spectra']
        groups = df['user'].values
        folds = list(GroupKFold(n_splits = k).split(x_train, y_train, groups=groups))
        return folds
    
def leave_one_patient_cv(df):
    x_train = df['spectra'].values
    if 'label' not in df.columns:
        raise Exception("This dataframe doesn't have a label column")
    else:
        y_train = df['spectra']
        groups = df['user'].values
        folds = list(LeaveOneGroupOut().split(x_train, y_train, groups = groups))
        return folds
        