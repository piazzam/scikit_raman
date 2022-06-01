
def svm(df, folds):
    for j, (train_idx, test_idx) in enumerate(folds):
        X_train_cv = df['spectra'].values[train_idx]
        X_test_cv = df['spectra'].values[test_idx]
        if 'label' not in df.columns:
            raise Exception("This dataframe doesn't have a label column")
        else:
            Y_train_cv = df['label'].values[train_idx]
            Y_test_cv = df['label'].values[test_idx]