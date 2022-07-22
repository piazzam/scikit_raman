from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, InputLayer, Conv1D, \
    MaxPooling1D, Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import copy

class DLModelKeras:
    """
    A class to represent a DLModel object in keras.
    ...

    Attributes
    ----------
    model : tf.Keras.Model
        model to train e test.
    batch_size: int
        dimension of the batch size
    epochs: int
        number of epochs
    callbacks: tf.keras.callbacks
        callbacks to apply in the fitting fase.
    """

    def __init__(self, model, batch_size, epochs, callbacks):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks

    @classmethod
    def load_model_benchmark(dlm,  n_dims, multi_gpu = False):
        """
        Load the benchmark model.
        :param n_dims:
                int
                    input dimension of the data
        :param multi_gpu:
                bool
                    if True multi_gpu model is used.
        :return:
                DLModelKers
                    Returns a new DLModelKeras.
        """

        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
        optimizer = Adam(lr=0.00020441990333108206)

        # ----- init model
        model = Sequential()
        model.add(InputLayer(input_shape=(n_dims,)))
        model.add(Reshape((n_dims, 1)))

        # ----- CNN layers
        model.add(Conv1D(filters=100,
                         kernel_size=100,
                         strides=1,
                         padding='same',
                         activation='relu'))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
        model.add(Conv1D(filters=100,
                         kernel_size=5,
                         strides=2,
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=6,
                               strides=3,
                               padding='same'))
        model.add(BatchNormalization(momentum=0.99, epsilon=0.01))
        model.add(Conv1D(filters=25,
                         kernel_size=9,
                         strides=5,
                         padding='same',
                         activation='relu'))
        model.add(MaxPooling1D(pool_size=3,
                               strides=2,
                               padding='same'))

        # ----- Flatten layer between CNN and Dense layers
        model.add(Flatten())
        model.add(Dropout(rate=0.1))
        # ----- Dense layers
        model.add(Dense(units=732))
        model.add(LeakyReLU())
        model.add(Dropout(rate=0.7))

        model.add(Dense(units=189))
        model.add(LeakyReLU())
        model.add(Dropout(rate=0.25))

        model.add(Dense(units=152))
        model.add(LeakyReLU())
        model.add(Dropout(rate=0.1))

        # ----- Classification layer
        model.add(Dense(units=3, activation='softmax'))

        # ----- Compile
        if multi_gpu:
            model = multi_gpu_model(model, gpus=[0, 1, 2, 3])
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        epochs = 273
        batch_size = 338
        es = EarlyStopping(monitor="val_categorical_accuracy", patience=100, verbose=1,
                                           restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, verbose=4, patience=80,
                                               cooldown=10)
        callbacks = [es, lr]
        return dlm(model, batch_size, epochs, callbacks)

    def train_model_leave_one_patient_out(self, dataset, number_classes, patient_level = True, get_patient_prediction = True, return_history = True, test_size = 0.1):
        """
        Train the model with Leave One Patient Out Cross Validation
        :param dataset:
                Dataset
                    A Dataset object on which train the model.
        :param number_classes:
                int
                    A integer that represent the number of classes of the problem.
        :param patient_level:
                bool
                    If true the results are returned at patient level granularity
        :param get_patient_prediction:
                bool
                    If true every prediction of the patient is returned.
        :param return_history:
                bool
                    if true the histories of the training is returned.
        :return:
                dict
                    Returns a dictionary with the results based on the different choiches.
        """
        folds = dataset.leave_one_patient_cv()
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        pat_pred_list = []
        pat_label_list = []
        pat_names_list = []
        histories = []
        names_list = []
        for j, (train_idx, test_idx) in enumerate(folds):
            #trained_model = self.model
            trained_model = copy.deepcopy(self.model)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            y_train_cv_cat = to_categorical(y_train_cv, number_classes)
            X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train_cv, y_train_cv_cat, test_size=test_size,
                                                                    # random_state = 42,
                                                                    stratify=y_train_cv)
            history = trained_model.fit(X_train_cv, y_train_cv,
                                epochs=self.epochs,
                                validation_data=(X_val, y_val),
                                batch_size=self.batch_size, verbose=1,
                                callbacks=self.callbacks)
            histories.append(history)
            names_list.append(np.unique(names_test_cv))
            pred = trained_model.predict(X_test_cv)
            y_pred = np.argmax(pred, axis=-1)
            if get_patient_prediction:
                tot_pred_list.extend(y_pred)
                tot_label_list.extend(y_test_cv)
                tot_names_list.extend(names_test_cv)
            if patient_level:
                counts = np.bincount(y_pred)
                pat_pred_list.append(np.argmax(counts))
                pat_label_list.append(y_test_cv[0])
                pat_names_list.append(np.unique(names_test_cv))
        dictionary = {}
        if patient_level:
            nested_dictionary = {'pred_list': pat_pred_list, 'label_list':pat_label_list, 'names_list':pat_names_list}
            dictionary['patient_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        if return_history:
            nested_dictionary = {'histories':histories, 'patients':names_list}
            dictionary['history'] = nested_dictionary
        return dictionary

    def train_model_cv(self, dataset, number_classes, k = 10, fold_level = True, get_patient_prediction = True, return_history = True):
        """
                Train the model with Leave One Patient Out Cross Validation
                :param dataset:
                        Dataset
                            A Dataset object on which train the model.
                :param number_classes:
                        int
                            A integer that represent the number of classes of the problem.
                :param patient_level:
                        bool
                            If true the results are returned at patient level granularity
                :param get_patient_prediction:
                        bool
                            If true every prediction of the patient is returned.
                :param return_history:
                        bool
                            if true the histories of the training is returned.
                :return:
                        dict
                            Returns a dictionary that is different based on the choiches.
                """
        folds = dataset.k_fold(k)
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        fold_pred_list = []
        fold_label_list = []
        fold_names_list = []
        histories = []
        names_list = []
        for j, (train_idx, test_idx) in enumerate(folds):
            trained_model = copy.deepcopy(self.model)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            y_train_cv_cat = to_categorical(y_train_cv, number_classes)
            X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train_cv, y_train_cv_cat, test_size=.1,
                                                                    # random_state = 42,
                                                                    stratify=y_train_cv)
            history = trained_model.fit(X_train_cv, y_train_cv,
                                epochs=self.epochs,
                                validation_data=(X_val, y_val),
                                batch_size=self.batch_size, verbose=1,
                                callbacks=self.callbacks)
            histories.append(history)
            names_list.append(np.unique(names_test_cv))
            pred = trained_model.predict(X_test_cv)
            y_pred = np.argmax(pred, axis=-1)
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
            nested_dictionary = {'pred_list': fold_pred_list, 'label_list': fold_label_list, 'names_list': fold_names_list}
            dictionary['fold_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        if return_history:
            nested_dictionary = {'histories':histories, 'patients':names_list}
            dictionary['history'] = nested_dictionary
        return dictionary

    def fit_model(self, X_train, y_train, X_val, y_val, return_history = True):
        """
        Train a model.
        :param X_train:
                np.array
                    The trainset.
        :param y_train:
                np.array
                    The labels. Must be in categorical way.
        :param X_val:
                np.array
                    The validation set.
        :param y_val:
                np.array
                    The validation labels. Must be in categorical way.
        :param return_history:
                bool
                    If true history of the training is returned.
        :return:
        """
        history = self.model.fit(X_train, y_train,
                                 epochs=self.epochs,
                                 validation_data=(X_val, y_val),
                                 batch_size=self.batch_size, verbose=1,
                                 callbacks=self.callbacks)
        if return_history:
            return history

    def test_model(self, X_test):
        """
        Test a model and return the prediction.
        :param X_test:
                np.array
                    The test set.
        :return:
                np.array
                    The prediction.
        """
        pred = self.model.predict(X_test)
        y_pred = np.argmax(pred, axis=-1)
        return y_pred




