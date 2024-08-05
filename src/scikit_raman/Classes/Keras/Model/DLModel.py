import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, InputLayer, Conv1D, MaxPooling1D, \
    Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from keras.models import clone_model
from tensorflow.keras.utils import to_categorical
from scikit_raman.Classes.DataAugmenter import *
from scikit_raman.Classes.Keras.Model.utility import get_optimizer
from tensorflow.keras.models import load_model
import scikit_raman.module.utility as utils
from scikit_raman.Classes.Keras.Model.callbacks import EpochCheckpointSaver
import tensorflow as tf

class DLModelKeras:
    def __init__(self, model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate):
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate

    @classmethod
    def load_model_benchmark(dlm,  n_dims, number_classes=3, data_augmentation=False, factor=0.5, 
                             set_seed=True, folder_path="models/checkpoint"):
        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
        learning_rate = 0.00020441990333108206
        optimizer = 'adam'

        # ----- init model
        model = Sequential()
        if data_augmentation:
            model.add(EMSC(factor, name="EMSC_augmentation"))
        model.add(InputLayer(shape=(n_dims,)))
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
        model.add(Dense(units=number_classes, activation='softmax'))

        epochs = 273
        batch_size = 338
        es = EarlyStopping(monitor="val_categorical_accuracy", patience=100, verbose=1,
                           restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, verbose=4, patience=80,
                               cooldown=10)
        ec = EpochCheckpointSaver(save_interval=39, folder_path=folder_path, model_name="Benchmark_CNN")
        callbacks = [es, lr, ec]
        if set_seed:
            utils.set_seed()
        return dlm(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)

    @classmethod
    def load_model(dlm, filename="model_saved/model", batch_size=256, epochs=200, callbacks=[], optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'], learning_rate=0.00020441990333108206,
        seed = 42):
        utils.set_seed()
        model = load_model(filename)
        return dlm(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)

    def compile_model(self):
        optimizer = get_optimizer(self.optimizer, self.learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self.loss, metrics=self.metrics)


    def load_weights(self, filename="model_saved/weights"):
        self.model.load_weights(filename, skip_mismatch=True)

    def train_model_leave_one_patient_out(self, dataset, number_classes, patient_level=True, get_patient_prediction=True,
                                          return_history=True, test_size=0.1, data_augmentation=False, f_name='emsc',
                                          f_params=None, save_model=False, model_path="model_saved/model/", save_weights=False,
                                          weights_path="model_saved/weights/", random_state=42, set_seed=True):
        if set_seed:
            utils.set_seed(random_state)
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
            names_test_cv = dataset.user[test_idx]
            trained_model = clone_model(self.model)
            optimizer = get_optimizer(self.optimizer, self.learning_rate)
            trained_model.compile(optimizer=optimizer,
                                  loss=self.loss, metrics=self.metrics)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            y_test_cv = dataset.labels[test_idx]
            X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train_cv, y_train_cv, test_size=test_size,
                                                                    random_state=random_state,
                                                                    stratify=y_train_cv)
            if data_augmentation:
                da = DataAugmenter(X_train_cv, y_train_cv)
                func = getattr(da, f_name)
                func(f_params)
                X_train_cv = da.spectra
                y_train_cv = da.labels
            y_train_cv_cat = to_categorical(y_train_cv, number_classes)
            y_val_cat = to_categorical(y_val, number_classes)
            history = trained_model.fit(X_train_cv, y_train_cv_cat,
                                        epochs=self.epochs,
                                        validation_data=(X_val, y_val_cat),
                                        batch_size=self.batch_size, verbose=1,
                                        callbacks=self.callbacks)
            histories.append(history)
            names_list.append(np.unique(names_test_cv)[0])
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
                pat_names_list.append(np.unique(names_test_cv)[0])
            if save_model:
                try:
                    trained_model.save(model_path+str(names_test_cv[0])+".keras")
                except FileNotFoundError:
                    os.makedirs(model_path)
                    trained_model.save(model_path+str(names_test_cv[0])+".keras")
            if save_weights:
                try:
                    trained_model.save_weights(weights_path+str(names_test_cv[0])+'.weights.h5')
                except FileNotFoundError:
                    os.makedirs(weights_path)
                    trained_model.save_weights(weights_path+str(names_test_cv[0])+".weights.h5")
        dictionary = {}
        if patient_level:
            nested_dictionary = {'pred_list': pat_pred_list,
                                 'label_list': pat_label_list, 'names_list': pat_names_list}
            dictionary['patient_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list,
                                 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        if return_history:
            nested_dictionary = {
                'histories': histories, 'patients': names_list}
            dictionary['history'] = nested_dictionary
        return dictionary

    def train_model_cv(self, dataset, number_classes, k=10, fold_level=True, get_patient_prediction=True, return_history=True,
                       data_augmentation=False, f_name='emsc', f_params=None, save_model=False, model_path="model_saved/model/",
                       save_weights=False, weights_path="model_saved/weights/", random_state=42, check_users_separated=True,
                       set_seed=True):
        if set_seed:
            utils.set_seed(random_state)
        folds = dataset.k_fold(k)
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        fold_pred_list = []
        fold_label_list = []
        fold_names_list = []
        histories = []
        names_list = []
        total_users = np.unique(dataset.user)
        for j, (train_idx, test_idx) in enumerate(folds):
            trained_model = clone_model(self.model)
            optimizer = get_optimizer(self.optimizer, self.learning_rate)
            trained_model.compile(optimizer=optimizer,
                                  loss=self.loss, metrics=self.metrics)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            y_test_cv = dataset.labels[test_idx]
            names_test_cv = dataset.user[test_idx]
            users_train = np.unique(dataset.user[train_idx])
            user_test = np.unique(dataset.user[test_idx])
            if check_users_separated:
                if len(users_train) + len(user_test) > len(total_users):
                    raise Exception("Mixed train and test")
            X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train_cv, y_train_cv, test_size=.1,
                                                                    random_state=random_state,
                                                                    stratify=y_train_cv)
            if data_augmentation:
                da = DataAugmenter(X_train_cv, y_train_cv)
                func = getattr(da, f_name)
                func(f_params)
                X_train_cv = da.spectra
                y_train_cv = da.labels
            y_train_cv_cat = to_categorical(y_train_cv, number_classes)
            y_val_cat = to_categorical(y_val, number_classes)
            history = trained_model.fit(X_train_cv, y_train_cv_cat,
                                        epochs=self.epochs,
                                        validation_data=(X_val, y_val_cat),
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
            if save_model:
                trained_model.save(model_path)
            if save_weights:
                trained_model.save_weights(weights_path)
        dictionary = {}
        if fold_level:
            nested_dictionary = {'pred_list': fold_pred_list,
                                 'label_list': fold_label_list, 'names_list': fold_names_list}
            dictionary['fold_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list,
                                 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        if return_history:
            nested_dictionary = {
                'histories': histories, 'patients': names_list}
            dictionary['history'] = nested_dictionary
        return dictionary

    def fit_model(self, X_train, y_train, X_val, y_val, return_history=True, model_name="model", save_model=False,
                  save_weights=False, model_path="model_saved/model/", weights_path="model_saved/weights/", random_state=42,
                  set_seed=True):
        if set_seed:
            utils.set_seed(random_state)
        history = self.model.fit(X_train, y_train,
                                 epochs=self.epochs,
                                 validation_data=(X_val, y_val),
                                 batch_size=self.batch_size, verbose=1,
                                 callbacks=self.callbacks)
        dictionary = {}
        if return_history:
            nested_dictionary = {'histories': [
                history], 'patients': [model_name]}
            dictionary['history'] = nested_dictionary
        if save_model:
            self.model.save(model_path)
        if save_weights:
            self.model.save_weights(weights_path)
        return dictionary

    def test_model(self, X_test):
        pred = self.model.predict(X_test)
        y_pred = np.argmax(pred, axis=-1)
        return y_pred

    def change_input_tl(self, new_input_dims, old_input_dims):
        new_model = Sequential()
        new_model.add(InputLayer(input=(new_input_dims,)))
        new_model.add(Dense(old_input_dims, name="dense_added"))
        for el in self.model.layers:
            new_model.add(el)

        self.model = new_model

    def change_output_tl(self, new_output_dims, new_activation_function="softmax"):
        input_shape = self.model.layers[0].shape
        new_model = tf.keras.models.Sequential(self.model.layers[:-1])
        new_model.build(input_shape)
        new_model.add(Dense(units=new_output_dims,
                      activation=new_activation_function, name="new_output_layer"))
        self.model = new_model
