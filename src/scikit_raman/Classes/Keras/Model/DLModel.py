import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, InputLayer, Conv1D, MaxPooling1D, \
    Reshape
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from keras.models import clone_model
from tensorflow.keras.utils import to_categorical
from scikit_raman.Classes.DataAugmenter import *
from scikit_raman.Classes.Keras.Model.utility import get_optimizer
from tensorflow.keras.models import load_model
import scikit_raman.module.utility as utils
import scikit_raman.module.models as models
from scikit_raman.Classes.Keras.Model.callbacks import EpochCheckpointSaver
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization, MaxPooling1D, Flatten, Dropout, Dense, LeakyReLU
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.optimizers import Adam

class DLModelKeras:
    """
    A class used to represent a tensorflow.Keras DL model.

    ...

    Attributes
    ----------
    model : tf.Keras
        A Deep Neural Network defined with Keras framework.
    batch_size : int
        Dimension of the batch size.
    epochs : int
        Number of epochs.
    callbacks : list
        List containing the callbacks applied during training.
    optimizer : string
        Optimizer name.
    loss : string
        Loss function name.
    metrics : list
        List of metrics to be monitored during training.
    learning_rate : float
        Value of the learning rate.

    Methods
    -------
    load_model_benchmark(dlm,  n_dims, number_classes=3)
        Load the benchmark model and return a DLModelKeras.
    load_model(dlm, filename="model_saved/model", batch_size=256, epochs=200, callbacks=[], optimizer='adam',
                loss='categorical_crossentropy', metrics=['categorical_accuracy'], learning_rate=0.00020441990333108206)
        Load a saved model, instantiates a DLModelKeras object and return it.
    compile_model(self)
        Compile the model.
    add_callback(self, callback)
        Add new callback.
    train_model_leave_one_patient_out(self, dataset, number_classes, patient_level=True, get_patient_prediction=True,
                                          return_history=True, val_size=0.1, data_augmentation=False, f_name='emsc',
                                          f_params=None, save_model=False, model_path="model_saved/models/final",
                                          save_weights=False,weights_path="model_saved/weights/", random_state=42,
                                          set_seed=True, model_name="Benchmark_CNN",
                                          checkpoint_folder_path="model_saved/models/checkpoint"):
        Train a model with leave-one-patient-out cross validation.
    train_model_cv(self, dataset, number_classes, k=10, fold_level=True, get_patient_prediction=True,
                    return_history=True,data_augmentation=False, f_name='emsc', f_params=None, save_model=False,
                       model_path="model_saved/models/",save_weights=False, weights_path="model_saved/weights/",
                       random_state=42, check_users_separated=True, model_name="Benchmark_CNN",
                       checkpoint_folder_path="model_saved/models/checkpoint", set_seed=True,val_size=0.1)
        Train a model with k-fold cross validation.
    fit_model(self, X_train, y_train, X_val, y_val, return_history=True, model_name="model", save_model=False,
                  save_weights=False, model_path="model_saved/model/", weights_path="model_saved/weights/",
                  random_state=42, set_seed=True)
        Train a model on a specified training set.
    test_model(self, X_test)
        Test a model on a specified test set and return the predictions.
    change_input_tl(self, new_input_dims, old_input_dims)
        Preliminary function for Transfer Learning. It changes the input size of a Deep Model.
    change_output_tl(self, new_input_dims, old_input_dims)
        Preliminary function for Transfer Learning. It changes the output size of a Deep Model.
    """
    def __init__(self, model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate):
        """
        Constructor function for DLModelKeras class.

        Parameters
        ----------
        model : tf.Keras
            A Deep Neural Network defined with Keras framework.
        batch_size : int
            Dimension of the batch size.
        epochs : int
            Number of epochs.
        callbacks : list
            List containing the callbacks applied during training.
        optimizer : string
            Optimizer name.
        loss : string
            Loss function name.
        metrics : list
            List of metrics to be monitored during training.
        learning_rate : float
            Value of the learning rate.
        """
        self.model = model
        self.batch_size = batch_size
        self.epochs = epochs
        self.callbacks = callbacks
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.learning_rate = learning_rate

    @classmethod
    def load_model_benchmark(dlm,  n_dims, number_classes=3, weight_initializer=True):
        """
        Load the benchmark model and return a DLModelKeras.

        Parameters
        ----------
        n_dims : int
            Input size of desired model.
        number_classes : int, optional (default is 3)
            Output size of desired model.
        weight_initializer : bool, optional (default is True)
            Whether to initialize the weights of Deep Benchmark Model.

        Returns
        -------
        scikit_raman.Keras.Model.DLModelKeras
            DLModelKeras initialized with benchmark model parameters.
        """
        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
        learning_rate = 0.00020441990333108206
        optimizer = 'adam'

        if weight_initializer:
            model = models.create_model_benchmark_weight_initialization(n_dims, number_classes)
        else:
            model = models.create_model_benchmark(n_dims, number_classes)

        epochs = 273
        batch_size = 338
        # es = EarlyStopping(monitor="val_categorical_accuracy", patience=100, verbose=1,
        #                    restore_best_weights=True)
        # lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, verbose=4, patience=80,
        #                        cooldown=10)
        # callbacks = [es, lr]
        callbacks = []
        return dlm(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)

    @classmethod
    def load_model_benchmark_functional(dlm,  n_dims, number_classes=3):
        """
        Load the benchmark model and return a DLModelKeras. The keras model
        has been defined according to the functional API.

        Parameters
        ----------
        n_dims : int
            Input size of desired model.
        number_classes : int, optional (default is 3)
            Output size of desired model.

        Returns
        -------
        scikit_raman.Keras.Model.DLModelKeras
            DLModelKeras initialized with benchmark model parameters.
        """
        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
        learning_rate = 0.00020441990333108206
        optimizer = Adam(learning_rate=learning_rate)
        initializer = HeUniform()

        input_layer = Input(shape=(n_dims,))
        x = Reshape((n_dims, 1))(input_layer)

        x = Conv1D(filters=100, kernel_size=100, strides=1, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = BatchNormalization(momentum=0.99, epsilon=0.01)(x)

        x = Conv1D(filters=100, kernel_size=5, strides=2, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = MaxPooling1D(pool_size=6, strides=3, padding='same')(x)
        x = BatchNormalization(momentum=0.99, epsilon=0.01)(x)

        x = Conv1D(filters=25, kernel_size=9, strides=5, padding='same', activation='relu', kernel_initializer=initializer)(x)
        x = MaxPooling1D(pool_size=3, strides=2, padding='same')(x)

        x = Flatten()(x)
        x = Dropout(rate=0.1)(x)

        x = Dense(units=732)(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.7)(x)

        x = Dense(units=189)(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.25)(x)

        x = Dense(units=152)(x)
        x = LeakyReLU()(x)
        x = Dropout(rate=0.1)(x)

        output_layer = Dense(units=number_classes, activation='softmax')(x)

        model = Model(inputs=input_layer, outputs=output_layer)

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        epochs = 273
        batch_size = 338
        callbacks = []

        # Restituisce il modello per l'addestramento
        return dlm(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)

    @classmethod
    def load_model(dlm, filename="model_saved/model", batch_size=256, epochs=200, callbacks=[], optimizer='adam',
                   loss='categorical_crossentropy', metrics=['categorical_accuracy'],
                   learning_rate=0.001):
        """
        Load a saved model, instantiates a DLModelKeras object and return it.

        Parameters
        ----------
        filename : str, optional (default is "model_saved/model")
            Path from which load the model.
        batch_size : int, optional (default is 256)
            Dimension of the batch size.
        epochs : int, optional (default is 200)
            Number of training epochs.
        callbacks : list, optional (default is [])
            List of callbacks applied during the training phase.
        optimizer : str, optional (default is 'adam')
            String corresponding to the desired optimizer
        loss : str, optional (default is 'categorical_crossentropy')
            String corresponding to the desired loss function.
        metrics : list, optional (default is ['categorical_accuracy'])
            List containing the metrics to be monitored during training.
        learning_rate : float, optional (default is 0.001)
            Learning rate value

        Returns
        -------
        scikit_raman.Keras.Model.DLModelKeras
            DLModelKeras initialized with benchmark model parameters.
        """
        utils.set_seed()
        model = load_model(filename)
        return dlm(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)

    def compile_model(self):
        """
        Compile the model.
        """
        optimizer = get_optimizer(self.optimizer, self.learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self.loss, metrics=self.metrics)


    def load_weights(self, filename="model_saved/weights"):
        """
        Load weights of the Neural Network.

        Parameters
        ----------
        filename : string
            Path from which load the weights.
        """
        self.model.load_weights(filename, skip_mismatch=True)

    def add_callback(self, callback):
        """
        Add callback to the list of callbacks.

        Parameters
        ----------
        callback : keras.callbacks.Callback
            Callback to add.
        """
        self.callbacks = [cb for cb in self.callbacks if not isinstance(cb, type(callback))]
        self.callbacks.append(callback)

    def train_model_leave_one_patient_out(self, dataset, number_classes, patient_level=True, get_patient_prediction=True,
                                          return_history=True, val_size=0.1, data_augmentation=False, f_name='emsc',
                                          f_params=None, save_model=False, model_path="model_saved/models/final",
                                          save_weights=False,weights_path="model_saved/weights/", random_state=42,
                                          set_seed=True, model_name="Benchmark_CNN",
                                          checkpoint_folder_path="model_saved/models/checkpoint",
                                          verbose=1):
        """
        Train a model in a leave-one-patient-out strategy and return the results.

        Parameters
        ----------
        dataset : scikit_raman.Dataset
            Dataset object on which compute the experiment.
        number_classes : int
            Number of classes of the problem.
        patient_level : bool, optional (default is True)
            Whether to return the results at a patient level.
        get_patient_prediction : bool, optional (default is True)
            Whether to return the results at a single spectra level.
        return_history : bool, optional (default is True)
            Whether to return the history of the training of the model.
        val_size : float, optional (default is 0.1)
            Percentage of dimension of validation set.
        data_augmentation : bool, optional (default is False)
            Whether to apply the data augmentation during the experimentation procedure.
        f_name : string, optional (default is 'emsc')
            Name of the data augmentation function.
        f_params : dict, optional (default is None)
            Parameters of data augmentation function.
        save_model : bool, optional (default is False)
            Whether to store the trained models or not.
        model_path : string, optional (default is "model_saved/models/final")
            Path for saving the trained models.
        save_weights : bool, optional (default is False)
            Whether to save the weights of trained models.
        weights_path : string, optional (default is "model_saved/weights/")
            Path for saving the weights of trained models.
        random_state : int, optional (default is 42)
            Seed
        set_seed : bool, optional (default is True)
            Whether to set a seed for reproducibility.
        model_name : string, optional (default is "Benchmark_CNN")
            Name of model for saving checkpoints.
        checkpoint_folder_path : string, optional (default is "model_saved/models/checkpoint")
            Path for saving the checkpoints.
        verbose : int, optional (default is 1)
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. "auto" becomes 1 for most cases.

        Returns
        -------
        dict
            Dictionary contained the desired results.
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
        histories = []
        names_list = []
        for j, (train_idx, test_idx) in enumerate(folds, start=1):
            es = EarlyStopping(monitor='loss', patience=100, verbose=1,
                               restore_best_weights=True)
            lr = ReduceLROnPlateau(monitor='loss', factor=0.5, verbose=4, patience=80,
                                   cooldown=10)
            names_test_cv = dataset.user[test_idx]
            patient_name = np.unique(names_test_cv)[0]
            print(f'\n[*] Patient {j}: {patient_name}')
            ec = EpochCheckpointSaver(save_interval=39, folder_path=checkpoint_folder_path, model_name=model_name, fold=patient_name)
            callbacks = [es, lr, ec]
            trained_model = clone_model(self.model)
            optimizer = get_optimizer(self.optimizer, self.learning_rate)
            trained_model.compile(optimizer=optimizer,
                                  loss=self.loss, metrics=self.metrics)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            y_test_cv = dataset.labels[test_idx]
            X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train_cv, y_train_cv, test_size=val_size,
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
                                        batch_size=self.batch_size, verbose=verbose,
                                        callbacks=callbacks)
            histories.append(history)
            names_list.append(patient_name)
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
                    os.makedirs(model_path, exist_ok=True)
                    trained_model.save(model_path+str(names_test_cv[0])+".keras")
            if save_weights:
                try:
                    trained_model.save_weights(weights_path+str(names_test_cv[0])+'.weights.h5')
                except FileNotFoundError:
                    os.makedirs(weights_path, exist_ok=True)
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
                       data_augmentation=False, f_name='emsc', f_params=None, save_model=False, model_path="model_saved/models/",
                       save_weights=False, weights_path="model_saved/weights/", random_state=42, check_users_separated=True,
                       model_name="Benchmark_CNN", checkpoint_folder_path="model_saved/models/checkpoint", set_seed=True,
                       val_size=0.1, verbose=1):
        """
        Train a model in a leave-one-patient-out strategy and return the results.

        Parameters
        ----------
        dataset : scikit_raman.Dataset
            Dataset object on which compute the experiment.
        number_classes : int
            Number of classes of the problem.
        k : int, optional (default is 10)
            Number of folds for the k-fold function.
        fold_level : bool, optional (default is True)
            Whether to return the results at a fold level.
        get_patient_prediction : boo, optional (default is True)
            Whether to return results at a single spectra level.
        return_history : bool, optional (default is True)
            Whether to return the history of the training of the model.
        data_augmentation : bool, optional (default is False)
            Whether to apply the data augmentation during the experimentation procedure.
        f_name : string, optional (default is 'emsc')
            Name of the data augmentation function.
        f_params : dict, optional (default is None)
            Parameters of data augmentation function.
        save_model : bool, optional (default is False)
            Whether to store the trained models or not.
        model_path : string, optional (default is "model_saved/models/final")
            Path for saving the trained models.
        save_weights : bool, optional (default is False)
            Whether to save the weights of trained models.
        weights_path : string, optional (default is "model_saved/weights/")
            Path for saving the weights of trained models.
        random_state : int, optional (default is 42)
            Seed value
        check_users_separated : bool, optional (default is True)
            Whether to check if users are not mixed between train and test set.
        model_name : string, optional (default is "Benchmark_CNN")
            Name of model for saving checkpoints.
        checkpoint_folder_path : string, optional (default is "model_saved/models/checkpoint")
            Path for saving the checkpoints.
        set_seed : bool, optional (default is True)
            Whether to set a seed for reproducibility.
        val_size : float, optional (default is 0.1)
            Percentage of dimension of validation set.
        verbose : int, optional (default is 1)
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch. "auto" becomes 1 for most cases.

        Returns
        -------
        dict
            Dictionary contained the desired results.
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
        histories = []
        names_list = []
        total_users = np.unique(dataset.user)
        for j, (train_idx, test_idx) in enumerate(folds, start=1):
            names_test_cv = dataset.user[test_idx]
            patient_name = np.unique(names_test_cv)[0]
            print(f'\n[*] Patient {j}: {patient_name}')
            ec = EpochCheckpointSaver(save_interval=39, folder_path=checkpoint_folder_path, model_name=model_name, fold=patient_name)
            self.add_callback(ec)
            trained_model = clone_model(self.model)
            optimizer = get_optimizer(self.optimizer, self.learning_rate)
            trained_model.compile(optimizer=optimizer,
                                  loss=self.loss, metrics=self.metrics)
            X_train_cv = dataset.spectra[train_idx]
            X_test_cv = dataset.spectra[test_idx]
            y_train_cv = dataset.labels[train_idx]
            y_test_cv = dataset.labels[test_idx]
            users_train = np.unique(dataset.user[train_idx])
            user_test = np.unique(dataset.user[test_idx])
            if check_users_separated:
                if len(users_train) + len(user_test) > len(total_users):
                    raise Exception("Mixed train and test")
            X_train_cv, X_val, y_train_cv, y_val = train_test_split(X_train_cv, y_train_cv, test_size=val_size,
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
                                        batch_size=self.batch_size, verbose=verbose,
                                        callbacks=self.callbacks)
            histories.append(history)
            names_list.append(patient_name)
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
                try:
                    trained_model.save(model_path + "fold_"+str(j) + ".keras")
                except FileNotFoundError:
                    os.makedirs(model_path, exist_ok=True)
                    trained_model.save(model_path + "fold_"+str(j) + ".keras")
            if save_weights:
                try:
                    trained_model.save_weights(weights_path + str(names_test_cv[0]) + '.weights.h5')
                except FileNotFoundError:
                    os.makedirs(weights_path, exist_ok=True)
                    trained_model.save_weights(weights_path + str(names_test_cv[0]) + ".weights.h5")
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
                  save_weights=False, model_path="model_saved/model/", weights_path="model_saved/weights/",
                  random_state=42, set_seed=True, checkpoint_folder_path="model_saved/models/checkpoint",
                  model_name_checkpoint="Benchmark_CNN"):
        """
        Train a model on a specified training set.

        Parameters
        ----------
        X_train : np.array
            <Array containing the independent features of the training set.>
        y_train : np.array
            Array containing the dependent features of the training set.
        X_val : np.array
            Array containing the independent features of the validation set.
        y_val : np.array
            Array containing the dependent features of the validation set.
        return_history : bool, optional (default is True)
            Whether to return the history of the training of the model.
        model_name : string, optional (default is "Benchmark_CNN")
            Name of model for saving checkpoints.
        save_model : bool, optional (default is False)
            Whether to store the trained models or not.
        save_weights : bool, optional (default is False)
            Whether to save the weights of trained models.
        model_path : string, optional (default is "model_saved/models/final")
            Path for saving the trained models.
        weights_path : string, optional (default is "model_saved/weights/")
            Path for saving the weights of trained models.
        random_state : int, optional (default is 42)
            Seed value
        set_seed : bool, optional (default is True)
            Whether to set a seed for reproducibility.
        checkpoint_folder_path : string, optional (default is "model_saved/models/checkpoint")
            Path for saving the checkpoints.
        model_name : string, optional (default is "Benchmark_CNN")
            Name of model for saving checkpoints.

        Returns
        -------
        dict
            If requested dictionary with training history.
        """
        if set_seed:
            utils.set_seed(random_state)
        # ec = EpochCheckpointSaver(save_interval=39, folder_path=checkpoint_folder_path, model_name=model_name,
        #                           fold=model_name_checkpoint)
        # self.callbacks.append(ec)
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
        """
        Test a model on a specified test set and return the predictions.

        Parameters
        ----------
        X_test : np.array
            Array containing the independent features of the training set.
        Returns
        -------
        np.array
            Array containig the predictions of the model (in integer format).
        """
        pred = self.model.predict(X_test)
        y_pred = np.argmax(pred, axis=-1)
        return y_pred

    def change_input_tl(self, new_input_dims, old_input_dims):
        """
        Preliminary function for Transfer Learning. It changes the input size of a Deep Model.

        Parameters
        ----------
        new_input_dims : int
            Desired input dimension.
        old_input_dims : int
            Actual input dimension.
        """
        new_model = Sequential()
        new_model.add(InputLayer(input=(new_input_dims,)))
        new_model.add(Dense(old_input_dims, name="dense_added"))
        for el in self.model.layers:
            new_model.add(el)

        self.model = new_model

    def change_output_tl(self, new_output_dims, new_activation_function="softmax"):
        """
        Preliminary function for Transfer Learning. It changes the output size of a Deep Model, if requested it changes
        also the activation function of last layer.

        Parameters
        ----------
        new_output_dims : int
            Desired output dimension.
        new_activation_function : string, optional (default is "softmax")
            Desired new activation function.
        """
        input_shape = self.model.layers[0].shape
        new_model = tf.keras.models.Sequential(self.model.layers[:-1])
        new_model.build(input_shape)
        new_model.add(Dense(units=new_output_dims,
                      activation=new_activation_function, name="new_output_layer"))
        self.model = new_model
