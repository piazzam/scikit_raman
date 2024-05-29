import copy
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, InputLayer, Conv1D, MaxPooling1D, \
    Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from keras.models import clone_model
from tensorflow.keras.utils import to_categorical
from scikit_raman.Classes.DataAugmenter import *
from scikit_raman.Classes.Keras.DataAugmentation.EMSC import *
from scikit_raman.Classes.Keras.DataAugmentation.Shift import *
from scikit_raman.Classes.Keras.SingleGPU.utility import get_optimizer
from tensorflow.keras.models import load_model
import wandb
from wandb.keras import WandbMetricsLogger


class DLModelKeras:
    """
    A class that represents a DLModel object in keras.
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
    optimizer: str
        optimizer of the model.
    loss: str
        loss function for the training of the model.
    metrics: str
        metrics for the evaluation of the model.
    """

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
    def load_model_benchmark(dlm,  n_dims, number_classes=3, data_augmentation=False, factor=0.5):
        """
        Load the benchmark model.
        :param n_dims: int
            number of feature for the problem.
        :param data_augmentation: bool, optional
            if true data augmentation on the fly is added to the model. The default value is False.
        :param factor: float, optional.
            percentage on which apply data augmentation. The default value is 0.5.
        :return: scikit_raman.Keras.DLMoldeKeras
            a model object representing the benchmark model.
        """
        loss = 'categorical_crossentropy'
        metrics = ['categorical_accuracy']
        learning_rate = 0.00020441990333108206
        # optimizer = Adam(learning_rate = learning_rate)
        # optimizer = get_optimizer('adam', learning_rate)
        optimizer = 'adam'

        # ----- init model
        model = Sequential()
        if data_augmentation:
            model.add(EMSC(factor, name="EMSC_augmentation"))
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
        model.add(Dense(units=number_classes, activation='softmax'))

        # ----- Compile
        # if data_augmentation:
        #    model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True)
        # else:
        #    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        epochs = 273
        batch_size = 338
        es = EarlyStopping(monitor="val_categorical_accuracy", patience=100, verbose=1,
                           restore_best_weights=True)
        lr = ReduceLROnPlateau(monitor="val_categorical_accuracy", factor=0.5, verbose=4, patience=80,
                               cooldown=10)
        callbacks = [es, lr]
        return dlm(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)

    @classmethod
    def load_model(dlm, filename="model_saved/model", batch_size=256, epochs=200, callbacks=[], optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'], learning_rate=0.00020441990333108206):
        """
        Load a model from classical storage keras.
        :param filename: str
            filename of json file.
        :param batch_size: int, optional.
            batch size of the model. The default value is 256.
        :param epochs: int, optional.
            epochs of the model. The default value is 200.
        :param callbacks: list
            list of the callbacks of the model. The default value is [].
        :param optimizer: keras.optimizers, optional
            optimizer of the model. The default value is Adam.
        :param loss: str, optional.
            loss function of the model. The defualt value is categorical_crossentropy.
        :param metrics: list, optional.
        metrics on which evaluate the model. The default value is ['categorical_accuracy']
        :return: scikit_raman.Keras.DLMoldeKeras
            A model object representing the json model.
        """
        model = load_model(filename)
        # optimizer_fun = get_optimizer(optimizer, learning_rate)
        return dlm(model, batch_size, epochs, callbacks, optimizer, loss, metrics, learning_rate)

    def compile_model(self):
        optimizer = get_optimizer(self.optimizer, self.learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss=self.loss, metrics=self.metrics)

    def load_weights(self, filename="model_saved/weights"):
        self.model.load_weights(filename, compile=False)

    def train_model_leave_one_patient_out(self, dataset, number_classes, patient_level=True, get_patient_prediction=True,
                                          return_history=True, test_size=0.1, data_augmentation=False, f_name='emsc',
                                          f_params=None, save_model=False, model_path="model_saved/model/", save_weights=False,
                                          weights_path="model_saved/weights/", random_state=42, log_to_wandb=False,
                                          wandb_project="project_one", wandbconfig={}, wandbcallbacks=[],
                                          reinit=True):
        """
        Train the model with Leave One Patient Out Cross Validation
        :param dataset: scikit_raman.Dataset
            a Dataset object on which train the model.
        :param number_classes: int
            an integer that represent the number of classes of the problem.
        :param patient_level: bool, optional
            if true the results are returned at patient level granularity. The default value is True.
        :param get_patient_prediction: bool, optional
            if true every prediction of the patient is returned. The default value is True.
        :param return_history: bool, optional
            if true the histories of the training is returned. The default value is True.
        :param test_size: float, optional
            dimension of the validation set. The default value is 0.1.
        :param data_augmentation: bool, optional
            if true offline data augmentation is applied. The default value is False.
        :param f_name: str, optional
            function name of the data augmentation strategy. It must be a scikit_raman.DataAugmenter function. The
            default value is 'emsc'
        :param f_params: dict, optional
            parameters in input to data augmentation function. The default value is None.
        :param save_model: bool, optional
            if true json format of the model is returned. The default value is False.
        :param save_weights: bool, optional
            if true numpy array with model weights is returned. The default value is False.
        :return: dict
            results based on the user choiches.
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
            names_test_cv = dataset.user[test_idx]
            if log_to_wandb:
                if wandbconfig == {}:
                    wandbconfig = {
                        'reinit': reinit,
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate}
                else:
                    wandbconfig_base = {
                        'reinit': reinit,
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'lr': self.learning_rate}
                    wandbconfig.update(wandbconfig_base)
                wandb.init(project=wandb_project,
                           name=wandb_project +
                           str(np.unique(names_test_cv)[0]),
                           config=wandbconfig)
                if wandbcallbacks == []:
                    self.callbacks.extend([WandbMetricsLogger()])
                else:
                    try:
                        self.callbacks.extend(wandbcallbacks)
                    except TypeError as exc:
                        raise TypeError("wandbcallbacks must be a list")
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
                trained_model.save(model_path+str(names_test_cv[0]))
            if save_weights:
                trained_model.save_weights(weights_path+str(names_test_cv[0]))
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
                       save_weights=False, weights_path="model_saved/weights/", random_state=42, log_to_wandb=False,
                       wandb_project="project_one", wandbconfig={}, wandbcallbacks=[],
                       reinit=True, check_users_separated=True):
        """
        Train the model with K-Fold Cross Validation.
        :param dataset: scikit_raman.Dataset
            Dataset object on which train the model.
        :param number_classes: int
            number of target classes of the task.
        :param k: int, optional
            number of fold for the k-kold cross validation. The default value is 10.
        :param fold_level: bool, optional
            if true results for every fold are returned. The default value is True.
        :param get_patient_prediction: bool, optional
            if true every prediction for every patient is returned. The default value is True.
        :param return_history: bool, optional
            if true history of the training procedure is returned. The default value is True.
        :param data_augmentation: bool, optional
            if true offline data augmentation is applied. The default value is False.
        :param f_name: str, optional
            string name of the data augmentation function. It must be a DataAugmenter function. The default value is
            'emsc'
        :param f_params: dict, optional
            dictionary with the parameters of the data augmentation function. The default value is None.
        :param save_model: bool, optional
            if true json format of the model is returned. The default value is False.
        :param save_weights: bool, optional
            if true numpy array with model weights is returned. The default value is False.
        :return: dict
            dictionary of the results based on the user choiches.
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
        total_users = np.unique(dataset.user)
        for j, (train_idx, test_idx) in enumerate(folds):
            if log_to_wandb:
                print("log_to_wandb")
                if wandbconfig == {}:
                    wandbconfig = {
                        'reinit': reinit,
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'learning_rate': self.learning_rate}
                else:
                    wandbconfig_base = {
                        'reinit': reinit,
                        'epochs': self.epochs,
                        'batch_size': self.batch_size,
                        'lr': self.learning_rate}
                    wandbconfig.update(wandbconfig_base)
                wandb.init(project=wandb_project,
                           name=wandb_project + str(j),
                           config=wandbconfig)
                if wandbcallbacks == []:
                    self.callbacks.extend([WandbMetricsLogger()])
                else:
                    try:
                        self.callbacks.extend(wandbcallbacks)
                    except TypeError as exc:
                        raise TypeError("wandbcallbacks must be a list")
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
                  save_weights=False, model_path="model_saved/model/", weights_path="model_saved/weights/"):
        """
        Fit a model.
        :param X_train: np.array
            trainset.
        :param y_train: np.array
            labels of the trainset.
        :param X_val: np.array
            validation set.
        :param y_val: np.array
            labels of the validation set.
        :param return_history: bool, optional
            if true history of the training is returned. The default value is True.
        :return: list
            if selected return the history of the training
        """
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
        Test a model on a test set.
        :param X_test: np.array
            testset.
        :return: np.array
            prediction.
        """
        pred = self.model.predict(X_test)
        y_pred = np.argmax(pred, axis=-1)
        return y_pred

    def change_input_tl(self, new_input_dims, old_input_dims):
        new_model = Sequential()
        new_model.add(InputLayer(input_shape=(new_input_dims,)))
        new_model.add(Dense(old_input_dims, name="dense_added"))
        for el in self.model.layers:
            new_model.add(el)

        self.model = new_model

    def change_output_tl(self, new_output_dims, new_activation_function="softmax"):
        input_shape = self.model.layers[0].input_shape
        new_model = tf.keras.models.Sequential(self.model.layers[:-1])
        new_model.build(input_shape)
        new_model.add(Dense(units=new_output_dims,
                      activation=new_activation_function, name="new_output_layer"))
        self.model = new_model
