from sklearn.model_selection import train_test_split
from scikit_raman.Classes.PyTorch.PytorchDataset import PytorchDataset
from scikit_raman.Classes.PyTorch.BenchmarkModel import BenchmarkModel
from scikit_raman.Classes.DataAugmenter import *
import copy
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from statistics import mean
import numpy as np
import torch.nn as nn
from torch.optim import Adam

class DLModel:
    """
        A class to represent a DLModel object in pytorch.
        ...

        Attributes
        ----------
        model : torch.nn.Module
            model to train e test.
        batch_size: int
            dimension of the batch size.
        epochs: int
            number of epochs.
        loss: torch.nn
            loss function to optimize during training.
        optimizer: torch.nn
            optimizer for the model.
        learning_rate: double
            value of the learning rate.
        gpu_ids: list
            list of the available gpu_ids on the machine.
        """

    def __init__(self, model, batch_size, epochs, loss, optimizer, learning_rate, gpu_ids):
        self.batch_size = batch_size
        self.epochs = epochs
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if torch.cuda.is_available():
            cuda = 'cuda:' + str(gpu_ids[0])
            if isinstance(model, nn.DataParallel) == False:
                model = nn.DataParallel(model, device_ids=gpu_ids)
            loss.cuda()
        self.device = torch.device(cuda if torch.cuda.is_available() else 'cpu')
        self.model = model.double()
        self.model.to(self.device)

    def train_model_leave_one_patient_out(self, dataset, early_stopping = True, patience = 100, scheduler = None, patient_level = True, get_patient_prediction = True, return_history = True, val_size = 0.1, data_augmentation_online = True, data_augmentation_offline = True, f_name = "emsc", f_params = None):
        """
        Train the model with Leave One Patient Out Cross Validation.
        :param dataset: scikit_raman.Dataset
            a Dataset object on which train the model.
        :param early_stopping: bool
            if true early stopping during train is applied.
        :param patience: int
            value of patience for the early stopping.
        :param scheduler: torch.optim
            scheduler to apply in order to optimize the training.
        :param patient_level: bool
            if true the prediction based on patients are returned.
        :param get_patient_prediction: bool
            if true every prediction is returned.
        :param return_history:
            if true the history (loss and accuracy) of training and validation are returned.
        :param val_size: float
            determines the dimension of the validation set.
        :param data_augmentation_online: bool
            if true data augmentation with online approach is applied.
        :param data_augmentation_offline: bool
            if true data augmentation with offline approach is applied.
        :param f_name: str
            name from DataAugmenter class to apply for data augmentation.
        :param f_params: dict
            parameters for the data augmentation function.
        :return: dict
            returns a dictionary with the results based on the different choiches.
        """
        folds = dataset.leave_one_patient_cv()
        train_dataset, validation_dataset, test_dataset = self.create_dataset_pytorch(dataset, val_size, folds)
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        pat_pred_list = []
        pat_label_list = []
        pat_names_list = []
        histories = []
        names_list = []
        loss_train_list = []
        loss_val_list = []
        acc_train_list = []
        acc_val_list = []
        for i in range(len(train_dataset)):
            trained_model = copy.deepcopy(self.model)
            optim = copy.deepcopy(self.optimizer)
            if data_augmentation_offline:
                da = DataAugmenter(train_dataset[i].x, train_dataset[i].y)
                func = getattr(da, f_name)
                train_dataset[i].x, train_dataset[i].y = func(f_params)
            train_loss, val_loss, train_acc, val_acc = self.train_function(trained_model, optim, train_dataset[i], validation_dataset[i], early_stopping, patience, scheduler, data_augmentation_online, f_name, f_params)
            history = {'loss':train_loss, 'val_loss':val_loss, 'accuracy':train_acc, 'val_accuracy':val_acc}
            histories.append(history)
            loss_train_list.append(train_loss)
            loss_val_list.append(val_loss)
            acc_train_list.append(train_acc)
            acc_val_list.append(val_acc)
            labels, predicted_values = self.predict_function(trained_model, test_dataset)
            if get_patient_prediction:
                tot_pred_list.extend(predicted_values)
                tot_label_list.extend(labels)
                tot_label_list.extend(train_dataset[i][2])
            if patient_level:
                counts = np.bincount(predicted_values)
                pat_pred_list.append(np.argmax(counts))
                pat_label_list.append(labels[0])
                pat_label_list.append(np.unique(train_dataset[i][2]))
        dictionary = {}
        if patient_level:
            nested_dictionary = {'pred_list': pat_pred_list, 'label_list': pat_label_list, 'names_list': pat_names_list}
            dictionary['patient_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        if return_history:
            nested_dictionary = {'histories': histories, 'patients': names_list}
            dictionary['history'] = nested_dictionary
        return dictionary

    def create_dataset_pytorch(self, dataset, val_size, folds):
        """
        Create the dataset object for the training, validation and test set.
        :param dataset: scikit_raman.Dataset
            a dataset to divide and create pytorch dataset.
        :param val_size: float
            determines the dimension of the validation set.
        :param folds: list
            folds of the cross validation strategy.
        :return: list
            training set, validation set and test set as pytorch dataset.
        """
        #folds = dataset.leave_one_patient_cv()
        train_set = []
        validation_set = []
        test_set = []
        for j, (train_idx, test_idx) in enumerate(folds):
            X_train_tmp = dataset.spectra[train_idx]
            Y_train_tmp = dataset.labels[train_idx]
            user_train_tmp = dataset.user[train_idx]

            X_test_tmp = dataset.spectra[test_idx]
            Y_test_tmp = dataset.labels[test_idx]
            user_test_tmp = dataset.user[test_idx]

            X_train_tmp, X_val_tmp, Y_train_tmp, Y_val_tmp, user_train_tmp, user_val_tmp = train_test_split(X_train_tmp, Y_train_tmp, user_train_tmp, test_size=val_size,
                                                                              stratify=Y_train_tmp)
            train_set_tmp = PytorchDataset(X_train_tmp, Y_train_tmp, user_train_tmp)
            train_set.append(train_set_tmp)
            val_set_tmp = PytorchDataset(X_val_tmp, Y_val_tmp, user_val_tmp)
            validation_set.append(val_set_tmp)
            test_set_tmp = PytorchDataset(X_test_tmp, Y_test_tmp, user_test_tmp)
            test_set.append(test_set_tmp)

            return train_set, validation_set, test_set

    def train_function(self, model, optim, train_set, validation_set, early_stopping = True, patience = 100, scheduler = None, data_augmentation_online = False, f_name = "emsc", f_params = None):
        """
        It execute a train step.
        :param model: torch.nn.Module
            model to train.
        :param optim: torch.nn
            optimizer to train.
        :param train_set: scikit_raman.PytorchDataset
            Dataset on which execute training function.
        :param validation_set: scikit_raman.PytorchDataset
            Validation on which validate the model.
        :param early_stopping: bool
            if true early stopping is applied
        :param patience: int
            patience value for the early stopping
        :param scheduler: torch.optim
            scheduler to apply in order to optimize the training.
        :param data_augmentation_online: bool
            if true data augmentation with online approach is applied.
        :param f_name: str
            name from DataAugmenter class to apply for data augmentation.
        :param f_params: dict
            parameters for the data augmentation function.
        :return: list
            value of loss and accuracy for both training and test.
        """
        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []
        epochs_no_improve_loss = 0
        min_val_loss = np.Inf
        training_generator = DataLoader(train_set, batch_size=self.batch_size)
        validation_generator = DataLoader(validation_set, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            loss_train_epoch = []
            acc_train_epoch = []

            for i, (ramanSpectraTrain, labelTrain, user) in enumerate(training_generator):

                if data_augmentation_online:
                    da = DataAugmenter(ramanSpectraTrain, labelTrain)
                    func = getattr(da, f_name)
                    ramanSpectraTrain, labelTrain = func(f_params)

                ramanSpectraTrain = ramanSpectraTrain.to(self.device)
                labelTrain = labelTrain.to(self.device)

                optim.zero_grad()
                output_train = model(ramanSpectraTrain)
                loss_train = self.loss(output_train, labelTrain)
                loss_train_epoch.append(loss_train.cpu().item())

                loss_train.backward()
                optim.step()

                output_label = torch.argmax(output_train, dim = 1)

                acc_train = accuracy_score(labelTrain.cpu().detach().numpy(), output_label.cpu().detach().numpy())
                acc_train_epoch.append(acc_train)

            loss_train = mean(loss_train_epoch)
            acc_train = mean(acc_train_epoch)
            train_losses.append(loss_train)
            train_acc.append(acc_train)

            with torch.no_grad():
                loss_val_epoch = []
                acc_val_epoch = []
                for j, (ramanSpectraVal, labelVal, user) in enumerate(validation_generator):
                    ramanSpectraVal = ramanSpectraVal.to(self.device)
                    labelVal = labelVal.to(self.device)

                    output_val = model(ramanSpectraVal)

                    loss_val = self.loss(output_val, labelVal)
                    loss_val_epoch.append(loss_val.cpu().item())

                    val_label = torch.argmax(output_val, dim=1)
                    acc_val = accuracy_score(labelVal.cpu().detach().numpy(), val_label.cpu().detach().numpy())
                    acc_val_epoch.append(acc_val)
                loss_val = mean(loss_val_epoch)
                acc_val = mean(acc_val_epoch)
            val_losses.append(loss_val)
            val_acc.append(acc_val)
            if scheduler != None:
                scheduler.step(val_acc)
            #early - stopping
            if early_stopping:
                if loss_val < min_val_loss:
                    epochs_no_improve_loss = 0
                    min_val_loss = loss_val
                else:
                    epochs_no_improve_loss += 1
                if epochs_no_improve_loss == patience:
                    break
        return train_losses, val_losses, train_acc, val_acc

    def predict_function(self, test_set):
        """
        It apply a prediction
        :param test_set: scikit_raman.PytorchDataset
            a test set on which predict.
        :return: list
            lables and predicted values.
        """
        self.model.eval()
        labels = []
        predicted_labels = []
        with torch.no_grad():
            for j, (ramanSpectraVal, labelVal, user) in enumerate(test_set):
                ramanSpectraVal = ramanSpectraVal.to(self.device)
                labelVal = labelVal.to(self.device)
                output_val = self.model(ramanSpectraVal)
                test_label = torch.argmax(output_val, dim=1)
                labels.append(labelVal)
                predicted_labels.append(test_label)
        return labels, predicted_labels

    @classmethod
    def load_model_benchmark(dlm, gpu_ids):
        """
        Load the benchmark model.
        :param gpu_ids: list
            list of available gpus.
        :return: DLModel
            return an object of the class DLModel with the benchmark data.
        """
        model = BenchmarkModel()
        batch_size = 338
        epochs = 273
        loss = nn.CrossEntropyLoss()
        lr = 0.00020441990333108206
        optimizer = Adam(model.parameters(), lr=lr)
        return dlm(model, batch_size, epochs, loss, optimizer, lr, gpu_ids)

    def train_model_cv(self, dataset, k = 10, early_stopping = True, patience = 100, scheduler = None, fold_level = True, get_patient_prediction = True, return_history = True, val_size = 0.1, data_augmentation_online = True, data_augmentation_offline = True, f_name = "emsc", f_params = None):
        """
        Train the model with K-Fold Cross Validation.
        :param dataset: scikit_raman.Dataset
            Dataset on which train the model.
        :param k: int
            number of folds for the k-fold approach.
        :param early_stopping: bool
            if true early stopping is applied.
        :param patience: int
            value of the patience for the early stopping.
        :param scheduler: torch.optim
            scheduler to apply in order to optimize the training.
        :param fold_level: bool
            if true prediction for every patient of every fold is returned.
        :param get_patient_prediction: bool
            if true all prediction are returned.
        :param return_history: bool
            if true history of the training is returned.
        :param val_size: float
            dimension of the validation set.
        :param data_augmentation_online: bool
            if true data augmentation with online approach is applied.
        :param data_augmentation_offline: bool
            if true data augmentation with offline approach is applied.
        :param f_name: str
            name from DataAugmenter class to apply for data augmentation.
        :param f_params: dict
            parameters for the data augmentation function.
        :return: dict
            returns a dictionary with the results based on the choiches.
        """
        folds = dataset.k_fold(k)
        train_dataset, validation_dataset, test_dataset = self.create_dataset_pytorch(dataset, val_size, folds)
        tot_pred_list = []
        tot_label_list = []
        tot_names_list = []
        fold_pred_list = []
        fold_label_list = []
        fold_names_list = []
        histories = []
        names_list = []
        loss_train_list = []
        loss_val_list = []
        acc_train_list = []
        acc_val_list = []
        for i in range(len(train_dataset)):
            trained_model = copy.deepcopy(self.model)
            optim = copy.deepcopy(self.optimizer)
            if data_augmentation_offline:
                da = DataAugmenter(train_dataset[i].x, train_dataset[i].y)
                func = getattr(da, f_name)
                train_dataset[i].x, train_dataset[i].y = func(f_params)
            train_loss, val_loss, train_acc, val_acc = self.train_function(trained_model, optim, train_dataset[i], validation_dataset[i], early_stopping, patience, scheduler, data_augmentation_online, f_name, f_params)
            history = {'loss':train_loss, 'val_loss':val_loss, 'accuracy':train_acc, 'val_accuracy':val_acc}
            histories.append(history)
            loss_train_list.append(train_loss)
            loss_val_list.append(val_loss)
            acc_train_list.append(train_acc)
            acc_val_list.append(val_acc)
            labels, predicted_values = self.predict_function(trained_model, test_dataset)
            if get_patient_prediction:
                tot_pred_list.extend(predicted_values)
                tot_label_list.extend(labels)
                tot_label_list.extend(train_dataset[i][2])
            if fold_level:
                labels = []
                list_pred = []
                for el in np.unique(train_dataset[i][2]):
                    l = []
                    l_v = []
                    for j in range(len(train_dataset[i][2])):
                        if train_dataset[i][2][j] == el:
                            l_v.append(labels[j])
                            l.append(predicted_values[j])
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
                fold_names_list.append(np.unique(train_dataset[i][2]))
        dictionary = {}
        if fold_level:
            nested_dictionary = {'pred_list': fold_pred_list, 'label_list': fold_label_list,
                                 'names_list': fold_names_list}
            dictionary['fold_level'] = nested_dictionary
        if get_patient_prediction:
            nested_dictionary = {'pred_list': tot_pred_list, 'label_list': tot_label_list, 'names_list': tot_names_list}
            dictionary['total_prediction'] = nested_dictionary
        if return_history:
            nested_dictionary = {'histories': histories, 'patients': names_list}
            dictionary['history'] = nested_dictionary
        return dictionary




