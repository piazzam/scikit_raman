from sklearn.model_selection import train_test_split
from scikit_raman.Classes.PyTorch.PytorchDataset import PytorchDataset
from scikit_raman.Classes.PyTorch.BenchmarkModel import BenchmarkModel
import copy
from torch.utils.data import DataLoader
import torch
from sklearn.metrics import accuracy_score
from statistics import mean
import numpy as np
import torch.nn as nn
from torch.optim import Adam

class DLModel:

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

    def train_model_leave_one_patient_out(self, dataset, number_classes, patient_level = True, get_patient_prediction = True, return_history = True, val_size = 0.1):
        train_dataset, validation_dataset, test_dataset = self.create_dataset_pytorch(dataset, val_size)
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
            train_loss, val_loss, train_acc, val_acc = self.train_function(trained_model, optim, train_dataset[i], validation_dataset[i])
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

    def create_dataset_pytorch(self, dataset, val_size):
        folds = dataset.leave_one_patient_cv()
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

    def train_function(self, model, optim, train_set, validation_set):
        train_losses = []
        val_losses = []
        train_acc = []
        val_acc = []
        training_generator = DataLoader(train_set, batch_size=self.batch_size)
        validation_generator = DataLoader(validation_set, batch_size=self.batch_size)
        for epoch in range(self.epochs):
            loss_train_epoch = []
            acc_train_epoch = []

            for i, (ramanSpectraTrain, labelTrain, user) in enumerate(training_generator):

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
        return train_losses, val_losses, train_acc, val_acc

    def predict_function(self, model, test_set):
        model.eval()
        labels = []
        predicted_labels = []
        with torch.no_grad():
            for j, (ramanSpectraVal, labelVal) in enumerate(validation_generator):
                ramanSpectraVal = ramanSpectraVal.to(self.device)
                labelVal = labelVal.to(self.device)
                output_val = model(ramanSpectraVal)
                test_label = torch.argmax(output_val, dim=1)
                labels.append(labelVal)
                predicted_labels.append(test_label)
        return labels, predicted_labels

    @classmethod
    def load_model_benchmark(dlm, gpu_ids):
        model = BenchmarkModel()
        batch_size = 338
        epochs = 273
        loss = nn.CrossEntropyLoss()
        lr = 0.00020441990333108206
        optimizer = Adam(model.parameters(), lr=lr)
        return dlm(model, batch_size, epochs, loss, optimizer, lr, gpu_ids)




