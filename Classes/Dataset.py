import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

class Dataset:
    """
    A class to represent a scikit-raman Dataset.

    ...

    Attributes
    ----------
    spectra : list
        list of the spectra of the dataset.
    x_axis : list
        list of the x_axis of all the spectra of the dataset.
    raw : list
        list of bool that represent if that spectra is raw or dark.
    user: list
        list of string that contains the code of every patients of the dataset.
    name: list
        list of string that represents if the spectra is raw or dark.
    category: list
        list of category of the spectra (Cov, CovNeg, C).
    labels: list
        list of labels in numeric form.
    """

    def __init__(self, spectra, x_axis, raw, user, name, category, labels, label_dictionary = {'cov':0, 'covNeg': 1, 'ctrl':2}):
        self.spectra = spectra
        self.spectra_to_numpy()
        self.x_axis = x_axis
        self.x_axis_to_numpy()
        self.raw = raw
        self.user = user
        self.name = name
        self.category = category
        #self.label_dictionary = label_dictionary
        if labels != []:
            self.labels = labels
        else:
            self.create_label(label_dictionary)


    @classmethod
    def load_file(ds, file_type, file_name, label_dictionary = {'cov':0, 'covNeg': 1, 'ctrl':2}):
        """
        Load a file and automatically create a Dataset object.
        :param ds: Dataset object to be returned.
        :param file_type: type of file to load.
        :param file_name: name of the file to load
        :param label_dictionary: dictionary for mapping category (str) in numeric label
        :return: a dataset object loaded from pickle. The pickle must be a pd.DataFrame well-formatted
                followed our policies.
        """
        if file_type == 'pkl':
            df = pd.read_pickle(file_name)
        elif file_type == 'csv':
            df = pd.read_csv(file_name)
        else:
            print("Error file type not supported")
        spectra = df['spectra'].to_numpy()
        x_axis = df['x-axis'].to_numpy()
        raw = df['raw'].to_numpy()
        user = df['user'].to_numpy()
        name = df['name'].to_numpy()
        category = df['category'].to_numpy()
        if 'label' in df.columns:
            labels = df['label'].to_numpy()
        else:
            labels = np.empty(len(spectra))
        return ds(spectra, x_axis, raw, user, name, category, labels, label_dictionary)

    def create_label(self, label_dictionary):
        """
        Create numeric labels for the dataset object.
        """
        labels = np.empty(len(self.spectra))
        for el in self.category:
            np.append(labels, label_dictionary[el])
        self.labels = labels

    def get_raw_data(self):
        """
        Return a new Dataset object with only raw data.
        :return: Dataset object
        """
        spectra = []
        x_axis = []
        raw = []
        user = []
        name = []
        category = []
        labels = []
        for i in range(len(self.spectra)):
            if self.raw[i] == True:
                spectra.append(self.spectra[i])
                x_axis.append(self.x_axis[i])
                raw.append(self.raw[i])
                user.append(self.user[i])
                name.append(self.name[i])
                category.append(self.category[i])
                labels.append(self.labels[i])
        ds = Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name), np.array(category), np.array(labels))
        return ds

    def get_dark_data(self):
        """
        Return new dataset with only dark data.
        :return: Dataset
        """
        spectra = []
        x_axis = []
        raw = []
        user = []
        name = []
        category = []
        labels = []
        for i in range(len(self.spectra)):
            if self.raw[i] == False:
                spectra.append(self.spectra[i])
                x_axis.append(self.x_axis[i])
                raw.append(self.raw[i])
                user.append(self.user[i])
                name.append(self.name[i])
                category.append(self.category[i])
                labels.append(self.labels[i])
        ds = Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name), np.array(category), np.array(labels))
        return ds

    def change_category_name(self, dictionary):
        """
        Change the name of the category in string form based on dictionary.
        :param dictionary: dict
            Map old names with new names.
        """
        new_category = []
        for el in self.category:
            new_category.append(dictionary[el])
        self.category = np.array(new_category)

    def change_user_name_string(self, dictionary):
        """
        Change the name of the user name based on dictionary.
        :param dictionary: dict
            Map old names with new names.
        """
        new_users = []
        for el in self.user:
            l = el.split('_')
            for d in dictionary:
                if d == l[0]:
                    l[0] = dictionary[d]
                    break
            string_name = l[0] + '_' + l[1]
            new_users.append(string_name)
        self.user = np.array(new_users)

    def change_category_and_user(self, dictionary):
        """
        Change both category and user names based on the dictionary.
        :param dictionary: dict
            Map old names with new names.
        :return:
        """
        self.change_category_name(dictionary)
        self.change_user_name_string(dictionary)

    def k_fold(self, k):
        """
        Implements the k-fold strategies. It preserves the patients.
        :param k: int
            Number of the fold
        :return:
            folds: (j, (train_ids, test_ids))
                Different fold for the different cv cycle.
        """
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(GroupKFold(n_splits=k).split(x_train, y_train, groups=groups))
            return folds

    def leave_one_patient_cv(self):
        """
        Apply leave-one-patient out cross-validation.
        :return:
            folds: (j, (train_ids, test_ids))
                Different folds for the different cv cycle. J corresponds to the number of
                patients.
        """
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(LeaveOneGroupOut().split(x_train, y_train, groups=groups))
            return folds

    def spectra_to_numpy(self):
        """
        Transform the spectra list in numpy.array
        :return:
        """
        spectra_list = []
        for el in self.spectra:
            spectra_list.append(np.array(el))
        spectra_list = np.array(spectra_list)
        self.spectra = spectra_list

    def x_axis_to_numpy(self):
        """
        Transform the spectra list in numpy.array
        :return:
        """
        x_axis_list = []
        for el in self.x_axis:
            x_axis_list.append(np.array(el))
        x_axis_list = np.array(x_axis_list)
        self.x_axis = x_axis_list

    def get_unique_category(self):
        """
        Get a numpy array with unique category values.
        :return:
            np.array:
                contains unique value of the categories
        """
        return np.unique(self.category)

    def get_unique_user(self):
        """
        Get a numpy array with unique user names.
        :return:
            np.array:
                contains unique value of the users.
        """
        return np.unique(self.user)

    def remove_elements(self, elements):
        self.spectra = np.delete(self.spectra, elements, axis = 0)
        self.x_axis = np.delete(self.x_axis, elements, axis = 0)
        self.raw = np.delete(self.raw, elements, axis = 0)
        self.name = np.delete(self.name, elements, axis = 0)
        self.user = np.delete(self.user, elements, axis = 0)
        self.category = np.delete(self.category, elements, axis = 0)
        self.labels = np.delete(self.labels, elements, axis = 0)
