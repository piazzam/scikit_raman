import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold
import pickle
from sklearn.utils import shuffle

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
        self.n_elements = len(spectra)
        #self.label_dictionary = label_dictionary
        if labels.size != 0:
            self.labels = labels
        else:
            self.create_label(label_dictionary)
        self.n_dims = self.spectra.shape[1]

    def __getitem__(self, items):
        return self.spectra[items], self.x_axis[items], self.raw[items], self.user[items], self.name[items], \
            self.category[items], self.labels[items]

    def __len__(self):
        return self.n_elements

    def extend(self, dataset):
        self.spectra = np.append(self.spectra, dataset.spectra, axis=0)
        self.x_axis = np.append(self.x_axis, dataset.x_axis, axis=0)
        self.raw = np.append(self.raw, dataset.raw, axis=0)
        self.user = np.append(self.user, dataset.user, axis=0)
        self.name = np.append(self.name, dataset.name, axis=0)
        self.category = np.append(self.category,dataset.category, axis=0)
        self.labels = np.append(self.labels, dataset.labels, axis=0)
        self.n_elements = self.n_elements + len(dataset)

    @classmethod
    def load_file(ds, file_type, file_name, label_dictionary = {'cov':0, 'covNeg': 1, 'ctrl':2}):
        """
        Load a file and automatically create a Dataset object.
        :param ds: scikit_raman.Dataset
            Dataset object to be returned.
        :param file_type: str
            type of file to load.
        :param file_name: str
            name of the file to load
        :param label_dictionary: dict, optional
            dictionary for mapping category (str) in numeric label.
        :return: a dataset object loaded from the file.
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
        if 'labels' in df.columns:
            labels = df['label'].to_numpy()
        else:
            labels = np.empty(shape = (0,0))
        return ds(spectra, x_axis, raw, user, name, category, labels, label_dictionary)

    def create_label(self, label_dictionary):
        """
        Create numeric labels for the dataset object.
        :param label_dictionary: dict
            dictionary on which rely for the numeric labels.
        :return:
        """
        labels = []
        for el in self.category:
            labels.append(label_dictionary[el])
        self.labels = np.array(labels)

    def to_df(self, save_label = True):
        if save_label:
            df = pd.DataFrame([], columns=['spectra', 'x-axis', 'raw', 'user', 'name', 'category', 'labels'])
        else:
            df = pd.DataFrame([], columns=['spectra', 'x-axis', 'raw', 'user', 'name', 'category'])
        df['spectra'] = self.spectra.tolist()
        df['x-axis'] = self.x_axis.tolist()
        df['raw'] = self.raw.tolist()
        df['user'] = self.user.tolist()
        df['name'] = self.name.tolist()
        df['category'] = self.category.tolist()
        if save_label:
            df['labels'] = self.labels.tolist()
        return df

    def save_pickle(self, filename, save_label = True):
        df = self.to_df(save_label)
        with open(filename, 'wb') as f:
            pickle.dump(df, f)

    def get_raw_data(self):
        """
        Return a new Dataset object with only raw data.
        :return: scikit_raman.Dataset
            object with raw data.
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
        :return: scikit_raman.Dataset
            dataset object with dark data.
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

    def get_by_indices(self, indices):
        spectra = []
        x_axis = []
        raw = []
        user = []
        name = []
        category = []
        labels = []
        for index in indices:
            spectra.append(self.spectra[index])
            x_axis.append(self.x_axis[index])
            raw.append(self.raw[index])
            user.append(self.user[index])
            name.append(self.name[index])
            category.append(self.category[index])
            labels.append(self.labels[index])
        ds = Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                     np.array(category), np.array(labels))
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
        :return: list
            Different folds for the different cv cycle.
        """
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(GroupKFold(n_splits=k).split(x_train, y_train, groups=groups))
            return folds

    def stratified_k_fold(self, k, random_state=42):
        """
        Implements the k-fold strategies. It preserves the patients.
        :param k: int
            Number of the fold
        :return: list
            Different folds for the different cv cycle.
        """
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(StratifiedGroupKFold(n_splits=k).split(x_train, y_train, groups=groups, random_state=random_state))
            return folds

    def leave_one_patient_cv(self):
        """
        Apply leave-one-patient out cross-validation.
        :return: list
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

    def get_unique_category(self):
        """
        Get a numpy array with unique category values.
        :return: np.array:
            contains unique value of the categories
        """
        return np.unique(self.category)

    def get_unique_labels(self):
        """
        Get a numpy array with unique labels values.
        :return: np.array:
            contains unique value of the labels
        """
        return np.unique(self.labels)

    def get_unique_user(self):
        """
        Get a numpy array with unique user names.
        :return: np.array:
            contains unique value of the users.
        """
        return np.unique(self.user)

    def remove_elements(self, elements):
        """
        Remove elements from the dataset.
        :param elements:list
            list of the elements to remove.
        """
        self.spectra = np.delete(self.spectra, elements, axis = 0)
        self.x_axis = np.delete(self.x_axis, elements, axis = 0)
        self.raw = np.delete(self.raw, elements, axis = 0)
        self.name = np.delete(self.name, elements, axis = 0)
        self.user = np.delete(self.user, elements, axis = 0)
        self.category = np.delete(self.category, elements, axis = 0)
        self.labels = np.delete(self.labels, elements, axis = 0)
        self.n_elements -= len(elements)

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

    def search_by_name(self, name):
        """
        Search and return subset of the dataset corresponding to the specified user.
        :param name:string
            String corresponding to the user name of the patients.
        :return: scikit_raman.Dataset
            A dataset object representing the subset of the Dataset.
        """
        ret_list = []
        for i in range(len(self.user)):
            if self.user[i] == name:
                ret_list.append(i)
        spectra = []
        x_axis = []
        raw = []
        user = []
        name = []
        category = []
        labels = []
        for i in ret_list:
            spectra.append(self.spectra[i])
            x_axis.append(self.x_axis[i])
            raw.append(self.raw[i])
            user.append(self.user[i])
            name.append(self.name[i])
            category.append(self.category[i])
            labels.append(self.labels[i])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name), np.array(category), np.array(labels))

    def search_by_name_indices(self, name):
        """
        Search and return subset of the dataset corresponding to the specified user.
        :param name:string
            String corresponding to the user name of the patients.
        :return: scikit_raman.Dataset
            A dataset object representing the subset of the Dataset.
        """
        ret_list = []
        for i in range(len(self.user)):
            if self.user[i] == name:
                ret_list.append(i)
        return ret_list

    def search_by_category_name(self, cat):
        """
        Search and return subset of the dataset corresponding to the specified label name.
        :param name:string
            String corresponding to the label.
        :return: scikit_raman.Dataset
            A dataset object representing the subset of the Dataset.
        """
        ret_list = []
        for i in range(len(self.category)):
            if self.category[i] == cat:
                ret_list.append(i)
        spectra = []
        x_axis = []
        raw = []
        user = []
        name = []
        category = []
        labels = []
        for i in ret_list:
            spectra.append(self.spectra[i])
            x_axis.append(self.x_axis[i])
            raw.append(self.raw[i])
            user.append(self.user[i])
            name.append(self.name[i])
            category.append(self.category[i])
            labels.append(self.labels[i])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name), np.array(category), np.array(labels))

    def search_by_category_label(self, lab):
        """
        Search and return subset of the dataset corresponding to the specified user.
        :param name:int
            Int corresponding to the user category of the patients.
        :return: scikit_raman.Dataset
            A dataset object representing the subset of the Dataset.
        """
        ret_list = []
        for i in range(len(self.labels)):
            if self.labels[i] == lab:
                ret_list.append(i)
        spectra = []
        x_axis = []
        raw = []
        user = []
        name = []
        category = []
        labels = []
        for i in ret_list:
            spectra.append(self.spectra[i])
            x_axis.append(self.x_axis[i])
            raw.append(self.raw[i])
            user.append(self.user[i])
            name.append(self.name[i])
            category.append(self.category[i])
            labels.append(self.labels[i])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                       np.array(category), np.array(labels))

    def create_mean_spectra(self):
        users = self.get_unique_user()
        spectra = []
        x_axis = []
        raw = []
        user_list = []
        name = []
        category = []
        labels = []
        for user in users:
            ds_user = self.search_by_name(user)
            spectrum = np.mean(ds_user.spectra, axis=0)
            spectra.append(spectrum)
            x_axis.append(ds_user.x_axis[0])
            raw.append(ds_user.raw[0])
            user_list.append(ds_user.user[0])
            name.append(ds_user.name[0])
            category.append(ds_user.category[0])
            labels.append(ds_user.labels[0])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user_list), np.array(name),
                       np.array(category), np.array(labels))