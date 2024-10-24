import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, StratifiedGroupKFold, KFold
import pickle
import scikit_raman.module.utility as utility


class Dataset:
    """
    A class that represents a Dataset object for scikit_raman library.

    ...

    Attributes
    ----------
    spectra : np.array
        Array of spectra contained in the dataset.
    x_axis : np.array
        Array of x_axis for each spectra contained in the dataset.
    user: np.Array
        Array of strings containing the code of patients.
    category: np.Array
        Array containing the patient's label in string form.
    assumed_drugs: np.Array
        Array containing the drugs taken by each patient.
    labels: np.Array
        Array contaning the patient's label in numerical form.

    Methods
    -------

    extend(self, dataset)
        Add another dataset in tail to the one instantiated.
    load_file(ds, file_type, file_name, parquet_engine='fastparquet',
                  label_dictionary={'cov': 0, 'covNeg': 1, 'ctrl': 2})
        Load a file and automatically instantiates a scikit_raman.Dataset object.
    load_drugs(self, drugs, drugs_category)
        Load the drugs taken by each patient.
    create_label(self, label_dictionary)
        Starting from the string labels it creates the numerical version.
    to_df(self, save_label=True)
        Convert the scikit_raman.Dataset into a pandas.Dataframe object.
    save_pickle(self, filename, save_label=True)
        Convert the scikit_raman.Dataset into a pandas.Dataframe object and then store it as a pickle file.
    save_parquet(self, filename, save_label=True, engine='fastparquet')
        Convert the scikit_raman.Dataset into a pandas.Dataframe object and then store it as a parquet file.
    get_raw_data(self)
        Return a subset of Dataset corresponding only to raw data.
    get_dark_data(self)
        Return a subset of Dataset corresponding only to dark data.
    get_by_indices(self, indices)
        Return a subset of Dataset by specifying the indexes of desired elements.
    change_category_name(self, dictionary)
        Change the name associated to a category.
    change_label_name(self, dictionary)
        Change the numeric value of a label.
    change_user_name_string(self, dictionary)
        Change the string name of a patient.
    change_category_and_user(self, dictionary)
        Change the name associated to a category and the corresponding user code.
    k_fold(self, k)
        Create folds for k-fold procedure from Dataset object. It preserves the patients among the folds.
    simple_k_fold(self, k)
        Create folds for k-fold procedure from Dataset object.
    stratified_k_fold(self, k)
        Create folds for k-fold procedure from Dataset object. It is stratified among labels.
    leave_one_patient_cv(self):
        Create folds for leave-one-patient-out cross validation procedure from Dataset object.
    get_unique_category(self):
        Return a numpy array with unique categories' name.
    get_unique_labels(self)
        Return a numpy array with unique label's numerical value.
    get_unique_user(self)
        Return a numpy array with unique user's string name.
    remove_elements(self, elements)
        Remove elements from Dataset
    spectra_to_numpy(self)
        Transform the spectra list in np.array
    x_axis_to_numpy(self)
        Transform the spectra list in numpy.array
    search_by_name(self, name)
        Search and return subset of the dataset corresponding to the specified user.
    search_by_name_indices(self, name)
        Search and return subset of the dataset corresponding to the specified user.
    search_by_category_name(self, cat)
        Search and return subset of the dataset corresponding to the specified label name.
    search_by_category_label(self, lab)
        Search and return subset of the dataset corresponding to the specified user.
    create_mean_spectra(self)
        Create a new dataset with teh averaged spectra for each patient as a new spectra.
    cut_based_xaxis(self, start, end):
        Cut spectra and x-axis based on value on positions on x-axis
    """

    def __init__(self, spectra, x_axis, raw, user, name, category, labels,
                 assumed_drugs=[], label_dictionary={'cov': 0, 'covNeg': 1, 'ctrl': 2}):
        """
        Constructor for Dataset class

        Parameters
        ----------
        spectra : list
            List of spectra.
        x_axis : list
            List of x-axis.
        raw : list
            List of boolean corresponding to be be a raw or a dark element.
        user : list
            List of user codes.
        name : list
            List of stirng corresponding to raw or dark data.
        category : list
            List of string corresponding to the category of patients.
        labels : list
            List of int corresponding to numerical labels.
        assumed_drugs: list, optional (default is [])
            List of int corresponding to assumed drugs
        label_dictionary dict, optional (default is {'cov': 0, 'covNeg': 1, 'ctrl': 2})
            Dictionary for mapping string labels into numerical form.
        """
        self.spectra = spectra
        self.spectra_to_numpy()
        self.x_axis = x_axis
        self.x_axis_to_numpy()
        self._raw = raw
        self.user = user
        self._name = name
        self.category = category
        self.assumed_drugs = np.array(assumed_drugs)
        self._n_elements = len(spectra)
        if labels.size != 0:
            self.labels = labels
        else:
            self.create_label(label_dictionary)
        self.n_dims = self.spectra.shape[1]

    def __getitem__(self, items):
        """
        Get subset of elements of the dataset based on integer position
        Parameters
        ----------
        items : int
            Integer position of elements to be returned

        Returns
        -------
            np.array
                Spectra at specified position
            np.array
                X-Axis at specified position
            np.array
                Raw at specified position
            np.array
                User at specified position
            np.array
                Name at specified position
            np.array
                Category at specified position
            np.array
                Labels at specified position
            np.array
                Assumed_drugs at specified position
        """
        try:
            return self.spectra[items], self.x_axis[items], self._raw[items], self.user[items], self._name[items], \
                   self.category[items], self.labels[items], self.assumed_drugs[items]
        except IndexError:
            return self.spectra[items], self.x_axis[items], self._raw[items], self.user[items], self._name[items], \
                   self.category[items], self.labels[items]

    def __len__(self):
        """
        Overriding of len function
        Returns
        -------
        int
            number of elements contained in the Dataset
        """
        return self._n_elements

    def extend(self, dataset):
        self.spectra = np.append(self.spectra, dataset.spectra, axis=0)
        self.x_axis = np.append(self.x_axis, dataset.x_axis, axis=0)
        self._raw = np.append(self._raw, dataset._raw, axis=0)
        self.user = np.append(self.user, dataset.user, axis=0)
        self._name = np.append(self._name, dataset._name, axis=0)
        self.category = np.append(self.category, dataset.category, axis=0)
        self.labels = np.append(self.labels, dataset.labels, axis=0)
        self._n_elements = self._n_elements + len(dataset)

    @classmethod
    def load_file(ds, file_type, file_name, parquet_engine='fastparquet',
                  label_dictionary={'cov': 0, 'covNeg': 1, 'ctrl': 2}):
        """
        Load a file and automatically create a Dataset object.

        Parameters
        ----------
        file_type : str
            String corresponding to the file type to be loaded.
        file_name : str
            String corresponding to the file path to be loaded.
        parquet_engine : str, optional (default is fastparquet)
            String corresponding to parquet engine for loading parquet file.
        label_dictionary : dict, optional (default is {'cov': 0, 'covNeg': 1, 'ctrl': 2})
            Dictionary for mapping string labels into numerical forms.

        Returns
        -------
        scikit_raman.Dataset
            Created dataset object.
        """
        if file_type == 'pkl':
            df = pd.read_pickle(file_name)
        elif file_type == 'csv':
            df = pd.read_csv(file_name)
        elif file_type == 'parquet':
            df = pd.read_parquet(file_name, engine=parquet_engine)
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
            labels = np.empty(shape=(0, 0))
        return ds(spectra, x_axis, raw, user, name, category, labels, [], label_dictionary)

    def load_drugs(self, drugs, drugs_category):
        """
        Load and assign to each its assumed drugs.

        Parameters
        ----------
        drugs: pandas.Dataframe
            Mapping each user into its assumed drugs
        drugs_category: list
            List of patient categories related to the drugs being loaded.
        """
        assumed_drugs = []
        for i, u in enumerate(self.user):
            if self.category[i] in drugs_category:
                this_user = drugs[drugs['Codice Labion'] == u]
                assumed_drug = [this_user['farmaco1'], this_user['farmaco2'], this_user['farmaco3'], this_user['farmaco4'],
                                this_user['farmaco5'], this_user['farmaco6']]
                assumed_drugs.append(assumed_drug)
        self.assumed_drugs = np.array(assumed_drugs)

    def create_label(self, label_dictionary):
        """
        Translate string labels into numerical version.

        Parameters
        ----------
        label_dictionary : dict
            Dictionary for mapping string labels into numerical version.
        """
        labels = []
        for el in self.category:
            labels.append(label_dictionary[el])
        self.labels = np.array(labels)

    def to_df(self, save_label=True):
        """
        Convert the Dataset into a pandas.Dataframe object.

        Parameters
        ----------
        save_label : bool, optional (defualt is True)
            Whether to also add numerical labels to the df.

        Returns
        -------
        pandas.Dataframe
            Translated dataframe.

        """
        if save_label:
            df = pd.DataFrame([], columns=['spectra', 'x-axis', 'raw', 'user', 'name', 'category', 'label'])
        else:
            df = pd.DataFrame([], columns=['spectra', 'x-axis', 'raw', 'user', 'name', 'category'])
        df['spectra'] = self.spectra.tolist()
        df['x-axis'] = self.x_axis.tolist()
        df['raw'] = self._raw.tolist()
        df['user'] = self.user.tolist()
        df['name'] = self._name.tolist()
        df['category'] = self.category.tolist()
        if save_label:
            df['labels'] = self.labels.tolist()
        return df

    def save_pickle(self, filename, save_label=True):
        """
        Convert the dataset object into a pandas.Dataframe and then store it as a pickle file.

        Parameters
        ----------
        filename : str
            Filename of the pickle file.
        save_label : bool, optional (default is True)
            Whether to include numerical labels into the saved file.
        """
        df = self.to_df(save_label)
        with open(filename, 'wb') as f:
            pickle.dump(df, f)

    def save_parquet(self, filename, save_label=True, engine='fastparquet'):
        """
        Convert the dataset object into a pandas.Dataframe and then store it as a parquet file.

        Parameters
        ----------
        filename : str
            Filename of the parquet file.
        save_label : bool, optional (default is True)
            Whether to include numerical labels into the saved file.
        engine : str, optional (default is 'fastparquet')
            Parquet - engine for saving the file.
        """
        df = self.to_df(save_label)
        df.to_parquet(filename, engine=engine)

    def get_raw_data(self):
        """
        Return a new dataset composed of only raw data.

        Returns
        -------
        scikit_raman.Dataset
            Object composed of raw data.
        """
        spectra = []
        x_axis = []
        raw = []
        user = []
        name = []
        category = []
        labels = []
        for i in range(len(self.spectra)):
            if self._raw[i] == True:
                spectra.append(self.spectra[i])
                x_axis.append(self.x_axis[i])
                raw.append(self._raw[i])
                user.append(self.user[i])
                name.append(self._name[i])
                category.append(self.category[i])
                labels.append(self.labels[i])
        ds = Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                     np.array(category), np.array(labels))
        return ds

    def get_dark_data(self):
        """
        Return a new dataset composed of only dark data.

        Returns
        -------
        scikit_raman.Dataset
            Object composed of dark data.
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
                raw.append(self._raw[i])
                user.append(self.user[i])
                name.append(self._name[i])
                category.append(self.category[i])
                labels.append(self.labels[i])
        ds = Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                     np.array(category), np.array(labels))
        return ds

    def get_by_indices(self, indices):
        """
        Return a new Dataset object composed of a subset of elements, specified by indexes value.

        Parameters
        ----------
        indices : list
            List of integer value.

        Returns
        -------
        scikit_raman.Dataset
            Reduced size object
        """
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
            raw.append(self._raw[index])
            user.append(self.user[index])
            name.append(self._name[index])
            category.append(self.category[index])
            labels.append(self.labels[index])
        ds = Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                     np.array(category), np.array(labels))
        return ds

    def change_category_name(self, dictionary):
        """
        Change the name of the category in string form.

        Parameters
        ----------
        dictionary : dict
            Map old names with new names.
        """
        new_category = []
        for el in self.category:
            new_category.append(dictionary[el])
        self.category = np.array(new_category)

    def change_label_name(self, dictionary):
        """
        Change the name of the category in string form.

        Parameters
        ----------
        dictionary : dict
            Map old names with new names.
        """
        new_label = []
        for el in self.labels:
            new_label.append(dictionary[el])
        self.labels = np.array(new_label)

    def change_user_name_string(self, dictionary):
        """
        Change the name of the user name

        Parameters
        ----------
        dictionary: dict
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
        Change both category and user names.

        Parameters
        ----------
        dictionary : dict
            Map old names with new names.
        """
        self.change_category_name(dictionary)
        self.change_user_name_string(dictionary)

    def k_fold(self, k):
        """
        Implements the k-fold strategies, with preservation of patients between different folds.

        Parameters
        ----------
        k : int
            Number of folds in k-fold procedure

        Returns
        -------
        np.array
            The training set indices for that split.
        np.array
            The testing set indices for that split.
        """
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(GroupKFold(n_splits=k).split(x_train, y_train, groups=groups))
            return folds

    def simple_k_fold(self, k):
        """
        Implements the k-fold strategies, without preservation of patients between different folds.

        Parameters
        ----------
        k : int
            Number of folds in k-fold procedure

        Returns
        -------
        np.array
            The training set indices for that split.
        np.array
            The testing set indices for that split.
        """
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            folds = list(KFold(n_splits=k).split(x_train, y_train))
            return folds

    def stratified_k_fold(self, k):
        """
        Implements the k-fold strategies, with stratification of labels between folds.

        Parameters
        ----------
        k : int
            Number of folds in k-fold procedure

        Returns
        -------
        np.array
            The training set indices for that split.
        np.array
            The testing set indices for that split.
        """
        x_train = self.spectra
        if not hasattr(self, 'labels'):
            raise Exception("This dataset doesn't have labels")
        else:
            y_train = self.labels
            groups = self.user
            folds = list(StratifiedGroupKFold(n_splits=k).split(x_train, y_train, groups=groups))
            return folds

    def leave_one_patient_cv(self):
        """
        Implements leave-one-patient out cross-validation.

        Returns
        -------

        np.array
            The training set indices for that split.
        np.array
            The testing set indices for that split.
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

        Returns
        -------
        np.array
            Array with unique value of categories.

        """
        return np.unique(self.category)

    def get_unique_labels(self):
        """
        Get a numpy array with unique labels values.

        Returns
        -------
        np.array
            Array with unique value of categories.

        """
        return np.unique(self.labels)

    def get_unique_user(self):
        """
        Get a numpy array with unique user names.

        Returns
        -------

        np.array
            Array with unique value of user code.
        """
        return np.unique(self.user)

    def remove_elements(self, elements):
        """
        Remove specified elements from the dataset.

        Parameters
        ----------
        elements: list
            List of elements to be removed.
        """
        self.spectra = np.delete(self.spectra, elements, axis=0)
        self.x_axis = np.delete(self.x_axis, elements, axis=0)
        self._raw = np.delete(self._raw, elements, axis=0)
        self._name = np.delete(self._name, elements, axis=0)
        self.user = np.delete(self.user, elements, axis=0)
        self.category = np.delete(self.category, elements, axis=0)
        self.labels = np.delete(self.labels, elements, axis=0)
        self._n_elements -= len(elements)

    def spectra_to_numpy(self):
        """
        Transform the spectra list in np.array
        """
        spectra_list = []
        for el in self.spectra:
            spectra_list.append(np.array(el))
        spectra_list = np.array(spectra_list)
        self.spectra = spectra_list

    def x_axis_to_numpy(self):
        """
        Transform the spectra list in numpy.array
        """
        x_axis_list = []
        for el in self.x_axis:
            x_axis_list.append(np.array(el))
        x_axis_list = np.array(x_axis_list)
        self.x_axis = x_axis_list

    def search_by_name(self, name):
        """
        Search and return subset of the dataset corresponding to the specified user.

        Parameters
        ----------
        name : str
            User code to be searched

        Returns
        -------
        scikit_raman.Dataset
            Dataset object comprising of the subset related to the specified user.
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
            raw.append(self._raw[i])
            user.append(self.user[i])
            name.append(self._name[i])
            category.append(self.category[i])
            labels.append(self.labels[i])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                       np.array(category), np.array(labels))

    def search_by_name_indices(self, name):
        """
        Search and return a list of int corresponding to indexes of the specified user.

        Parameters
        ----------
        name : str
            User code to be searched

        Returns
        -------
        list
            List of integers containing indexes of user's name.

        """
        ret_list = []
        for i in range(len(self.user)):
            if self.user[i] == name:
                ret_list.append(i)
        return ret_list

    def search_by_category_name(self, cat):
        """
        Search and return subset of the dataset corresponding to the specified label name.

        Parameters
        ----------
        cat : str
            String corresponding to the label.

        Returns
        -------
        scikit_raman.Dataset
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
            raw.append(self._raw[i])
            user.append(self.user[i])
            name.append(self._name[i])
            category.append(self.category[i])
            labels.append(self.labels[i])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                       np.array(category), np.array(labels))

    def search_by_category_label(self, lab):
        """
        Search and return subset of the dataset corresponding to the specified user.

        Parameters
        ----------
        lab : int
            Int corresponding to the user category of the patients.

        Returns
        -------
        scikit_raman.Dataset
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
            raw.append(self._raw[i])
            user.append(self.user[i])
            name.append(self._name[i])
            category.append(self.category[i])
            labels.append(self.labels[i])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user), np.array(name),
                       np.array(category), np.array(labels))

    def create_mean_spectra(self):
        """
        Create a new Dataset object with average spectra for each patient.

        Returns
        -------
        scikit_raman.Dataset
            New dataset object

        """
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
            raw.append(ds_user._raw[0])
            user_list.append(ds_user.user[0])
            name.append(ds_user._name[0])
            category.append(ds_user.category[0])
            labels.append(ds_user.labels[0])
        return Dataset(np.array(spectra), np.array(x_axis), np.array(raw), np.array(user_list), np.array(name),
                       np.array(category), np.array(labels))

    def cut_based_xaxis(self, start, end):
        """
        Cut spectra and x-axis based on value on positions on x-axis

        Parameters
        ----------
        start : int
            Starting point on x-axis
        end : int
            Ending point on x-axis

        """
        new_spectra = []
        new_x_axis = []
        for x, y in zip(self.x_axis, self.spectra):
            _, idx_start = utility.find_nearest(x, start)
            _, idx_end = utility.find_nearest(x, end)
            new_spectra.append(y[idx_start:idx_end])
            new_x_axis.append(x[idx_start:idx_end])
        self.spectra = np.array(new_spectra)
        self.x_axis = np.array(new_x_axis)
