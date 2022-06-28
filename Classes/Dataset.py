import pandas as pd

class Dataset:
    """

    """

    def __init__(self, file_type, file_name, label_dictionary = {'cov':0, 'covNeg': 1, 'ctrl':2}):
        self.file_type = file_type
        self.file_name = file_name
        self.label_dictionary = label_dictionary
        self.load_file()

    def __init__(self, file_type, file_name,spectra, x_axis, raw, user, name, category, labels, label_dictionary = {'cov':0, 'covNeg': 1, 'ctrl':2}):
        self.file_type = file_type
        self.file_name = file_name
        self.spectra = spectra
        self.x_axis = x_axis
        self.raw = raw
        self.user = user
        self.name = name
        self.category = category
        self.labels = labels
        self.label_dictionary = label_dictionary

    def __init__(self, spectra, x_axis, raw, user, name, category, labels):
        self.spectra = spectra
        self.x_axis = x_axis
        self.raw = raw
        self.user = user
        self.name = name
        self.category = category
        self.labels = labels

    def load_file(self):
        if self.file_type == 'pkl':
            df = pd.read_pickle(self.file_name)
        elif self.file_type == 'csv':
            df = pd.read_csv(self.file_name)
        else:
            print("Error file type not supported")
        self.spectra = df['spectra'].tolist()
        self.x_axis = df['x-axis'].tolist()
        self.raw = df['raw'].tolist()
        self.user = df['user'].tolist()
        self.name = df['name'].tolist()
        self.category = df['category'].tolist()
        if 'label' in df.columns:
            self.labels = df['label'].tolist()
        else:
            self.create_label()

    def create_label(self):
        labels = []
        for el in self.category:
            labels.append(self.label_dictionary[el])
        self.labels = labels

    def get_raw_data(self):
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
        ds = Dataset(self.file_type, self.file_name, spectra, x_axis, raw, user, name, category, labels)
        return ds

    def get_dark_data(self):
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
        ds = Dataset(self.file_type, self.file_name, spectra, x_axis, raw, user, name, category, labels)
        return ds

    def change_category_name(self, dictionary):
        new_category = []
        for el in self.category:
            new_category.append(dictionary[el])
        self.category = new_category

    def change_user_name_string(self, dictionary):
        new_users = []
        for el in self.user:
            l = el.split('_')
            for d in dictionary:
                if d == l[0]:
                    l[0] = dictionary[d]
                    break
            string_name = l[0] + '_' + l[1]
            new_users.append(string_name)
        self.user = new_users

    def change_category_and_user(self, dictionary):
        self.change_category_name(dictionary)
        self.change_user_name_string(dictionary)
