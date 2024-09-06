import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PlottingSpectra:
    """
    A class for automatically plot a Raman spectra dataset with Matplotlib.
    ...

    Attributes
    ----------
    dataset : scikit_raman.Dataset
        Dataset object to plot.

    Methods
    -------
    plot_single_spectra_number(self, n, save = False, filename = "plot.png")
        Plot a single spectra identified by its index.
    plot_single_patient_name(self, name, save = False, filename = "plot.png")
        Plot a subset of spectra identified by the name of the patient.
    plot_dataset(self, save = False, filename = "plot.png", title = "Dataset")
        Plot the entire dataset.
    plot_category_name(self, category_name, save = False, filename = "plot.png")
        Plot a subset of the dataset corresponding to the specified category by its string name.
    plot_category_label(self, category_label, save = False, filename = "plot.png")
        Plot a subset of the dataset corresponding to the specified category by its numerical label.
    plot_mean_error_category(self, category_label, label = "", color_map_value = 0,save=False, filename = "plot.png")
        Plot mean for a single category with its error value.
    plot_mean_error_different_categories(self, color_map_value = 0, save=False, filename = "plot.png")
        Plot mean for each category with its error value.
    plot_mean_different_categories(self, save=True, filename="plot.png", show_legend=True)
        Plot the mean spectrum for each category.
    """
    def __init__(self, dataset):
        """
        Constructor for class PlottingSpectra

        Parameters
        ----------
        dataset : scikit_raman.Dataset
            Dataset object to plot.
        """
        self.dataset = dataset

    def plot_single_spectra_number(self, n, save = False, filename = "plot.png"):
        """
        Plot a single spectra identified by its index.

        Parameters
        ----------
        n : int
            Index of the spectro.
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        """
        x = self.dataset.x_axis[n]
        y = self.dataset.spectra[n]

        fig, ax = plt.subplots()
        plt.title(self.dataset.user[n], fontdict={'fontsize': 15})
        plt.xlabel("Raman shift (cm$^{-1}$)", fontdict={'fontsize': 15})
        plt.ylabel("Intensity", fontdict={'fontsize': 15})
        plt.rcParams.update({'font.size': 15})
        ax.plot(x, y, linewidth=2.0)
        plt.show()

        if save == True:
            plt.savefig(filename)

    def plot_single_patient_name(self, name, save = False, filename = "plot.png"):
        """
        Plot a single spectra identified by its name.

        Parameters
        ----------
        name : string
            Name of the patient.
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        """
        ds = self.dataset.search_by_name(name)
        fig, ax = plt.subplots()
        plt.title(ds.user[0], fontdict={'fontsize': 15})
        plt.xlabel("Raman shift (cm$^{-1}$)", fontdict={'fontsize': 15})
        plt.ylabel("Intensity", fontdict={'fontsize': 15})
        plt.rcParams.update({'font.size': 15})
        ax.plot(ds.x_axis[0], np.transpose(ds.spectra), linewidth=2.0)
        plt.show()

        if save == True:
            plt.savefig(filename)

    def plot_dataset(self, save = False, filename = "plot.png", title = "Dataset"):
        """
        Plot the entire dataset.

        Parameters
        ----------
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        title : string, optional (defualt is "Dataset")
            Title of the plot.
        """
        fig, ax = plt.subplots()
        plt.title(title, fontdict={'fontsize': 15})
        plt.xlabel("Raman shift (cm$^{-1}$)", fontdict={'fontsize': 15})
        plt.ylabel("Intensity", fontdict={'fontsize': 15})
        plt.rcParams.update({'font.size': 15})
        ax.plot(self.dataset.x_axis[0], np.transpose(self.dataset.spectra), linewidth=2.0)
        if save == True:
            plt.savefig(filename)
        plt.show()

    def plot_category_name(self, category_name, save = False, filename = "plot.png",):
        """
        Plot a subset of the dataset corresponding to the specified category by its string name.

        Parameters
        ----------
        category_name : string
            String name of the category.
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        """
        ds = self.dataset.search_by_category_name(category_name)

        fig, ax = plt.subplots()
        plt.title(category_name, fontdict={'fontsize': 15})
        plt.xlabel("Raman shift (cm$^{-1}$)", fontdict={'fontsize': 15})
        plt.ylabel("Intensity", fontdict={'fontsize': 15})
        plt.rcParams.update({'font.size': 15})
        ax.plot(ds.x_axis[0], np.transpose(ds.spectra), linewidth=2.0)
        plt.show()

        if save == True:
            plt.savefig(filename)

    def plot_category_label(self, category_label, save = False, filename = "plot.png"):
        """
        Plot a subset of the dataset corresponding to the specified category by its numerical label.

        Parameters
        ----------
        category_label : int
            Numerical value of the label.
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        """
        ds = self.dataset.search_by_category_label(category_label)

        fig, ax = plt.subplots()
        plt.title(category_label, fontdict={'fontsize': 15})
        plt.xlabel("Raman shift (cm$^{-1}$)", fontdict={'fontsize': 15})
        plt.ylabel("Intensity", fontdict={'fontsize': 15})
        plt.rcParams.update({'font.size': 15})
        ax.plot(ds.x_axis[0], np.transpose(ds.spectra), linewidth=2.0)
        plt.show()

        if save == True:
            plt.savefig(filename)


    def plot_mean_error_category(self, category_label, label = "", color_map_value = 0,save=False, filename = "plot.png"):
        """
        Plot mean for a single category with its error value.

        Parameters
        ----------
        category_label : str
            Category label for searching over the dataset.
        label : str, optional (default is "").
            Label wrote on the plot. If it is equal to "" it will be used the category_label parameter.
        color_map_value : int, optional (default is 0).
            Starting color map value.
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        """
        if label == "":
            label = category_label
        cat_patient = self.dataset.search_by_category_name(category_label)
        cat_array = []
        for el in cat_patient.spectra:
            cat_array.append(np.array(el))
        df = pd.DataFrame(cat_array)
        plt.rcParams["figure.figsize"] = (20, 10)
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['font.size'] = 20
        cmap = plt.get_cmap("tab20")
        x_als = np.array(cat_patient.x_axis[0])
        mean_als = df.mean()
        std_als = df.std()
        plt.errorbar(x_als,
                     mean_als.values,
                     yerr=std_als.values,
                     ecolor=cmap(2 * (color_map_value + 1) - 1),
                     color=cmap(2 * color_map_value),
                     label=label)
        plt.show()
        if save == True:
            plt.savefig(filename)

    def plot_mean_error_different_categories(self, color_map_value = 0, save=False, filename = "plot.png"):
        """
        Plot mean for each category with its error value.

        Parameters
        ----------
        color_map_value : int, optional (default is 0).
            Starting color map value.
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        """
        unique_cat = np.unique(self.dataset.category)
        plt.rcParams["figure.figsize"] = (20, 10)
        plt.rcParams['legend.fontsize'] = 20
        plt.rcParams['font.size'] = 20
        cmap = plt.get_cmap("tab20")
        x_als = np.array(self.dataset.x_axis[0])
        for cat in unique_cat:
            cat_ds = self.dataset.search_by_category_name(cat)
            cat_array = []
            for el in cat_ds.spectra:
                cat_array.append(np.array(el))
            df = pd.DataFrame(cat_array)
            mean_als = df.mean()
            std_als = df.std()
            plt.errorbar(x_als,
                     mean_als.values,
                     yerr=std_als.values,
                     ecolor=cmap(2 * (color_map_value + 1) - 1),
                     color=cmap(2 * color_map_value),
                     label=cat)
            color_map_value += 1
        plt.legend()
        plt.show()
        if save == True:
            plt.savefig(filename)

    def plot_mean_different_categories(self, save=True, filename="plot.png", show_legend=True):
        """
        Plot the mean spectrum for each category.

        Parameters
        ----------
        save : bool, optional (default is False)
            Whether to save or not the plot.
        filename : int, string (default is "plot.png")
            Filename of the file to be stored.
        show_legend : bool, optional (default is True)
            Whether to show the legend or not.
        """
        unique_cat = np.unique(self.dataset.category)
        lines = []
        x = self.dataset.x_axis[0]
        plt.xlabel("Raman shift (cm$^{-1}$)", fontdict={'fontsize': 15})
        plt.ylabel("Intensity", fontdict={'fontsize': 15})
        plt.rcParams.update({'font.size': 15})
        for cat in unique_cat:
            ds_cat = self.dataset.search_by_category_name(cat)
            average_line = np.mean(ds_cat.spectra, axis=0)
            lines.append(average_line)
        for i in range(len(lines)):
            line = lines[i]
            label = unique_cat[i]
            plt.plot(x, np.transpose(line), label=label)
        if show_legend:
            plt.legend()
        if save == True:
            plt.savefig(filename)
        plt.show()