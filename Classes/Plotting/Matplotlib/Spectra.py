import matplotlib.pyplot as plt
import numpy as np

class PlottingSpectra:
    """
        A class to plot the dataset in with Matplotlib.

        ...

        Attributes
        ----------
        dataset : scikit_raman.Dataset
            Dataset object to plot.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def plot_single_spectra_number(self, n, save = False, filename = "plot.png"):
        """
        Plot a single spectra taking it with the number offset.
        :param n: int
            The number offset of the spectra to plot.
        :param save: boolean, optional
            If true the generated plot is save.
        :param filename: String, optional
            If save parameter is true the plot is save with this filename.
        :return:
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
        Plot a subset of spectra taking it with the name of the patients.
        :param name: string
            The name of the user to plot.
        :param save: boolean, optional
            If true the generated plot is save.
        :param filename: string, optional.
            If save parameter is true the plot is save with this filename.
        :return:
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
        Plot the entire dataset
        :param save: boolean, optional
            If true the generated plot is save.
        :param filename: string, optional.
            If save parameter is true the plot is save with this filename.
        :param title: String, optional
            The title of the plot.
        :return:
        """
        fig, ax = plt.subplots()
        plt.title(title, fontdict={'fontsize': 15})
        plt.xlabel("Raman shift (cm$^{-1}$)", fontdict={'fontsize': 15})
        plt.ylabel("Intensity", fontdict={'fontsize': 15})
        plt.rcParams.update({'font.size': 15})
        ax.plot(self.dataset.x_axis[0], np.transpose(self.dataset.spectra), linewidth=2.0)
        plt.show()

        if save == True:
            plt.savefig(filename)

    def plot_category_name(self, category_name, save = False, filename = "plot.png",):
        """
        Plot a subset of the dataset corresponding to the category specified.
        :param category_name: String
            Name of the category to plot.
        :param save: boolean, optional
            If true the generated plot is save.
        :param filename: string, optional.
            If save parameter is true the plot is save with this filename.
        :return:
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

    def plot_category_label(self, category_label, save = False, filename = "plot.png",):
        """
        Plot a subset of the dataset corresponding to the category specified.
        :param category_label: int
            Label of the category to plot.
        :param save: boolean, optional
            If true the generated plot is save.
        :param filename: string, optional.
            If save parameter is true the plot is save with this filename.
        :return:
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