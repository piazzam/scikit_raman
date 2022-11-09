import matplotlib.pyplot as plt
import numpy as np

class PlottingSpectra:

    def __init__(self, dataset):
        self.dataset = dataset

    def plot_single_spectra_number(self, n, save = False, filename = "plot.png"):
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