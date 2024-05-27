import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PlottingGradCam:

    def __init__(self, gcpip, heatmaps, spectra, x_axis):
        self.gcpip = gcpip
        self.heatmaps = heatmaps
        self.spectra = spectra
        self.x_axis = x_axis

    def visualize_single_CAM(self, i, n_top=100, wellClassified=True, title="Grad-Cam", pathToSave="gradCam_plot.png", show=True,
                             save=True):
        fig = plt.figure(figsize=(20, 10))
        cam = self.heatmaps[i]
        valueToHighlight = self.gcpip.get_highest_values_number(cam, n_top)
        x_value = [x for x, y in valueToHighlight]
        if wellClassified:
            color = 'palegreen'
        else:
            color = 'lightcoral'
        plt.plot(self.spectra[i], 'k')
        for i in range(len(x_value)):
            plt.axvspan(x_value[i] - 0.2, x_value[i] + 0.2, color=color, label="_" * i + "Most important variables")
        plt.legend(['Raw spectra'])
        plt.xlabel('Raman shift (cm-1)')
        plt.ylabel('Intensity (a.u.)')

        plt.title('Original spectra with highlight on most important variable')
        plt.suptitle(title)
        plt.rcParams.update({'font.size': 25})
        if save:
            plt.savefig(pathToSave)
        if show:
            plt.show()
        plt.close('all')

    def visualize_all_CAM(self, n_top=100, wellClassified=True, title="Grad-Cam", base_path="gradCam/", show=True,
                             save=True):
        for i in range(len(self.heatmaps)):
            fig = plt.figure(figsize=(20, 10))
            cam = self.heatmaps[i]
            valueToHighlight = self.gcpip.get_highest_values_number(cam, n_top)
            x_value = [x for x, y in valueToHighlight]
            if wellClassified:
                color = 'palegreen'
            else:
                color = 'lightcoral'
            plt.plot(self.spectra[i], 'k')
            for i in range(len(x_value)):
                plt.axvspan(x_value[i] - 0.2, x_value[i] + 0.2, color=color, label="_" * i + "Most important variables")
            plt.legend(['Raw spectra'])
            plt.xlabel('Raman shift (cm-1)')
            plt.ylabel('Intensity (a.u.)')

            plt.title('Original spectra with highlight on most important variable')
            plt.suptitle(title)
            plt.rcParams.update({'font.size': 25})
            if save:
                plt.savefig(base_path+str(i)+".png")
            if show:
                plt.show()
            plt.close('all')