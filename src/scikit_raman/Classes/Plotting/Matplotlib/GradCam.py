import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class PlottingGradCam:
    """
    A class for automatically plot results obtained by Grad-CAM.
    ...

    Attributes
    ----------
    gcpip : scikit_raman.Keras.Explainability.GradCamPipeline
        Object for automatically apply functions related to grad-CAM.
    heatmaps : np.array
        Array containing heatmaps obtained by grad-CAM.
    spectra : np.array
        Array containing the spectra corresponding to the heatmaps.
    x_axis : np.array
        Array containing x-axis corresponfing to the spectra.

    Methods
    -------
    visualize_single_CAM(self, i, n_top=100, wellClassified=True, title="Grad-Cam", pathToSave="gradCam_plot.png",
                        show=True, save=True)
        Plot and save the grad-CAM value for a single spectrum.
    visualize_all_CAM(self, n_top=100, wellClassified=True, title="Grad-Cam", base_path="gradCam/", show=True,
                        save=True)
        Plot and save grad-CAM value for all heatmaps included in the heatmaps array.
    """
    def __init__(self, gcpip, heatmaps, spectra, x_axis):
        """
        Constructor for PlottingGradCam class.

        Parameters
        ----------
        gcpip : scikit_raman.Keras.Explainability.GradCamPipeline
            Object for automatically apply functions related to grad-CAM.
        heatmaps : np.array
            Array containing heatmaps obtained by grad-CAM.
        spectra : np.array
            Array containing the spectra corresponding to the heatmaps.
        x_axis : np.array
            Array containing x-axis corresponfing to the spectra.
        """
        self.gcpip = gcpip
        self.heatmaps = heatmaps
        self.spectra = spectra
        self.x_axis = x_axis

    def visualize_single_CAM(self, i, n_top=100, wellClassified=True, title="Grad-Cam", pathToSave="gradCam_plot.png",
                             show=True, save=True):
        """
        Plot and save Grad-CAM value for a single heatmap.

        Parameters
        ----------
        i : int
            Index of value to be plot.
        n_top : int, optional (default is 100)
            Number of top values to highlight on the plot.
        wellClassified : bool, optional (default is True)
            Whether to plot only positive classified samples or not.
        title : string, optional (default is "Grad-Cam")
            Title on the plot.
        pathToSave : string, optional (default is "gradCam_plot.png")
            Path for saving the plot.
        show : bool, optional (default is True)
            Whether to show or not.
        save : bool, optional (default is True)
            Whether to save the plot or not.
        """
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
        """
        Plot and save Grad-CAM value for the entire set of heatmaps.

        Parameters
        ----------
        n_top : int, optional (default is 100)
            Number of top values to highlight on the plot.
        wellClassified : bool, optional (default is True)
            Whether to plot only positive classified samples or not.
        title : string, optional (default is "Grad-Cam")
            Title on the plot.
        pathToSave : string, optional (default is "gradCam_plot.png")
            Path for saving the plot.
        show : bool, optional (default is True)
            Whether to show or not.
        save : bool, optional (default is True)
            Whether to save the plot or not.
        """
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