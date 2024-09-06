import plotly.express as px
class PlottingPCA:
    """
    A class for automatically plot the PCA results.
    ...

    Attributes
    ----------
    pca_result : scikit_raman.Processor.PCA_result
        PCA result object.

    Methods
    -------
    scatter_plot2D(self, color_type = "category", x_axis_title = "PCA-component 1", y_axis_title="PCA-component 2",
                     save = False, filename="plot.png")
        Plot the 2D PCA results into a 2D scatter plot.
    scatter_plot3D(self, color_type = "category", x_axis_title = "PCA-component 1", y_axis_title="PCA-component 2",
                     z_axis_title = "PCA-component 3", save = False, filename="plot.png")
        Plot the 2D PCA results into a 2D scatter plot.
    """

    def __init__(self, pca_result):
        """
        Constructor for PlottingPCA class

        Parameters
        ----------
        pca_result : scikit_raman.Processor.PCA_result
            PCA result object.
        """
        self.pca_result = pca_result

    def scatter_plot2D(self, color_type = "category", x_axis_title = "PCA-component 1", y_axis_title="PCA-component 2",
                     save = False, filename="plot.png"):
        """
        Plot the 2D PCA results into a 2D scatter plot.

        Parameters
        ----------
        color_type : str, optional (default is "category")
            Element for which distinguish between different colors on the scatter plot.
        x_axis_title : str, optional (default is "PCA-component 1)
            Label of x-axis
        y_axis_title : str, optional (default is "PCA-component 2"):
            Label of y-axis
        save : bool, optional (default is False)
            Whether to save the generated plot.
        filename : string, optional (default is plot.png)
            Filename of the file to be saved.
        """
        fig = px.scatter(self.pca_result.result, x=0, y=1, color=color_type)
        fig.update_layout(xaxis_title=x_axis_title, yaxis_title=y_axis_title)
        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(font=dict(size=15))
        fig.show()
        if save:
            fig.write_image(filename)

    def scatter_plot3D(self, color_type = "category", x_axis_title = "PCA-component 1", y_axis_title="PCA-component 2",
                     z_axis_title = "PCA-component 3", save = False, filename="plot.png"):
        """
        Plot the 3D PCA results into a 3D scatter plot.

        Parameters
        ----------
        color_type : str, optional (default is "category")
            Element for which distinguish between different colors on the scatter plot.
        x_axis_title : str, optional (default is "PCA-component 1)
            Label of x-axis
        y_axis_title : str, optional (default is "PCA-component 2"):
            Label of y-axis
        save : bool, optional (default is False)
            Whether to save the generated plot.
        filename : string, optional (default is plot.png)
            Filename of the file to be saved.
        """
        fig = px.scatter_3d(self.pca_result.result, x=0, y=1, z=2, color=color_type,
                            labels = {"0": x_axis_title, "1": y_axis_title, "2":z_axis_title})
        fig.update_layout(xaxis_title=x_axis_title, yaxis_title=y_axis_title)
        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.update_layout(font=dict(size=15))
        fig.show()
        if save:
            fig.write_image(filename)