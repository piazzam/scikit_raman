import plotly.graph_objects as go

class PlottingSpectra:
    """
    A class for automatically plot a Raman spectra dataset with Plotly.
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

    def plot_single_spectra_number(self, n, save = False, filename = "plot.png", color="#000000"):
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
        color : str, optional (default is #000000)
            RGB of spectra on the plot.
        """
        x = self.dataset.x_axis[n]
        y = self.dataset.spectra[n]
        fig = go.Figure()
        fig.layout.title = self.dataset.user[n]
        fig.add_trace(go.Line(x=x, y=y, line=dict(color=color)))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font = dict(size=25))
        fig.show()

        if save:
            fig.write_image(filename)

    def plot_single_patient_name(self, name, save = False, filename = "plot.png",color="#000000"):
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
        fig = go.Figure()
        fig.update_layout()
        fig.layout.title = name
        for i in range(len(ds)):
            x = ds.x_axis[i]
            y = ds.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color=color)))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)



    def plot_dataset(self, save = False, filename = "plot.png", title = "Dataset", color="#000000"):
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
        fig = go.Figure()
        fig.layout.title = title
        fig.update_layout()
        for i in range(len(self.dataset)):
            x = self.dataset.x_axis[i]
            y = self.dataset.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color=color)))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)


    def plot_category_name(self, category_name, save = False, filename = "plot.png",color="#000000"):
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
        fig = go.Figure()
        fig.update_layout()
        fig.layout.title = category_name
        for i in range(len(ds)):
            x = ds.x_axis[i]
            y = ds.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color=color)))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)



    def plot_category_label(self, category_label, save = False, filename = "plot.png", color="#000000"):
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
        fig = go.Figure()
        fig.update_layout()
        fig.layout.title = category_label
        for i in range(len(ds)):
            x = ds.x_axis[i]
            y = ds.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color=color)))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)