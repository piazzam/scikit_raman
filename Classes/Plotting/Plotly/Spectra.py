import plotly.graph_objects as go

class PlottingSpectra:
    """
        A class to plot the dataset in with Plotly.

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
        fig = go.Figure()
        fig.layout.title = self.dataset.user[n]
        fig.add_trace(go.Line(x=x, y=y, line=dict(color="#000000")))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font = dict(size=25))
        fig.show()

        if save:
            fig.write_image(filename)

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
        fig = go.Figure()
        fig.update_layout()
        fig.layout.title = name
        for i in range(len(ds)):
            x = ds.x_axis[i]
            y = ds.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color="#000000")))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)



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
        fig = go.Figure()
        fig.layout.title = title
        fig.update_layout()
        for i in range(len(self.dataset)):
            x = self.dataset.x_axis[i]
            y = self.dataset.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color="#000000")))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)


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
        fig = go.Figure()
        fig.update_layout()
        fig.layout.title = category_name
        for i in range(len(ds)):
            x = ds.x_axis[i]
            y = ds.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color="#000000")))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)



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
        fig = go.Figure()
        fig.update_layout()
        fig.layout.title = category_label
        for i in range(len(ds)):
            x = ds.x_axis[i]
            y = ds.spectra[i]
            fig.add_trace(go.Line(x=x, y=y, line=dict(color="#000000")))
        fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title='Raman shift (cm<sup>-1</sup>)', yaxis_title="Intensity")
        fig.update_layout(font=dict(size=25))
        fig.show()
        if save:
            fig.write_image(filename)
