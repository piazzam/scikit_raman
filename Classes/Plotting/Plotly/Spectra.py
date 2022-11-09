import plotly.graph_objects as go

class PlottingSpectra:
    def __init__(self, dataset):
        self.dataset = dataset

    def plot_single_spectra_number(self, n, save = False, filename = "plot.png"):
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