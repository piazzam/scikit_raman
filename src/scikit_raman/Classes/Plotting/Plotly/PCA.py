import plotly.express as px
class PlottingPCA:

    def __init__(self, pca_result):
        self.pca_result = pca_result

    def scatter_plot2D(self, color_type = "category", x_axis_title = "PCA-component 1", y_axis_title="PCA-component 2",
                     save = False, filename="plot.png"):
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