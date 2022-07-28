import pandas as pd
from scipy import interpolate

class Alluminium:
    """
    A class that represent an Alluminium object.
    ...

    Attributes
    ----------
    x-axis : np.array
        x-axis of the Alluminium.
    spectra: np.array
        spectra of the Alluminium.
    """

    def __init__(self, x_axis, spectra):
        self.x_axis = x_axis
        self.spectra = spectra

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, item):
        return self.spectra[item], self.x_axis[item]

    @classmethod
    def read_from_txt(all, filename):
        """
        Read an alluminium spectra from txt file.
        :param filename: str
            complete filename of the file
        :return: an Alluminium object readed from filename.
        """
        df = pd.read_table(filename, delimiter='\t', names=['x-axis', 'spectral'])
        x_axis = list(df['x-axis'].replace(",", ".", regex=True).astype(float))
        y_axis = list(df['spectral'].replace(",", ".", regex=True).astype(float))
        return all(x_axis, y_axis)

    def resample(self, x_axis):
        """
        Apply the resample of the alluminium spectra.
        :param x_axis: np.array
            an array representing the x-axis from which interpolate.
        """
        x = self.x_axis['x-axis']
        y = self.spectra['spectra']
        fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_axis)
        self.x_axis = x_axis
        self.spectra = y_new
