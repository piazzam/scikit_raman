import pandas as pd
from scipy import interpolate

class Alluminium:
    """
    A class for loading an Alluminium file.
    ...

    Attributes
    ----------
    x-axis : np.array
        X-axis of the Alluminium.
    spectra: np.array
        Spectra of the Alluminium.

    Methods
    -------
    read_from_txt(all, filename)
        Read an alluminium spectra from txt file.
    resample(self, x_axis)
        Apply the resample of the alluminium spectra.
    """
    def __init__(self, x_axis, spectra):
        """
        Constructor of Alluminium class.

        Parameters
        ----------
        x_axis : np.array
            X-axis of the Allumiunium.
        spectra : np.array
            spectra of the Alluminium.
        """
        self.x_axis = x_axis
        self.spectra = spectra

    def __len__(self):
        """
        Overriding of len function.

        Returns
        -------

        int
            Length of the array spectra.
        """
        return len(self.spectra)

    def __getitem__(self, item):
        """
        Overriding of getitem function.

        Parameters
        ----------
        item : int
            position to be returned by the function

        Returns
        -------

        np.array
            selected spectra
        np.array
            selected x-axis
        """
        return self.spectra[item], self.x_axis[item]

    @classmethod
    def read_from_txt(all, filename):
        """
        Read an alluminium spectra from txt file.

        Parameters
        ----------
        filename : str
            Filename of Alluminium file.

        Returns
        -------
        scikit_raman.Alluminium
            Alluminium object read from filename.

        """
        df = pd.read_table(filename, delimiter='\t', names=['x-axis', 'spectral'])
        x_axis = list(df['x-axis'].replace(",", ".", regex=True).astype(float))
        y_axis = list(df['spectral'].replace(",", ".", regex=True).astype(float))
        return all(x_axis, y_axis)

    def resample(self, x_axis):
        """
        Apply the resample function on the alluminium.

        Parameters
        ----------
        x_axis : np.array
            x-axis from which apply the interpolation.
        """
        x = self.x_axis['x-axis']
        y = self.spectra['spectra']
        fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_axis)
        self.x_axis = x_axis
        self.spectra = y_new
