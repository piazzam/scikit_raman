import pandas as pd
from scipy import interpolate

class Alluminium:

    def __init__(self, x_axis, spectra):
        self.x_axis = x_axis
        self.spectra = spectra

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, item):
        return self.spectra[item], self.x_axis[item]

    @classmethod
    def read_from_txt(all, filename):
        df = pd.read_table(filename, delimiter='\t', names=['x-axis', 'spectral'])
        x_axis = list(df['x-axis'].replace(",", ".", regex=True).astype(float))
        y_axis = list(df['spectral'].replace(",", ".", regex=True).astype(float))
        return all(x_axis, y_axis)

    def resample(self, x_axis):
        x = self.x_axis['x-axis']
        y = self.spectra['spectra']
        fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_axis)
        self.x_axis = x_axis
        self.spectra = y_new
