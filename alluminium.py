import pandas as pd
from scipy import interpolate


def read_from_txt(filename):
    df = pd.read_table(filename, delimiter = '\t', names = ['x-axis', 'spectral'])
    x_axis = list(df['x-axis'].replace(",",".", regex=True).astype(float))
    y_axis = list(df['spectral'].replace(",",".", regex=True).astype(float))
    return_df = pd.DataFrame(columns = ['x-axis', 'spectra'])
    return_df = return_df.append({'x-axis':x_axis, 'spectra':y_axis})
    return return_df
    
def resample(df, x_axis):
    x = df['x-axis']
    y = df['spectra']    
    fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                              fill_value='extrapolate')
    y_new = fi(x_axis)
    df['x-axis'] = x_axis
    df['spectra'] = y_new
    return df        