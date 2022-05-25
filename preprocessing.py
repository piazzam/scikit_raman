import numpy as np
import pandas as pd
from sklearn import preprocessing
import peakutils
from tqdm import tqdm
from scipy.signal import medfilt
from scipy import interpolate
from itertools import groupby
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


def resample_shift(df, start = 400, end = 1600, points = 991):
    """Calculate the new set of points for the spectras contained in the dataframe
     providen in input. Apply the shift of the spectra. 

    Parameters:
    df : pd.DataFrame 
        dataframe formatted according to our policy

    Returns:
    df : pd.DataFrame
        dataframe formatted according to our policy with the new x-axis and the new
        spectra
    """

    x_new = list(np.linspace(start, end, points))

    result = []
    for index, row in df.iterrows():

        y = row['spectra']
        x = row['x-axis']

        fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_new)
        result.append(list(y_new))
        
        
    df['spectra'] = result
    df['x-axis'] = [x_new] * len(df['x-axis'])
    return df

def delete_uninformative_spectra(df):
    """
    Delete uninformative spectra. Uninformative spectras are defined by:
        10% of zeros or 10% repeated continuos values.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe formatted according to our policy.

    Returns
    -------
    df : pd.DataFrame
        A dataframe without uninformative spectras.

    """
    tot = 0
    df_to_remove = []
    #for i in range(len(df)):
    for index, row in df.iterrows():
        current_spectra = row['spectra']
        nz = len(current_spectra) - np.count_nonzero(current_spectra)
        if nz >= (10*len(current_spectra))/100:
            df_to_remove.append(index)
            tot += 1
    df.drop(df_to_remove, inplace = True)
    df_to_remove = []
    for index, row in df.iterrows():
        current_spectra = row['spectra']
        counts = [(k, sum(1 for i in g)) for k,g in groupby(current_spectra)]
        mc = max([c[1] for c in counts])
        if mc >= (10*len(current_spectra))/100:
            df_to_remove.append(index)
            tot += 1
    print("Tot = "+str(tot) + " spettri rimossi")
    df.drop(df_to_remove, inplace = True)
    return df


def remove_alluminium(df, df_allu):
    """
    Removes the alluminium substrate

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe formatted according to our policy.
    df_allu : pd.DataFrame
        An alluminium dataframe formatted according to our policy.

    Returns
    -------
    df : pd.DataFrame
        A new dataframe without alluminium substrate.

    """
    allu = df_allu.iloc[0]['spectra']
    for i in range(len(df)):
        current_spectra = df.iloc[i]['spectra']
        new_spectra = []
        for s,al in zip(current_spectra,allu):
            new_spectra.append(s-al)
        df.iloc[i]['spectra'] = new_spectra
    return df

def modified_z_score(intensity):
    """
    Function propaedeutics for spike removal function

    Parameters
    ----------
    intensity : TYPE
        DESCRIPTION.

    Returns
    -------
    modified_z_scores : TYPE
        DESCRIPTION.

    """
    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    try:
        modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    except RuntimeWarning:
        print(intensity)
        print(mad_int)
        modified_z_scores = 0.6745 * (intensity - median_int) / 1
    return modified_z_scores

def fixer(X,m,index,threshold = 3.5):
    """
    Function propaedeutics for spike removal function   

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    m : TYPE
        DESCRIPTION.
    index : TYPE
        DESCRIPTION.
    threshold : TYPE, optional
        DESCRIPTION. The default is 3.5.

    Returns
    -------
    X_out : TYPE
        DESCRIPTION.

    """
    spikes = abs(np.array(modified_z_score(np.diff(X)))) > threshold
    X_out = X.copy() # So we don’t overwrite y
    ns = 0
    for i in np.arange(len(spikes)-m):
        if spikes[i] != 0: # If we have a spike in position i
            w = np.arange(i-m,i+1+m) # we select 2 m + 1 points around our spike
            w2 = w[spikes[w] == 0] # From such interval, we choose the ones which are not spikes
            if(len(w2) != 0): #avoid erroneous detected spikes ([2m+1]+ consecutive spikes)
                X_out[i] = np.mean(np.array(X)[w2]) # and we average their values
                ns +=1
    #if(ns > 0):
    #    print('Found '+str(ns)+' spikes')
    return X_out

def spike_removal(df):
    """
    Removes the spikes from the spectra.

    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe formatted according to our policy.

    Returns
    -------
    df : pd.DataFrame
        A Dataframe formatted according to our policy. In the spectra fields
        the spike spectra are removed.

    """
    X = list(df['spectra'])
    X_c = []
    for x in X:
        x_fix = fixer(x,5, X.index(x), threshold=3.5)
        X_c.append(x_fix)
    df['spectra'] = X_c
    return df

def remove_baseline_polynomial(df, deg = 6, max_it = 100000, tol = pow(10, -11)):
    """
    Removes baseline (background noise) from the spectra. It apply the 
    polinomyal fitting approach.

    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe formatted according to out policy.
    deg : int, optional
        degree of the polynomial. The default is 6.
    max_it : int, optional
        maximum number of iterations. The default is 100000.
    tol : int, optional
        tolerance value. The default is pow(10, -11).

    Returns
    -------
    df : pd.DataFrame
        A DataFrame formatted according to our policy. It contains in the 
        spectra fields the field without background noise.

    """
    spectra = list(df['spectra'])
    X_out = []
    for i in tqdm(range(len(spectra))):
        if(np.any(spectra[i])):
            baseline_values = peakutils.baseline(np.array(spectra[i]), deg = deg, max_it = max_it, tol = tol)
            z = zip(spectra[i], baseline_values)
            X_out.append([x[0] - x[1] for x in z])
    df['spectra'] = X_out
    return df

def remove_baseline_median_filtering(df, filter_size = 5):
    """
    Apply the median filtering approach to remove the background noise. 

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formated according to our policy.
    filter_size : Int, optional
        Dimension of the filter size. The default is 5.

    Returns
    -------
    df : pd.DataFrame
        A DataFrame formatted according to our policy. It contains in the 
        spectra fields the field without background noise.

    """
    X_mf = []
    label_set = set(df['label'])
    X = list(df['spectra'])
    for i in range(len(label_set)):
        subX = []
        for j in range(len(X)):
            if(df.iloc[j]['label'] == i):
                subX.append(X[j])
        for j in range(len(subX)):
            subX[j] = medfilt(subX[j, filter_size])
        X_mf.extend(subX)
    df['spectra'] = X_mf
    return X_mf

def smoothing_savitzky_golay(df, window_length = 9, polyorder = 2):
    """
    Removes the baseline (background noise). It applies the Savitzy-Golay
    smoothing.

    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe formated according to our policy.
    window_length : int, optional
        Window length for the Savitzky-Golay function. The default is 9.
    polyorder : int, optional
        polyorder for the Savitzky-Golay function. The default is 2.

    Returns
    -------
    df : pd.DataFrame
        A Dataframe formatted according to out policy. In the spectra field 
        the spectra are smoothed with Savitzky-Golay filter.

    """
    X = list(df['spectra'])
    X_filter = savgol_filter(X, window_length = window_length, polyorder = polyorder)
    X_filter_bis = []
    for el in X_filter:
        X_filter_bis.append(list(el))
    df['spectra'] = X_filter_bis
    return df

def snv_normalization(df):
    """
    Apply the normalization with the SNV approach.

    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe formatted according to our policy.

    Returns
    -------
    df : pd.DataFrame
        A Dataframe formatted according to out policy. In the spectra column 
        the value are substituted with the normalized values.

    """
    X = list(df['spectra'])
    data_snv = np.zeros_like(X)
    for i in range(len(X)):
        # Apply correction
        data_snv[i,:] = list((X[i] - np.mean(X[i])) / np.std(X[i]))
    l = data_snv.tolist()
    df['spectra'] = l
    return df
    

def min_max_normalization(df):
    """
    Apply the min-max normalization.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted according to our policy. 

    Returns
    -------
    df : TYPE
        A DataFrame formatted according to our policy. In the spectra column 
        the value are substituted with the normalized values. 

    """
    X = list(df['spectra'])
    norm = X.copy()
    scaler = MinMaxScaler()
    for i in range(len(X)):
        # Apply correction
        x = np.array(X[i])
        norm[i] = scaler.fit_transform(np.reshape(x, (-1,1)))
        norm[i] = list(norm[i].reshape(991))
    df['spectra'] = norm
    return df

def l2_normalization(df):
    """
    Apply the l2 - normalization.

    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe formatted according to our policy.

    Returns
    -------
    df : pd.DaraFrame
        A dataframe formatted according to our policy. In the spectra column 
        the value are substituted with the normalized values. 

    """
    X = list(df['spectra'])
    X_norm = preprocessing.normalize(X, norm='l2')
    df['spectra'] = X_norm.tolist()
    return df

def peak_normalization(df):
    """
    Apply the peak - normalization.

    Parameters
    ----------
    df : pd.DataFrame
        A Dataframe formatted according to our policy.

    Returns
    -------
    df : pd.DaraFrame
        A dataframe formatted according to our policy. In the spectra column 
        the value are substituted with the normalized values. 

    """
    X = list(df['spectra'])
    X_norm = preprocessing.normalize(X, norm='max')
    df['spectra'] = X_norm.tolist()
    return df
