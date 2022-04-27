import numpy as np
import pickle
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.manifold import TSNE
from scipy.spatial.distance import euclidean
import peakutils
from tqdm import tqdm
import statistics as st
from scipy.signal import medfilt
from scipy import interpolate
from ast import literal_eval
from itertools import groupby
import seaborn as sn
import matplotlib.pyplot as plt

def resample_shift(df, start = 400, end = 1600, points = 991):
    """Calculate the new set of points for the spectras contained in the dataframe
     providen in input. Apply the shift of the spectra. 

    Parameters:
    df : pd.DataFrame 
        dataframe formatted by our policy

    Returns:
    df : pd.DataFrame
        dataframe formatted by our policy with the new x-axis and the new
        spectra
    """

    x_new = np.linspace(start, end, points)

    result = []
    for index, row in df.iterrows():

        y = row['spectra']
        x = row['x-axis']

        fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_new)
        result.append(y_new)
        
        
    df['spectra'] = result
    df['x-axis'] = [x_new] * len(df['x-axis'])
    return df

def delete_uninformative_spectra(X,y,user):
    """
    Delete uninformative spectra. 
    The uninformativa spectra are defined by up to 10% of null values or up
    to 10% of saturate values.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    user : TYPE
        DESCRIPTION.

    Returns
    -------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    user : TYPE
        DESCRIPTION.

    """
    tot = 0
    for i in range(len(X))[::-1]:
        nz = len(X[i]) - np.count_nonzero(X[i])
        if(nz >= (10*len(X[i]))/100):
            X.pop(i)
            y.pop(i)
            user.pop(i)
            print('Item removed! '+str(nz)+' zeros found')
            tot += 1
    #remove repeated continuous values (signal saturated)
    for i in range(len(X))[::-1]:
        counts = [(k, sum(1 for i in g)) for k,g in groupby(X[i])]
        mc = max([c[1] for c in counts])
        if(mc >= (10*len(X[i]))/100):
            X.pop(i)
            y.pop(i)
            user.pop(i)
            print('Item removed! '+str(mc)+' identical consecutive intensities found')
            tot += 1 
    print('\n'+str(tot)+' uninformative spectras removed')
    return X,y,user

def remove_alluminium(X, y, allu):
    """
    Remove signal of the alluminium substrate.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.
    y : TYPE
        DESCRIPTION.
    allu : TYPE
        DESCRIPTION.

    Returns
    -------
    X_new : List(float)
        List that contains the new points without alluminium substrate

    """
    X_new = []
    for spectra in X:
        spectra_new = []
        for s,al in zip(spectra, allu):
            spectra_new.append(s-al)
    return X_new

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
    modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
    return modified_z_scores

def fixer(X,m,index,threshold = 3.5, plot=False):
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
    plot : TYPE, optional
        DESCRIPTION. The default is False.

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
    if(ns > 0):
        print('Found '+str(ns)+' spikes')
        if(plot == True):
            sn.set()
            plt.figure()
            plt.title('Spectra '+ str(index))
            plt.plot(spikes*1000)
            plt.plot(X, alpha = 0.5)
            plt.figure()
            plt.plot(X_out)
    return X_out

def spike_removal(X):
    """
    Remove spikes from spectra.

    Parameters
    ----------
    X : TYPE
        DESCRIPTION.

    Returns
    -------
    X_c : TYPE
        DESCRIPTION.

    """
    X_c = []
    for x in X:
        x_fix = fixer(x,5, X.index(x), threshold=3.5)
        X_c.append(x_fix)
    return X_c

def remove_baseline(spectra, deg = 6, max_it = 100000, tol = pow(10, -11)):
    """
    Remove baseline (bacground noise) from the spectra

    Parameters
    ----------
    spectra : TYPE
        DESCRIPTION.
    deg : TYPE, optional
        DESCRIPTION. The default is 6.
    max_it : TYPE, optional
        DESCRIPTION. The default is 100000.
    tol : TYPE, optional
        DESCRIPTION. The default is pow(10, -11).

    Returns
    -------
    X_out : TYPE
        DESCRIPTION.

    """
    #df_final = pd.DataFrame(columns = df.columns, index = df.index)
    #print(df_final)
    X_out = []
    for i in tqdm(range(len(spectra))):
        if(np.any(spectra[i])):
            baseline_values = peakutils.baseline(spectra[i], deg = deg, max_it = max_it, tol = tol)
            z = zip(spectra[i], baseline_values)
            #df_final.loc[:, 'Spectra'] = [x[0] - x[1] for x in z] 
            X_out.append([x[0] - x[1] for x in z])
    return X_out

#median filtering of the spectra
def median_filtering(X, y, labels, filter_size = 5, plot = True):
    X_mf = []
    for i in range(len(set(y))):
        if(plot == True):
            plt.figure(figsize=[10,5])
            plt.title('Spectra '+labels[i])
        subX = []
        for j in range(len(X)):
            if(y[j] == i):
                subX.append(X[j])

        for j in range(len(subX)):
            subX[j] = medfilt(subX[j], filter_size)
            if(plot == True):
                plt.plot(subX[j])
        X_mf.extend(subX)
    return X_mf

import pandas as pd
import pickle

with open('../dataset/data_raman_portable.pkl', 'rb') as pickle_file:
    content_due = pickle.load(pickle_file)
    
#with open('../dataset/raman_shift_portable.pkl', 'rb') as pickle_file:
#    content_tre = pickle.load(pickle_file)
    
with open('../dataset/data_raman_raw_portable.pkl', 'rb') as pickle_file:
    content_uno = pickle.load(pickle_file)
    
with open('../dataset/data_raman_aramis.pkl', 'rb') as pickle_file:
    aramis = pickle.load(pickle_file)
    
#print(aramis[0])
#print(aramis[1])    