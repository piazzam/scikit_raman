import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def pca_fit_transform(df, n_components = 2):
    """
    Apply fit_transform function on the spectra data. The returned dataframe 
    is formatted according to our policy with a column for every component of 
    the pca.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted according to our policy.
    n_components : int, optional
        Number of components for the PCA. The default is 2.

    Returns
    -------
    df_pca : pd.DataFrame
        A DataFrame formatted according to our policy. It has a column for 
        every components and category, user and label. 

    """
    pca=PCA(n_components=n_components)
    spectra = list(df['spectra'])
    pca=pca.fit_transform(spectra)
    df.drop(columns = ['spectra', 'x-axis'], inplace = True)
    df['components'] = np.nan
    df['components'] = df['components'].astype('object')
    for i in range(len(df)):
        df.at[i,'components'] = pca[i].tolist()
    return df

def pca_fit(df, n_components = 2):
    """
    Fit the model with the spectra.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted according to our policy. 
    n_components : int, optional
        Number of components for the PCA. The default is 2.

    Returns
    -------
    pca_el : PCA object
        Object of the class PCA decomposition.

    """
    pca=PCA(n_components=n_components)
    spectra = list(df['spectra'])
    pca_el=pca.fit(spectra)
    return pca_el

def pca_transform(df, pca):
    """
    Transform the spectra field of the DataFrame passed in input. It use 
    the fitted model pca passed as input.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted according to our policy.
    pca : PCA object
        Object of the class PCA decomposition.

    Returns
    -------
    df_pca : pd.DataFrame
        A DataFrame formatted according to our policy. It has a column for every 
        component with user, category and label.

    """
    spectra = list(df['spectra'])
    pca_result = pca.transform(spectra)
    n_components = pca_result.shape[1]
    df.drop(columns = ['spectra', 'x-axis'], inplace = True)
    df['components'] = np.nan
    df['components'] = df['components'].astype('object')
    for i in range(len(df)):
        df.at[i,'components'] = pca[i].tolist()
    return df

def tsne_fit_transform(df, n_components = 2):
    """
    Apply fit_transform function for T-SNE embedding on the spectra data. 
    The returned dataframe is formatted according to our policy with a column 
    for every component of the T-SNE.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame formatted according to our policy.
    n_components : int, optional
        Number of components for the T-SNE. The default is 2.

    Returns
    -------
    df_pca : pd.DataFrame
        A DataFrame formatted according to our policy. It has a column for 
        every components and category, user and label. 

    """
    tsne=TSNE(n_components=n_components)
    spectra = list(df['spectra'])
    tsne_res = tsne.fit_transform(spectra)
    df.drop(columns = ['spectra', 'x-axis'], inplace = True)
    df['components'] = np.nan
    df['components'] = df['components'].astype('object')
    for i in range(len(df)):
        df.at[i,'components'] = pca[i].tolist()
    return df    