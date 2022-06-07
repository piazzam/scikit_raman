import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

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
    #return pca

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
        df.at[i,'components'] = tsne_res[i].tolist()
    return df

def plot_pca_matplotlib(df):
    #x = []
    #y = []
    #for i in range(len(df)):
    #    x.append(df.iloc[i]['components'][0])
    #    y.append(df.iloc[i]['components'][1])
    com_list = get_pca_components(df)
    if(len(com_list) == 2):
        x = com_list[0]
        y = com_list[1]
        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=df['label'])
        legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Category")
        ax.add_artist(legend1)
        plt.show()
    elif(len(com_list) == 3):
        x = com_list[0]
        y = com_list[1]
        z = com_list[2]
        fig, ax = plt.subplots()
        scatter = ax.scatter(x, y, c=df['label'])
        legend1 = ax.legend(*scatter.legend_elements(),
                        loc="lower left", title="Category")
        ax.add_artist(legend1)
        plt.show()
    else:
        print("Error - impossible to show more than 3 components or less than 2")        
        
    
def plot_pca_plotly(df, title = "PCA plot"):
    #x = []
    #y = []
    #for i in range(len(df)):
    #    x.append(df.iloc[i]['components'][0])
    #    y.append(df.iloc[i]['components'][1])
    com_list = get_pca_components(df)
    if(len(com_list) == 2):
        x = com_list[0]
        y = com_list[1]
        fig = go.Figure()
        fig.update_layout(title_text=title)
        fig = px.scatter(x = x, y = y, color = df['category'])
        fig.show()
    elif(len(com_list) == 3):
        x = com_list[0]
        y = com_list[1]
        z = com_list[2]
        fig = go.Figure()
        fig.update_layout(title_text=title)
        fig = px.scatter(x = x, y = y, z=z, color = df['category'])
        fig.show()
    else:
        print("Error - impossible to show more than 3 components or less than 2")
        
    
def get_pca_components(df):
    n_components = len(df.iloc[0]["components"])
    component_list = [[]] * n_components
    for i in range(len(df)):
        for j in range(n_components):
            print(j)
            component_list[j].append(df.iloc[i]['components'][j])
            print(component_list[j])
        break
    return component_list
        