import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def pca(df, n_components = 2):
    pca=PCA(n_components=n_components)
    spectra = list(df['spectra'])
    pca=pca.fit_transform(spectra)
    if 'label' in df.columns:
        df_pca=pd.concat([pd.DataFrame(pca, columns = ["component" + str(i) for i in range(n_components)]),df.user, df.category, df.label], axis=1)
        
    else:
        df_pca=pd.concat([pd.DataFrame(pca, columns = ["component" + str(i) for i in range(n_components)]),df.user, df.category], axis=1)
    return df_pca