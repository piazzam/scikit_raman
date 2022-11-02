import numpy as np

def spectra_to_numpy(df):
    spectra = df.spectra
    spectra_list = []
    for el in spectra:
        spectra_list.append(np.array(el))
    spectra_list = np.array(spectra_list)
    return spectra_list        