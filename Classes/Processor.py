from scipy import interpolate
import numpy as np
from itertools import groupby
import tqdm
import peakutils
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Processor:

    def __init__(self, dataset):
        self.dataset = dataset

    def resample_one_shift(self, y, x, start=400, end=1600, points=991):
        x_new = list(np.linspace(start, end, points))
        fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_new)
        return y_new, x_new

    def resample_shift(self, start=400, end=1600, points=991):

        x_new = list(np.linspace(start, end, points))

        result = []
        for y,x in zip(self.dataset.spectra, self.dataset.x_axis):
            fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                      fill_value='extrapolate')
            y_new = fi(x_new)
            result.append(list(y_new))
        self.dataset.spectra = result
        self.dataset.x_axis = [x_new] * len(self.dataset.x_axis)

    def delete_uninformative_spectra(self):
        tot = 0
        df_to_remove = []
        i = 0
        for current_spectra in self.dataset.spectra:
            nz = len(current_spectra) - np.count_nonzero(current_spectra)
            if nz >= (10 * len(current_spectra)) / 100:
                df_to_remove.append(i)
                tot += 1
            i += 1
        #df.drop(df_to_remove, inplace=True)
        for el in df_to_remove:
            self.dataset.spectra.pop(el)
        df_to_remove = []
        i = 0
        for current_spectra in self.dataset.spectra:
            counts = [(k, sum(1 for i in g)) for k, g in groupby(current_spectra)]
            mc = max([c[1] for c in counts])
            if mc >= (10 * len(current_spectra)) / 100:
                df_to_remove.append(i)
                tot += 1
            i += 1
        print("Tot = " + str(tot) + " spettri rimossi")
        for el in df_to_remove:
            self.dataset.spectra.pop(el)
        #df = df.reset_index()
        #return df

    def modified_z_score(self, intensity):
        median_int = np.median(intensity)
        mad_int = np.median([np.abs(intensity - median_int)])
        modified_z_scores = 0.6745 * (intensity - median_int) / mad_int
        return modified_z_scores

    def fixer(self, X, m, index, threshold=3.5):
        spikes = abs(np.array(self.modified_z_score(np.diff(X)))) > threshold
        X_out = X.copy()  # So we don’t overwrite y
        ns = 0
        for i in np.arange(len(spikes) - m):
            if spikes[i] != 0:  # If we have a spike in position i
                w = np.arange(i - m, i + 1 + m)  # we select 2 m + 1 points around our spike
                w2 = w[spikes[w] == 0]  # From such interval, we choose the ones which are not spikes
                if (len(w2) != 0):  # avoid erroneous detected spikes ([2m+1]+ consecutive spikes)
                    X_out[i] = np.mean(np.array(X)[w2])  # and we average their values
                    ns += 1
        # if(ns > 0):
        #    print('Found '+str(ns)+' spikes')
        return X_out

    def spike_removal(self):
        X = self.dataset.spectra
        X_c = []
        for x in X:
            x_fix = self.fixer(x, 5, X.index(x), threshold=3.5)
            X_c.append(x_fix)
        self.dataset.spectra = X_c

    def remove_baseline_polynomial(self, deg=6, max_it=100000, tol=pow(10, -11)):
        spectra = self.dataset.spectra
        X_out = []
        for i in tqdm(range(len(spectra))):
            if (np.any(spectra[i])):
                baseline_values = peakutils.baseline(np.array(spectra[i]), deg=deg, max_it=max_it, tol=tol)
                z = zip(spectra[i], baseline_values)
                X_out.append([x[0] - x[1] for x in z])
        self.dataset.spectra = X_out

    def snv_normalization(self):
        X = self.dataset.spectra
        data_snv = np.zeros_like(X)
        for i in range(len(X)):
            # Apply correction
            data_snv[i, :] = list((X[i] - np.mean(X[i])) / np.std(X[i]))
        l = data_snv.tolist()
        self.dataset.spectra = l

    def min_max_normalization(self):
        X = self.dataset.spectra
        norm = X.copy()
        scaler = MinMaxScaler()
        for i in range(len(X)):
            # Apply correction
            x = np.array(X[i])
            norm[i] = scaler.fit_transform(np.reshape(x, (-1, 1)))
            norm[i] = list(norm[i].reshape(991))
        self.dataset.spectra = norm

    def l2_normalization(self):
        X = self.dataset.spectra
        X_norm = preprocessing.normalize(X, norm='l2')
        self.dataset.spectra = X_norm

    def peak_normalization(self):
        X = self.dataset.spectra
        X_norm = preprocessing.normalize(X, norm='max')
        self.dataset.spectra = X_norm.tolist()

    def pca_fit_transform(self, n_components=2):
        pca = PCA(n_components=n_components)
        spectra = self.dataset.spectra
        pca = pca.fit_transform(spectra)
        return pca

    def pca_fit(self, n_components=2):
        pca = PCA(n_components=n_components)
        spectra = self.dataset.spectra
        pca_el = pca.fit(spectra)
        return pca_el

    def pca_transform(self, pca):
        spectra = self.dataset.spectra
        pca_result = pca.transform(spectra)
        return pca_result

    def tsne_fit_transform(self, n_components=2):
        tsne = TSNE(n_components=n_components)
        spectra = self.dataset.spectra
        tsne_res = tsne.fit_transform(spectra)
        return tsne_res