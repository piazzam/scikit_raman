from scipy import interpolate
import numpy as np
from itertools import groupby
from tqdm import tqdm
import peakutils
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class Processor:
    """
    This class represent a processor. This permits to apply the preprocessing steps to one
    dataset.

    ...

    Attributes:
    ----------
    dataset: scikit_raman.Dataset
        An object of the class Dataset of scikit_raman.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def resample_one_shift(self, y, x, start=400, end=1600, points=991):
        x_new = list(np.linspace(start, end, points))
        fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_new)
        return y_new, x_new

    def resample_shift(self, start=400, end=1600, points=991):
        """Calculate the new set of points for the spectras contained the dataset. Apply the
            shift of the spectra.

            Parameters:
            start: int
                Starting point of the x-axis. Default value is 400.
            end: int
                Ending point of the x-axis. Default value is 1600.
            points: int
                Number of points to take in the x-axis. Default value is 991.
            """
        x_new = np.linspace(start, end, points)

        result = []
        for y,x in zip(self.dataset.spectra, self.dataset.x_axis):
            fi = interpolate.interp1d(x, y, kind='linear', bounds_error=False,
                                      fill_value='extrapolate')
            y_new = fi(x_new)
            result.append(y_new)
        self.dataset.spectra = np.array(result)
        self.dataset.x_axis = np.array([x_new] * len(self.dataset.x_axis))

    def delete_uninformative_spectra(self):
        """
            Delete uninformative spectra from the dataset. Uninformative spectras are defined by:
            10% of zeros or 10% repeated continuos values.
        """
        tot = 0
        df_to_remove = []
        i = 0
        for current_spectra in self.dataset.spectra:
            nz = len(current_spectra) - np.count_nonzero(current_spectra)
            if nz >= (10 * len(current_spectra)) / 100:
                df_to_remove.append(i)
                tot += 1
            i += 1
        self.dataset.remove_elements(df_to_remove)
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
        self.dataset.remove_elements(df_to_remove)

    def modified_z_score(self, intensity):
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

    def fixer(self, X, m, index, threshold=3.5):
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
        """
            Removes the spikes from the spectra.
        """
        X = self.dataset.spectra
        X_c = []
        for x in X:
            #x_fix = self.fixer(x, 5, X.index(x), threshold=3.5)
            x_fix = self.fixer(x, 5, np.where(X == x), threshold=3.5)
            X_c.append(x_fix)
        self.dataset.spectra = np.array(X_c)

    def remove_baseline_polynomial(self, deg=6, max_it=100000, tol=pow(10, -11)):
        """
        Removes baseline (background noise) from the spectra. It apply the
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
        """
        spectra = self.dataset.spectra
        X_out = []
        for i in tqdm(range(len(spectra))):
            if (np.any(spectra[i])):
                baseline_values = peakutils.baseline(np.array(spectra[i]), deg=deg, max_it=max_it, tol=tol)
                z = zip(spectra[i], baseline_values)
                X_out.append([x[0] - x[1] for x in z])
        self.dataset.spectra = np.array(X_out)

    def snv_normalization(self):
        """
        Apply the normalization with the SNV approach.
        """
        X = self.dataset.spectra
        data_snv = np.zeros_like(X)
        for i in range(len(X)):
            # Apply correction
            data_snv[i, :] = list((X[i] - np.mean(X[i])) / np.std(X[i]))
        l = data_snv
        self.dataset.spectra = l

    def min_max_normalization(self):
        """
            Apply the min-max normalization.
        """
        X = self.dataset.spectra.tolist()
        norm = X.copy()
        scaler = MinMaxScaler()
        for i in range(len(X)):
            # Apply correction
            x = np.array(X[i])
            norm[i] = scaler.fit_transform(np.reshape(x, (-1, 1)))
            norm[i] = list(norm[i].reshape(991))
        self.dataset.spectra = np.array(norm)

    def l2_normalization(self):
        """
           Apply the l2 - normalization.
        """
        X = self.dataset.spectra
        X_norm = preprocessing.normalize(X, norm='l2')
        self.dataset.spectra = X_norm

    def peak_normalization(self):
        """
            Apply the peak - normalization.
        """
        X = self.dataset.spectra
        X_norm = preprocessing.normalize(X, norm='max')
        self.dataset.spectra = X_norm

    def pca_fit_transform(self, n_components=2):
        """
        Apply the pca on the spectra data.
        :param n_components: int, optional. The default value is 2.
            Number of components of the pca.
        :return: ndarray
            Transformed values.
        """
        pca = PCA(n_components=n_components)
        spectra = self.dataset.spectra
        pca = pca.fit_transform(spectra)
        return pca

    def pca_fit(self, n_components=2):
        """
        Create the pca object.
        :param n_components: int
            Number of components for the pca object.
        :return: sklearn.PCA object
            PCA object fitted on the data.
        """
        pca = PCA(n_components=n_components)
        spectra = self.dataset.spectra
        pca_el = pca.fit(spectra)
        return pca_el

    def pca_transform(self, pca):
        """
        Transform the data with the pca object.
        :param pca: sklearn.Decomposition.PCA object
            PCA object to apply.
        :return: ndarray
            The components of the pca.
        """
        spectra = self.dataset.spectra
        pca_result = pca.transform(spectra)
        return pca_result

    def tsne_fit_transform(self, n_components=2):
        """
        Apply the t-sne reduction.
        :param n_components: int
            Number of components for the t-sne.
        :return: ndarray
            The components of the t-sne.
        """
        tsne = TSNE(n_components=n_components)
        spectra = self.dataset.spectra
        tsne_res = tsne.fit_transform(spectra)
        return tsne_res

    def remove_alluminium(self, alluminium):
        allu = alluminium.spectra
        for i in range(len(self.dataset)):
            current_spectra = self.dataset[i][0]
            new_spectra = []
            for s, al in zip(current_spectra, allu):
                new_spectra.append(s - al)
            self.dataset.spectra[i] = np.array(new_spectra)