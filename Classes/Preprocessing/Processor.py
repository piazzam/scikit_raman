from scipy import interpolate
import numpy as np
from itertools import groupby
from tqdm import tqdm
import peakutils
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scikit_raman.Classes.Preprocessing.PCA_result import *
from scikit_raman.Classes.Preprocessing.TSNE_result import *
from scipy.signal import find_peaks, peak_prominences
from scipy.signal import savgol_filter

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
        self.dataset.n_dims = self.dataset.spectra.shape[1]

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
        X_out = X.copy()
        ns = 0
        for i in np.arange(len(spikes) - m):
            if spikes[i] != 0:
                w = np.arange(i - m, i + 1 + m)
                w2 = w[spikes[w] == 0]
                if (len(w2) != 0):
                    X_out[i] = np.mean(np.array(X)[w2])
                    ns += 1
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

    def smoothing_savitzky_golay(self, window_length=9, polyorder=2):
        X = list(self.dataset.spectra)
        X_filter = savgol_filter(X, window_length=window_length, polyorder=polyorder)
        X_filter_bis = []
        for el in X_filter:
            X_filter_bis.append(el)
        self.dataset.spectra = np.array(X_filter_bis)
        self.dataset.spectra_to_numpy()

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
            norm[i] = list(norm[i].reshape(self.dataset.n_dims))
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
        :return: scikit_raman.PCA_result
            An object that represents the results of PCA
        """
        pca = PCA(n_components=n_components)
        spectra = self.dataset.spectra
        pca = pca.fit_transform(spectra)
        pca_result = PCA_result(pca, self.dataset)
        return pca_result

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
        :return: scikit_raman.PCA_result
            An object that represents the results of PCA
        """
        spectra = self.dataset.spectra
        pca_result = pca.transform(spectra)
        pca_result = PCA_result(pca_result, self.dataset)
        return pca_result

    def tsne_fit_transform(self, n_components=2):
        """
        Apply the t-sne reduction.
        :param n_components: int
            Number of components for the t-sne.
        :return: scikit_raman.TSNE_result
            An object that represents the result of tsne
        """
        tsne = TSNE(n_components=n_components)
        spectra = self.dataset.spectra
        tsne_res = tsne.fit_transform(spectra)
        tsne_result = TSNE_result(tsne_res, self.dataset)
        return tsne_result

    def remove_alluminium(self, alluminium):
        """
        Remove alluminium from the background of the spectra.
        :param alluminium: scikit_raman.Alluminium
            this object represent an alluminium signal.
        :return:
        """
        allu = alluminium.spectra
        for i in range(len(self.dataset)):
            current_spectra = self.dataset[i][0]
            new_spectra = []
            for s, al in zip(current_spectra, allu):
                new_spectra.append(s - al)
            self.dataset.spectra[i] = np.array(new_spectra)

    def realignment(self, window_size = 10):
        y = self.dataset.spectra[0].tolist()
        x = self.dataset.x_axis[0].tolist()
        val = min(x, key=lambda x: abs(x - 1001))  # val = number with minimum distance from 1001
        index_val = x.index(val)
        one_value = x.index(val) - window_size
        two_value = x.index(val) + window_size
        y_small = y[one_value:two_value]
        x_small = x[one_value:two_value]
        p = find_peaks(y_small)
        if len(p[0]) > 1:
            prom = -1
            prom_index = 0
            for i in range(len(p[0])):
                index_p = x.index(x_small[p[0][i]])
                y = np.array(y)
                prom_calc = peak_prominences(y, [index_p])
                if prom_calc[0][0] > prom:
                    prom = prom_calc[0][0]
                    prom_index = i
        else:
            prom_index = 0
        x_value = x_small[p[0][prom_index]]
        y_value = y_small[p[0][prom_index]]
        index_true = x.index(x_value)
        new_spectra = []
        new_spectra.append(y)
        new_axis = []
        new_axis.append(x)
        for i in range(1, len(self.dataset)):
            y = self.dataset.spectra[i].tolist()
            x = self.dataset.x_axis[i].tolist()
            val = min(x, key=lambda x: abs(x - 1001))
            one_value = x.index(val) - window_size
            two_value = x.index(val) + window_size
            y_small = y[one_value:two_value]
            x_small = x[one_value:two_value]
            p = find_peaks(y_small)
            new_y = []
            if len(p[0]) > 1:
                prom = -1
                prom_index = 0
                for k in range(len(p[0])):
                    index = x.index(x_small[p[0][k]])
                    y = np.array(y)
                    prom_calc = peak_prominences(y, [index])
                    if prom_calc[0][0] > prom:
                        prom = prom_calc[0][0]
                        prom_index = k
            else:
                prom_index = 0
            if len(p[0]) >= 1:
                x_value_curr = x_small[p[0][prom_index]]
                y_value_curr = y_small[p[0][prom_index]]
                index_curr = x.index(x_value_curr)
                if index_true == index_curr:
                    new_spectra.append(y)
                    new_axis.append(x)
                else:
                    if index_true > index_curr:
                        diff = index_true - index_curr
                        new_index = (len(y)) - diff
                        new_y = y[:new_index]
                        for j in range(diff):
                            new_y = np.insert(new_y, 0, y[0])
                        new_spectra.append(new_y)
                        new_axis.append(x)
                    else:
                        diff = index_curr - index_true
                        new_y = y[diff:]
                        for j in range(diff):
                            new_y = np.append(new_y, y[len(y) - 1])
                        new_spectra.append(new_y)
                        new_axis.append(x)
            else:
                print(i)
                print("ERROR - Max peak not found")
                new_spectra.append(y)
                new_axis.append(x)
        self.dataset.spectra = np.array(new_spectra)
        self.dataset.spectra_to_numpy()
        self.dataset.x_axis = np.array(new_axis)
        self.dataset.x_axis_to_numpy()
        self.dataset.n_elements = len(new_spectra)