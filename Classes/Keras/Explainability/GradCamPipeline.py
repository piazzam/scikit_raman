import numpy as np
from scipy.signal import argrelextrema
import operator

class GradCamPipeline:

    def __init__(self, heatmaps, spectra):
        self.heatmaps = heatmaps
        self.spectra = spectra
        self.clustered_values = None
        self.clustered_signals = None

    def create_heatmap_peaks(self):
        clustered_signals = []
        clustered_values = []
        mins = []
        for heatmap, spectrum in zip(self.heatmaps, self.spectra):
            min = argrelextrema(spectrum, np.less)[0]
            mins.append(min)
            j = 0
            z = 0
            new_vector = np.zeros(len(spectrum))
            new_vector_cluster = []
            for idx in min:
                v = heatmap[j:idx]
                for i in range(j, idx):
                    new_vector[i] = np.mean(v)
                    new_vector_cluster.append(z)
                j = idx
                z += 1
            clustered_signals.append(new_vector)
            clustered_values.append(new_vector_cluster)
        self.clustered_signals = clustered_signals
        self.clustered_values = clustered_values
        return clustered_values, clustered_signals, mins

    def create_clusters(self):
        c_values = []
        for i in range(len(self.clustered_values)):
            el_np = np.array(self.clustered_values[i])
            unique = np.unique(el_np)
            actual_c_values = []
            for u in unique:
                index = self.clustered_values[i].index(u)
                actual_c_values.append(self.clustered_signals[i][index])
            c_values.append(actual_c_values)
        return c_values

    def get_highest_values_number(self, values, n):
        indexed = list(enumerate(values))
        top = sorted(indexed, key=operator.itemgetter(1))
        topn = top[-n:]
        return list(reversed(topn))

    def create_top_n(self, heatmaps_peaks, n):
        complete_starting = []
        complete_ending = []
        highest_value = []
        complete_peaks_number = []
        for heatmap_peaks, clustered_signal, clustered_value in zip(heatmaps_peaks, self.clustered_signals, self.clustered_values):
            h_val = []
            r = self.get_highest_values_number(heatmap_peaks, n)
            for i in r:
                h_val.append(i[1])
            highest_value.append(h_val)
            starting = []
            ending = []
            peak_number = []
            for e in h_val:
                occ = np.where(clustered_signal == e)
                starting.append(occ[0][0])
                ending.append(occ[0][-1])
                peak_number.append(clustered_value[occ[0][0]])
            complete_starting.append(starting)
            complete_ending.append(ending)
            complete_peaks_number.append(peak_number)
        return highest_value, complete_starting, complete_ending, complete_peaks_number

    def all_pipeline(self, n):
        clustered_values, clustered_signal, _ = self.create_heatmap_peaks()
        clusters = self.create_clusters()
        hg, cs, ce, pn = self.create_top_n(clusters, n)
        return hg, cs, ce, pn