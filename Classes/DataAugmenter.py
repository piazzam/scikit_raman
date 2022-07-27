import copy
import numpy as np

class DataAugmenter:
    """
    A class that represent a Data Augmenter object. It's possible to augment data in different form.
    ...

    Attributes
    ----------
    spectra : list
        list of the spectra of the dataset.
    labels: list
        list of labels in numeric form.
    """

    def __init__(self, spectra, labels):
        self.spectra = spectra
        self.labels = labels

    def dataaugment(self, betashift, slopeshift, multishift):
        """
        Function propaedeutic to data augmentation.
            :param betashift:
            :param slopeshift:
            :param multishift:
            :return
                np.array:
                    an array containing augmented signals.
        """
        # baseline shift
        signal = self.spectra
        beta = np.random.random(size=(signal.shape[0], 1)) * 2 * betashift - betashift
        slope = np.random.random(size=(signal.shape[0], 1)) * 2 * slopeshift - slopeshift + 1
        # relative positions
        axis = np.array(range(signal.shape[1])) / float(signal.shape[1])
        # offset
        offset = slope * (axis) + beta - axis - slope / 2. + 0.5

        # multiplicative coefficient
        multi = np.random.random(size=(signal.shape[0], 1)) * 2 * multishift - multishift + 1
        augmented_signal = multi * signal + offset

        return augmented_signal

    def augment_signals(self, times, keep_original=True, betashift=0.0005, slopeshift=0.002, multishift=0.005):
        """
        It apply data augmentation strategy. Augment the all dataset many times as specified by times.
        :param times:
            int
                number of reply of replicas of the dataset.
        :param keep_original:
            bool
                keep original dataset or not in the augmented dataset.
        :param betashift:
        :param slopeshift:
        :param multishift:
        """
        if keep_original:
            aug_list = copy.copy(self.spectra)
            y_list = copy.copy(self.labels)
        else:
            y_list = np.array([])
        for i in range(times):
            if keep_original==False and i == 0:
                aug_list = self.dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)
            else:
                aug_list = np.concatenate((aug_list, self.dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)))
        for i in range(times):
            y_list = np.concatenate((y_list, self.labels), axis=0)
        self.spectra = aug_list
        self.labels = y_list

    def emsc(self, params):
        """
        Apply emsc data augmentation.
        :param params: dict
            Parameters on which apply the emsc function.
        :return:
        """
        if params == None:
            keep_original = True
            times = 30
            betashift = 0.005
            slopeshift = 0.002
            multishift = 0.05
        else:
            if 'keep_original' in params:
                keep_original = params['keep_original']
            else:
                keep_original = True
            if 'times' in params:
                times = params['times']
            else:
                times = 30
            if 'betashift' in params:
                betashift = params['betashift']
            else:
                betashift=0.0005
            if 'slopeshift' in params:
                slopeshift = params['slopeshift']
            else:
                slopeshift = 0.002
            if 'multishift' in params:
                multishift = params['multishift']
            else:
                multishift = 0.05
        if keep_original:
            aug_list = copy.copy(self.spectra)
            y_list = copy.copy(self.labels)
        else:
            y_list = np.array([])
        for i in range(times):
            if keep_original==False and i == 0:
                aug_list = self.dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)
            else:
                aug_list = np.concatenate((aug_list, self.dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)))
        for i in range(times):
            y_list = np.concatenate((y_list, self.labels), axis=0)
        self.spectra = aug_list
        self.labels = y_list
