import copy
import numpy as np
import random

class DataAugmenter:
    """
    A class that represent a Data Augmenter object. It's possible to augment data in different form.
    ...

    Attributes
    ----------
    spectra : list
        List of the spectra of the dataset.
    labels: list
        List of labels in numeric form.

    Methods
    -------
    augment_signals(self, times, keep_original=True, betashift=0.0005, slopeshift=0.002, multishift=0.005)
        Applies EMSC data augmentation strategy. Augment current dataset many times as specified by times.
    emsc(self, params)
        Apply emsc data augmentation.
    emsc_single_spectra(self, params)
        Apply emsc data augmentation.
    emsc_single_class(self, params)
        Apply emsc data augmentation.
    """

    def __init__(self, spectra, labels):
        """
        Constructor of class DataAugmenter

        Parameters
        ----------
        spectra : list
            List of spectra in dataset.
        labels : list
            List of labels in numerical form.
        """
        self.spectra = spectra
        self.labels = labels

    def _dataaugment(self, betashift, slopeshift, multishift):
        """
        Function propaedeutic to data augmentation. It augments the set of spectra of one time.

        Parameters
        ----------
        betashift : float
            Parameter for varying the shape of the spectra.
        slopeshift : float
            Parameter for varying the shape of the spectra.
        multishift : float
            Parameter for varying the shape of the spectra.

        Returns
        -------
        np.array
            An array containing augmented signals.

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

    def _dataaugment_single_spectra(self, betashift, slopeshift, multishift, index='random'):
        """
        Function propaedeutic to data augmentation. It augments a single spectrum of one time.

        Parameters
        ----------
        betashift : float
            Parameter for varying the shape of the spectra.
        slopeshift : float
            Parameter for varying the shape of the spectra.
        multishift : float
            Parameter for varying the shape of the spectra.
        index : int or str, optional (default is 'random')
            If it is equal to random the function randomly select the index, otherwise augment the signal specified by
            index.

        Returns
        -------
        np.array
            An array containing augmented signals.

        """
        # baseline shift
        if index == 'random':
            random_index = random.randint(0, len(self.spectra) - 1)
        else:
            random_index = index
        signal = self.spectra[random_index]
        label = self.labels[random_index]
        beta = np.random.random(size=(1, 1)) * 2 * betashift - betashift
        slope = np.random.random(size=(1, 1)) * 2 * slopeshift - slopeshift + 1
        # relative positions
        axis = np.array(range(len(signal))) / float(len(signal))
        # offset
        offset = slope * (axis) + beta - axis - slope / 2. + 0.5

        # multiplicative coefficient
        multi = np.random.random(size=(1, 1)) * 2 * multishift - multishift + 1
        augmented_signal = multi * signal + offset

        self.labels = np.append(self.labels, label)

        return augmented_signal

    def _dataaugment_single_class(self, signal, betashift, slopeshift, multishift):
        """
        Function propaedeutic to data augmentation. It augments spectra of a single class.

        Parameters
        ----------
        betashift : float
            Parameter for varying the shape of the spectra.
        slopeshift : float
            Parameter for varying the shape of the spectra.
        multishift : float
            Parameter for varying the shape of the spectra.
        index : int or str, optional (default is 'random')
            If it is equal to random the function randomly select the index, otherwise augment the signal specified by
            index.

        Returns
        -------
        np.array
            An array containing augmented signals.

        """
        # baseline shift
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
        It applies EMSC data augmentation strategy. Augment current dataset many times as specified by times.

        Parameters
        ----------
        times : int
            Number of reply of replicas of the dataset.
        keep_original : bool, optional (default is True)
            Whether to keep the original dataset into the augmented signal
        betashift : float
            Parameter for varying the shape of the spectra.
        slopeshift : float
            Parameter for varying the shape of the spectra.
        multishift : float
            Parameter for varying the shape of the spectra.
        """
        if keep_original:
            aug_list = copy.copy(self.spectra)
            y_list = copy.copy(self.labels)
        else:
            y_list = np.array([])
        for i in range(times):
            if keep_original==False and i == 0:
                aug_list = self._dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)
            else:
                aug_list = np.concatenate((aug_list, self._dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)))
        for i in range(times):
            y_list = np.concatenate((y_list, self.labels), axis=0)
        self.spectra = aug_list
        self.labels = y_list

    def emsc(self, params):
        """
        Function to apply emsc data augmentation.

        Parameters
        ----------
        params : dict
            Parameters for the application of data augmentation.
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
                aug_list = self._dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)
            else:
                aug_list = np.concatenate((aug_list, self._dataaugment(betashift = betashift, slopeshift=slopeshift, multishift=multishift)))
        for i in range(times):
            y_list = np.concatenate((y_list, self.labels), axis=0)
        self.spectra = aug_list
        self.labels = y_list

    def emsc_single_spectra(self, params):
        """
        Apply EMSC data augmentation on a single spectra.

        Parameters
        ----------
        params : dict
            Parameters for the application of data augmentation.
        """
        if params == None:
            keep_original = True
            times = 300
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
                times = 300
            if 'betashift' in params:
                betashift = params['betashift']
            else:
                betashift = 0.0005
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
            if keep_original == False and i == 0:
                aug_list = self._dataaugment_single_spectra(betashift=betashift, slopeshift=slopeshift,
                                                           multishift=multishift)
            else:
                aug_list = np.concatenate(
                    (aug_list, self._dataaugment_single_spectra(betashift=betashift, slopeshift=slopeshift,
                                                               multishift=multishift)))
        for i in range(times):
            y_list = np.concatenate((y_list, self.labels), axis=0)
        self.spectra = aug_list
        #self.labels = y_list

    def emsc_single_class(self, params):
        """
        Apply EMSC data augmentation on a single class.

        Parameters
        ----------
        params : dict
            Parameters for the application of data augmentation.
        """
        if params == None:
            keep_original = True
            times = 300
            betashift = 0.005
            slopeshift = 0.002
            multishift = 0.05
            label = 0
        else:
            if 'keep_original' in params:
                keep_original = params['keep_original']
            else:
                keep_original = True
            if 'times' in params:
                times = params['times']
            else:
                times = 300
            if 'betashift' in params:
                betashift = params['betashift']
            else:
                betashift = 0.0005
            if 'slopeshift' in params:
                slopeshift = params['slopeshift']
            else:
                slopeshift = 0.002
            if 'multishift' in params:
                multishift = params['multishift']
            else:
                multishift = 0.05
            if 'label' in params:
                label = params['label']
            else:
                label = 0
        list_indices_label = np.where(self.labels == label)
        list_label = []
        list_spectra = []
        for el in list_indices_label[0]:
            list_label.append(self.labels[el])
            list_spectra.append(self.spectra[el])
        array_label = np.array(list_label)
        array_spectra = np.array(list_spectra)
        if keep_original:
            aug_list = copy.copy(array_spectra)
            y_list = copy.copy(array_label)
        else:
            y_list = np.array([])
        for i in range(times):
            if keep_original == False and i == 0:
                aug_list = self._dataaugment_single_class(array_spectra, betashift=betashift, slopeshift=slopeshift,
                                                           multishift=multishift)
            else:
                aug_list = np.concatenate(
                    (aug_list, self._dataaugment_single_class(array_spectra, betashift=betashift, slopeshift=slopeshift,
                                                               multishift=multishift)))
        for i in range(times):
            y_list = np.concatenate((y_list, array_label), axis=0)
        spectra_final = []
        label_final = []
        for single_spectra, single_label in zip(self.spectra, self.labels):
            if single_label != label:
                spectra_final.append(single_spectra)
                label_final.append(single_label)
        spectra_final = np.array(spectra_final)
        label_final = np.array(label_final)
        aug_list = np.concatenate((aug_list, spectra_final))
        y_list = np.concatenate((y_list, label_final))
        self.spectra = aug_list
        self.labels = y_list