import tensorflow as tf
import numpy as np

class EMSC(tf.keras.layers.Layer):
    """
    A class to apply EMSC data augmentation approach on the fly with Keras.
    ...

    Attributes
    ----------
    factor: float
        percentage of the data on which apply on the fly data augmentation.
    seed: int, optional
        if None, no seed is set. Otherwise set a seed for tf.random. Default value is None.
    keep_original: bool
        if True the original dataset is kept in the dataset.
    slopeshift: float, optional
        parameter to smooth the emsc function. The default values is 0.002.
    betashift: float, optional
        parameter to smooth the emsc function. The default value is 0.005.
    multishift: float, optional
        parameter to smooth the emsc function. The default values is 0.005
    """

    def __init__(self, factor, seed = None, betashift=0.0005, slopeshift=0.002, multishift=0.005, **kwargs):
        super(EMSC, self).__init__(**kwargs)
        self.factor = factor
        self.betashift = betashift
        self.slopeshift = slopeshift
        self.multishift = multishift
        self.seed = seed

    def call(self, spectra, Training=None):
        new_spectra = []
        if self.seed != None:
            tf.random.set_seed(self.seed)
        for el in spectra.numpy():
            if tf.random.uniform([]) > self.factor:
                new_spectra.append(self.dataaugment(el))
            else:
                new_spectra.append(el)
        new_spectra = tf.convert_to_tensor(new_spectra)
        return new_spectra

    def dataaugment(self, signal):
        """
        Function propaedeutic to data augmentation.
        :param signal: np.array
            spectra to augment.
        :param betashift:
        :param slopeshift:
        :param multishift:
        :return np.array:
            an array containing augmented signals.
        """
        # baseline shift
        #beta = np.random.random(size=(signal.shape[0], 1)) * 2 * self.betashift - self.betashift
        #slope = np.random.random(size=(signal.shape[0], 1)) * 2 * self.slopeshift - self.slopeshift + 1
        beta = np.random.random(size=(1, 1)) * 2 * self.betashift - self.betashift
        slope = np.random.random(size=(1, 1)) * 2 * self.slopeshift - self.slopeshift + 1
        # relative positions
        #axis = np.array(range(signal.shape[1])) / float(signal.shape[1])
        axis = np.array(range(signal.shape[0])) / float(signal.shape[0])
        # offset
        offset = slope * (axis) + beta - axis - slope / 2. + 0.5

        # multiplicative coefficient
        #multi = np.random.random(size=(signal.shape[0], 1)) * 2 * self.multishift - self.multishift + 1
        multi = np.random.random(size=(1, 1)) * 2 * self.multishift - self.multishift + 1
        augmented_signal = multi * signal + offset

        return augmented_signal

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'factor': self.times,
            'seed': self.keep_original,
            'betashift': self.betashift,
            'slopeshift': self.slopeshift,
            'multishift': self.multishift,
        })
        return config