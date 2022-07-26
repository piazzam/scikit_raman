import tensorflow as tf
import numpy as np
import copy


class EMSC(tf.keras.layers.Layer):

    def __init__(self, times, keep_original=True, betashift=0.0005, slopeshift=0.002, multishift=0.005, **kwargs):
        super(EMSC, self).__init__(**kwargs)
        self.times = times
        self.keep_original = keep_original
        self.betashift = betashift
        self.slopeshift = slopeshift
        self.multishift = multishift

    def call(self, spectra, training=None):
        if self.keep_original:
            aug_list = copy.copy(spectra)
            #y_list = copy.copy(labels)
        #else:
            #y_list = np.array([])
        for i in range(self.times):
            if self.keep_original == False and i == 0:
                aug_list = self.dataaugment(spectra.numpy(), betashift=self.betashift, slopeshift=self.slopeshift,
                                            multishift=self.multishift)
            else:
                aug_list = np.concatenate((aug_list,
                                           self.dataaugment(spectra.numpy(), betashift=self.betashift, slopeshift=self.slopeshift,
                                                            multishift=self.multishift)))
        #for i in range(self.times):
            #y_list = np.concatenate((y_list, self.labels), axis=0)
        aug_list = tf.convert_to_tensor(aug_list)
        return aug_list #, y_list

    def dataaugment(self, signal, betashift, slopeshift, multishift):
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
        #signal = self.spectra
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

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'times': self.times,
            'keep_original': self.keep_original,
            'betashift': self.betashift,
            'slopeshift': self.slopeshift,
            'multishift': self.multishift,
        })
        return config
