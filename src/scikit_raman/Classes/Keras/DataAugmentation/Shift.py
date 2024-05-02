import tensorflow as tf
import numpy as np
from scipy import interpolate

class Shift(tf.keras.layers.Layer):

    def __init__(self, factor, x_axis, points = 991, reduced_value = 16, seed = None, **kwargs):
        super(Shift, self).__init__(**kwargs)
        self.factor = factor
        self.reduced_value = reduced_value
        self.seed = seed
        self.x_axis = x_axis
        self.points = points

    def call(self, spectra, Training=None):
        new_spectra = []
        if self.seed != None:
            tf.random.set_(self.seed)
        for el in spectra.numpy():
            if tf.random.uniform([]) > self.factor:
                new_spectra.append(self.random_shift(el))
            else:
                new_spectra.append(el)
        print(len(new_spectra))
        print(len(new_spectra[0]))
        new_spectra = tf.convert_to_tensor(new_spectra)
        return new_spectra

    def random_shift(self, signal):
        a = tf.random.uniform([], minval=0, maxval=self.reduced_value)
        b = self.reduced_value - a
        augmented_signal = self.resample_one_shift(signal, start = 400+a, end = 1600-b)
        return augmented_signal

    def resample_one_shift(self, y, start=400, end=1600):
        x_new = list(np.linspace(start, end, self.points))
        fi = interpolate.interp1d(self.x_axis, y, kind='linear', bounds_error=False,
                                  fill_value='extrapolate')
        y_new = fi(x_new)
        return y_new

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'factor': self.factor,
            'reduced_value': self.reduced_value,
            'points': self.points,
            'x_axis': self.x_axis,
            'seed': self.seed
        })
        return config