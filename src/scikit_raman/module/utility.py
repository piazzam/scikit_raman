import numpy as np
from tensorflow import keras
import tensorflow as tf
from sklearn.utils import check_random_state
import random
import os

def spectra_to_numpy(df):
    spectra = df.spectra
    spectra_list = []
    for el in spectra:
        spectra_list.append(np.array(el))
    spectra_list = np.array(spectra_list)
    return spectra_list

def set_seed_keras(seed):
    keras.utils.set_random_seed(seed) #this also set ranomd seed for numpy - tensorflow and python
    tf.config.experimental.enable_op_determinism()

def set_seed_scikit_learn(seed):
    random_state = check_random_state(seed)

def set_seed_numpy(seed):
    np.random.seed(seed)
def set_seed_random(seed):
    random.seed(seed)
def set_seed(keras=True, seed_keras=42, scikit_learn = True, seed_scikit_learn=42, numpy_set = True, seed_numpy=42,
             random_set = True, random_seed=42):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    if keras:
        set_seed_keras(seed_keras)
    if scikit_learn:
        set_seed_scikit_learn(seed_scikit_learn)
    if numpy_set:
        set_seed_numpy(seed_numpy)
    if random_set:
        set_seed_random(random_seed)