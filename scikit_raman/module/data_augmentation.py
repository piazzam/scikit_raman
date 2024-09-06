import numpy as np

def data_augment(signal, betashift = 0.0005, slopeshift = 0.002, multishift = 0.005):
    #baseline shift
    beta = np.random.random(size=(signal.shape[0],1))*2*betashift-betashift
    slope = np.random.random(size=(signal.shape[0],1))*2*slopeshift-slopeshift + 1
    #relative positions
    axis = np.array(range(signal.shape[1]))/float(signal.shape[1])
    #offset
    offset = slope*(axis) + beta - axis - slope/2. + 0.5

    #multiplicative coefficient
    multi = np.random.random(size=(signal.shape[0],1))*2*multishift-multishift + 1
    augmented_signal = multi*signal + offset

    return augmented_signal

