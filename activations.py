import numpy as np

activations = [
    np.sin,     # classic yes
    np.tanh,    # yes
    # bitcrusher
    lambda x: np.floor(x * 8) / 8,
    #inverse gaussian
    lambda x: -1./pow(2., (pow(x, 2.)))+1,
    np.arctan,  # HELL YES
    np.arcsin,  # yes
    np.cos,     # crazy but sure
    np.exp,     # senselessly crazy
    np.sqrt,    # sure
    np.abs,     # meh
    #inverse square
    lambda x: 1. / pow(x, 2),
    #inverse cube
    lambda x: 1. / pow(x, 3),
    np.arccos,  # no
    np.log,     # no
]