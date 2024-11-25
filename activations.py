import numpy as np
from enum import Enum, auto

class ActivationEnum(Enum):
    LINEAR = auto()
    RELU = auto()
    GELU = auto()
    ABS = auto()
    SIN = auto()
    SIN_PI = auto()
    TANH = auto()
    ARCTAN = auto()
    BITCRUSHER = auto()
    INV_GAUSS = auto()
    ARCSIN = auto()
    COS = auto()
    COS_PI = auto()
    SQRT = auto()
    EXP = auto()
    INV_SQUARE = auto()
    INV_CUBE = auto()
    ARCCOS = auto()

activations = {
    ActivationEnum.LINEAR: {'function': lambda x: x},
    ActivationEnum.RELU: {'function': lambda x: np.maximum(0, x)},
    ActivationEnum.GELU: {'function': lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))},
    ActivationEnum.SIN: {'function': np.sin},
    ActivationEnum.SIN_PI: {'function': lambda x: np.sin(x * np.pi)},
    ActivationEnum.TANH: {'function': np.tanh},
    ActivationEnum.BITCRUSHER: {'function': lambda x: np.floor(x * 8) / 8},
    ActivationEnum.INV_GAUSS: {'function': lambda x: -1./pow(2., (pow(x, 2.)))+1},
    ActivationEnum.ARCTAN: {'function': np.arctan},
    ActivationEnum.ARCSIN: {'function': np.arcsin},
    ActivationEnum.COS: {'function': np.cos},
    ActivationEnum.COS_PI: {'function': lambda x: np.cos(x * np.pi)},
    ActivationEnum.EXP: {'function': np.exp},
    ActivationEnum.SQRT: {'function': np.sqrt},
    ActivationEnum.ABS: {'function': np.abs},
    ActivationEnum.INV_SQUARE: {'function': lambda x: 1. / pow(x, 2)},
    ActivationEnum.INV_CUBE: {'function': lambda x: 1. / pow(x, 3)},
    ActivationEnum.ARCCOS: {'function': np.arccos},
}
# activations = [
#     np.sin,     # classic yes
#     np.tanh,    # yes
#     # bitcrusher
#     lambda x: np.floor(x * 8) / 8,
#     #inverse gaussian
#     lambda x: -1./pow(2., (pow(x, 2.)))+1,
#     np.arctan,  # HELL YES
#     np.arcsin,  # yes
#     np.cos,     # crazy but sure
#     np.exp,     # senselessly crazy
#     np.sqrt,    # sure
#     np.abs,     # meh
#     #inverse square
#     lambda x: 1. / pow(x, 2),
#     #inverse cube
#     lambda x: 1. / pow(x, 3),
#     np.log,     # no
# ]