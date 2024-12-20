import numpy as np

kernel_presets: dict = {
    0: np.array([
        [-0.795, -0.671, 0.501],
        [-0.993, -0.792,  0.609],
        [0.392, 0.74, -0.987]
    ], dtype=np.float64),
    1: np.array([
        [-0.04884933, -0.44665813,  0.61806521],
        [-0.91224085, -0.06480854,  0.28417327],
        [ 0.37040991, -0.73425871,  0.31814526]
    ], dtype=np.float64),
    2: np.array([
        [ 0.3,  -1.,    0.51],
        [-0.78, -1.66, -0.78],
        [-1.43,  0.38, -0.07]
    ], dtype=np.float64),
    3: np.array([
        [ 0.17075394,  0.21331279,  0.25395169],
        [ 0.22687523,  0.09328171, -0.36148304],
        [ 0.20261168,  0.21646832,  0.22501386]
    ], dtype=np.float64),
    4: np.array([
        [-0.27924606,  0.13331279,  0.28395169],
        [ 0.40687523,  0.29328171,  0.16851696],
        [-0.29738832,  0.30646832,  0.13501386]
    ], dtype=np.float64),
    5: np.array([
        [ 0.21075394,  0.17331279, -0.05604831],
        [ 0.51687523,  0.11328171,  0.14851696],
        [-0.76738832,  0.46646832,  0.20501386]
    ], dtype=np.float64),
    6: np.array([
        [ 0.19075394,  0.17331279, -0.05604831],
        [ 0.54687523,  0.03328171,  0.14851696],
        [-0.76738832,  0.49646832,  0.18501386]
    ], dtype=np.float64),
    7: np.array([
        [-0.45459072, -0.91901521,  0.32544112],
        [-0.69739655,  0.24759039, -0.65519655],
        [-0.42030883,  0.85556822,  0.57564318]
    ], dtype=np.float64),
    8: np.array([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.float64),
    9: np.array([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]
    ], dtype=np.float64)
}