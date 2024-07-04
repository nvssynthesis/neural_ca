import numpy as np
import pygame as pg
from scipy.signal import convolve2d, fftconvolve


SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
BACKDROP_WIDTH = int(SCREEN_WIDTH * 0.8)
BACKDROP_HEIGHT = int(SCREEN_HEIGHT * 1.0)

KERNEL_DISPLAY_WIDTH = SCREEN_WIDTH - BACKDROP_WIDTH
KERNEL_DISPLAY_HEIGHT = KERNEL_DISPLAY_WIDTH


def surf_to_normalized_array(surf: pg.Surface) -> np.ndarray:
    array = pg.surfarray.array3d(surf)
    array = array.astype(np.float64)
    array *= 1/255
    return array

def normalized_array_to_surf(array: np.ndarray) -> pg.Surface:
    # array = np.clip(array, 0, 1 - 1/255)
    array *= 255
    # array = array.astype(np.uint8)
    surf = pg.surfarray.make_surface(array)
    return surf

def add_surfaces(surf1: pg.Surface, surf2: pg.Surface, amt_1: float = 0.5, amt_2: float = 0.5) -> pg.Surface:
    array1 = surf_to_normalized_array(surf1)
    array2 = surf_to_normalized_array(surf2)
    array = array1 * amt_1 + array2 * amt_2
    array = np.clip(array, 0, 1)
    return normalized_array_to_surf(array)


def roll_surface(surf: pg.Surface, n: int=1) -> pg.Surface:
    array = surf_to_normalized_array(surf)
    array = np.roll(array, -n, axis=1)
    return normalized_array_to_surf(array)

def convolve_array(array: np.ndarray, base_kernel: np.ndarray) -> np.ndarray:
    # Check if using FFT is likely to be faster
    if base_kernel.size > 30:  # Example threshold, adjust based on profiling
        convolve_func = fftconvolve
    else:
        convolve_func = convolve2d  # Assuming convolve2d is imported

    # Prepare the kernel, no need to duplicate
    kernel = base_kernel.T

    channels = []
    for chan in range(array.shape[2]):
        channel = array[:, :, chan]
        # Apply convolution using the selected method
        convolved_channel = convolve_func(channel, kernel, mode='same')
        channels.append(convolved_channel)
        
    return np.stack(channels, axis=-1)

def make_terrain():
    s = np.random.rand(13, 13, 1)
    # make 3d array
    s = np.concatenate([s, s, s], axis=2)

    # center around 0.5
    # s -= s.mean()
    # s += 0.5
    # s = np.sqrt(s)

    s *= 255

    s = s.astype(np.uint8)
    s = normalized_array_to_surf(s)
    s = s.convert(24)
    s = pg.transform.smoothscale(s, (BACKDROP_WIDTH, BACKDROP_HEIGHT))
    s = surf_to_normalized_array(s)

    # s = np.clip(s, 0, 1)

    return s


def clear(screen):
    screen.fill((0, 0, 0))

def display_kernel(screen, kernel):
    block_size = KERNEL_DISPLAY_WIDTH // 3
    color = np.array([1,0,-1], dtype=np.float64)

    for i in range(3):
        for j in range(3):
            this_color = color * kernel[i, j]
            this_color = np.clip(this_color, 0, 1)
            this_color *= 255
            this_color = this_color.astype(np.uint8)
            this_color = tuple(this_color)
            block_surf = pg.Surface((block_size, block_size))
            block_surf.fill(this_color)

            pos_x = BACKDROP_WIDTH + j * block_size
            pos_y = i * block_size
            screen.blit(block_surf, (pos_x, pos_y))

