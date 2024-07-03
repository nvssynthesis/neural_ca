import pygame as pg
import numpy as np
from scipy.signal import convolve2d

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
    bk = base_kernel.T

    kernel = np.array([
        bk,
        bk,
        bk
    ])
    kernel[0] = kernel[0] * 1
    kernel[1] = kernel[0] * 1
    kernel[2] = kernel[0] * 1
    if np.max(array) > 1:
        breakpoint()
    channels = []
    for chan in range(array.shape[2]):
        channel = array[:, :, chan]
        # Apply convolve2d to each channel with its own kernel
        # Assuming kernel is a list of kernels for each channel
        convolved_channel = convolve2d(channel, kernel[0], mode='same', boundary='wrap')
        channels.append(convolved_channel)        
    array = np.stack(channels, axis=-1)
    return array

def apply_nonlinearity(array: np.ndarray, nonlinearity: callable) -> np.ndarray:
    array = nonlinearity(array)
    return array


def clear(screen):
    screen.fill((0, 0, 0))

def main():
    # make window appear
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pg.time.Clock()
    pg.display.set_caption("neural worms")

    clear(screen)
    # backdrop occupies 80% of the screen and is initially all black
    backdrop = np.zeros((BACKDROP_WIDTH, BACKDROP_HEIGHT, 3), dtype=np.float64)
    backdrop = normalized_array_to_surf(backdrop)



    base_kernel = np.array([[0, 0, 0], 
                            [0, 0, 0], 
                            [0, 0, 0]], dtype=np.float64)

    # keep window on screen
    running = True
    while running:
        # affect the matrix in small increments using the 3x3 grid of keys: q, w, e, a, s, d, z, x, c
        # if up or down arrow keys are clicked while one of these keys are clicked, the value of the 
        # kernel for that entry is increased or decreased
        keys = pg.key.get_pressed()
        increment = 0.01
        if keys[pg.K_UP] or keys[pg.K_DOWN]:
            should_print = False
            if keys[pg.K_q]:
                base_kernel[0, 0] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_w]:
                base_kernel[0, 1] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_e]:
                base_kernel[0, 2] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_a]:
                base_kernel[1, 0] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_s]:
                base_kernel[1, 1] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_d]:
                base_kernel[1, 2] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_z]:
                base_kernel[2, 0] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_x]:
                base_kernel[2, 1] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if keys[pg.K_c]:
                base_kernel[2, 2] += increment if keys[pg.K_UP] else -increment
                should_print = True
            if should_print:
                print(base_kernel)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            # if space bar is clicked, clear screen
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_SPACE:
                    clear(screen)
                    clear(backdrop)
                # if 'r' is clicked, randomize the backdrop
                if event.key == pg.K_r:
                    backdrop = np.random.rand(BACKDROP_WIDTH, BACKDROP_HEIGHT, 3)
                    backdrop *= backdrop
                    # print random 3x3 selection of backdrop
                    print(backdrop[:3, :3, :])
                    backdrop = normalized_array_to_surf(backdrop)
                # if 'k' is clicked, randomize the kernel
                if event.key == pg.K_k:
                    base_kernel = np.random.rand(3, 3)
                    base_kernel *= 2.0
                    base_kernel -= 1.0
                    print(base_kernel)
                # for a few number keys, a preset kernel is chosen
                if event.key == pg.K_0:
                    base_kernel = np.array([
                        [-0.795, -0.671, 0.501],
                        [-0.993, -0.792,  0.609],
                        [0.392, 0.74, -0.987]
                    ])


        clear(screen)
        
        # get mouse position
        mouse_pos = pg.mouse.get_pos()

        pg.draw.circle(screen, color=(50, 128, 200), center=mouse_pos, radius=3, width=1, 
            draw_top_right=False, draw_top_left=False, draw_bottom_right=False, draw_bottom_left=False)
        
        # hold drawing if mouse was clicked
        if pg.mouse.get_pressed()[0]:
            # draw circle on backdrop
            # random color
            color = np.random.rand(3)
            color = color * 255
            pg.draw.circle(backdrop, color=color, center=mouse_pos, radius=3, width=1)

        backdrop = surf_to_normalized_array(backdrop)
        backdrop = convolve_array(backdrop, base_kernel)
        backdrop = apply_nonlinearity(backdrop, lambda x: np.sin(x))
        backdrop = normalized_array_to_surf(backdrop)
        
        screen.blit(backdrop, (0, 0))
        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()