import pygame as pg
import numpy as np
from scipy.signal import convolve2d
from kernel_presets import kernel_presets

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


def main():
    # make window appear
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pg.time.Clock()
    pg.display.set_caption("neural worms")

    clear(screen)

    terrain = make_terrain()
    terrain_alpha = 0.1  

    # backdrop occupies 80% of the screen and is initially all black
    backdrop = np.zeros((BACKDROP_WIDTH, BACKDROP_HEIGHT, 3), dtype=np.float64)
    backdrop = normalized_array_to_surf(backdrop)


    base_kernel = np.array([[0, 0, 0], 
                            [0, 1, 0], 
                            [0, 0, 0]], dtype=np.float64)

    # keep window on screen
    running = True
    while running:
        # affect the matrix in small increments using the 3x3 grid of keys: q, w, e, a, s, d, z, x, c
        # if up or down arrow keys are clicked while one of these keys are clicked, the value of the 
        # kernel for that entry is increased or decreased
        keys = pg.key.get_pressed()
        kernel_incr = 0.01
        should_print = False
        # keys -> kernel positions map
        key_to_position = {
            pg.K_q: (0, 0), pg.K_w: (0, 1), pg.K_e: (0, 2),
            pg.K_a: (1, 0), pg.K_s: (1, 1), pg.K_d: (1, 2),
            pg.K_z: (2, 0), pg.K_x: (2, 1), pg.K_c: (2, 2)
        }
        for key, position in key_to_position.items():
            if keys[key] and (keys[pg.K_UP] or keys[pg.K_DOWN]):
                row, col = position
                base_kernel[row, col] += kernel_incr if keys[pg.K_UP] else -kernel_incr
                should_print = True
        if keys[pg.K_t]:
            terr_incr = 0.005
            if keys[pg.K_UP]:
                terrain_alpha += terr_incr
                terrain_alpha = min(terrain_alpha, 1)
                print(terrain_alpha)
            elif keys[pg.K_DOWN]:
                terrain_alpha -= terr_incr
                terrain_alpha = max(terrain_alpha, 0)
                print(terrain_alpha)
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
                    # screen.blit(terrain, (0, 0))
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
                # for a few number keys, a preset kernel is chosen. if shift is being held the current kernel is saved to that preset.
                number_keys = [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9]
                if event.key in number_keys:
                    index = number_keys.index(event.key)
                    if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
                        kernel_presets[index] = base_kernel
                    else:
                        base_kernel = kernel_presets[index]


        clear(screen)
        # screen.blit(terrain, (0, 0))
        
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
        # backdrop = backdrop + terrain_alpha * terrain
        backdrop = (1 - terrain_alpha) * backdrop + terrain_alpha * (backdrop * terrain)
        lamdas = [
            np.sin,     # classic yes
            np.cos,     # crazy but sure
            np.tanh,    # yes

            # bitcrusher
            lambda x: np.floor(x * 8) / 8,
            #inverse gaussian
            lambda x: -1./pow(2., (pow(x, 2.)))+1,

            np.exp,     # senselessly crazy
            np.sqrt,    # sure
            np.arctan,  # HELL YES
            np.arcsin,  # yes
            np.arccos,  # no
            np.log,     # no
            np.abs,     # meh
        ]
        backdrop = apply_nonlinearity(backdrop, np.arcsin)
        
        # sample a line in the middle of the backdrop
        line = backdrop[:, BACKDROP_HEIGHT // 2, :]
        
        backdrop = normalized_array_to_surf(backdrop)

        backdrop.set_alpha(240)
        screen.blit(backdrop, (0, 0))
        display_kernel(screen, base_kernel)
        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()