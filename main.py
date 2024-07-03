import pygame as pg
import numpy as np
from scipy.signal import convolve2d

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

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
    # 5x5 kernel
    # base_kernel = np.array(
    #     [
    #         [2/5,   1/5,  2/5],
    #         [-0.0, -0.0,  -0.0],
    #         [-.0,  -0.0,  -0.0],
    #     ])
    # make random 3x3 base kernel
    
    # base_kernel = np.flip(base_kernel, axis=0)
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

class Draw():
    def __init__(self):
        self.last_screen = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.float64)
    
    def __call__(self, screen):
        self.draw(screen)

    def draw(self, screen: pg.Surface):
        interp_factor = 0.95
        fadeout_factor = 0.5
        s = surf_to_normalized_array(screen) * interp_factor
        # set any pixels that are less than 0.1 to 0
        s += self.last_screen * (1 - interp_factor)
        s = np.where(s < 0.09, 0, s)
        s = np.clip(s, 0, 1)
        s = normalized_array_to_surf(s)
        screen.blit(s, (0, 0))
        self.last_screen = surf_to_normalized_array(s) * fadeout_factor


def clear(screen):
    screen.fill((0, 0, 0))

def main():
    # make window appear
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    fps = pg.time.Clock()
    pg.display.set_caption("neural worms")

    clear(screen)
    backdrop = screen.copy()

    base_kernel = np.random.rand(3, 3)

    # draw = Draw()

    # keep window on screen
    running = True
    while running:
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
                    backdrop = np.random.rand(SCREEN_WIDTH, SCREEN_HEIGHT, 3)
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
        # add backdrop to screen
        # screen.blit(add_surfaces(screen, backdrop, amt_1=1.0, amt_2=1.0), (0, 0))

        # draw(screen)


        backdrop = surf_to_normalized_array(backdrop)
        # backdrop = np.sin(0.07125 * backdrop * 2 * np.pi)
        backdrop = convolve_array(backdrop, base_kernel)
        backdrop = apply_nonlinearity(backdrop, lambda x: np.cos(x))
        # print(f'{np.max(backdrop[:,:,0])}, {np.max(backdrop[:,:,1])}, {np.max(backdrop[:,:,2])}')
        backdrop = normalized_array_to_surf(backdrop)
        
        screen.blit(backdrop, (0, 0))
        pg.display.flip()
        fps.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()