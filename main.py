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
    array = np.clip(array, 0, 1 - 1/255)
    array *= 255
    # array = array.astype(np.uint8)
    surf = pg.surfarray.make_surface(array)
    return surf

def add_surfaces(surf1: pg.Surface, surf2: pg.Surface, amt_1: float = 0.5, amt_2: float = 0.5) -> pg.Surface:
    array1 = surf_to_normalized_array(surf1)
    array2 = surf_to_normalized_array(surf2)
    array = array1 * amt_1 + array2 * amt_2
    return normalized_array_to_surf(array)


def roll_surface(surf: pg.Surface, n: int=1) -> pg.Surface:
    array = surf_to_normalized_array(surf)
    array = np.roll(array, -n, axis=1)
    return normalized_array_to_surf(array)

def convolve_array(array: np.ndarray) -> np.ndarray:
    kernel = np.array([
        # r
        [[0,   0.1,   0.1],
         [0.6, 1,     0], 
         [-1,   0,     -0.6]],
    ])  
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

    draw = Draw()

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
        
        # get mouse position
        mouse_pos = pg.mouse.get_pos()

        pg.draw.circle(screen, color=(50, 128, 200), center=mouse_pos, radius=5, width=1, 
            draw_top_right=False, draw_top_left=False, draw_bottom_right=False, draw_bottom_left=False)
        
        # hold drawing if mouse was clicked
        if pg.mouse.get_pressed()[0]:
            # draw circle on backdrop
            pg.draw.circle(backdrop, color=(128, 128, 128), center=mouse_pos, radius=1, width=1)
        # add backdrop to screen
        screen.blit(add_surfaces(screen, backdrop, amt_1=1.0, amt_2=1.0), (0, 0))

        draw(screen)

        # backdrop = roll_surface(backdrop, n=1)
        backdrop = surf_to_normalized_array(backdrop)
        backdrop = apply_nonlinearity(backdrop, np.sin)
        backdrop = convolve_array(backdrop)
        backdrop = normalized_array_to_surf(backdrop)

        pg.display.flip()
        fps.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()