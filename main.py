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

class Draw():
    def __init__(self):
        self.last_screen = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.float64)
    
    def __call__(self, screen):
        self.draw(screen)

    def draw(self, screen: pg.Surface):
        interp_factor = 0.95
        fadeout_factor = 0.7
        s = surf_to_normalized_array(screen) * interp_factor
        s += self.last_screen * (1 - interp_factor)
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
    pg.display.set_caption("neural worms")

    clear(screen)

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
        
        # get mouse position
        mouse_pos = pg.mouse.get_pos()

        pg.draw.circle(screen, color=(50, 128, 200), center=mouse_pos, radius=5, width=1, 
            draw_top_right=False, draw_top_left=False, draw_bottom_right=False, draw_bottom_left=False)
        
        # hold drawing if mouse was clicked
        if pg.mouse.get_pressed()[0]:
            pass

        draw(screen)

        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    main()