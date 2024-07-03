import pygame as pg
import numpy as np
from scipy.signal import convolve2d

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

class Draw():
    def __init__(self):
        self.last_screen = np.zeros((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    def __call__(self, screen):
        self.draw(screen)

    def draw(self, screen):
        pg.draw.rect(screen, (255, 255, 255), (100, 100, 10, 10))



def main():
    # make window appear
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pg.display.set_caption("neural worms")

    screen.fill((0, 0, 0))

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
                    screen.fill((0, 0, 0))
        
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