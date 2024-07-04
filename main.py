import pygame as pg
import numpy as np
from kernel_presets import kernel_presets
from activations import activations

from util import SCREEN_WIDTH, SCREEN_HEIGHT, BACKDROP_WIDTH, BACKDROP_HEIGHT, \
    surf_to_normalized_array, normalized_array_to_surf, convolve_array, display_kernel, make_terrain, clear


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
        
        # get mouse position
        mouse_pos = pg.mouse.get_pos()

        pg.draw.circle(screen, color=(50, 128, 200), center=mouse_pos, radius=3, width=1, 
            draw_top_right=False, draw_top_left=False, draw_bottom_right=False, draw_bottom_left=False)
        
        if pg.mouse.get_pressed()[0]:
            # draw circle on backdrop
            # random color
            color = np.random.rand(3)
            color = color * 255
            pg.draw.circle(backdrop, color=color, center=mouse_pos, radius=3, width=1)

        backdrop = surf_to_normalized_array(backdrop)
        backdrop = convolve_array(backdrop, base_kernel)
        backdrop = (1 - terrain_alpha) * backdrop + terrain_alpha * (backdrop * terrain)

        tf = np.arcsin
        backdrop = tf(backdrop)
        
        backdrop = normalized_array_to_surf(backdrop)

        backdrop.set_alpha(240)
        screen.blit(backdrop, (0, 0))
        display_kernel(screen, base_kernel)
        pg.display.flip()
        clock.tick(60)

    pg.quit()


if __name__ == "__main__":
    main()