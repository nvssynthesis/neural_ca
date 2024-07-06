import pygame as pg
import pygame_gui as pgui
import numpy as np
from kernel_presets import kernel_presets
from activations import activations
# timer
from time import time

from util import SCREEN_WIDTH, SCREEN_HEIGHT, BACKDROP_WIDTH, BACKDROP_HEIGHT, \
    surf_to_normalized_array, normalized_array_to_surf, convolve_array, display_kernel, make_terrain, clear


def expand_kernel(base_kernel: np.ndarray, expansion_factor: int) -> np.ndarray:
    expansion_kernel = np.zeros((expansion_factor, expansion_factor), dtype=np.float64)
    expansion_kernel[0,0] = 1
    expanded_kernel = np.kron(base_kernel, expansion_kernel)
    return expanded_kernel


def handle_key_presses(keys: dict, neural_ca_params: dict, verbose=False):
    # affect the matrix in small increments using the 3x3 grid of keys: q, w, e, a, s, d, z, x, c
    # if up or down arrow keys are clicked while one of these keys are clicked, the value of the 
    # kernel for that entry is increased or decreased
    kernel_incr = 0.01
    # keys -> kernel positions map
    key_to_position = {
        pg.K_q: (0, 0), pg.K_w: (0, 1), pg.K_e: (0, 2),
        pg.K_a: (1, 0), pg.K_s: (1, 1), pg.K_d: (1, 2),
        pg.K_z: (2, 0), pg.K_x: (2, 1), pg.K_c: (2, 2)
    }
    for key, position in key_to_position.items():
        if keys[key] and (keys[pg.K_UP] or keys[pg.K_DOWN]):
            row, col = position

            neural_ca_params['base_kernel'][row, col] += kernel_incr if keys[pg.K_UP] else -kernel_incr

            neural_ca_params['expanded_kernel'] = expand_kernel(neural_ca_params['base_kernel'], 2)

            if verbose:
                print(f'base kernel: {neural_ca_params['base_kernel']}')

    if keys[pg.K_t]:
        terr_incr = 0.005
        if keys[pg.K_UP] or keys[pg.K_DOWN]:
            terrain_alpha = neural_ca_params['terrain_alpha']
            terrain_alpha += terr_incr if keys[pg.K_UP] else -terr_incr
            terrain_alpha = min(terrain_alpha, 1)
            terrain_alpha = max(terrain_alpha, 0)
            neural_ca_params['terrain_alpha'] = terrain_alpha
            if verbose:
                print(f'terrain_alpha: {terrain_alpha}')



def handle_event(event: pg.event.Event, screen: pg.Surface, keys: dict, neural_ca_params: dict, 
                 hello_button=None,
                 verbose=False):
    if event.type == pg.QUIT:
        pg.quit()
        quit(0)
    if event.type == pgui.UI_BUTTON_PRESSED:
        if event.ui_element == hello_button:
            print('Hello World!')
    # if space bar is clicked, clear screen
    if event.type == pg.KEYDOWN:
        if event.key == pg.K_SPACE:
            clear(screen)
            clear(neural_ca_params['backdrop'])
            # screen.blit(terrain, (0, 0))
        # if 'r' is clicked, randomize the backdrop
        if event.key == pg.K_r:
            bd = np.random.rand(BACKDROP_WIDTH, BACKDROP_HEIGHT, 3)
            bd *= bd
            # print random 3x3 selection of backdrop
            if verbose:
                print(bd[:3, :3, :])
            neural_ca_params['backdrop'] = normalized_array_to_surf(bd)
        # if 'k' is clicked, randomize the kernel
        if event.key == pg.K_k:
            new_base_kernel = np.random.rand(3, 3)
            new_base_kernel *= 2.0
            new_base_kernel -= 1.0
            neural_ca_params['base_kernel'] = new_base_kernel
            neural_ca_params['expanded_kernel'] = expand_kernel(new_base_kernel, 2)
            if verbose:
                print(new_base_kernel)
        # for a few number keys, a preset kernel is chosen. if shift is being held the current kernel is saved to that preset.
        number_keys = [pg.K_0, pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9]
        if event.key in number_keys:
            index = number_keys.index(event.key)
            if keys[pg.K_LSHIFT] or keys[pg.K_RSHIFT]:
                kernel_presets[index] = neural_ca_params['base_kernel']
            else:
                neural_ca_params['base_kernel'] = kernel_presets[index]

def handle_mouse(mouse_pos: tuple, mouse_pressed: tuple, screen: pg.Surface, backdrop: pg.Surface):
    pg.draw.circle(screen, color=(50, 128, 200), center=mouse_pos, radius=3, width=1, 
        draw_top_right=False, draw_top_left=False, draw_bottom_right=False, draw_bottom_left=False)
    
    if mouse_pressed[0]:
        # draw circle on backdrop
        # random color
        color = np.random.rand(3)
        color = color * 255
        pg.draw.circle(backdrop, color=color, center=mouse_pos, radius=1, width=1)



def main(verbose=False):
    # make window appear
    pg.init()
    screen = pg.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    clock = pg.time.Clock()
    gui_manager = pgui.UIManager((SCREEN_WIDTH, SCREEN_HEIGHT))

    pg.display.set_caption("neural worms")

    initial_base = np.array([[0, 0, 0], 
                            [0, 1, 0], 
                            [0, 0, 0]], dtype=np.float64)
    
    neural_ca_state = {
        'terrain_alpha': 0.1,
        'base_kernel': initial_base.copy(),
        'expanded_kernel': expand_kernel(initial_base.copy(), 2),
        'backdrop': None,
    }

    clear(screen)

    terrain = make_terrain()

    if True:
        # backdrop occupies 80% of the screen and is initially all black
        bd = np.zeros((BACKDROP_WIDTH, BACKDROP_HEIGHT, 3), dtype=np.float64)
        bd = normalized_array_to_surf(bd)
        neural_ca_state['backdrop'] = bd

    while True:
        time_delta = clock.tick(60) / 1000.0

        # get frame rate
        if verbose:
            fps = 1.0 / time_delta
            print(f'fps: {fps}')

        keys = pg.key.get_pressed()
        handle_key_presses(keys, neural_ca_params=neural_ca_state, verbose=True)
        for event in pg.event.get():
            handle_event(event, screen, keys, neural_ca_params=neural_ca_state, 
                         hello_button=None, 
                         verbose=False)
            gui_manager.process_events(event)

        gui_manager.update(time_delta)
        
        clear(screen)
        
        handle_mouse(mouse_pos=pg.mouse.get_pos(), mouse_pressed=pg.mouse.get_pressed(), 
                     screen=screen, backdrop=neural_ca_state['backdrop'])

        base_kernel = neural_ca_state['base_kernel']
        expanded_kernel = neural_ca_state['expanded_kernel']
        terrain_alpha = neural_ca_state['terrain_alpha']
        tf = np.arcsin  # get from menu instead of hardcoding

    
        bd = surf_to_normalized_array(neural_ca_state['backdrop'])

        # bd = convolve_array(bd, expanded_kernel)

        bd = convolve_array(bd, base_kernel)
        # bd = (1 - terrain_alpha) * bd + terrain_alpha * (bd * terrain)
        bd = tf(bd)
        bd = normalized_array_to_surf(bd)

        bd.set_alpha(240)

        # bd_to_blit = bd.copy()

        gui_manager.draw_ui(bd)
        screen.blit(bd, (0, 0))

        neural_ca_state['backdrop'] = bd

        display_kernel(screen, base_kernel)

        pg.display.flip()


if __name__ == "__main__":
    main()