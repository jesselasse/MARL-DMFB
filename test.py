import sys, pygame,os
import pygame_menu
from common.rollout import Evaluator
from agent.agent import Agents
from common.arguments import common_args, evaluate_args, set_default, config
import yaml

pygame.init()
screen = pygame.display.set_mode((800,800))
pygame.display.set_caption('Droplet routing for DMFBs')

def get_evaluate_args():
    parser = common_args()
    parser = evaluate_args(parser)
    args = parser.parse_args([_menu.get_widget('mode').get_value()[0][0]])
    args.length = _menu.get_widget('length').get_value()
    args.width = _menu.get_widget('width').get_value()
    args.drop_num = _menu.get_widget('dropnum').get_value()
    args = set_default(args)
    ENV = config(args.name, args.version)
    filename ='TrainParas/4d.yaml'
    with open(filename) as f:
        netdata, data = yaml.safe_load_all(f.read())
    args.__dict__.update(netdata)
    args.show = True
    args.evaluate_task = 5
    args.load_model_name='4d0b/4_9_'
    return args, ENV


def run():
    print('drop number:', _menu.get_widget('dropnum').get_value())
    print('chip size:', _menu.get_widget('length').get_value(), '*', _menu.get_widget('width').get_value())
    args, ENV = get_evaluate_args()
    # ----一次运行FF
    env = ENV(args.width, args.length, args.drop_num,
                  n_blocks=args.block_num, fov=args.fov, stall=args.stall, show=args.show, savemp4=args.show_save)
    args.__dict__.update(env.get_env_info())
    evaluator = Evaluator(env, Agents(args), args.episode_limit)
    average_episode_rewards, average_episode_steps, _, success_rate = evaluator.evaluate(args.evaluate_task)
    print('The averege total_rewards of {} is  {}'.format(
        args.alg, average_episode_rewards))
    print('The each epoch total_steps is: {}'.format(
        average_episode_steps))
    print('The successful rate is: {}'.format(success_rate))
    env.close()
    os.chdir('..')
    # ----


def onchange_dropselect(*args) -> None:
    """
    called if the select is changed.
    """
    b = _menu.get_widget('start')
    b.readonly = False
    b.is_selectable = True
    b.set_cursor(pygame_menu.locals.CURSOR_HAND)


def button_onmouseover(w: 'pygame_menu.widgets.Widget', _) -> None:
    """
    Set the background color of buttons if entered.
    """
    w.set_background_color((98, 103, 106))


def button_onmouseleave(w: 'pygame_menu.widgets.Widget', _) -> None:
    """
    Set the background color of buttons if leaved.
    """
    w.set_background_color('red')


theme = pygame_menu.Theme(
    background_color=pygame_menu.themes.TRANSPARENT_COLOR,
    title=False,
    widget_font=pygame_menu.font.FONT_FIRACODE,
    widget_font_color=(255, 255, 255),
    widget_font_size=25,
    widget_margin=(0, 10),
    #     widget_alignment=pygame_menu.locals.ALIGN_LEFT,
    widget_selection_effect=pygame_menu.widgets.NoneSelection()
)
_menu = pygame_menu.Menu(
    height=700,
    mouse_motion_selection=True,
    position=(10, 25, False),
    theme=theme,
    title='',
    width=600
)
_menu.add.selector('MODE:',[('dmfb','DMFB'), ('meda','MEDA')],selector_id='mode')

_menu.add.range_slider('# Droplet: ', 2, [2, 3, 4, 5, 6, 7, 8, 9, 10], width=200, range_margin=(40, 0),
                       rangeslider_id='dropnum')
_menu.add.range_slider('Chip width:', 10, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], width=250, range_margin=(25, 0),
                       rangeslider_id='width')
_menu.add.range_slider('Chip length:', 10, [10, 20, 30, 40, 50, 60, 70, 80, 90, 100], width=250, range_margin=(10, 0),
                       rangeslider_id='length')


b1 = _menu.add.button(
    'Start',
    run,
    button_id='start',
    cursor=pygame_menu.locals.CURSOR_HAND,
    font_size=20,
    margin=(0, 75),
    shadow_width=10,
    align=pygame_menu.locals.ALIGN_CENTER,
    background_color='red'
)
b1.set_onmouseover(button_onmouseover)
b1.set_onmouseleave(button_onmouseleave)

if __name__ == '__main__':
    while True:

        screen.fill((0, 0, 0))

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                break

        if _menu.is_enabled():
            _menu.update(events)
            _menu.draw(screen)

        pygame.display.update()