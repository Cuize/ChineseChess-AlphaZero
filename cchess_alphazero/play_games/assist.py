import sys
import pygame
import random
import os.path
import time
import copy
import numpy as np

from pygame.locals import *
from logging import getLogger
from collections import defaultdict
from datetime import datetime

from cchess_alphazero.play_games import play

import cchess_alphazero.environment.static_env as senv
from cchess_alphazero.environment.chessboard import Chessboard
from cchess_alphazero.environment.chessman import *
from cchess_alphazero.agent.model import CChessModel
from cchess_alphazero.agent.player import CChessPlayer, VisitState
from cchess_alphazero.agent.api import CChessModelAPI
from cchess_alphazero.config import Config
from cchess_alphazero.environment.env import CChessEnv
from cchess_alphazero.environment.lookup_tables import Winner, ActionLabelsRed, flip_move
from cchess_alphazero.lib.model_helper import load_best_model_weight
from cchess_alphazero.lib.tf_util import set_session_config

logger = getLogger(__name__)
main_dir = os.path.split(os.path.abspath(__file__))[0]
SCREENRECT = Rect(0, 0, 800, 577)
PIECE_STYLE = 'WOOD'
BOARD_STYLE = 'WOOD'


def load_image(file, sub_dir=None):
    '''loads an image, prepares it for play'''
    if sub_dir:
        file = os.path.join(main_dir, 'images', sub_dir, file)
    else:
        file = os.path.join(main_dir, 'images', file)
    try:
        surface = pygame.image.load(file)
    except pygame.error:
        raise SystemExit('Could not load image "%s" %s' %
                         (file, pygame.get_error()))
    return surface.convert()


def load_images(*files):
    imgs = []
    style = PIECE_STYLE
    for file in files:
        imgs.append(load_image(file, style))
    return imgs


class Chessman_Sprite(pygame.sprite.Sprite):
    is_selected = False
    images = []
    is_transparent = False

    def __init__(self, images, chessman, w=57, h=57):
        pygame.sprite.Sprite.__init__(self)
        self.chessman = chessman
        self.images = images
        self.image = self.images[0]
        self.rect = Rect(chessman.col_num * 57, (9 - chessman.row_num) * 57, 57, 57)

    def move(self, col_num, row_num, w=57, h=57):
        # print self.chessman.name, col_num, row_num
        old_col_num = self.chessman.col_num
        old_row_num = self.chessman.row_num
        is_correct_position = self.chessman.move(col_num, row_num)
        if is_correct_position:
            self.rect.move_ip((col_num - old_col_num)
                              * 57, (old_row_num - row_num) * 57)
            self.rect = self.rect.clamp(SCREENRECT)
            self.chessman.chessboard.clear_chessmans_moving_list()
            self.chessman.chessboard.calc_chessmans_moving_list()
            return True
        return False

    def update(self):
        if self.is_selected:
            self.image = self.images[1]
        else:
            self.image = self.images[0]


def creat_sprite_group(sprite_group, chessmans_hash, w=57, h=57):
    for chess in chessmans_hash.values():
        if chess.is_red:
            if isinstance(chess, Rook):
                images = load_images("RR.GIF", "RRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("RC.GIF", "RCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("RN.GIF", "RNS.GIF")
            elif isinstance(chess, King):
                images = load_images("RK.GIF", "RKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("RB.GIF", "RBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("RA.GIF", "RAS.GIF")
            else:
                images = load_images("RP.GIF", "RPS.GIF")
        else:
            if isinstance(chess, Rook):
                images = load_images("BR.GIF", "BRS.GIF")
            elif isinstance(chess, Cannon):
                images = load_images("BC.GIF", "BCS.GIF")
            elif isinstance(chess, Knight):
                images = load_images("BN.GIF", "BNS.GIF")
            elif isinstance(chess, King):
                images = load_images("BK.GIF", "BKS.GIF")
            elif isinstance(chess, Elephant):
                images = load_images("BB.GIF", "BBS.GIF")
            elif isinstance(chess, Mandarin):
                images = load_images("BA.GIF", "BAS.GIF")
            else:
                images = load_images("BP.GIF", "BPS.GIF")
        chessman_sprite = Chessman_Sprite(images, chess, w, h)
        sprite_group.add(chessman_sprite)


def select_sprite_from_group(sprite_group, col_num, row_num):
    for sprite in sprite_group:
        if sprite.chessman.col_num == col_num and sprite.chessman.row_num == row_num:
            return sprite


def translate_hit_area(screen_x, screen_y, w=57, h=57):
    return screen_x // 57, 9 - screen_y // 57


def main(config: Config,winstyle=0):
    AI = play.PlayWithHuman(config)
    AI.env.reset()
    AI.load_model()
    AI.pipe = AI.model.get_pipes()
    AI.ai = CChessPlayer(AI.config, search_tree=defaultdict(VisitState), pipes=AI.pipe,
                              enable_resign=True, debugging=True)
    pygame.init()
    bestdepth = pygame.display.mode_ok(SCREENRECT.size, winstyle, 32)
    screen = pygame.display.set_mode(SCREENRECT.size, winstyle, bestdepth)
    pygame.display.set_caption("中国象棋-AlphaZero")

    # create the background, tile the bgd image
    bgdtile = load_image(f'{BOARD_STYLE}.GIF')
    board_background = pygame.Surface([521, 577])
    board_background.blit(bgdtile, (0, 0))
    widget_background = pygame.Surface([700 - 521, 577])
    white_rect = Rect(0, 0, 700 - 521, 577)
    widget_background.fill((255, 255, 255), white_rect)

    #create text label
    font_file = os.path.join(main_dir, 'PingFang.ttc')
    font = pygame.font.Font(font_file, 16)
    font_color = (0, 0, 0)
    font_background = (255, 255, 255)
    t = font.render("着法记录", True, font_color, font_background)
    t_rect = t.get_rect()
    t_rect.centerx = (700 - 521) / 2
    t_rect.y = 10
    widget_background.blit(t, t_rect)

    # background = pygame.Surface(SCREENRECT.size)
    # for x in range(0, SCREENRECT.width, bgdtile.get_width()):
    #     background.blit(bgdtile, (x, 0))
    screen.blit(board_background, (0, 0))
    screen.blit(widget_background, (521, 0))

    # load button image. press button to enable ai assisted move
    pressBT = load_image('PRESS.PNG')
    pressedBT = load_image('PRESSED.PNG')
    screen.blit(pressBT, (720,0))

    pygame.display.flip()

    
    AI.chessmans = pygame.sprite.Group()
    framerate = pygame.time.Clock()

    creat_sprite_group(AI.chessmans, AI.env.board.chessmans_hash)
    current_chessman = None
    AI.env.board.calc_chessmans_moving_list()
    # print(AI.env.board.legal_moves())

    AI.history = [AI.env.get_state()]
    no_act = None
    cont = True


    while not AI.env.board.is_end():
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                AI.env.board.print_record()
                AI.ai.close(wait=False)
                sys.exit()
            elif event.type == VIDEORESIZE:
                    pass
            elif event.type == MOUSEBUTTONDOWN:
                pressed_array = pygame.mouse.get_pressed()
                for index in range(len(pressed_array)):
                    if index == 0 and pressed_array[index]:
                        mouse_x, mouse_y = pygame.mouse.get_pos()
                        if 720 < mouse_x < 780 and 0 < mouse_y < 60:
                            # amount to pressed ai-assist button
                            print('ai-assist-move')
                            # screen.blit(pressedBT, (720,0))
                            cont = AI.actual_ai_move()
                            # pygame.display.update()
                        else:
                            col_num, row_num = translate_hit_area(mouse_x, mouse_y)
                            chessman_sprite = select_sprite_from_group(
                                AI.chessmans, col_num, row_num)
                            if current_chessman is None and chessman_sprite != None:
                                if chessman_sprite.chessman.is_red == AI.env.board.is_red_turn:
                                    current_chessman = chessman_sprite
                                    chessman_sprite.is_selected = True
                            elif current_chessman != None and chessman_sprite != None:
                                if chessman_sprite.chessman.is_red == AI.env.board.is_red_turn:
                                    current_chessman.is_selected = False
                                    current_chessman = chessman_sprite
                                    chessman_sprite.is_selected = True
                                else:
                                    move = str(current_chessman.chessman.col_num) + str(current_chessman.chessman.row_num) +\
                                               str(col_num) + str(row_num)
                                    success = current_chessman.move(col_num, row_num)
                                    AI.history.append(move)
                                    if success:
                                        AI.chessmans.remove(chessman_sprite)
                                        chessman_sprite.kill()
                                        current_chessman.is_selected = False
                                        current_chessman = None
                                        AI.history.append(AI.env.get_state())
                                        # print(AI.env.board.legal_moves())
                            elif current_chessman != None and chessman_sprite is None:
                                move = str(current_chessman.chessman.col_num) + str(current_chessman.chessman.row_num) +\
                                           str(col_num) + str(row_num)
                                success = current_chessman.move(col_num, row_num)
                                AI.history.append(move)
                                if success:
                                    current_chessman.is_selected = False
                                    current_chessman = None
                                    AI.history.append(AI.env.get_state())
                                    # print(AI.env.board.legal_moves())
        # records = AI.env.board.record.split('\n')
        # font = pygame.font.Font(font_file, 12)
        # i = 0
        # for record in records[-20:]:
        #     rec_label = font.render(record, True, font_color, font_background)
        #     t_rect = rec_label.get_rect()
        #     t_rect.centerx = (700 - 521) / 2
        #     t_rect.y = 35 + i * 15
        #     widget_background.blit(rec_label, t_rect)
        #     i += 1
        # screen.blit(widget_background, (521, 0))
        # screen.blit(pressedBT, (521,0))
        AI.draw_widget(screen, widget_background)
        framerate.tick(20)
        # clear/erase the last drawn sprites
        AI.chessmans.clear(screen, board_background)

        # update all the sprites
        # screen.blit(pressBT, (720,0))
        AI.chessmans.update()
        AI.chessmans.draw(screen)
        pygame.display.update()

        if not cont:
            break

    AI.ai.close(wait=False)
    logger.info(f"Winner is {AI.env.board.winner} !!!")
    AI.env.board.print_record()

if __name__ == '__main__':
    main()