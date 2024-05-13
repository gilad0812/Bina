import numpy as np
import pygame
import time


WIDTH, HEIGHT = 800, 800
ROWS, COLS = 8, 8
SQUARE_SIZE = WIDTH // COLS
LINE_WIDTH = 2
PADDING = SQUARE_SIZE // 5

# RGB
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (211, 211, 211)
GREEN = (0, 128, 0)

pygame.init()


class Graphics:
    def __init__(self, win, board):
        self.board = board
        rows, cols = board.shape
        self.win = win
        self.rows = rows
        self.cols = cols

        self.font = pygame.font.SysFont(None, 48)

    def draw_board(self):
        self.win.fill(LIGHTGRAY)
        for row in range(ROWS):
            for col in range(COLS):
                if (row + col) % 2 == 0:
                    pygame.draw.rect(self.win, WHITE, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))
                else:
                    pygame.draw.rect(self.win, BLACK, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    def draw_piece(self, row_col, player):
        center = self.calc_pos(row_col)
        radius = (SQUARE_SIZE) // 2 - PADDING
        color = self.calc_color(player)
        pygame.draw.circle(self.win, RED, center, radius + 2)
        pygame.draw.circle(self.win, color, center, radius)

    def calc_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE + SQUARE_SIZE // 2
        x = col * SQUARE_SIZE + SQUARE_SIZE // 2
        return x, y

    def calc_base_pos(self, row_col):
        row, col = row_col
        y = row * SQUARE_SIZE
        x = col * SQUARE_SIZE
        return x, y

    def calc_row_col(self, pos):
        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        return row, col

    def calc_color(self, player):
        if player == 1:
            return WHITE
        elif player == -1:
            return BLACK
        else:
            return LIGHTGRAY

    def draw(self):
        self.draw_board()
        for row in range(ROWS):
            for col in range(COLS):
                player = self.board[row][col]
                if player != 0:
                    self.draw_piece((row, col), player)

    def draw_square(self, row_col, color):
        pos = self.calc_base_pos(row_col)
        pygame.draw.rect(self.win, color, (*pos, SQUARE_SIZE, SQUARE_SIZE))

    def draw_text(self, text, color, pos):
        text_surface = self.font.render(text, True, color)
        text_rect = text_surface.get_rect(center=pos)
        self.win.blit(text_surface, text_rect)

    def blink(self, row_col, color):
        row, col = row_col
        player = self.board[row][col]
        for i in range (2):
            self.draw_square((row, col), color)
            if player:
                self.draw_piece((row ,col), player) 
            pygame.display.update()
            time.sleep(0.1)
            self.draw_square((row, col), LIGHTGRAY)
            if player:
                self.draw_piece((row,col), player) 
            pygame.display.update()
            time.sleep(0.1)

    