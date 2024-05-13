import pygame
from HalmaMove import halma
from Graphics import *

class Human_Agent:

    def __init__(self, player: int) -> None:
        self.player = player
        self.old_pos = None

    def get_Action (self, events= None, graphics: Graphics = None, enviroment : halma = None, state = None):
        board = enviroment.state.board
        for event in events:
            if event.type == pygame.MOUSEBUTTONDOWN:
                new = pygame.mouse.get_pos()   
                new_pos = graphics.calc_row_col(new)      
                if self.old_pos == None or board[new_pos] == self.player:                      
                    if enviroment.choose_piece(new_pos, enviroment.state):
                        self.old_pos = new_pos
                    else:
                        graphics.blink(new_pos,RED)
                else:
                    if enviroment.is_valid_move(new_pos,self.old_pos, enviroment.state):
                        graphics.blink(new_pos,GREEN)
                        return self.old_pos,new_pos
                    else:
                        graphics.blink(new_pos,RED)      
            
        return None    

    def get_old_pos(self):
        return self.old_pos             

        
    
                
