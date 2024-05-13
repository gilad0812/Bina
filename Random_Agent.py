import pygame
from HalmaMove import halma
from Graphics import *
import random
from State import State

class Random_Agent:

    def __init__(self, player: int, env = None) -> None:
        self.player = player
        self.env = env
        

    def get_Action (self, event= None, graphics: Graphics = None, env : halma = None,state: State = None, train = None):
        # if env and not state:
        #     legal_actions = env.get_legal_actions(env.state)
        # if env and state:
        #     legal_actions = env.get_legal_actions(state)
        legal_actions = state.legal_actions
        if len(legal_actions) == 0:
            return ((0,0),(0,0))
        return random.choice(legal_actions)

        
    
                
