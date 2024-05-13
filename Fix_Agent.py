import numpy as np
from HalmaMove import halma
from State import State
import random


class Fix_Agent:
    def __init__(self, environment: halma = None, player = 1, train = False, random = 0.60) -> None:
        self.env: halma  = environment
        self.player = player
        self.train = train
        self.random = random

    def simple_heuristic(self,state: State,env: halma) -> float:
        player = state.get_opponent()
        opponent = state.player
        n = halma()
        
        player_distance = np.mean([abs(i - 7) + abs(j-7) for i, j in zip(*np.where(state.board == player))])
        opponent_distance = np.mean([i +j for i, j in zip(*np.where(n.state.board == opponent))])
        
        
        unique_actions = len(set(state.legal_actions))
        w = 0
        if player == 1 and env.does_black_wins(state=state):
            w = -10
        if player == -1 and env.does_white_wins(state=state):
            w = 10


        return opponent_distance - player_distance - 0.05 * unique_actions + w
        
    def get_Action (self, events = None, graphics=None, env: halma = None, state: State = None, epoch = 0, train = True):
        legal_actions = state.legal_actions
        if len(legal_actions) == 0:
            return ((0,0),(0,0))
        if self.train and train and random.random() < self.random:
             return random.choice(legal_actions)
        next_states, _ = self.env.get_all_next_states(state)
        values = []
        for next_state in next_states:
                values.append(self.simple_heuristic(next_state,env = env))
        if self.player == 1:
            maxIndex = values.index(max(values))
            return legal_actions[maxIndex]
        else:
            minIndex = values.index(min(values))
            return legal_actions[minIndex]

    
       
        