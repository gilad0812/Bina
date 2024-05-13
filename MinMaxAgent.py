from HalmaMove import halma
from State import State
from State import State
import random
import numpy as np 

MAXSCORE = 1000

class MinMaxAgent:

    def __init__(self, player, depth = 2, environment: halma = None):
        self.player = player
        if self.player == 1:
            self.opponent = 2
        else:
            self.opponent = 1
        self.depth = depth
        self.environment : halma = environment


    def evaluate(self, gameState: State):
        a_pos = []
        b_pos = []
        a_goal_pos = []
        b_goal_pos = []
        for i in range(8):
            for j in range(8):
                if gameState.board[i][j] == 1:
                    a_pos.append((i, j))
                elif gameState.board[i][j] == -1:
                    b_pos.append((i, j))

        t = 7
        for i in range(4, 8):
            for j in reversed(range(t, 8)):
                a_goal_pos.append((i, j))
            t -= 1
        k = 4
        for i in range(4):
            for j in range(k):
                b_goal_pos.append((i, j))
            k -= 1

        a_blocking = 0
        b_blocking = 0
        for pos1 in a_pos:
            for pos2 in b_pos:
                if pos1[0] == pos2[0] + 1 and abs(pos1[1] - pos2[1]) <= 1:
                    a_blocking += 1
                elif pos1[0] == pos2[0] + 2 and pos1[1] == pos2[1]:
                    a_blocking += 1
        for pos1 in b_pos:
            for pos2 in a_pos:
                if pos1[0] == pos2[0] - 1 and abs(pos1[1] - pos2[1]) <= 1:
                    b_blocking += 1
                elif pos1[0] == pos2[0] - 2 and pos1[1] == pos2[1]:
                    b_blocking += 1

        a_guarding = 0
        b_guarding = 0
        for pos in a_pos:
            if pos in a_goal_pos or pos[0] >= 5:
                a_guarding += 4
        for pos in b_pos:
            if pos in b_goal_pos or pos[0] <= 2:
                b_guarding += 4

                # Precompute distances between all positions
        a_distances = np.zeros((len(a_pos), len(b_goal_pos)))
        b_distances = np.zeros((len(b_pos), len(a_goal_pos)))

        for i, pos1 in enumerate(a_pos):
            for j, pos2 in enumerate(b_goal_pos):
                a_distances[i, j] = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        for i, pos1 in enumerate(b_pos):
            for j, pos2 in enumerate(a_goal_pos):
                b_distances[i, j] = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

        # Compute pos_score efficiently
        pos_score = np.sum(np.min(a_distances, axis=1)) + np.sum(np.min(b_distances, axis=1))
        final = 0
        if gameState.player == 1:
            if all(x == y for x, y in zip(a_pos, a_goal_pos)):
                final = 2
        if gameState.player == -1:
            if all(x == y for x, y in zip(b_pos, b_goal_pos)):
                final = -2
        noise = random.uniform(-3, 3)
        return (pos_score + a_blocking - b_blocking + a_guarding - b_guarding + final * 10 + noise)


       
    def get_Action(self, event, graphics, env: halma,state: State):
        reached = set()
        value, bestAction = self.minmax(env.state, reached, 0, -MAXSCORE,MAXSCORE)
        return bestAction
    

    def minmax(self, gameState: State, reached: set, depth, alpha, beta):
        if gameState.player == self.player:
            value = -MAXSCORE
        else:
            value = MAXSCORE

        if depth == self.depth or self.environment.is_end_of_game(gameState):
            value = self.evaluate(gameState)
            return value, None

        bestAction = None
        legal_actions = self.environment.get_legal_actions(gameState)
        for action in legal_actions:
            newGameState = self.environment.get_next_state(action, gameState)
            if newGameState not in reached:
                reached.add(newGameState)
                if self.player == gameState.player:         
                    newValue, newAction = self.minmax(newGameState, reached, depth + 1, alpha, beta)
                    if newValue > value:
                        value = newValue
                        bestAction = action
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                else:                       
                    newValue, newAction = self.minmax(newGameState, reached, depth + 1, alpha, beta)
                    if newValue < value:
                        value = newValue
                        bestAction = action
                    beta = min(beta, value)
                    if beta <= alpha:
                        break

        return value, bestAction