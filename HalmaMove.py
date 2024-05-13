from State import State
import numpy as np
from Constant import MAX_STEPS


class halma:

    def __init__(self, state:State = None) -> None:
        if state == None:
            self.state = self.get_init_state((8, 8))
        else:
            self.state = state

        self.set_legal_actions(self.state)
        
        
            

    def choose_piece(self, row_col: tuple[int, int], state: State):
        if(state.board[row_col[0]][row_col[1]] == state.player):
            return True
        return False
   
    def get_init_state(self, Rows_Cols):
        rows, cols = Rows_Cols
        board = np.zeros([rows, cols],int)
        k = 4
        for i in range(4):            
            for j in range(k):
                board[i][j] = 1
            k -= 1
        t = 7
        for i in range(4,8):            
            for j in reversed(range(t,8)):
                board[i][j] = -1
            t -= 1        
        state = State (board, 1)
        self.set_legal_actions(state)
        return state

    def move_piece(self,chosen: tuple[int, int], row_col: tuple[int, int], state: State):
        
        if self.is_valid_move(row_col,chosen,state):
            
            state.board[chosen[0]][chosen[1]] = 0
            
            state.board[row_col[0]][row_col[1]] = state.player
            state.action = (chosen,row_col)
            state.switch_player()
            self.set_legal_actions(state)
            state.step += 1
            return True
        return False
    
    def is_free(self, row_col: tuple[int, int], state: State):
        row, col = row_col
        return state.board[row, col] == 0

    def is_valid_move(self, row_col, chosen,state: State):
        b = False
        if self.is_free(row_col, state) and self.is_not_backwards(chosen, row_col, state):
            if self.is_near_move(chosen,row_col):
                b = True
            elif self.is_jump_move(chosen,row_col, state):           
                b = True
        return b       
        
    def is_not_backwards(self, chosen,row_col, state : State):
        if state.is_reversed:
            if state.player == -1 and chosen[0] > row_col[0] and chosen[1] > row_col[1]:    
                return False
            if state.player == 1 and chosen[0] < row_col[0] and chosen[1] < row_col[1]:    
                return False
            return True
        else:
            if state.player == 1 and chosen[0] > row_col[0] and chosen[1] > row_col[1]:    
                return False
            if state.player == -1 and chosen[0] < row_col[0] and chosen[1] < row_col[1]:    
                return False
            return True

    def is_jump_move(self, chosen,row_col, state : State):       
        o1 = int(abs(chosen[0] + row_col[0])/2)
        o2 = int(abs(chosen[1] + row_col[1])/2)
        if abs(chosen[0] - row_col[0]) == 2 and abs(chosen[1] - row_col[1]) == 2:
            if state.board[o1][o2] == state.get_opponent():
                return True
        return False

    def is_near_move(self, current_pos, new_pos):
        
        if abs(current_pos[0] - new_pos[0]) == 1 and abs(current_pos[1] - new_pos[1]) == 1:
            return True
        return False

    def get_jump_pos(self, current_pos, new_pos):
        jump_pos = []
        jump_pos.append(((current_pos[0] + new_pos[0]) // 2, (current_pos[1] + new_pos[1]) // 2))
        return jump_pos

    def does_black_wins(self, state: State):
        rows_cols = state.board.shape
        black = 0
        k = 4
        for i in range(4):            
            for j in range(k):
                if state.board[i][j] == 0:
                    return False
                if state.board[i][j] == -1:
                    black+=1
            k -= 1
        if black > 0:
            return True
        return  False
    
    def does_white_wins(self, state: State):
        rows, cols = state.board.shape
        white = 0
        t = 7
        for i in range(4,8):            
            for j in reversed(range(t,8)):
                if state.board[i][j] == 0:
                    return False
                if state.board[i][j] == 1:
                    white += 1
            t -= 1
        if white > 0:
            return True
        return  False
    
    def is_end_of_game(self, state: State):
        return self.does_black_wins(state) or self.does_white_wins(state)
    
    
    def get_player_pos(self, state: State):
        player = state.player
        indices = np.where(state.board == player)
        return list(zip(indices[0], indices[1]))
    

    def set_legal_actions(self, state: State):
        moves = []
        # for i in range(8):
        #     for j in range(8):
        #         if state.board[i][j] == state.player:
        #             for di in range(8):
        #                 for dj in range(8):
        #                     if self.is_valid_move((di,dj),(i,j),state):
        #                         moves.append(((i,j),(di,dj)))
        # return moves
    
        players = self.get_player_pos(state)
        dir = [(1,1), (-1,-1), (-1, 1), (1, -1), (2,2), (-2,-2), (-2, 2), (2, -2)]
        
        for player in players:
            i, j  = player
            for d in dir:
                di = i + d[0]
                dj = j + d[1]
                if 0<=di<=7 and 0<=dj<=7 and self.is_valid_move((di,dj),(i,j),state):
                   m = ((i,j),(di,dj))
                   moves.append(((i,j),(di,dj)))
        state.legal_actions = moves
         

    def get_legal_actions(self, state: State):
        return state.legal_actions
        


    def get_next_state(self, action, state: State):
        next_state = state.copy()
        self.move_piece(action[0],action[1], next_state)
        self.set_legal_actions(next_state)
        return next_state
    
    def get_all_next_states (self, state: State) -> tuple:
        legal_actions = state.legal_actions
        next_states = []
        for action in legal_actions:
            next_states.append(self.get_next_state(action, state))
        return next_states, legal_actions
    
    def reward (self, state : State, action = None) -> tuple:
        # if action:
        #     next_state = self.get_next_state(action, state)
        # else:
            
        next_state = state
        if (self.is_end_of_game(next_state)):
            if self.does_white_wins(next_state):
                return 1, True  
            elif self.does_black_wins(next_state):
                return -1, True  
            else:
                return 0, True  
        return 0, False