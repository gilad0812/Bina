import numpy as np
import torch

class State:

    def __init__(self, board, player = 1,legal_actions = [], is_reversed = False):        
        self.player = player
        self.board = board
        self.legal_actions = legal_actions 
        self.action : tuple[tuple[int, int],tuple[int, int]] = None
        self.is_reversed = is_reversed
        self.step = 0
    
    def switch_player(self):
        if self.player == 1:
            self.player = -1
        else:
            self.player = 1

    def get_opponent (self):
        if self.player == 1:
            return -1
        else:
            return 1
    
    def get_player (self):
        if self.player == 1:
            return 1
        else:
            return -1
        
    def score (self) -> tuple[int, int]:
        self.board.sum()
        

    def __eq__(self, other) ->bool:
        b1 = np.equal(self.board, other.board).all()
        b2 = self.player == other.player
        return b1 and b2

    def __hash__(self) -> int:
        return hash(repr(self.board) + repr(self.player))
    
    def copy (self):
        newBoard = np.copy(self.board)
        legal = self.legal_actions.copy()
        state = State(board=newBoard, player=self.player, 
        legal_actions=legal, is_reversed= self.is_reversed) 
        state.step = self.step
        return state
    
    def reverse (self):
        reversed = self.copy()
        # if self.action:
        #     from_ , to = self.action
        #     black_action = self.calc(self.action)
        #     reversed.board[from_[0],from_[1]] = 1
        #     reversed.board[to[0],to[1]] = 0
        reversed.board = (np.flip(reversed.board*-1))
        # reversed.is_reversed = self.is_reversed == False
        reversed.player = reversed.player * -1
        return reversed
    
    def calc(self):
    # Unpack the move tuple
        piece_pos, target_pos = self.action
        
        # Calculate the mirrored move for black to white
        mirrored_piece_pos = (7 - piece_pos[1], 7 - piece_pos[0])
        mirrored_target_pos = (7 - target_pos[1], 7 - target_pos[0])
        
        return (mirrored_piece_pos, mirrored_target_pos)

    def toTensor (self, device = torch.device('cpu')) -> tuple:
        board_np = self.board.reshape(-1)
        board_tensor = torch.tensor(board_np.copy(), 
                dtype=torch.float32, device=device)
        # step_tensor = torch.tensor([self.step])
        # board_tensor = torch.cat((board_tensor, step_tensor ))
        actions_np = np.array(self.legal_actions)
        actions_tensor = torch.from_numpy(actions_np)
        actions_tensor = actions_tensor.reshape(-1,4)
        return board_tensor, actions_tensor
        # return board_tensor, actions_np

    [staticmethod]
    def tensorToState (state_tuple, player):
        board_tensor = state_tuple[0]
        # step = state_tuple[0][64]
        board = board_tensor.reshape([8,8]).cpu().numpy()
        legal_actions_tensor = state_tuple[1]
        # legal_actions_tensor = legal_actions_tensor.reshape(-1, 2, 2) 
        legal_actions = []
        for action_tensor in legal_actions_tensor:
            action = (action_tensor[0].item(), action_tensor[1].item()), (action_tensor[2].item(), action_tensor[3].item())
            legal_actions.append(action)
        # legal_actions = legal_actions_tensor.cpu().numpy()
        # legal_actions = list(map(tuple, legal_actions))
        # legal_actions = state_tuple[1]
        state = State(board, player=player, legal_actions=legal_actions)
        # state.step = step.item() 
        return state