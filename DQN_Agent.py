import torch
import random
import math
from DQN import DQN
from Constant import *
from State import State
from HalmaMove import halma
from Graphics import Graphics
class DQN_Agent:
    def __init__(self, player = 1, parametes_path = None, train = True, env= None,R = None):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.player = player
        self.train = train
        self.R = R
        self.setTrainMode()

    def setTrainMode (self):
          if self.train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action (self,event = None, graphics : Graphics = None, env: halma = None  ,state: State = None , epoch = 0, train = True, black_state = None) -> tuple:
       
        # if env and not state:
        #     actions = env.get_legal_actions(env.state)
        # elif state and env:
        #     state.legal_actions = env.get_legal_actions(state=state)
        actions = state.legal_actions
        if len(state.legal_actions) == 0:
            return ((0,0),(0,0))
        if self.train and train:
            epsilon = self.epsilon_greedy(epoch)
            rnd = random.random()
            if rnd < epsilon:
                return random.choice(actions)
        
        if self.R:
            black_state = state.reverse()
            i = 0
            while(i<len(actions)):
                a = (actions[i])
                b1 = a[0]
                b2 = a[1]
                black_state.legal_actions[i]=(((7-b1[0],7-b1[1]),(7-b2[0],7-b2[1])))
                i+=1
            state_tensor, action_tensor = black_state.toTensor()
        else:
            state_tensor, action_tensor = state.toTensor()

        expand_state_tensor = state_tensor.unsqueeze(0).repeat((len(action_tensor),1))
        
        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)
        max_index = torch.argmax(Q_values)       
        action = actions[max_index]
        return action

    def get_Actions (self, states_tensor: State, dones) -> torch.tensor:
        actions = []
        boards_tensor = states_tensor[0]
        actions_tensor = states_tensor[1]
        for i, board in enumerate(boards_tensor):
            if dones[i].item():
                actions.append(((0,0),(0,0)))
            else:
                actions.append(self.get_Action(state=State.tensorToState(state_tuple=(boards_tensor[i],actions_tensor[i]),player=self.player), train=False))
        return torch.tensor(actions).reshape(-1, 4)

    def epsilon_greedy(self,epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        if epoch > decay:
            return final
        return start - (start - final)*epoch/decay
        # res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
    
    def loadModel (self, file):
        self.model = torch.load(file)
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)
