from Random_Agent import Random_Agent
from DQN_Agent import DQN_Agent
from HalmaMove import halma
from MinMaxAgent import MinMaxAgent
from Fix_Agent import Fix_Agent
import os
import torch

class Tester:
    def __init__(self, env: halma, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test (self, games_num):
        env = self.env
        player = self.player1
        player1_win = 0
        player2_win = 0
        games = 0
        step = 0
        while games < games_num:
            print (games)
            step += 1

            action = player.get_Action(env=env, train = False,state = env.state)
            if action == -1:
                env.state = env.get_init_state((8,8))
                break
            env.move_piece(action[0],action[1], env.state)
            player = self.switchPlayers(player)
            if env.is_end_of_game(env.state):
                if env.does_white_wins(env.state):
                    player1_win += 1
                elif env.does_black_wins(env.state):
                    player2_win += 1
                env.state = env.get_init_state((8,8))
                games += 1
                player = self.player1
                step = 0
        return player1_win, player2_win        

    def switchPlayers(self, player):
        if player == self.player1:
            return self.player2
        else:
            return self.player1

    def __call__(self, games_num):
        return self.test(games_num)

if __name__ == '__main__':
    File_Num = 10
    checkpoint_path = f"Data/checkpoint{File_Num}.pth"
    env = halma()
    player1 = DQN_Agent(player=1)
    player2 = Random_Agent(player=-1, env = halma())

    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
    player1.DQN.load_state_dict(checkpoint['model_state_dict'])
    test = Tester(env,player1, player2)
    print(test.test(100))
    