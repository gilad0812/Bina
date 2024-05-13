import pygame
from Graphics import *
from HalmaMove import halma
from Human_Agent import Human_Agent
from MinMaxAgent import MinMaxAgent
from Random_Agent import Random_Agent
from DQN_Agent import DQN_Agent
from Fix_Agent import Fix_Agent
import torch
import time
import os
FPS = 60

# file 9 - vs rnd gamma 0.99 FAILED
# file 10 - vs rnd gamma 0.95 
# file 11 - vs fix gamma 0.95 FAILED
# file 12 - vs rnd gamma 0.95 R black FAILED
# file 13 - vs Fix gamma 0.95, epsiln = 1000 , fix.random = 0.60
# file 14 - vs rnd gamma 0.95 black R FAILED
# file 15 - vs rnd gamma 0.95 black R FAILED
# file 16 - vs rnd gamma 0.95 lr = 0.001 C = 25
# file 17 - vs rnd gamma 0.95 black R FAILED
# file 18 - vs rnd gamma 0.95 black FAILED
# file 19 - vs rnd gamma 0.95 black R
# file 20 - vs rnd gamma 0.95 BW
# file 21 - vs rnd gamma 0.95 black 


win = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Halma Move')
environment = halma()
graphics = Graphics(win, board = environment.state.board)
# player1 = Human_Agent(player=1)
# player2 = Human_Agent(player=-1)
# player2 = Fix_Agent(environment=environment,player=-1, train=True)
player2 = Random_Agent(player=-1)
# player1 = Random_Agent(player=1)
player1 = DQN_Agent(player=1, train=False)
# player2 = DQN_Agent(player=-1, train=False, R = True)
# player2 = DQN_Agent(player=-1, train=False)
# player2 = MinMaxAgent(player = -1,depth = 3, environment=environment)
# player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)

File_Num = 23
checkpoint_path = f"Data/checkpoint{File_Num}.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    player1.DQN.load_state_dict(checkpoint['model_state_dict'])


def main ():
    run = True
    clock = pygame.time.Clock()
    graphics.draw()
    player = player1
    start = time.time()

    while(run):
        clock.tick(FPS)       
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
               run = False
        action = player.get_Action(events, graphics, environment, state=environment.state)
        if action:
            environment.move_piece(action[0],action[1],state=environment.state)
            player = switchPlayers(environment.state) 
        graphics.draw()
        pygame.display.update()
        if environment.is_end_of_game(environment.state):
            run = False
    
    time.sleep(2)
    pygame.quit()
    print("End of game")
    score = environment.state.score()
    print ("score = ", score)
    print (time.time() - start)

def switchPlayers(state):
    if state.player == 1:
       return player1
    else:
        return player2

if __name__ == '__main__':
    main()
