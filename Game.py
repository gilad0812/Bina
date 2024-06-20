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

player1_file_num = 16
player2_file_num = 21

def main (player1,player2):
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
    
    
def GUI ():
    global player1, player2
    player1 = Human_Agent(player=1)
    player2 = Human_Agent(player=-1)
    # player1 = Human_Agent(player=1)
    # player2 = Human_Agent(player=-1)
    # player2 = Fix_Agent(environment=environment,player=-1, train=True)
    # player2 = Random_Agent(player=-1)
    # player1 = Random_Agent(player=1)
    # player1 = DQN_Agent(player=1, train=False)
    # player2 = DQN_Agent(player=-1, train=False, R = True)
    # player2 = DQN_Agent(player=-1, train=False)
    # player2 = MinMaxAgent(player = -1,depth = 3, environment=environment)
    # player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)

    # model = DQN(environment)
    # model = torch.load(file)
    # player1 = DQNAgent(model, player=1, train=False)
    # player2 = DQNAgent(model, player=2, train=False)

    colors = [['blue', 'gray', 'gray', 'gray'], ['blue', 'gray', 'gray', 'gray']]
    player1_chosen = 0
    player2_chosen = 0
    clock = pygame.time.Clock()
    run = True
    while(run):
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
               run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                if 300<pos[0]<500 and 470<pos[1]<500:
                    main(player1, player2) 
                if 100<pos[0]<300 and 200<pos[1]<240:
                    player1 = Human_Agent(player=1)
                    player1_chosen=0
                if 500<pos[0]<800 and 200<pos[1]<240:
                    player2 = Human_Agent(player=1)
                    player2_chosen=0
                if 100<pos[0]<300 and 250<pos[1]<290:
                    player1 = MinMaxAgent(player = 1,depth = 3, environment=environment)
                    player1_chosen=1
                if 500<pos[0]<800 and 250<pos[1]<290:
                    player2 = MinMaxAgent(player = -1,depth = 3, environment=environment)
                    player2_chosen=1
                if 100<pos[0]<300 and 300<pos[1]<340:
                    player1 = DQN_Agent(player=1, train=False)
                    checkpoint_path = f"Data/checkpoint{player1_file_num}.pth"
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path)
                        player1.DQN.load_state_dict(checkpoint['model_state_dict'])
                    player1_chosen=2
                if 500<pos[0]<800 and 300<pos[1]<340:
                    player2 = DQN_Agent(player=-1, train=False)
                    checkpoint_path = f"Data/checkpoint{player2_file_num}.pth"
                    if os.path.exists(checkpoint_path):
                        checkpoint = torch.load(checkpoint_path)
                        player2.DQN.load_state_dict(checkpoint['model_state_dict'])
                    player2_chosen=2
                if 100<pos[0]<300 and 350<pos[1]<390:
                    player1 = Random_Agent(player=1)
                    player1_chosen=3
                if 500<pos[0]<800 and 340<pos[1]<390:
                    player2 = Random_Agent(player=-1)
                    player2_chosen=3
                if 100<pos[0]<300 and 400<pos[1]<440:
                    player1 = Fix_Agent(environment=environment,player=1, train=True)
                    player1_chosen=4
                if 500<pos[0]<800 and 400<pos[1]<440:
                    player2 = Fix_Agent(environment=environment,player=-1, train=True)
                    player2_chosen=4


        colors = [['gray', 'gray', 'gray', 'gray','gray'], ['gray', 'gray', 'gray', 'gray','gray']]
        colors[0][player1_chosen]='#4CAF50'
        colors[1][player2_chosen]='#4CAF50'




        win.fill('LightGray')
        write(win, "Halma Move", pos=(300, 50), color=BLACK, background_color=None)

        write(win, 'Player 1',(150,150),color=BLACK)
        pygame.draw.rect(win, colors[0][0], (100,200,200,40))
        write(win, 'Human', (120,200),color=BLACK)
        pygame.draw.rect(win, colors[0][1], (100,250,200,40))
        write(win, 'Alpha_Beta', (120,250),color=BLACK)
        pygame.draw.rect(win, colors[0][2], (100,300,200,40))
        write(win, 'DQN', (120,300),color=BLACK)
        pygame.draw.rect(win, colors[0][3], (100,350,200,40))
        write(win, 'Random', (120,350),color=BLACK)
        pygame.draw.rect(win, colors[0][4], (100,400,200,40))
        write(win, 'Fix', (120,400),color=BLACK)

        write(win, 'Player 2',(550,150),color=BLACK)
        pygame.draw.rect(win, colors[1][0], (500,200,200,40))
        write(win, 'Human', (520,200),color=BLACK)
        pygame.draw.rect(win, colors[1][1], (500,250,200,40))
        write(win, 'Alpha_Beta', (520,250),color=BLACK)
        pygame.draw.rect(win, colors[1][2], (500,300,200,40))
        write(win, 'DQN', (520,300),color=BLACK)
        pygame.draw.rect(win, colors[1][3], (500,350,200,40))
        write(win, 'Random', (520,350),color=BLACK)
        pygame.draw.rect(win, colors[1][4], (500,400,200,40))
        write(win, 'Fix', (520,400),color=BLACK)

        
        pygame.draw.rect(win, 'gray', (300,470,200,40))
        write(win, 'Play', (350,470),color=BLACK)

        image = pygame.image.load("pics/halma.jpg")
        win.blit(image, (200, 520))

        pygame.display.update()

    pygame.quit()

def write (surface, text, pos = (50, 20), color = BLACK, background_color = None):
    font = pygame.font.SysFont("Harrington", 34)
    text_surface = font.render(text, True, color, background_color)
    surface.blit(text_surface, pos)

if __name__ == '__main__':
    GUI()
    
