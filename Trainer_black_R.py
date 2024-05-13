from HalmaMove import halma
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
import torch
from Tester import Tester
import os



def main ():
    
    env = halma()
    player1 = DQN_Agent(player=-1, env=env,parametes_path=None)
    player_hat = DQN_Agent(player=-1, env=env, train=False)
    Q = player1.DQN
    Q_hat = Q.copy()
    # Q_hat.train = False
    player_hat.DQN = Q_hat
    
    # player2 = Fix_Agent(player=-1, env=env, train=True, random=0)   #0.1
    player2 = Random_Agent(player=1, env=env)   
    buffer = ReplayBuffer(path=None) # None
    

    ########### params #####################
    epochs = 2000000
    start_epoch = 0
    C = 25
    learning_rate = 0.001
    batch_size = 32
    MIN_Buffer = 4000
    results = [] 
    avgLosses = [] 
    avgLoss = 0 
    loss = 0
    res = 0
    best_res = -100
    loss_count = 0
    
    
    # init optimizer
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optim,10000*30, gamma=0.90)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim,[30*5000, 30*100000, 30*250000, 30*500000], gamma=0.5)

    ######### checkpoint Load ############
    File_Num = 22
    checkpoint_path = f"Data/checkpoint{File_Num}.pth"
    buffer_path = f"Data/buffer{File_Num}.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        start_epoch = checkpoint['epoch']+1
        player1.DQN.load_state_dict(checkpoint['model_state_dict'])
        player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        buffer = torch.load(buffer_path)
        results = checkpoint['results']
        avgLosses = checkpoint['avglosses']
        best_res = checkpoint['best_res']
        avgLoss = avgLosses[-1]
    player1.DQN.train()
    player_hat.DQN.eval()

    for epoch in range(start_epoch, epochs):
        print(f'epoch = {epoch}', end='\r')
        state = env.get_init_state((8,8))
        action = player2.get_Action(state=state)
        state_1 = env.get_next_state(state=state, action=action)
        state_1_R = state_1.reverse()
        step = 0
        while not env.is_end_of_game(state_1_R.reverse()):
            ############################## Sample Environement ##########################
            print(step, end='\r')
            step += 1
            env.set_legal_actions(state_1_R)
            action_1_R = player1.get_Action(state=state_1_R,env=env, epoch=epoch, black_state=state_1_R) # fix add param
            after_state_1_R = env.get_next_state(state=state_1_R, action=action_1_R)
            reward_1_R, end_of_game_1_R = env.reward(after_state_1_R)
            if end_of_game_1_R:
                res += reward_1_R
                buffer.push(state_1_R, action_1_R, reward_1_R, after_state_1_R, True)
                break
            state_2 = after_state_1_R.reverse()
            env.set_legal_actions(state_2)
            action_2 = player2.get_Action(env = env,state=state_2)
            after_state_2 = env.get_next_state(state=state_2, action=action_2)
            after_state_2_R = after_state_2.reverse() 
            env.set_legal_actions(after_state_2_R)
            reward_2_R, end_of_game_2 = env.reward(state=after_state_2_R)
            if end_of_game_2:
                res += reward_2_R
            buffer.push(state_1_R, action_1_R, reward_2_R, after_state_2_R, end_of_game_2)
            state_1_R = after_state_2_R

            if len(buffer) < MIN_Buffer:
                continue
            
            ############################ Train NN #####################################
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            Q_values = Q(states[0], actions)
            next_actions = player_hat.get_Actions(next_states, dones) #fixed bug
            
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states[0], next_actions) #todo: use the values calculated in get_Actions

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            scheduler.step()
            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 
            
        if epoch % C == 0:
            Q_hat.load_state_dict(Q.state_dict())

        if (epoch+1) % 100 == 0:
            print(f'\nres= {res}')
            avgLosses.append(avgLoss)
            results.append(res)
            if best_res < res:      
                best_res = res
                # player1.save_param(path_Save)
            res = 0

        # if (epoch+1) % 1000 == 0:
        #     test = tester(100)
        #     test_score = test[0]-test[1]
        #     if best_random < test_score and tester_fix(1) == (1,0):
        #         best_random = test_score
        #         player1.save_param(path_best_random)
        #     print(test)
        #     random_results.append(test_score)

        if epoch % 500 == 0 and epoch > 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': player1.DQN.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'avglosses': avgLosses,
                'results':results,
                'best_res':best_res
            }
            torch.save(checkpoint, checkpoint_path)
            torch.save(buffer, buffer_path)
       
        
        print (f'epoch={epoch} step={step} loss={loss:.5f} avgloss={avgLoss:.5f}', end=" ")
        print (f'learning rate={scheduler.get_last_lr()[0]} path={checkpoint_path} res= {res} best_res = {best_res}')

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': player1.DQN.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'avglosses': avgLosses,
        'results':results,
        'best_res':best_res
    }
    torch.save(checkpoint, checkpoint_path)
    torch.save(buffer, buffer_path)

if __name__ == '__main__':
    main()


