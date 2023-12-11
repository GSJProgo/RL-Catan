import numpy as np
import random
import math 
from collections import namedtuple, deque
from itertools import count
import time
from itertools import product

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical

from RL_agent.DQN.config import *
from RL_agent.DQN.log import *

from Catan_Env.state_changer import state_changer
from DQN.replay_memory import ReplayMemory  

from Catan_Env.catan_env import new_game

from Catan_Env.action_selection import action_selecter
from Catan_Env.random_action import random_assignment

#plotting
import wandb 
import plotly.graph_objects as go
wandb.init(project="RL-Catan", name="RL_version_0.1.1", config={})
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(2)

cur_boardstate = state_changer()[0]
cur_vectorstate = state_changer()[1]

agent2_policy_net = NEURAL_NET.to(device)
agent1_policy_net = NEURAL_NET.to(device)

target_net = NEURAL_NET.to(device)
target_net.load_state_dict(agent1_policy_net.state_dict())

optimizer = optim.Adam(agent1_policy_net.parameters(), lr = LR_START, amsgrad=True)
memory = ReplayMemory(100000)

steps_done = 0

#different types of reward shaping: Immidiate rewards vps, immidiate rewards legal/illegal, immidiate rewards ressources produced, rewards at the end for winning/losing (+vps +legal/illegal)

def select_action(boardstate, vectorstate):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done / EPS_DECAY)

    lr = LR_END + (LR_START - LR_END) * math.exp(-1. * steps_done / LR_DECAY)
    
    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if sample > eps_threshold:
        with torch.no_grad():
            if game.cur_player == 0:
                phase.actionstarted += 1
                action = agent1_policy_net(boardstate, vectorstate).max(1).indices.view(1,1)
                if action >= 4*11*21:
                    final_action = action - 4*11*21 + 5
                    position_y = 0
                    position_x = 0
                else:
                    final_action = math.ceil((action/11/21)+1)
                    position_y = math.floor((action - ((final_action-1)*11*21))/21)
                    position_x = action % 21 
                action_selecter(final_action, position_x, position_y)
                action_counts[action] += 1
                if phase.actionstarted >= 5:
                    action_selecter(5,0,0)
                return action
            #elif game.cur_player == 1:
            #    action =  agent2_policy_net(boardstate, vectorstate).max(1).indices.view(1,1) 
            #    if action >= 4*11*21:
            #        final_action = action - 4*11*21 + 5
            #        position_y = 0
            #        position_x = 0
            #    else:
            #        final_action = math.ceil((action/11/21)+1)
            #        position_y = math.floor((action - ((final_action-1)*11*21))/21)
            #        position_x = action % 21 
            #    action_selecter(final_action, position_x, position_y)
            #    action_counts[action] += 1
            #    return action
    else:
        final_action,position_x,position_y = random_assignment()
        if final_action > 4:
            action = final_action + 4*11*21 - 5
        else:
            action = (final_action-1)*11*21 + position_y*21 + position_x 
        random_action_counts[action] += 1
        action_tensor = torch.tensor([[action]], device=device, dtype=torch.long)
        game.random_action_made = 1
        return action_tensor
    
episode_durations = []

def plotting():
    print()

log_called = 0
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s[0] is not None and s[1] is not None, zip(batch.next_boardstate, batch.next_vectorstate))), device=device, dtype=torch.bool)
    non_final_next_board_states = torch.cat([s for s in batch.next_boardstate if s is not None])
    non_final_next_vector_states = torch.cat([s for s in batch.next_vectorstate if s is not None])

    state_batch = (torch.cat(batch.cur_boardstate), torch.cat(batch.cur_vectorstate))
    action_batch = (torch.cat(batch.action))
    reward_batch = torch.cat(batch.reward)
    state_action_values = agent1_policy_net(*state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    next_state_values[non_final_mask] = target_net(non_final_next_board_states, non_final_next_vector_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    #adding sum of state action and sum of expected state action values to wandb
    game.average_q_value_loss.insert(0, loss.mean().item())
    while len(game.average_q_value_loss) > 1000:
        game.average_q_value_loss.pop(1000)

    game.average_reward_per_move.insert(0, phase.reward)
    while len(game.average_reward_per_move) > 1000:
        game.average_reward_per_move.pop(1000)

    game.average_expected_state_action_value.insert(0, expected_state_action_values.mean().item())
    while len(game.average_expected_state_action_value) > 1000:
        game.average_expected_state_action_value.pop(1000)

    optimizer.zero_grad()
    start_time = time.time()
    loss.backward()
    final_time = time.time() - start_time

    optimizer.step()

start_time = time.time()


num_episodes = 1000
for i_episode in range (num_episodes):
    new_game()
    time_new_start = time.time()
    print(i_episode)
    if i_episode % 50 == 49:
        target_net.load_state_dict(agent1_policy_net.state_dict())
        
    if i_episode % 20 == 19:
        torch.save(agent1_policy_net.state_dict(), f'agent{i_episode}_policy_net_0_1_1.pth')
        #agent2_policy_net.load_state_dict(torch.load(f'agent{i_episode}_policy_net_0_0_4.pth'))

    for t in count():
        
        if game.cur_player == 1:
            
            final_action,position_x,position_y = random_assignment()
            if final_action > 4:
                action = final_action + 4*11*21 - 5
            else:
                action = (final_action-1)*11*21 + position_y*21 + position_x 
            random_action_counts[action] += 1
            action = torch.tensor([[action]], device=device, dtype=torch.long)
            game.random_action_made = 1
            phase.actionstarted = 0
            if phase.statechange == 1:
                #calculate reward and check done
                #next_board_state, next_vector_state, reward, done = state_changer()[0], state_changer()[1], phase.reward, game.is_finished  #[this is were I need to perform an action and return the next state, reward, done
                #reward = torch.tensor([reward], device = device)
                #next_board_state = torch.tensor(next_board_state, device = device, dtype = torch.float).unsqueeze(0)
                #next_vector_state = torch.tensor(next_vector_state, device = device, dtype = torch.float).unsqueeze(0)

                if game.is_finished == 1: #this is mormally the var done
                    game.cur_player = 0
                    cur_boardstate =  state_changer()[0]
                    cur_vectorstate = state_changer()[1]
                    cur_boardstate = cur_boardstate.clone().detach().unsqueeze(0).to(device).float()        
                    cur_vectorstate = cur_vectorstate.clone().detach().unsqueeze(0).to(device).float()
                    next_board_state, next_vector_state, reward, done = state_changer()[0], state_changer()[1], phase.reward, game.is_finished  #[this is were I need to perform an action and return the next state, reward, done
                    reward = torch.tensor([reward], device = device)
                    print(reward)
                    next_board_state = next_board_state.clone().detach().unsqueeze(0).to(device).float()
                    next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(device).float()
                    if done == 1:
                        phase.gamemoves = t
                        print("done0")
                        next_board_state = None
                        next_vector_state = None
                    memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
                    cur_boardstate = next_board_state
                    cur_vectorstate = next_vector_state
                    optimize_model()
                    next_board_state = None
                    next_vector_state = None
                #cur_boardstate = next_board_state
                #cur_vector_state = next_vector_state
                if game.is_finished == 1: #this is mormally the var done
                    phase.gamemoves = t
                    print("done1")
                    game.is_finished = 0
                    episode_durations.append(t+1)
                    break
            #else:
            #    phase.reward -= 0.0001
            #    sample = random.random()
            #    if sample < 0.3:
            #        next_board_state, next_vector_state, reward, done = state_changer()[0], state_changer()[1], phase.reward, game.is_finished
            #        reward = torch.tensor([reward], device = device)
            #        next_board_state = torch.tensor(next_board_state, device = device, dtype = torch.float).unsqueeze(0)
            #        next_vector_state = torch.tensor(next_vector_state, device = device, dtype = torch.float).unsqueeze(0)
            #        memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
        elif game.cur_player == 0:
            cur_boardstate =  state_changer()[0]
            cur_vectorstate = state_changer()[1]
            cur_boardstate = cur_boardstate.clone().detach().unsqueeze(0).to(device).float()        
            cur_vectorstate = cur_vectorstate.clone().detach().unsqueeze(0).to(device).float()
            action = select_action(cur_boardstate, cur_vectorstate)
            #calculate reward and check done
            if phase.statechange == 1:
                #phase.reward += 0.0001
                next_board_state, next_vector_state, reward, done = state_changer()[0], state_changer()[1], phase.reward, game.is_finished  #[this is were I need to perform an action and return the next state, reward, done
                reward = torch.tensor([reward], device = device)
                next_board_state = next_board_state.clone().detach().unsqueeze(0).to(device).float()
                next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(device).float()
                if done == 1:
                    phase.gamemoves = t
                    print("done0")
                    next_board_state = None
                    next_vector_state = None
                memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
                cur_boardstate = next_board_state
                cur_vectorstate = next_vector_state
                optimize_model()

                #target_net_state_dict = target_net.state_dict()
                #policy_net_state_dict = agent1_policy_net.state_dict()
                #I might do a mix later on
                #for key in policy_net_state_dict:
                #    target_net_state_dict[key] = TAU*policy_net_state_dict[key] + (1-TAU)*target_net_state_dict[key]
                #target_net.load_state_dict(target_net_state_dict)

                #target_net_state_dict = target_net.state_dict()
                #policy_net_state_dict = agent1_policy_net.state_dict()
                #for key in policy_net_state_dict:
                #    target_net_state_dict[key] = TAU*policy_net_state_dict[key] + (1-TAU)*target_net_state_dict[key]
                #target_net.load_state_dict(target_net_state_dict)


                if done == 1:
                    phase.gamemoves = t
                    game.is_finished = 0
                    episode_durations.append(t+1)
                    break
            else:
                #phase.reward -= 0.00002 #does this gradient get to small? Should I rather add a reward for successful moves?
                sample = random.random()
                if sample < 0.05:
                    next_board_state, next_vector_state, reward, done = state_changer()[0], state_changer()[1], phase.reward, game.is_finished
                    reward = torch.tensor([reward], device = device)
                    next_board_state = next_board_state.clone().detach().unsqueeze(0).to(device).float()
                    next_vector_state = next_vector_state.clone().detach().unsqueeze(0).to(device).float()
                    memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
        
        steps_done += phase.statechange
        phase.statechangecount += phase.statechange
        phase.statechange = 0
        game.random_action_made = 0
        phase.reward = 0
        
    a = int(t/100)
    log(i_episode)
    elapsed_time = time.time() - start_time
    wandb.log({"Elapsed Time": elapsed_time}, step=i_episode)
    wandb.log({"t": t}, step = i_episode)
    #print(t)
    #print(player0.victorypoints)
    #print(player1.victorypoints)
    game.average_time.insert(0, time.time() - time_new_start) 
    if len(game.average_time) > 10:
        game.average_time.pop(10)
    game.average_moves.insert(0, t+1)
    if len(game.average_moves) > 10:
        game.average_moves.pop(10)
    if i_episode > 1:

        game.average_legal_moves_ratio.insert(0, (phase.statechangecount - statechangecountprevious)/t)
        if len(game.average_legal_moves_ratio) > 20:
            game.average_legal_moves_ratio.pop(20)
    statechangecountprevious = phase.statechangecount
    phase.statechange = 0
    game.random_action_made = 0
    phase.reward = 0
    
    
print('Complete')




