import numpy as np
import random
import math 
from collections import namedtuple, deque
from itertools import count
import time
from itertools import product

import sys
import os
import sys
print(sys.path)

sys.path.append('/home/victor/maturarbeit/Catan')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from config import *
from RL_agent.DQN.log import log, logging

from Catan_Env.state_changer import state_changer

from Catan_Env.action_selection import action_selecter
from Catan_Env.random_action import random_assignment

from Catan_Env.catan_env import board, game, phase, player0, player1, players, new_game

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

class Agent(mp.Process):
    def __init__(self, global_actor_critic, optimizer, input_dims, n_actions, 
                gamma, lr, name, global_ep_idx, env_id):
        super(Agent, self).__init__()
        self.local_actor_critic = ActorCritic(input_dims, n_actions, gamma)
        self.global_actor_critic = global_actor_critic
        self.name = 'w%02i' % name
        self.episode_idx = global_ep_idx
        self.env = gym.make(env_id)
        self.optimizer = optimizer

    def run(self):
        t_step = 1
        while self.episode_idx.value < N_GAMES:
            done = False
            observation = self.env.reset()
            score = 0
            self.local_actor_critic.clear_memory()
            while not done:
                action = self.local_actor_critic.choose_action(observation)
                observation_, reward, done, info = self.env.step(action)
                score += reward
                self.local_actor_critic.remember(observation, action, reward)
                if t_step % T_MAX == 0 or done:
                    loss = self.local_actor_critic.calc_loss(done)
                    self.optimizer.zero_grad()
                    loss.backward()
                    for local_param, global_param in zip(self.local_actor_critic.parameters(),self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    self.optimizer.step()
                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    self.local_actor_critic.clear_memory()
                t_step += 1
                observation = observation_
            with self.episode_idx.get_lock():
                self.episode_idx.value += 1
            print(self.name, 'episode ', self.episode_idx.value, 'reward %.1f' % score)

if __name__ == '__main__':
    lr = 1e-4
    env_id = 'CartPole-v0'
    n_actions = 2
    input_dims = [4]
    N_GAMES = 3000
    T_MAX = 5
    global_actor_critic = ActorCritic(input_dims, n_actions)
    global_actor_critic.share_memory()
    optim = SharedAdam(global_actor_critic.parameters(), lr=lr, 
                        betas=(0.92, 0.999))
    global_ep = mp.Value('i', 0)

    workers = [Agent(global_actor_critic,
                    optim,
                    input_dims,
                    n_actions,
                    gamma=0.99,
                    lr=lr,
                    name=i,
                    global_ep_idx=global_ep,
                    env_id=env_id) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    [w.join() for w in workers]

UPDATE_GLOBAL_ITER = 5
GAMMA = 0.9
MAX_EP = 3000

env = gym.make('CartPole-v0')
N_S = env.observation_space.shape[0]
N_A = env.action_space.n

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = ActorCritic()           # local network
        self.env = gym.make('CartPole-v0').unwrapped

    def run(self):
        total_step = 1
        while self.g_ep.value < MAX_EP:
            s = self.env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                if self.name == 'w00':
                    self.env.render()
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = self.env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)


if __name__ == "__main__":
    gnet = Net(N_S, N_A)        # global network
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i) for i in range(mp.cpu_count())]
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

    import matplotlib.pyplot as plt
    plt.plot(res)
    plt.ylabel('Moving average ep reward')
    plt.xlabel('Step')
    plt.show()


memory = ReplayMemory(100000)



#different types of reward shaping: Immidiate rewards vps, immidiate rewards legal/illegal, immidiate rewards ressources produced, rewards at the end for winning/losing (+vps +legal/illegal)



def select_action(boardstate, vectorstate):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * log.steps_done / EPS_DECAY)

    lr = LR_END + (LR_START - LR_END) * math.exp(-1. * log.steps_done / LR_DECAY)
    
    # Update the learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    if sample > eps_threshold:
        with torch.no_grad():
            if game.cur_player == 0:
                #phase.actionstarted += 1
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
                log.action_counts[action] += 1
                #if phase.actionstarted >= 5:
                #    action_selecter(5,0,0)
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
        log.random_action_counts[action] += 1
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
            log.random_action_counts[action] += 1
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
        
        log.steps_done += phase.statechange
        phase.statechangecount += phase.statechange
        phase.statechange = 0
        game.random_action_made = 0
        phase.reward = 0
        
    a = int(t/100)
    logging(i_episode)
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