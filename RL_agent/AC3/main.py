import numpy as np
import random
import math 
from collections import namedtuple, deque
from itertools import count
import time
from itertools import product
import gc


from utils import v_wrap, push_and_pull, record, init_weights

from SharedAdam import SharedAdam
from torch.optim.lr_scheduler import ExponentialLR

import sys
import os
import sys
print(sys.path)

sys.path.append('/home/victor/Maturarbeit/Catan/RL-Catan')

from log import Log
import torch.multiprocessing as mp
import torch.distributions as Categorical


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.distributions import Categorical
from torch.multiprocessing import multiprocessing

from config import *

from Catan_Env.state_changer import state_changer

from Catan_Env.action_selection import action_selecter
from Catan_Env.random_action import random_assignment

from Catan_Env.catan_env import Catan_Env, create_env

from Neural_Networks.A3C_Medium_Seperated import ActorCritic
#plotting
import wandb 
import plotly.graph_objects as go
import os
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("available_gpus", available_gpus)
run = wandb.init(project="RL-Catan_AC3", name="RL_version_9.7.2", config={}, group='finalrun9.7.2')

torch.manual_seed(2)

def select_action(action, env):

    with torch.no_grad():
        
        if action >= 4*11*21:
            final_action = action - 4*11*21 + 5
            position_y = 0
            position_x = 0
        else:
            final_action = math.ceil(((action + 1)/11/21))
            position_y = math.floor((action - ((final_action-1)*11*21))/21)
            position_x = action % 21 
        action_selecter(env, final_action, position_x, position_y)
        return action
       

UPDATE_GLOBAL_ITER = 1000
GAMMA = 0.99
MAX_EP = 500000

class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, device, logger, global_device):
        super(Worker, self).__init__()
        self.number = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.actor_optimizer = torch.optim.Adam(self.gnet.actor_parameters, lr=4e-4, betas=(0.9, 0.999), weight_decay=0.0001) 
        self.critic_optimizer = torch.optim.Adam(self.gnet.critic_parameters, betas=(0.9, 0.999), lr=1e-5, weight_decay=0.0001) 
        self.critic_optimizer = opt #need to think if this works like this

        
        self.env = Catan_Env()
        self.device = torch.device("cuda:{}".format(device))
        self.global_device = torch.device("cuda:{}".format(global_device))  
        self.lnet = ActorCritic().to(self.device)       
        self.oppnet = ActorCritic().to(self.device)
        self.logger = logger

        self.average_loss = []
        self.average_v_s_ = []
        self.average_c_loss = []
        self.average_a_loss = []
        self.average_entropy = []
        self.average_l2 = []
        self.average_v_s_end = []
        self.average_actorregu = []
        self.average_criticregu = []
        self.average_value_loss = []

        self.gametime = []

        self.count = 0

        self.hasassigned = 0
        self.haslrupdated = 0
       
        print("self.device", self.device)

    def run(self):
        log_dict = {}
        step_dict = {}
        value_dict = {}
        total_step = 1
        self.average_loss = []
        self.average_v_s_ = []
        self.average_c_loss = []
        self.average_a_loss = []
        self.average_entropy = []
        self.average_l2 = []
        self.average_actorregu = []
        self.average_criticregu = []
        self.average_value_loss = []
        buffer_boardstate, buffer_vectorstate, buffer_a, buffer_r, buffer_logits, buffer_values = [], [], [], [], [], []
        while self.g_ep.value < MAX_EP:  
            if self.g_ep.value % 1000 == 0:
                torch.save(self.gnet.state_dict(), f'A3Cagent{self.g_ep.value}_policy_net_9_7_2.pth')
            
            print("episode", self.g_ep.value)
            self.env.new_game()
            boardstate = state_changer(self.env)[0].to(self.device)
            vectorstate = state_changer(self.env)[1].to(self.device)

            ep_r = 0.
            if self.opt.param_groups[0]['lr'] > 5e-5:
                self.opt.param_groups[0]['lr'] = 1e-4 * 0.9998 ** (self.g_ep.value)
                self.actor_optimizer.param_groups[0]['lr'] = 4e-4 * 0.9999 ** (self.g_ep.value)
                self.critic_optimizer.param_groups[0]['lr'] = 4e-4 * 0.9999 ** (self.g_ep.value)
            elif self.opt.param_groups[0]['lr'] > 1e-5:
                self.opt.param_groups[0]['lr'] = 5e-5 * 0.99998 ** (self.g_ep.value)
                self.actor_optimizer.param_groups[0]['lr'] = 5e-5 * 0.99998 ** (self.g_ep.value)
                self.critic_optimizer.param_groups[0]['lr'] = 5e-5 * 0.99998 ** (self.g_ep.value)
            else:
                self.opt.param_groups[0]['lr'] = 1e-5 * 0.999998 ** (self.g_ep.value)
                self.actor_optimizer.param_groups[0]['lr'] = 1e-5 * 0.999998 ** (self.g_ep.value)
                self.critic_optimizer.param_groups[0]['lr'] = 1e-5 * 0.999998 ** (self.g_ep.value)


            print("new_lr", self.opt.param_groups[0]['lr'])
            print("new_actor_lr", self.actor_optimizer.param_groups[0]['lr'])
            print("new_critic_lr", self.critic_optimizer.param_groups[0]['lr'])

            for i in range (len(buffer_a)):
                buffer_a.pop()
            for i in range (len(buffer_boardstate)):
                buffer_boardstate.pop()
            for i in range (len(buffer_vectorstate)):
                buffer_vectorstate.pop()
            for i in range (len(buffer_r)):
                buffer_r.pop()
            for i in range (len(buffer_logits)):
                buffer_logits.pop()
            for i in range (len(buffer_values)):
                buffer_values.pop()
            
            #print("buffer_a", len(buffer_a))    
            #print("buffer_boardstate", len(buffer_boardstate))
            #print("buffer_vectorstate", len(buffer_vectorstate))
            #print("buffer_r", len(buffer_r))
            #print("buffer_logits", len(buffer_logits))
            #print("buffer_values", len(buffer_values))

            

            torch.cuda.memory_summary(device=None, abbreviated=False)


            while True:
                self.count += 1
                if self.env.game.cur_player == 0:
                    self.env.phase.statechange = 0
                    a, meanlogits, logits, values = self.lnet.choose_action(boardstate, vectorstate, self.env, total_step) #select action
                    buffer_logits.append(logits)
                    buffer_values.append(values)
                    log_dict[f'meanlogits{self.name}'] = meanlogits
                    select_action(a, self.env)

                    boardstate_,vectorstate_, r, done =  state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.env.game.is_finished
                    self.logger.action_counts[a] += 1 
                    self.env.phase.reward = 0
                    boardstate_ = boardstate_.to(self.device)
                    vectorstate_ = vectorstate_.to(self.device)
                    ep_r += r #episode reward
                    buffer_a.append(a)
                    buffer_boardstate.append(boardstate)
                    buffer_vectorstate.append(vectorstate)
                    buffer_r.append(r)
                    
                    if total_step % UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                        # sync
                        v_s_, loss, c_loss, a_loss, entropy, l2, valueloss = push_and_pull(self.actor_optimizer, self.critic_optimizer , self.lnet, self.gnet, done, boardstate_, vectorstate_, buffer_boardstate, buffer_vectorstate, buffer_a, buffer_r, GAMMA, self.device, self.global_device, total_step, 0, buffer_logits, buffer_values)

                        self.average_loss.insert(0, loss)
                        if len(self.average_loss) > 200:
                            self.average_loss.pop()

                        v_s_ = v_s_.cpu().mean().item()
                        self.average_v_s_.insert(0, v_s_)
                        if len(self.average_v_s_) > 200:
                            self.average_v_s_.pop()

                        self.average_c_loss.insert(0, c_loss)
                        if len(self.average_c_loss) > 200:
                            self.average_c_loss.pop()
                       
                        self.average_entropy.insert(0, entropy)
                        if len(self.average_entropy) > 200:
                            self.average_entropy.pop()

                        self.average_l2.insert(0, l2)
                        if len(self.average_l2) > 200:
                            self.average_l2.pop()

                        self.average_a_loss.insert(0, a_loss)
                        if len(self.average_a_loss) > 2000:
                            self.average_a_loss.pop()

                        self.average_value_loss.insert(0, valueloss)
                        if len(self.average_value_loss) > 200:
                            self.average_value_loss.pop()

                        if self.env.game.is_finished == 1:  # done and print information
                            print("loss", loss)
                            print("c_loss", c_loss)
                            print("a_loss", a_loss)
                            print("entropy", entropy)
                            self.env.game.is_finished = 0
                            print(self.name, "has achieved total steps of", total_step)
                            print(self.env.player0.victorypoints)
                            print(self.env.player1.victorypoints)
                            print("v_s_", v_s_)
                            print("loss", loss)
                            print("done")
                            print("total reward =", ep_r)
                            logging(self.env, self.logger, ep_r, v_s_, loss, total_step, self.average_loss, self.average_v_s_, self.average_c_loss, self.average_a_loss, self.average_entropy, self.average_l2, self.average_value_loss)
                            record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                            value_dict[f'v_s_{self.name}'] = v_s_
                            step_dict[f'total_step{self.name}'] = total_step
                            self.count = 0
                            total_step = 0
                            break

                    if self.count > 10000:
                        self.env.game.is_finished = 1
                        print("Game did not finish")
                        print(self.name, "has achieved total steps of", total_step)
                        print(self.env.player0.victorypoints)
                        print(self.env.player1.victorypoints)
                        print("v_s_", v_s_)
                        print("loss", loss)
                        print("done")
                        print("total reward =", ep_r)
                        self.count = 0
                        total_step = 0
                        break
                    #if self.g_ep.value % 20 == 0:
                    #    if self.haslrupdated == 1:
                    #        self.opt.param_groups[0]['lr'] = self.opt.param_groups[0]['lr'] * 0.9
                    #        print("updating learning rate")
                    #        print("new_lr", self.opt.param_groups[0]['lr'])
                    #        self.haslrupdated = 0
                    boardstate = boardstate_
                    vectorstate = vectorstate_ 
                    
                    total_step += 1
                else: 
                    if self.g_ep.value < 16000 + self.number*4000:
                        boardstate = state_changer(self.env)[0].to(self.device)
                        vectorstate = state_changer(self.env)[1].to(self.device)
                        a = random_assignment(self.env)
                    else:
                        if self.g_ep.value % 16000 == self.number*4000:
                            print("This is called to early")
                            if self.hasassigned == 0:
                                self.lnet.to(self.device)  # Ensure lnet is on the correct device
                                self.oppnet.load_state_dict(self.lnet.state_dict())
                                self.oppnet.to(self.device) 
                                self.hasassigned = 1
                        if self.g_ep.value % 4000 == 1:
                            self.hasassigned = 0
                        boardstate = state_changer(self.env)[0].to(self.device)
                        vectorstate = state_changer(self.env)[1].to(self.device)
                        a, meanlogits = self.oppnet.choose_action(boardstate, vectorstate, self.env, total_step) #select action
                        select_action(a, self.env)
                    if self.env.game.is_finished == 1:  # done and print information
                        boardstate_,vectorstate_, r, done =  state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.env.game.is_finished
                        print(self.env.phase.reward)
                        self.env.phase.reward = 0
                        buffer_a.append(a)
                        buffer_boardstate.append(boardstate)
                        buffer_vectorstate.append(vectorstate)
                        buffer_r.append(r)
                        ep_r += r
                        v_s_, loss, c_loss, a_loss, entropy, l2, valueloss = push_and_pull(self.actor_optimizer, self.critic_optimizer, self.lnet, self.gnet, done, boardstate_, vectorstate_, buffer_boardstate, buffer_vectorstate, buffer_a, buffer_r, GAMMA, self.device, self.global_device, total_step, 1, buffer_logits, buffer_values)
                        
                        
                        self.average_loss.insert(0, loss)
                        if len(self.average_loss) > 200:
                            self.average_loss.pop()

                        v_s_ = v_s_.cpu().mean().item()
                        self.average_v_s_.insert(0, v_s_)
                        if len(self.average_v_s_) > 200:
                            self.average_v_s_.pop()

                        self.average_c_loss.insert(0, c_loss)
                        if len(self.average_c_loss) > 200:
                            self.average_c_loss.pop()
                       
                        self.average_entropy.insert(0, entropy)
                        if len(self.average_entropy) > 200:
                            self.average_entropy.pop()

                        self.average_l2.insert(0, l2)
                        if len(self.average_l2) > 200:
                            self.average_l2.pop()

                        self.average_a_loss.insert(0, a_loss)
                        if len(self.average_a_loss) > 200:
                            self.average_a_loss.pop()

                        self.average_value_loss.insert(0, valueloss)
                        if len(self.average_value_loss) > 200:
                            self.average_value_loss.pop()



                        print("loss", loss)
                        print("c_loss", c_loss)
                        print("a_loss", a_loss)
                        print("entropy", entropy)
                        print(self.name, "has achieved total steps of", total_step)
                        print("v_s_", v_s_)
                        print("loss", loss)
                        print("done")
                        print("total reward =", ep_r)
                        self.env.game.is_finished = 0
                        logging(self.env, self.logger, ep_r, 0, 0, total_step, self.average_loss, self.average_v_s_, self.average_c_loss, self.average_a_loss, self.average_entropy, self.average_l2, self.average_value_loss)
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        self.count = 0
                        total_step = 0
                        break
                
        self.res_queue.put(None)
    def update_learning_rate(self, new_lr):
        for param_group in self.opt.param_groups:
            param_group['lr'] = new_lr


    




if __name__ == "__main__":
    torch.set_printoptions(precision=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn') #not necessary
    gnet = ActorCritic().to(device) # global network
    
    # Load the weights
    #gnet.load_state_dict(torch.load("A3Cagent180_policy_net_0_1_1.pth"))
    
    gnet.share_memory()         # share the global parameters in multiprocessing
    opt = SharedAdam(gnet.parameters(), lr=1e-4, betas=(0.92, 0.999))      # global optimizer
    scheduler = ExponentialLR(opt, gamma=0.99998)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    logger = Log()
    # parallel training
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, j, logger, 0) for i in range(4) for j in range(2)]
    print("workers", workers)
    [w.start() for w in workers]
    res = []                    # record episode reward to plot
    while True:
        scheduler.step()
        for worker in workers:
            worker.update_learning_rate(opt.param_groups[0]['lr'])
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break
    [w.join() for w in workers]

def logging(env, logger, global_ep_r, v_s_, loss, total_step, average_loss, average_v_s_, average_c_loss, average_a_loss, average_entropy, average_l2, average_value_loss):
    logger.total_episodes += 1
    global_ep = logger.total_episodes

    player0 = env.players[0]
    player1 = env.players[1]
    phase = env.phase
    game = env.game
    random_testing = env.random_testing
    player0_log = env.player0_log
    player1_log = env.player1_log
    

    logger.average_legal_moves_ratio.insert(0, env.total_step/(phase.statechangecount + 1))
    if len(logger.average_legal_moves_ratio) > 5:
        logger.average_legal_moves_ratio.pop()
    logger.average_moves.insert(0, env.total_step)
    if len(logger.average_moves) > 5:
        logger.average_moves.pop()

    run.log({"game.average_moves": sum(logger.average_moves)/5})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(logger.action_counts))), y=logger.action_counts, mode='markers', name='Action Counts'))
    fig.add_trace(go.Scatter(x=list(range(len(logger.random_action_counts))), y=logger.random_action_counts, mode='markers', name='Random Action Counts'))
    
    if player0.wins == 1:
        logger.average_win_ratio.insert(0, 1)
    else:
        logger.average_win_ratio.insert(0, 0)
    if len(logger.average_win_ratio) > 5:
        logger.average_win_ratio.pop()
    logger.player0_totalwins += player0.wins
    player0.wins = 0
    logger.player1_totalwins += player1.wins
    player1.wins = 0

    run.log({"game.average_win_ratio": sum(logger.average_win_ratio)/5}, step=global_ep)
    run.log({"Player 0 Wins": logger.player0_totalwins}, step=global_ep)
    run.log({"Player 1 Wins": logger.player1_totalwins}, step=global_ep)
    run.log({"Episode Duration": logger.episode_durations}, step=global_ep)
    run.log({"Action Counts": wandb.Plotly(fig)}, step=global_ep)
    logger.total_move_finished += random_testing.move_finished


    random_testing.move_finished = 0
    run.log({"random_testing.move_finsihed":logger.total_move_finished}, step=global_ep)
    logger.total_statechangecount += phase.statechangecount
    phase.statechangecount = 0
    run.log({"phase.statechangecount": logger.total_statechangecount}, step=global_ep)
    logger.average_reward.insert(0, phase.reward)
    if len(logger.average_reward) > 5:
        logger.average_reward.pop()
    phase.reward = 0
    run.log({"phase.reward": sum(logger.average_reward)/5}, step=global_ep)
    logger.average_victory_reward.insert(0, phase.victoryreward)
    if len(logger.average_victory_reward) > 5:
        logger.average_victory_reward.pop()
    phase.victoryreward = 0
    run.log({"phase.victoyreward": sum(logger.average_victory_reward)/5}, step=global_ep)
    logger.average_victor_point_reward.insert(0, phase.victorypointreward)
    if len(logger.average_victor_point_reward) > 5:
        logger.average_victor_point_reward.pop()
    phase.victorypointreward = 0
    run.log({"phase.victorypointreward": sum(logger.average_victor_point_reward)/5}, step=global_ep)
    logger.average_illegal_moves_reward.insert(0, phase.illegalmovesreward)
    if len(logger.average_illegal_moves_reward) > 5:
        logger.average_illegal_moves_reward.pop()
    phase.illegalmovesreward = 0
    run.log({"phase.illegalmovesreward": sum(logger.average_illegal_moves_reward)/5}, step=global_ep)
    logger.average_legal_moves_reward.insert(0, phase.legalmovesreward)
    if len(logger.average_legal_moves_reward) > 5:
        logger.average_legal_moves_reward.pop()
    phase.legalmovesreward = 0
    run.log({"phase.legalmovesreward": sum(logger.average_legal_moves_reward)/5}, step=global_ep)
    logger.average_total_reward.insert(0, global_ep_r)
    if len(logger.average_total_reward) > 5:
        logger.average_total_reward.pop()
    global_ep_r = 0
    run.log({"global_ep_r.value": sum(logger.average_total_reward)/5}, step=global_ep)
    


    run.log({"average_v_s_end": sum(logger.average_v_s_end)/5}, step=global_ep)

    run.log({"average_loss_end": sum(logger.average_loss_end)/5}, step=global_ep)

    run.log({"average_v_s_": sum(average_v_s_)/200}, step=global_ep)
    run.log({"average_loss": sum(average_loss)/200}, step=global_ep)
    run.log({"average_c_loss": sum(average_c_loss)/200}, step=global_ep)
    run.log({"average_a_loss": sum(average_a_loss)/200}, step=global_ep)
    run.log({"average_entropy": sum(average_entropy)/200}, step=global_ep)
    run.log({"average_l2": sum(average_l2)/200}, step=global_ep)
    run.log({"average_value_loss": sum(average_value_loss)/200}, step=global_ep)

    run.log({"Function Call Counts": wandb.Plotly(fig)}, step=global_ep)
    
    run.log({"game.average_legal_moves_ratio": sum(logger.average_legal_moves_ratio)/5}, step=global_ep)

    logger.average_time.insert(0, time.time() - logger.time)
    if len(logger.average_time) > 5:
        logger.average_time.pop(5)
    logger.time = time.time()
    run.log({"game.average_time": sum(logger.average_time)/5}, step=global_ep)
    
    run.log({"game.average_q_value_loss": sum(game.average_q_value_loss)/1000}, step=global_ep)
    logger.player0_average_victory_points.insert(0, player0.victorypoints)
    if len(logger.player0_average_victory_points) > 5:
        logger.player0_average_victory_points.pop(5)
    logger.player1_average_victory_points.insert(0, player1.victorypoints)
    if len(logger.player1_average_victory_points) > 5:
        logger.player1_average_victory_points.pop(5)
    logger.player0_average_resources_found.insert(0, player0_log.total_resources_found)
    if len(logger.player0_average_resources_found) > 5:
        logger.player0_average_resources_found.pop(5)
    logger.player1_average_resources_found.insert(0, player1_log.total_resources_found)
    if len(logger.player1_average_resources_found) > 5:
        logger.player1_average_resources_found.pop(5)
    logger.player0_average_development_cards_bought.insert(0, player0_log.total_development_cards_bought)
    if len(logger.player0_average_development_cards_bought) > 5:
        logger.player0_average_development_cards_bought.pop(5)
    logger.player1_average_development_cards_bought.insert(0, player1_log.total_development_cards_bought)
    if len(logger.player1_average_development_cards_bought) > 5:
        logger.player1_average_development_cards_bought.pop(5)
    logger.player0_average_development_cards_used.insert(0, player0_log.total_development_cards_used)
    if len(logger.player0_average_development_cards_used) > 5:
        logger.player0_average_development_cards_used.pop(5)
    logger.player1_average_development_cards_used.insert(0, player1_log.total_development_cards_used)
    if len(logger.player1_average_development_cards_used) > 5:
        logger.player1_average_development_cards_used.pop(5)
    logger.player0_average_settlements_built.insert(0, player0_log.total_settlements_built)
    if len(logger.player0_average_settlements_built) > 5:
        logger.player0_average_settlements_built.pop(5)
    logger.player1_average_settlements_built.insert(0, player1_log.total_settlements_built)
    if len(logger.player1_average_settlements_built) > 5:
        logger.player1_average_settlements_built.pop(5)
    logger.player0_average_cities_built.insert(0, player0_log.total_cities_built)
    if len(logger.player0_average_cities_built) > 5:
        logger.player0_average_cities_built.pop(5)
    logger.player1_average_cities_built.insert(0, player1_log.total_cities_built)
    if len(logger.player1_average_cities_built) > 5:
        logger.player1_average_cities_built.pop(5)
    logger.player0_average_roads_built.insert(0, player0_log.total_roads_built)
    if len(logger.player0_average_roads_built) > 5:
        logger.player0_average_roads_built.pop(5)
    logger.player1_average_roads_built.insert(0, player1_log.total_roads_built)
    if len(logger.player1_average_roads_built) > 5:
        logger.player1_average_roads_built.pop(5)
    logger.player0_average_resources_traded.insert(0, player0_log.total_resources_traded)
    if len(logger.player0_average_resources_traded) > 5:
        logger.player0_average_resources_traded.pop(5)
    logger.player1_average_resources_traded.insert(0, player1_log.total_resources_traded)
    if len(logger.player1_average_resources_traded) > 5:
        logger.player1_average_resources_traded.pop(5)
    logger.player0_average_longest_road.insert(0, player0.roads_connected)
    if len(logger.player0_average_longest_road) > 5:
        logger.player0_average_longest_road.pop(5)
    logger.player1_average_longest_road.insert(0, player1.roads_connected)
    if len(logger.player1_average_longest_road) > 5:
        logger.player1_average_longest_road.pop(5)

    logger.player0_average_knights_played.insert(0, player0_log.total_knights_played)
    if len(logger.player0_average_knights_played) > 5:
        logger.player0_average_knights_played.pop(5)
    logger.player1_average_knights_played.insert(0, player1_log.total_knights_played)
    if len(logger.player1_average_knights_played) > 5:
        logger.player1_average_knights_played.pop(5)


    run.log({"player0_log.average_victory_points": sum(player0_log.average_victory_points)/5}, step=global_ep)
    run.log({"player1_log.average_victory_points": sum(player1_log.average_victory_points)/5}, step=global_ep)
    run.log({"player0_log.average_resources_found": sum(player0_log.average_resources_found)/5}, step=global_ep)
    run.log({"player1_log.average_resources_found": sum(player1_log.average_resources_found)/5}, step=global_ep)
    run.log({"player0_log.average_resources_traded": sum(player0_log.average_resources_traded)/5}, step=global_ep)
    run.log({"player1_log.average_resources_traded": sum(player1_log.average_resources_traded)/5}, step=global_ep)
    run.log({"player0_log.average_development_cards_bought": sum(player0_log.average_development_cards_bought)/5}, step=global_ep)
    run.log({"player1_log.average_development_cards_bought": sum(player1_log.average_development_cards_bought)/5}, step=global_ep)
    run.log({"player0_log.average_development_cards_used": sum(player0_log.average_development_cards_used)/5}, step=global_ep)
    run.log({"player1_log.average_development_cards_used": sum(player1_log.average_development_cards_used)/5}, step=global_ep)
    
    run.log({"player0_log.average_roads_built": sum(player0_log.average_roads_built)/5}, step=global_ep)
    run.log({"player1_log.average_roads_built": sum(player1_log.average_roads_built)/5}, step=global_ep)
    run.log({"player0_log.average_settlements_built": sum(player0_log.average_settlements_built)/5}, step=global_ep)
    run.log({"player1_log.average_settlements_built": sum(player1_log.average_settlements_built)/5}, step=global_ep)
    run.log({"player0_log.average_cities_built": sum(player0_log.average_cities_built)/5}, step=global_ep)
    run.log({"player1_log.average_cities_built": sum(player1_log.average_cities_built)/5}, step=global_ep)
    run.log({"player0_log.average_knights_played": sum(player0_log.average_knights_played)/5}, step=global_ep)
    run.log({"player1_log.average_knights_played": sum(player1_log.average_knights_played)/5}, step=global_ep)
    run.log({"player0_log.average_longest_road": sum(player0_log.average_longest_road)/5}, step=global_ep)

    run.log({"game.average_reward_per_move": sum(game.average_reward_per_move)/1000}, step=global_ep)
    run.log({"game.average_expected_state_action_value": sum(game.average_expected_state_action_value)/1000}, step=global_ep)

    env.total_step = 0
    
    #for i in range (len(action_counts)):
    #    wandb.log({f"Action {i-1}": action_counts[i-1]})
    #
    #for i in range (len(random_action_counts)):
    #    wandb.log({f"Random Action {i-1}": random_action_counts[i-1]})

