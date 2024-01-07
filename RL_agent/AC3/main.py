import math 
import time
import torch
import torch.multiprocessing as mp
import sys
sys.path.append('/home/victor/Maturarbeit/Catan/RL-Catan')

from utils import push_and_pull, record
from SharedAdam import SharedAdam
from log import Log
from config import *

from Neural_Networks.A3C_Medium_Seperated import ActorCritic

from Catan_Env.catan_env import Catan_Env
from Catan_Env.action_selection import action_selecter
from Catan_Env.random_action import random_assignment
from Catan_Env.state_changer import state_changer

#plotting
import wandb 
import plotly.graph_objects as go

# Get the available GPUs
available_gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
print("available_gpus", available_gpus)

# Initialize wandb for logging
run = wandb.init(project="RL-Catan_AC3", name="RL_version_9.9.8", config={}, group='finalrun9.9.8')

# Set the random seed for reproducibility
torch.manual_seed(RANDOM_SEED)


class Worker(mp.Process):
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name, device, logger, global_device):
        """
        Worker class for parallel training of the A3C agent.

        Args:
        - gnet: Global network shared among all workers
        - opt: Global optimizer shared among all workers
        - global_ep: Global episode counter
        - global_ep_r: Global episode reward
        - res_queue: Queue for storing the training results
        - name: Worker name/number
        - device: Device (GPU) to be used for training
        - logger: Logger for tracking training progress
        - global_device: Global device (GPU) for synchronization

        """
        super(Worker, self).__init__()

        # Initialize worker attributes
        self.number = name
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.actor_optimizer = torch.optim.Adam(self.gnet.actor_parameters, lr=LR_START, betas=OPT_BETA, weight_decay=WEIGHT_DECAY) 
        self.critic_optimizer = torch.optim.Adam(self.gnet.critic_parameters, betas=OPT_BETA, lr=LR_START, weight_decay=WEIGHT_DECAY) 

        self.env = Catan_Env()

        self.device = torch.device("cuda:{}".format(device))
        self.global_device = torch.device("cuda:{}".format(global_device))  
        self.lnet = ActorCritic().to(self.device)       
        self.oppnet = ActorCritic().to(self.device)
        self.logger = logger

        # Initialize lists for tracking average values
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

        self.hasassigned = 0
       
        print("self.number", self.number)
        print("self.device", self.device)


    # This section of code defines the run method of a class.
    # The run method is responsible for executing the main training loop of the agent.
    # It initializes various variables and lists for tracking statistics.
    # It also sets the learning rate of the actor and critic optimizers based on the current episode number.
    # The loop continues until the maximum number of episodes is reached.

    def run(self):
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
            print("episode", self.g_ep.value)

            # Save the model every MODEL_SAVE_RATE episodes
            if self.g_ep.value % MODEL_SAVE_RATE == 0:
                torch.save(self.gnet.state_dict(), f'A3Cagent{self.g_ep.value}_policy_net_9_9_8.pth')
            
            self.env.new_game()
            boardstate = state_changer(self.env)[0].to(self.device)
            vectorstate = state_changer(self.env)[1].to(self.device)
            ep_r = 0.

            # Adjust the learning rate of the optimizers based on the current episode number
            if self.actor_optimizer.param_groups[0]['lr'] > LR_UPDATE_TRESHOLD_1:
                self.actor_optimizer.param_groups[0]['lr'] = LR_START * LR_DECAY_RATE_1 ** (self.g_ep.value)
                self.critic_optimizer.param_groups[0]['lr'] = LR_START * LR_DECAY_RATE_1 ** (self.g_ep.value)
            elif self.actor_optimizer.param_groups[0]['lr'] > LR_UPDATE_TRESHOLD_2:
                self.actor_optimizer.param_groups[0]['lr'] = LR_UPDATE_TRESHOLD_1 * LR_DECAY_RATE_2 ** (self.g_ep.value)
                self.critic_optimizer.param_groups[0]['lr'] = LR_UPDATE_TRESHOLD_1 * LR_DECAY_RATE_2 ** (self.g_ep.value)
            else:
                self.actor_optimizer.param_groups[0]['lr'] = LR_UPDATE_TRESHOLD_2 * LR_DECAY_RATE_3 ** (self.g_ep.value)
                self.critic_optimizer.param_groups[0]['lr'] = LR_UPDATE_TRESHOLD_2 * LR_DECAY_RATE_3 ** (self.g_ep.value)

            print("new_actor_lr", self.actor_optimizer.param_groups[0]['lr'])
            print("new_critic_lr", self.critic_optimizer.param_groups[0]['lr'])

            # Clear the buffers for storing previous states, actions, rewards, etc.
            buffer_a.clear()
            buffer_boardstate.clear()
            buffer_vectorstate.clear()
            buffer_r.clear()
            buffer_logits.clear()
            buffer_values.clear()
            
            while True:
                if self.env.game.cur_player == 0:
                    self.env.phase.statechange = 0

                    # Choose an action using the local network
                    a, logits, values = self.lnet.choose_action(boardstate, vectorstate, self.env, total_step) 
                    buffer_logits.append(logits)
                    buffer_values.append(values)
                    select_action(a, self.env)
                    boardstate_,vectorstate_, r, done =  state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.env.game.is_finished
                    self.logger.action_counts[a] += 1 
                    self.env.phase.reward = 0
                    boardstate_ = boardstate_.to(self.device)
                    vectorstate_ = vectorstate_.to(self.device)
                    ep_r += r 

                    #append information to the buffers
                    buffer_a.append(a)
                    buffer_boardstate.append(boardstate)
                    buffer_vectorstate.append(vectorstate)
                    buffer_r.append(r)
                    
                    if total_step % UPDATE_GLOBAL_ITER == 0 or done: 
                        # Update the global network using the buffers
                        v_s_, loss, c_loss, a_loss, entropy, l2, valueloss = push_and_pull(self.actor_optimizer, self.critic_optimizer , self.lnet, self.gnet, done, boardstate_, vectorstate_, buffer_boardstate, buffer_vectorstate, buffer_a, buffer_r, GAMMA, self.device, self.global_device, buffer_logits, buffer_values)

                        #save information for logging
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

                        if self.env.game.is_finished == 1:  
                            print("loss", loss)
                            print("c_loss", c_loss)
                            print("a_loss", a_loss)
                            print("entropy", entropy)
                            print(self.name, "has achieved total steps of", total_step)
                            print(self.env.player0.victorypoints)
                            print(self.env.player1.victorypoints)
                            print("v_s_", v_s_)
                            print("loss", loss)
                            print("done")
                            print("total reward =", ep_r)

                            # Log the statistics and record the episode result
                            logging(self.env, self.logger, total_step, self.average_loss, self.average_v_s_, self.average_c_loss, self.average_a_loss, self.average_entropy, self.average_l2, self.average_value_loss)
                            record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)

                            total_step = 0
                            self.env.game.is_finished = 0
                            break

                    boardstate = boardstate_
                    vectorstate = vectorstate_ 
                    total_step += 1
                else: 
                    if self.g_ep.value < OPPONENT_AGENT_CHANGE_THRESHOLD + self.number * 200:
                        # If the RL agent has not reached the threshold for changing the opponent agent,
                        # choose a random action and update the board state and vector state.
                        boardstate = state_changer(self.env)[0].to(self.device)
                        vectorstate = state_changer(self.env)[1].to(self.device)
                        a = random_assignment(self.env)
                        self.hasassigned = 0
                    else:           
                        if self.hasassigned == 0:
                            # If the RL agent has reached the threshold and has not assigned the opponent network yet,
                            # assign the local network to the opponent network and update the board state and vector state.
                            self.lnet.to(self.device) 
                            self.oppnet.load_state_dict(self.lnet.state_dict())
                            self.oppnet.to(self.device) 
                            self.hasassigned = 1

                        # Choose an action using the opponent network and update the board state and vector state.
                        boardstate = state_changer(self.env)[0].to(self.device)
                        vectorstate = state_changer(self.env)[1].to(self.device)
                        a, logits, values = self.oppnet.choose_action(boardstate, vectorstate, self.env, total_step) 
                        select_action(a, self.env)

                    if self.env.game.is_finished == 1: 
                        boardstate_,vectorstate_, r, done =  state_changer(self.env)[0], state_changer(self.env)[1], self.env.phase.reward, self.env.game.is_finished
                        print(self.env.phase.reward)
                        self.env.phase.reward = 0

                        buffer_values.append(values)
                        buffer_logits.append(logits)
                        buffer_a.append(a)
                        buffer_boardstate.append(boardstate)
                        buffer_vectorstate.append(vectorstate)
                        buffer_r.append(r)
                        ep_r += r

                        # Update the global network using the buffers.
                        v_s_, loss, c_loss, a_loss, entropy, l2, valueloss = push_and_pull(self.actor_optimizer, self.critic_optimizer, self.lnet, self.gnet, done, boardstate_, vectorstate_, buffer_boardstate, buffer_vectorstate, buffer_a, buffer_r, GAMMA, self.device, self.global_device, buffer_logits, buffer_values)

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

                        # Log the statistics
                        logging(self.env, self.logger, ep_r, self.average_loss, self.average_v_s_, self.average_c_loss, self.average_a_loss, self.average_entropy, self.average_l2, self.average_value_loss)
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)

                        self.env.game.is_finished = 0
                        total_step = 0
                        break
                
        self.res_queue.put(None)

if __name__ == "__main__":
    # Set the precision for printing tensors
    torch.set_printoptions(precision=5)

    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp.set_start_method('spawn')

    # Create the global network and move it to the device
    gnet = ActorCritic().to(device) 
    gnet.share_memory()

    # Create the optimizer for the global network
    opt = SharedAdam(gnet.parameters(), lr=LR_START, betas=OPT_BETA, weight_decay=WEIGHT_DECAY)

    # Create shared counters and a queue for storing the results
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()

    # Create a logger for recording training statistics
    logger = Log()

    # Create worker processes
    workers = [Worker(gnet, opt, global_ep, global_ep_r, res_queue, i, j, logger, 0) for i in range(NUM_WORKERS_PER_GPU) for j in range(NUM_GPUS)]
    print("workers", workers)

    # Start the worker processes
    [w.start() for w in workers]

    # Collect the results from the worker processes
    res = []               
    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    # Join the worker processes
    [w.join() for w in workers]

#Converts the action into 1 of 45 possible actions and the coordinates of the board actions
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


def logging(env, logger, global_ep_r, average_loss, average_v_s_, average_c_loss, average_a_loss, average_entropy, average_l2, average_value_loss):
    logger.total_episodes += 1
    global_ep = logger.total_episodes

    # Get information from the environment
    player0 = env.players[0]
    player1 = env.players[1]
    phase = env.phase
    game = env.game
    random_testing = env.random_testing
    player0_log = env.player0_log
    player1_log = env.player1_log

    logger.average_legal_moves_ratio.insert(0, env.total_step / (phase.statechangecount + 1))
    if len(logger.average_legal_moves_ratio) > 5:
        logger.average_legal_moves_ratio.pop()
    run.log({"game.average_legal_moves_ratio": sum(logger.average_legal_moves_ratio) / 5}, step=global_ep)

    logger.average_moves.insert(0, env.total_step)
    if len(logger.average_moves) > 5:
        logger.average_moves.pop()
    run.log({"game.average_moves": sum(logger.average_moves) / 5}, step=global_ep)

    # Create a scatter plot for action counts and random action counts
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(logger.action_counts))), y=logger.action_counts, mode='markers', name='Action Counts'))
    fig.add_trace(go.Scatter(x=list(range(len(logger.random_action_counts))), y=logger.random_action_counts, mode='markers', name='Random Action Counts'))

    if player0.wins == 1:
        logger.average_win_ratio.insert(0, 1)
    else:
        logger.average_win_ratio.insert(0, 0)
    if len(logger.average_win_ratio) > 5:
        logger.average_win_ratio.pop()
    run.log({"game.average_win_ratio": sum(logger.average_win_ratio) / 5}, step=global_ep)

    logger.player0_totalwins += player0.wins
    player0.wins = 0
    logger.player1_totalwins += player1.wins
    player1.wins = 0
    run.log({"Player 0 Wins": logger.player0_totalwins}, step=global_ep)
    run.log({"Player 1 Wins": logger.player1_totalwins}, step=global_ep)

    run.log({"Episode Duration": logger.episode_durations}, step=global_ep)
    run.log({"Action Counts": wandb.Plotly(fig)}, step=global_ep)
    logger.total_move_finished += random_testing.move_finished
    random_testing.move_finished = 0
    run.log({"random_testing.move_finsihed": logger.total_move_finished}, step=global_ep)

    logger.total_statechangecount += phase.statechangecount
    phase.statechangecount = 0
    run.log({"phase.statechangecount": logger.total_statechangecount}, step=global_ep)

    logger.average_victory_reward.insert(0, phase.victoryreward)
    if len(logger.average_victory_reward) > 5:
        logger.average_victory_reward.pop()
    phase.victoryreward = 0
    run.log({"phase.victoyreward": sum(logger.average_victory_reward) / 5}, step=global_ep)

    logger.average_victor_point_reward.insert(0, phase.victorypointreward)
    if len(logger.average_victor_point_reward) > 5:
        logger.average_victor_point_reward.pop()
    phase.victorypointreward = 0
    run.log({"phase.victorypointreward": sum(logger.average_victor_point_reward) / 5}, step=global_ep)

    logger.average_illegal_moves_reward.insert(0, phase.illegalmovesreward)
    if len(logger.average_illegal_moves_reward) > 5:
        logger.average_illegal_moves_reward.pop()
    phase.illegalmovesreward = 0
    run.log({"phase.illegalmovesreward": sum(logger.average_illegal_moves_reward) / 5}, step=global_ep)

    logger.average_legal_moves_reward.insert(0, phase.legalmovesreward)
    if len(logger.average_legal_moves_reward) > 5:
        logger.average_legal_moves_reward.pop()
    phase.legalmovesreward = 0
    run.log({"phase.legalmovesreward": sum(logger.average_legal_moves_reward) / 5}, step=global_ep)

    logger.average_total_reward.insert(0, global_ep_r)
    if len(logger.average_total_reward) > 5:
        logger.average_total_reward.pop()
    global_ep_r = 0
    run.log({"global_ep_r.value": sum(logger.average_total_reward) / 5}, step=global_ep)

    run.log({"average_v_s_": sum(average_v_s_) / 200}, step=global_ep)
    run.log({"average_loss": sum(average_loss) / 200}, step=global_ep)
    run.log({"average_c_loss": sum(average_c_loss) / 200}, step=global_ep)
    run.log({"average_a_loss": sum(average_a_loss) / 200}, step=global_ep)
    run.log({"average_entropy": sum(average_entropy) / 200}, step=global_ep)
    run.log({"average_l2": sum(average_l2) / 200}, step=global_ep)
    run.log({"average_value_loss": sum(average_value_loss) / 200}, step=global_ep)

    run.log({"Function Call Counts": wandb.Plotly(fig)}, step=global_ep)

    logger.average_time.insert(0, time.time() - logger.time)
    if len(logger.average_time) > 5:
        logger.average_time.pop(5)
    logger.time = time.time()
    run.log({"game.average_time": sum(logger.average_time) / 5}, step=global_ep)

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

    run.log({"player0_log.average_victory_points": sum(player0_log.average_victory_points) / 10}, step=global_ep)
    run.log({"player1_log.average_victory_points": sum(player1_log.average_victory_points) / 10}, step=global_ep)
    run.log({"player0_log.average_resources_found": sum(player0_log.average_resources_found) / 10}, step=global_ep)
    run.log({"player1_log.average_resources_found": sum(player1_log.average_resources_found) / 10}, step=global_ep)
    run.log({"player0_log.average_resources_traded": sum(player0_log.average_resources_traded) / 10}, step=global_ep)
    run.log({"player1_log.average_resources_traded": sum(player1_log.average_resources_traded) / 10}, step=global_ep)
    run.log({"player0_log.average_development_cards_bought": sum(player0_log.average_development_cards_bought) / 10}, step=global_ep)
    run.log({"player1_log.average_development_cards_bought": sum(player1_log.average_development_cards_bought) / 10}, step=global_ep)
    run.log({"player0_log.average_development_cards_used": sum(player0_log.average_development_cards_used) / 10}, step=global_ep)
    run.log({"player1_log.average_development_cards_used": sum(player1_log.average_development_cards_used) / 10}, step=global_ep)
    run.log({"player0_log.average_roads_built": sum(player0_log.average_roads_built) / 10}, step=global_ep)
    run.log({"player1_log.average_roads_built": sum(player1_log.average_roads_built) / 10}, step=global_ep)
    run.log({"player0_log.average_settlements_built": sum(player0_log.average_settlements_built) / 10}, step=global_ep)
    run.log({"player1_log.average_settlements_built": sum(player1_log.average_settlements_built) / 10}, step=global_ep)
    run.log({"player0_log.average_cities_built": sum(player0_log.average_cities_built) / 10}, step=global_ep)
    run.log({"player1_log.average_cities_built": sum(player1_log.average_cities_built) / 10}, step=global_ep)
    run.log({"player0_log.average_knights_played": sum(player0_log.average_knights_played) / 10}, step=global_ep)
    run.log({"player1_log.average_knights_played": sum(player1_log.average_knights_played) / 10}, step=global_ep)
    run.log({"player0_log.average_longest_road": sum(player0_log.average_longest_road) / 10}, step=global_ep)

    run.log({"game.average_expected_state_action_value": sum(game.average_expected_state_action_value) / 1000}, step=global_ep)
    
    # Reset total_step
    env.total_step = 0
