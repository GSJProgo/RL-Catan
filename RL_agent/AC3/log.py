import math
import plotly.graph_objects as go

from config import *
import wandb

class Log:
    def __init__(self):
        #The average is taken over the last 10 games 
        self.average_victory_points = []
        self.average_resources_found = []
        #self.average_resources_found_move = 0 | I can calculate this  
        self.final_board_state = 0 #tommorow
        self.AI_function_calls = 0 #same here
        self.successful_AI_function_calls = 0 #same here
        self.average_development_cards_bought = []
        self.average_roads_built = []
        self.average_settlements_built = []
        self.average_cities_built = []
        self.average_knights_played = []
        self.average_development_cards_used = [] #victory point cards are seen as automatically used
        self.average_resources_traded = []
        self.average_longest_road = []

        self.total_resources_found = 0
        self.total_development_cards_bought = 0
        self.total_roads_built = 0
        self.total_settlements_built = 0
        self.total_cities_built = 0
        self.total_development_cards_used = 0
        self.total_resources_traded = 0
        self.total_knights_played = 0

        self.steps_done = 0

        self.action_counts = []
        self.random_action_counts = []
        self.episode_durations = []




log = Log()
log.action_counts = [0] * TOTAL_ACTIONS
log.random_action_counts = [0] * TOTAL_ACTIONS

from Catan_Env.catan_env import player0, player1, phase, random_testing, game, player0_log, player1_log

def logging(num_episode):
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * log.steps_done / EPS_DECAY)
    wandb.log({"eps_threshold": eps_threshold}, step=num_episode)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(log.action_counts))), y=log.action_counts, mode='markers', name='Action Counts'))
    fig.add_trace(go.Scatter(x=list(range(len(log.random_action_counts))), y=log.random_action_counts, mode='markers', name='Random Action Counts'))
    wandb.log({"Player 0 Wins": player0.wins}, step=num_episode)
    wandb.log({"Player 1 Wins": player1.wins}, step=num_episode)
    wandb.log({"Episode Duration": log.episode_durations}, step=num_episode)
    wandb.log({"Action Counts": wandb.Plotly(fig)}, step=num_episode)
    wandb.log({"random_testing.move_finsihed":random_testing.move_finished}, step=num_episode)
    wandb.log({"phase.statechangecount": phase.statechangecount}, step=num_episode)

    wandb.log({"phase.reward": phase.reward}, step=num_episode)
    wandb.log({"phase.victoyreward": phase.victoryreward}, step=num_episode)
    wandb.log({"phase.victorypointreward": phase.victorypointreward}, step=num_episode)
    wandb.log({"phase.illegalmovesreward": phase.illegalmovesreward}, step=num_episode)
    wandb.log({"phase.legalmovesreward": phase.legalmovesreward}, step=num_episode)

    phase.victoryreward = 0
    phase.victoryreward = 0
    phase.illegalmovesreward = 0
    phase.legalmovesreward = 0

    wandb.log({"Function Call Counts": wandb.Plotly(fig)}, step=num_episode)

    wandb.log({"game.average_win_ratio": sum(game.average_win_ratio)/20}, step=num_episode)
    wandb.log({"game.average_legal_moves_ratio": sum(game.average_legal_moves_ratio)/20}, step=num_episode)

    wandb.log({"game.average_time": sum(game.average_time)/10}, step=num_episode)
    wandb.log({"game.average_moves": sum(game.average_moves)/10}, step=num_episode)
    wandb.log({"game.average_q_value_loss": sum(game.average_q_value_loss)/1000}, step=num_episode)

    wandb.log({"player0_log.average_victory_points": sum(player0_log.average_victory_points)/10}, step=num_episode)
    wandb.log({"player1_log.average_victory_points": sum(player1_log.average_victory_points)/10}, step=num_episode)
    wandb.log({"player0_log.average_resources_found": sum(player0_log.average_resources_found)/10}, step=num_episode)
    wandb.log({"player1_log.average_resources_found": sum(player1_log.average_resources_found)/10}, step=num_episode)
    wandb.log({"player0_log.average_resources_traded": sum(player0_log.average_resources_traded)/10}, step=num_episode)
    wandb.log({"player1_log.average_resources_traded": sum(player1_log.average_resources_traded)/10}, step=num_episode)
    wandb.log({"player0_log.average_development_cards_bought": sum(player0_log.average_development_cards_bought)/10}, step=num_episode)
    wandb.log({"player1_log.average_development_cards_bought": sum(player1_log.average_development_cards_bought)/10}, step=num_episode)
    wandb.log({"player0_log.average_development_cards_used": sum(player0_log.average_development_cards_used)/10}, step=num_episode)
    wandb.log({"player1_log.average_development_cards_used": sum(player1_log.average_development_cards_used)/10}, step=num_episode)
    wandb.log({"player0_log.average_roads_built": sum(player0_log.average_roads_built)/10}, step=num_episode)
    wandb.log({"player1_log.average_roads_built": sum(player1_log.average_roads_built)/10}, step=num_episode)
    wandb.log({"player0_log.average_settlements_built": sum(player0_log.average_settlements_built)/10}, step=num_episode)
    wandb.log({"player1_log.average_settlements_built": sum(player1_log.average_settlements_built)/10}, step=num_episode)
    wandb.log({"player0_log.average_cities_built": sum(player0_log.average_cities_built)/10}, step=num_episode)
    wandb.log({"player1_log.average_cities_built": sum(player1_log.average_cities_built)/10}, step=num_episode)
    wandb.log({"player0_log.average_knights_played": sum(player0_log.average_knights_played)/10}, step=num_episode)
    wandb.log({"player1_log.average_knights_played": sum(player1_log.average_knights_played)/10}, step=num_episode)
    wandb.log({"player0_log.average_longest_road": sum(player0_log.average_longest_road)/10}, step=num_episode)

    wandb.log({"game.average_reward_per_move": sum(game.average_reward_per_move)/1000}, step=num_episode)
    wandb.log({"game.average_expected_state_action_value": sum(game.average_expected_state_action_value)/1000}, step=num_episode)
    
    #for i in range (len(action_counts)):
    #    wandb.log({f"Action {i-1}": action_counts[i-1]})
    #
    #for i in range (len(random_action_counts)):
    #    wandb.log({f"Random Action {i-1}": random_action_counts[i-1]})

