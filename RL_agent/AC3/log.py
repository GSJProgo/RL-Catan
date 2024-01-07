import math
import plotly.graph_objects as go

from config import *
import wandb

import time


class Log:
    def __init__(self):
        self.average_victory_points = []
        self.average_resources_found = []
        self.final_board_state = 0 
        self.AI_function_calls = 0 
        self.successful_AI_function_calls = 0 
        self.average_development_cards_bought = []
        self.average_roads_built = []
        self.average_settlements_built = []
        self.average_cities_built = []
        self.average_knights_played = []
        self.average_development_cards_used = [] 
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

        self.action_counts = [0] * TOTAL_ACTIONS
        self.random_action_counts = [0] * TOTAL_ACTIONS
        self.episode_durations = []

        self.player0_totalwins = 0
        self.player1_totalwins = 0

        self.total_move_finished = 0
        self.total_statechangecount = 0

        self.average_reward = []
        self.average_victory_reward =[]
        self.average_victor_point_reward = []
        self.average_illegal_moves_reward = []
        self.average_legal_moves_reward = []

        self.average_total_reward = []

        self.average_win_ratio = []
        self.average_legal_moves_ratio = []

        self.average_time = []

        self.average_moves = []

        self.average_value_loss = []

        self.player0_average_victory_points = []
        self.player1_average_victory_points = []

        self.player0_average_resources_found = []
        self.player1_average_resources_found = []

        self.player0_average_resources_traded = []
        self.player1_average_resources_traded = []

        self.player0_average_development_cards_bought = []
        self.player1_average_development_cards_bought = []

        self.player0_average_development_cards_used = []
        self.player1_average_development_cards_used = []

        self.player0_average_roads_built = []
        self.player1_average_roads_built = []

        self.player0_average_settlements_built = []
        self.player1_average_settlements_built = []

        self.player0_average_cities_built = []
        self.player1_average_cities_built = []

        self.player0_average_knights_played = []
        self.player1_average_knights_played = []

        self.player0_average_longest_road = []
        self.player1_average_longest_road = []

        self.average_reward_per_move = []
        self.average_expected_state_action_value = []

        self.average_v_s_end = []
        self.average_loss_end = []
        self.average_loss = []
        self.average_v_s_ = []
        

        self.time = time.time()

        self.total_episodes = 0




