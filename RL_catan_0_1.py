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

#plotting
import wandb 
import plotly.graph_objects as go
wandb.init(project="RL-Catan", name="RL_version_0.0.2", config={})
import os

_NUM_ROWS = 11
_NUM_COLS = 21

class Board: 
    def __init__(self):

        #_______________________input_________________________
        self.tiles_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_ore = np.zeros((_NUM_ROWS, _NUM_COLS))

        self.tiles_probability_1 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_2 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_3 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_4 = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_probability_5 = np.zeros((_NUM_ROWS, _NUM_COLS))

        self.rober_position = np.zeros((_NUM_ROWS, _NUM_COLS))

        self.harbor_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_three_one = np.zeros((_NUM_ROWS, _NUM_COLS))


        #_________________ game specific ________________
        #board
        self.ZEROBOARD = np.zeros((_NUM_ROWS, _NUM_COLS))
        #tiles
        self.TILES_POSSIBLE = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_dice = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_dice_probabilities = np.zeros((_NUM_ROWS, _NUM_COLS))

        #settlements
        self.settlements_free = np.ones((_NUM_ROWS, _NUM_COLS))
        self.settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.settlements_used = np.zeros((_NUM_ROWS, _NUM_COLS))

        #roads
        self.roads_available = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.roads_free = np.zeros((_NUM_ROWS, _NUM_COLS))

        #harbors
        self.harbors_possible = np.zeros((9, 2, 2))

        #longest road
        self.longest_road = np.zeros((_NUM_ROWS,_NUM_COLS))
        self.increasing_roads = np.zeros((_NUM_ROWS,_NUM_COLS))

    
class Distribution: 
    def __init__(self):
        self.tile_numbers = [0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,5,5,5]
        self.tile_random_numbers = np.random.choice(self.tile_numbers,19,replace=False)

        self.harbor_numbers = [1,2,3,4,5,6,6,6,6]
        self.harbor_random_numbers = np.random.choice(self.harbor_numbers,9,replace=False)

        self.plate_numbers = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]
        self.plate_random_numbers = np.random.choice(self.plate_numbers, 18, replace=False)

        self.development_card_numbers = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,4,4,5,5]
        self.development_card_random_number = np.random.choice(self.development_card_numbers,25,replace=False)
        self.development_cards_bought = 0

class Player:
    def __init__(self):
        #________________________input board__________________________
        self.settlements = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.roads = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.cities = np.zeros((_NUM_ROWS, _NUM_COLS))       
        #_______________________input vector__________________________
        self.resource_lumber = 0
        self.resource_wool = 0
        self.resource_grain = 0
        self.resource_brick = 0
        self.resource_ore = 0

        self.settlements_left = 5
        self.roads_left = 15
        self.cities_left = 4

        self.army_size = 0

        self.knight_cards_old = 0
        self.victorypoints_cards_old = 0
        self.yearofplenty_cards_old = 0
        self.monopoly_cards_old = 0
        self.roadbuilding_cards_old = 0

        self.knight_cards_new = 0
        self.victorypoints_cards_new = 0
        self.yearofplenty_cards_new = 0
        self.monopoly_cards_new = 0
        self.roadbuilding_cards_new = 0

        self.harbor_lumber = 0
        self.harbor_wool = 0
        self.harbor_grain = 0
        self.harbor_brick = 0
        self.harbor_ore = 0
        self.harbor_three_one = 0

        self.largest_army = 0

        self.roads_connected = 0
        self.longest_road = 0

        self.knight_cards_played = 0

        self.victorypoints_before = 0
        self.victorypoints = 0

        self.development_card_played = 0
        self.knight_move_pending = 0
        self.monopoly_move_pending = 0
        self.roadbuilding_move_pending = 0
        self.roadbuilding1 = 0
        self.yearofplenty_move_pending = 0

        self.yearofplenty_started = 0
        self.yearofplenty1 = 0
        self.yearofplenty2 = 0

        self.discard_resources_started = 0
        self.discard_resources_turn = 0
        self.discard_first_time = 0
        self.total_resources = 0

        self.discard_resources_lumber = 0
        self.discard_resources_wool = 0
        self.discard_resources_grain = 0
        self.discard_resources_brick = 0
        self.discard_resources_ore = 0

        #__________________game-specific resource_____________
        #roads
        self.roads_possible = np.zeros((_NUM_ROWS, _NUM_COLS))

        #rewards 
        self.rewards_possible = np.zeros((_NUM_ROWS,_NUM_COLS))

        self.roadbuilding_d = 0
        self.roadbuilding_e = 0

        self.wins = 0



    
    class Action: 
        def __init__(self):
            #________________________Output board_______________
            self.rober_move = np.zeros((_NUM_ROWS,_NUM_COLS))

            self.road_place = np.zeros((_NUM_ROWS,_NUM_COLS))
            self.settlement_place = np.zeros((_NUM_ROWS,_NUM_COLS))
            self.city_place = np.zeros((_NUM_ROWS,_NUM_COLS))

            #______________________Vector_____________________
            self.end_turn = 0

            #how many development cards the agent wants to buy 
            self.development_card_buy = 0

            #Play a development card
            self.knight_cards_activate = 0 
            self.road_building_cards_activate = 0
            self.monopoly_cards_activate = 0
            self.yearofplenty_cards_activate = 0


            #Which resources do you want to take (Chooses twice)
            self.yearofplenty_lumber = 0
            self.yearofplenty_wool = 0
            self.yearofplenty_grain = 0
            self.yearofplenty_brick = 0
            self.yearofplenty_ore = 0

            #Which resource do you want to take when playing monopoly
            self.monopoly_lumber = 0
            self.monopoly_wool = 0
            self.monopoly_grain = 0
            self.monopoly_brick = 0
            self.monopoly_ore = 0
            
        class Keepresources:
            def __init__(self):
                self.keep_lumber = 0
                self.keep_wool = 0
                self.keep_grain = 0
                self.keep_brick = 0
                self.keep_ore = 0

        class Trading: 
            def __init__(self):
                self.give_lumber_get_wool = 0
                self.give_lumber_get_grain = 0
                self.give_lumber_get_brick = 0
                self.give_lumber_get_ore  = 0
                self.give_wool_get_lumber = 0
                self.give_wool_get_grain = 0
                self.give_wool_get_brick = 0
                self.give_wool_get_ore = 0
                self.give_grain_get_lumber = 0
                self.give_grain_get_wool = 0
                self.give_grain_get_brick = 0
                self.give_grain_get_ore = 0
                self.give_brick_get_lumber = 0
                self.give_brick_get_wool = 0
                self.give_brick_get_grain = 0
                self.give_brick_get_ore = 0
                self.give_ore_get_lumber = 0
                self.give_ore_get_wool = 0
                self.give_ore_get_grain = 0
                self.give_ore_get_brick = 0

class Random: 
    def __init__(self):
        self.random_action = 0
        self.random_position_x = 0
        self.random_position_y = 0

class Random_Testing: 
    def __init__(self):
        self.numberofturns = 0
        self.numberofgames = 0

        self.development_card_choose = 0
        self.settlement_place = 0
        self.settlement_place_placement = 0
        self.settlement_possible_check = 0
        self.buy_settlement = 0
        self.development_card_choose = 0
        self.buy_development_cards = 0
        self.buy_city = 0
        self.city_place = 0
        self.road_place = 0
        self.road_place_placement = 0
        self.road_possible_check = 0
        self.buy_road = 0
        self.activate_road_building_func = 0
        self.activate_monopoly_func = 0
        self.activate_yearofplenty_func = 0
        self.tile_update_rewards = 0
        self.update_longest_road = 0
        self.find_longest_road = 0
        self.check_longest_road = 0
        self.find_largest_army = 0
        self.trav = 0
        self.move_finished = 0
        self.randomly_pick_resources = 0
        self.discard_resources = 0
        self.trade_resources = 0
        self.move_rober = 0
        self.play_knight = 0
        self.steal_card = 0
        self.roll_dice = 0

        self.successful_development_card_choose = 0
        self.successful_settlement_place = 0
        self.successful_settlement_place_placement = 0
        self.successful_settlement_possible_check = 0
        self.successful_buy_settlement = 0
        self.successful_development_card_choose = 0
        self.successful_buy_development_cards = 0
        self.successful_buy_city = 0
        self.successful_city_place = 0
        self.successful_road_place = 0
        self.successful_road_place_placement = 0
        self.successful_road_possible_check = 0
        self.successful_buy_road = 0
        self.successful_activate_road_building_func = 0
        self.successful_activate_monopoly_func = 0
        self.successful_activate_yearofplenty_func = 0
        self.successful_tile_update_rewards = 0
        self.successful_update_longest_road = 0
        self.successful_find_longest_road = 0
        self.successful_check_longest_road = 0
        self.successful_find_largest_army = 0
        self.successful_trav = 0
        self.successful_move_finished = 0
        self.successful_randomly_pick_resources = 0
        self.successful_discard_resources = 0
        self.successful_trade_resources = 0
        self.successful_move_rober = 0
        self.successful_play_knight = 0
        self.successful_steal_card = 0
        self.successful_roll_dice = 0

        self.resource_lumber_total = 0
        self.resource_wool_total = 0
        self.resource_grain_total = 0
        self.resource_brick_total = 0
        self.resource_ore_total = 0

        self.howmuchisthisaccsessed = 0


        self.resources_buy_road = 0
        self.resources_buy_settlement = 0
        self.resources_buy_city = 0
        self.resources_buy_dc = 0


        
        

#config Variables
class Game: 
    def __init__(self):
        self.cur_player = 0

        self.cur_agent = 0

        self.is_finished = 0
        self.settlementplaced = 0
        self.placement_phase_pending = 0
        self.placement_phase_turns_made = 0

        self.placement_phase_settlement_turn = 0
        self.placement_phase_road_turn = 0

        self.seven_rolled = 0

        self.placement_phase_settlement_coordinate1 = 0
        self.placement_phase_settlement_coordinate2 = 0


class Phase():
    def __init__(self):
        self.development_card_played = 0
        self.reward = 0
        self.statechange = 0
        self.statechangecount = 0


random_testing = Random_Testing()
board = Board()
phase = Phase()
distribution = Distribution()
player0 = Player()
player1 = Player()
players = [player0,player1]
player0_action = player0.Action()
player1_action = player1.Action()
player_action = [player0_action, player1_action]
game = Game()
player0_keepresources = player0.Action.Keepresources()
player1_keepresources = player1.Action.Keepresources()
player_keepresources = [player0_keepresources, player1_keepresources]
player0_trading = player0.Action.Trading()
player1_trading = player1.Action.Trading()
player_trading = [player0_trading, player1_trading]

random_agent = Random()

call_counts = {}

def count_calls(func):
    def wrapper(*args, **kwargs):
        call_counts[func.__name__] = call_counts.get(func.__name__, 0) + 1
        return func(*args, **kwargs)
    return wrapper


@count_calls
def harbors_building():
    # Define harbor locations

    board.harbors_possible[0] = [[0,4],[0,6]]
    board.harbors_possible[1] = [[0,10],[0,12]]
    board.harbors_possible[2] = [[2,16],[2,18]]
    board.harbors_possible[3] = [[2,2],[4,2]]
    board.harbors_possible[4] = [[6,2],[8,2]]
    board.harbors_possible[5] = [[10,4],[10,6]]
    board.harbors_possible[6] = [[10,10],[10,12]]
    board.harbors_possible[7] = [[8,16],[8,18]]
    board.harbors_possible[8] = [[4,20],[6,20]]

@count_calls  
def tiles_buidling():
    for i in range(1,10,2):
        for j in range(2 + abs(5-i),20 - abs(5-i),4):
            board.TILES_POSSIBLE[i][j] = 1

@count_calls
def settlements_building():
    for i in range(0,11,2):
        for j in range(-1 + abs(5-i),23 - abs(5-i),2):
            board.settlements_available[i][j] = 1  

@count_calls
def roads_building():
    for i in range(0,10,1):
        for j in range(0,20,1):
            if board.settlements_available[i + 1][j] == 1 and board.settlements_available[i - 1][j] == 1:
                board.roads_available[i][j] = 1
            elif board.settlements_available[i + 1][j + 1] == 1 and board.settlements_available[i - 1][j + 1] == 1:
                board.roads_available[i][j+1] = 1
            elif board.settlements_available[i][j + 1] == 1 and board.settlements_available[i][j - 1] == 1:
                board.roads_available[i][j] = 1
            elif board.settlements_available[i + 1][j + 1] == 1 and board.settlements_available[i + 1][j - 1] == 1:
                board.roads_available[i+1][j] = 1

    board.roads_available = board.roads_available*(1-board.TILES_POSSIBLE)

@count_calls          
def tile_distribution():
    a = 0
    for i in range (1,11,1):
        for j in range(1,21,1):
            if board.TILES_POSSIBLE[i][j] == 1:
                if distribution.tile_random_numbers[a-1] == 0:
                    board.rober_position[i][j] = 1
                elif distribution.tile_random_numbers[a-1] == 1:
                    board.tiles_lumber[i][j] = 1
                elif distribution.tile_random_numbers[a-1] == 2:
                    board.tiles_wool[i][j] = 1
                elif distribution.tile_random_numbers[a-1] == 3:
                    board.tiles_grain[i][j] = 1
                elif distribution.tile_random_numbers[a-1] == 4:
                    board.tiles_brick[i][j] = 1
                elif distribution.tile_random_numbers[a-1] == 5:
                    board.tiles_ore[i][j] = 1
                a += 1 

@count_calls          
def harbor_distribution():
    for i in range(0,9,1):
        x1 = int(board.harbors_possible[i][0][0])
        y1 = int(board.harbors_possible[i][0][1])
        x2 = int(board.harbors_possible[i][1][0])
        y2 = int(board.harbors_possible[i][1][1])

        if distribution.harbor_random_numbers[i] == 1:
            board.harbor_lumber[x1][y1] = 1
            board.harbor_lumber[x2][y2] = 1
        elif distribution.harbor_random_numbers[i] == 2:
            board.harbor_wool[x1][y1] = 1
            board.harbor_wool[x2][y2] = 1
        elif distribution.harbor_random_numbers[i] == 3:
            board.harbor_grain[x1][y1] = 1
            board.harbor_grain[x2][y2] = 1
        elif distribution.harbor_random_numbers[i] == 4:
            board.harbor_brick[x1][y1] = 1
            board.harbor_brick[x2][y2] = 1
        elif distribution.harbor_random_numbers[i] == 5:
            board.harbor_ore[x1][y1] = 1
            board.harbor_ore[x2][y2] = 1
        elif distribution.harbor_random_numbers[i] == 6:
            board.harbor_three_one[x1][y1] = 1
            board.harbor_three_one[x2][y2] = 1

@count_calls        
def plate_distribution():
    a = 0
    for i in range (1,11,1):
        for j in range (1,21,1):
            if board.TILES_POSSIBLE[i][j] == 1 and board.rober_position[i][j] == 0:
                board.tiles_dice[i][j] = distribution.plate_random_numbers[a-1]
                board.tiles_dice_probabilities[i][j] = 6-abs(7-board.tiles_dice[i][j])

                if board.tiles_dice_probabilities[i][j] == 1:
                    board.tiles_probability_1[i][j] = 1
                elif board.tiles_dice_probabilities[i][j] == 2:
                    board.tiles_probability_2[i][j] = 1
                elif board.tiles_dice_probabilities[i][j] == 3:
                    board.tiles_probability_3[i][j] = 1
                elif board.tiles_dice_probabilities[i][j] == 4:
                    board.tiles_probability_4[i][j] = 1
                elif board.tiles_dice_probabilities[i][j] == 5:
                    board.tiles_probability_5[i][j] = 1
                a += 1

@count_calls
def development_card_choose():
    random_testing.development_card_choose += 1
    player = players[game.cur_player]
    if distribution.development_card_random_number[distribution.development_cards_bought] == 1:
        player.knight_cards_new += 1 
    elif distribution.development_card_random_number[distribution.development_cards_bought] == 2:
        player.victorypoints_cards_new += 1 
        player.victorypoints += 1
    elif distribution.development_card_random_number[distribution.development_cards_bought] == 3:
        player.yearofplenty_cards_new += 1 
    elif distribution.development_card_random_number[distribution.development_cards_bought] == 4:
        player.monopoly_cards_new += 1 
    elif distribution.development_card_random_number[distribution.development_cards_bought] == 5:
        player.roadbuilding_cards_new += 1 
        
    distribution.development_cards_bought += 1

    phase.development_card_played = 1

    return 1

@count_calls
def tile_update_rewards(a,b):
    random_testing.tile_update_rewards += 1
    player = players[game.cur_player]
    adjacent_offsets = [
        (-1, 0), (1, 0),
        (1, 2), (1, -2),
        (-1, -2), (-1, 2),
    ]
    for da, db in adjacent_offsets:
        if da < 0 and a == 0:
            continue
        if da > 0 and a == 10:
            continue
        if db < 0 and b <= 1:
            continue
        if db > 0 and b >= 19: 
            continue

        x = da + a
        y = db + b
        player.rewards_possible[x][y] += 1
    

@count_calls
def settlement_place(a,b):
    random_testing.settlement_place += 1
    player = players[game.cur_player]
    board.settlements_used = (1-player0.settlements)*(1-player1.settlements)
    board.settlements_free = board.settlements_available * board.settlements_used
    if player.settlements_left > 0:
        if board.settlements_free[a][b] == 1 and settlement_possible_check(a,b,0) == 1:
            player.settlements[a][b] = 1
            player.settlements_left -= 1
            tile_update_rewards(a,b)
            player.victorypoints += 1
            random_testing.successful_settlement_place += 1
            phase.statechange = 1
            return 1 
        return 0

@count_calls
def settlement_place_placement(a,b):
    random_testing.settlement_place_placement += 1
    player = players[game.cur_player]
    board.settlements_used = (1-player0.settlements)*(1-player1.settlements)
    board.settlements_free = board.settlements_available * board.settlements_used
    if board.settlements_free[a][b] == 1 and settlement_possible_check(a,b,1) == 1:
        player.settlements[a][b] = 1
        tile_update_rewards(a,b)
        player.victorypoints += 1
        phase.statechange = 1
        return 1 
    return 0

@count_calls
def settlement_possible_check(a,b,c):
    random_testing.settlement_possible_check += 1

    #Only update for a and b instead of doing it with i for everything
    tworoads = 0
    player = players[game.cur_player]    

    neighboring_settlements = [
        (0, 2), (0, -2),
        (2, 0), (-2, 0),
    ]
    for da, db in neighboring_settlements:
        if (a == 0 or a == 1) and da < 0:
            da = 0
        elif (a == 10 or a == 9) and da > 0:
            da = 0
        elif (b == 0 or b == 1) and db < 0: 
            db = 0
        elif (b == 20 or b == 19) and db > 0:
            db = 0
        x = da + a
        y = db + b
        if player0.settlements[x][y] == 1 or player1.settlements[x][y] == 1:
            return 0
    
    if c != 1: 
        if b!= 20 and player.roads[a][b + 1] == 1:
            if b != 18 and b != 19 and player.roads[a][b+3] == 1:
                tworoads = 1
            elif a != 10 and b != 19 and player.roads[a+1][b+2] == 1:
                tworoads = 1
            elif a != 0 and b != 19 and player.roads[a-1][b+2] == 1:
                tworoads = 1
        if b != 0 and player.roads[a][b - 1] == 1:
            if b != 2 and b != 1 and player.roads[a][b - 3] == 1:
                tworoads = 1
            elif a != 10 and b != 1 and player.roads[a + 1][b - 2] == 1:
                tworoads = 1
            elif a != 0 and b != 1 and player.roads[a - 1][b - 2] == 1:
                tworoads = 1
        if a != 0 and player.roads[a - 1][b] == 1:
            if a != 2 and a != 1 and player.roads[a - 3][b] == 1:
                tworoads = 1
            elif b != 20 and a != 1 and player.roads[a - 2][b + 1] == 1:
                tworoads = 1
            elif b != 0 and a != 1 and player.roads[a - 2][b - 1] == 1:
                tworoads = 1
        if a != 10 and player.roads[a + 1][b] == 1:
            if a != 8 and a != 9 and player.roads[a + 3][b] == 1:
                tworoads = 1
            elif b != 20 and a != 9 and player.roads[a + 2][b + 1] == 1:
                tworoads = 1
            elif b != 0 and a != 9 and player.roads[a + 2][b - 1] == 1:
                tworoads = 1

        if tworoads == 1: 
            return 1
        else: 
            return 0
    return 1

@count_calls                     
def road_place(a,b):

    random_testing.road_place += 1
    player = players[game.cur_player]
    possible = 0
    possible = road_possible_check(a,b)
    if player.roads_left > 0:
        if possible == 1:
            player.roads[a][b] = 1
            player.roads_left -= 1
            update_longest_road()
            random_testing.successful_road_place += 1
            phase.statechange = 1
            return 1 
    return 0

@count_calls
def road_place_card(a,b,c,d):
    player = players[game.cur_player]
    if a == c and b == d:
        return 0 
    possible = 0
    possible = road_possible_check(a,b)
    possible2 = 0
    possible2 = road_possible_check(c,d)
    if player.roads_left > 1:
        if possible == 1 and possible2 == 1:
            player.roads[a][b] = 1
            player.roads[a][b] = 1
            player.roads_left -= 2
            update_longest_road()
            player.roadbuilding_cards_old -= 1
            random_testing.successful_road_place += 1
            phase.statechange = 1
            return 1 
    return 0

@count_calls
def road_place_placement(settlement_a,settlement_b,road_a,road_b):
    random_testing.road_place_placement += 1
    player = players[game.cur_player]
    if ((((road_a + 1) == settlement_a or (road_a - 1)  == settlement_a) and road_b == settlement_b) or (((road_b + 1) == settlement_b or (road_b - 1)  == settlement_b) and road_a == settlement_a)):
        player.roads[road_a][road_b] = 1
        player.roads_left -= 1
        update_longest_road()
        phase.statechange = 1
        return 1 
    return 0

@count_calls
def road_possible_check(a,b):
    random_testing.road_possible_check += 1

    board.roads_free = board.roads_available * (1 - player0.roads) * (1 - player1.roads)
    player = players[game.cur_player]
    opponent = players[1-game.cur_player]    
    #I could work with boards and multiply them at the end to check 
    player.roads_possible = (1-board.ZEROBOARD)
    
    if board.roads_free[a][b] == 0:
        return 0 
    if b != 1 and b != 0:
        if player.roads[a][b-2] == 1:
            if opponent.settlements[a][b-1] == 0:
                return 1
    if b != 20 and b != 19:
        if player.roads[a][b+2] == 1:
            if opponent.settlements[a][b + 1] == 0:
                return 1
    if b != 20 and a != 20:
        if player.roads[a+1][b+1] == 1:
            if opponent.settlements[a+1][b] == 0 and opponent.settlements[a][b+1] == 0:
                return 1 
    if b != 0 and a != 20:
        if player.roads[a+1][b-1] == 1:
            if opponent.settlements[a+1][b] == 0 and opponent.settlements[a][b-1] == 0:
                return 1 
    if b != 0 and a != 0:
        if player.roads[a-1][b-1] == 1:
            if opponent.settlements[a+1][b] == 0 and opponent.settlements[a][b-1] == 0:
                return 1 
    if b != 20 and a != 0:
        if player.roads[a-1][b+1] == 1:
            if opponent.settlements[a-1][b] == 0 and opponent.settlements[a][b+1] == 0:
                return 1 
    return 0 

@count_calls
def city_place(a,b):
    random_testing.city_place += 1
    #still need to add a max cities check, the same comes to settlements
    player = players[game.cur_player]
    if player.cities_left > 0:    
        if player.settlements[a][b] == 1:
            player.cities[a][b] = 1
            player.cities_left -= 1
            player.settlements[a][b] = 0
            player.settlements_left += 1
            tile_update_rewards(a,b)
            player.victorypoints += 1
            random_testing.successful_city_place += 1
            phase.statechange = 1
            return 1
        return 0 

@count_calls
def roll_dice(): 
    random_testing.roll_dice += 1
    roll = np.random.choice(np.arange(2, 13), p=[1/36,2/36,3/36,4/36,5/36,6/36,5/36,4/36,3/36,2/36,1/36])

    for i in range (0,11,1):
        for j in range(0,21,1):
            if board.tiles_dice[i][j] == roll and board.rober_position[i][j] == 0:
                #
                if player0.rewards_possible[i][j] != 0:
                    if board.tiles_lumber[i][j] == 1:
                        player0.resource_lumber += player0.rewards_possible[i][j]
                    elif board.tiles_wool[i][j] == 1:
                        player0.resource_wool += player0.rewards_possible[i][j]
                    elif board.tiles_grain[i][j] == 1:
                        player0.resource_grain += player0.rewards_possible[i][j]
                    elif board.tiles_brick[i][j] == 1:
                        player0.resource_brick += player0.rewards_possible[i][j]
                    elif board.tiles_ore[i][j] == 1:
                        player0.resource_ore += player0.rewards_possible[i][j]

                if player1.rewards_possible[i][j] != 0:
                    if board.tiles_lumber[i][j] == 1:
                        player1.resource_lumber += player1.rewards_possible[i][j]
                    elif board.tiles_wool[i][j] == 1:
                        player1.resource_wool += player1.rewards_possible[i][j]
                    elif board.tiles_grain[i][j] == 1:
                        player1.resource_grain += player1.rewards_possible[i][j]
                    elif board.tiles_brick[i][j] == 1:
                        player1.resource_brick += player1.rewards_possible[i][j]
                    elif board.tiles_ore[i][j] == 1:
                        player1.resource_ore += player1.rewards_possible[i][j]
    return roll

@count_calls
def buy_development_cards():
    random_testing.buy_development_cards += 1
    player = players[game.cur_player]
    possible = 0
    if player.resource_wool > 0 and player.resource_grain > 0 and player.resource_ore > 0 and distribution.development_cards_bought < 25:
        possible = development_card_choose()
        if possible == 1:
            find_largest_army()
            player.resource_wool -= 1
            player.resource_grain -= 1 
            player.resource_ore -= 1 
            phase.statechange = 1
            return 1
    return 0 
        

@count_calls
def buy_road(a,b):
    random_testing.buy_road += 1
    possible = 0
    player = players[game.cur_player]
    if player.resource_brick > 0 and player.resource_lumber > 0:
            possible = road_place(a,b)
            if possible == 1:
                player.resource_brick -= 1
                player.resource_lumber -= 1
                phase.statechange = 1
                return 1
    return 0 

@count_calls
def buy_settlement(a,b):
    random_testing.buy_settlement += 1
    player = players[game.cur_player]
    possible = 0

    if player.resource_brick > 0 and player.resource_lumber > 0 and player.resource_grain > 0 and player.resource_wool > 0:
        possible = settlement_place(a,b)
        if possible == 1:
            player.resource_lumber -= 1
            player.resource_brick -= 1
            player.resource_wool -= 1 
            player.resource_grain -= 1
            phase.statechange = 1
            return 1 
    return 0 

@count_calls          
def buy_city(a,b):
    random_testing.buy_city += 1
    player = players[game.cur_player]
    possible = 0
    if player.resource_grain > 1 and player.resource_ore > 2:
        possible = city_place(a,b)
        if possible == 1:
            player.resource_grain -= 2
            player.resource_ore -= 3  
            phase.statechange = 1
            return 1
    return 0 

@count_calls
def steal_card():
    random_testing.steal_card += 1
    player = players[game.cur_player]
    opponent = players[1-game.cur_player]
    
    opponent_resources_total = opponent.resource_lumber + opponent.resource_brick + opponent.resource_wool + opponent.resource_grain + opponent.resource_ore
    if opponent_resources_total != 0:
        random_resource = np.random.choice(np.arange(1, 6), p=[opponent.resource_lumber/opponent_resources_total, opponent.resource_brick/opponent_resources_total, opponent.resource_wool/opponent_resources_total, opponent.resource_grain/opponent_resources_total, opponent.resource_ore/opponent_resources_total])
        if random_resource == 1:
            opponent.resource_lumber = opponent.resource_lumber - 1
            player.resource_lumber = player.resource_lumber + 1
        elif random_resource == 2:
            opponent.resource_brick = opponent.resource_brick - 1
            player.resource_brick = player.resource_brick + 1
        elif random_resource == 3:
            opponent.resource_wool = opponent.resource_wool - 1
            player.resource_wool = player.resource_wool + 1
        elif random_resource == 4:
            opponent.resource_grain = opponent.resource_grain - 1
            player.resource_grain = player.resource_grain + 1
        elif random_resource == 5:
            opponent.resource_ore = opponent.resource_ore - 1
            player.resource_ore = player.resource_ore + 1
        random_testing.steal_card += 1

@count_calls
def play_knight(a,b):
    random_testing.play_knight += 1
    player = players[game.cur_player]
    possible = 0
    if player.knight_cards_old > 0: #this is wrong, need to update that
        possible = move_rober(a,b)
        if possible == 1:
            steal_card()
            player.knight_cards_old -= 1
            player.knight_cards_played += 1
            phase.statechange = 1
            return 1
    return 0

@count_calls
def move_rober(a,b):
    random_testing.move_rober += 1
    if board.rober_position[a][b] != 1 and board.TILES_POSSIBLE[a][b] == 1:
        board.rober_position = board.rober_position * board.ZEROBOARD
        board.rober_position[a][b] = 1
        random_testing.successful_move_rober += 1
        phase.statechange = 1
        return 1
    return 0

@count_calls
def activate_yearofplenty_func(resource1,resource2):
    random_testing.activate_yearofplenty_func += 1
    #need to take a look at this later. I'm not sure how to comvert those resources. 
    player = players[game.cur_player]
    if player.yearofplenty_cards_old > 0:
        player.yearofplenty_cards_old = player.yearofplenty_cards_old - 1 
        if resource1 == 1:
            player.resource_lumber += 1
        if resource1 == 1:
            player.resource_lumber = player.resource_lumber + 1
        elif resource1 == 2:
            player.resource_brick = player.resource_brick + 1
        elif resource1 == 3:
            player.resource_wool = player.resource_wool + 1
        elif resource1 == 4:
            player.resource_grain = player.resource_grain + 1
        elif resource1 == 5:
            player.resource_ore = player.resource_ore + 1
        if resource2 == 1:
            player.resource_lumber = player.resource_lumber + 1
        elif resource2 == 2:
            player.resource_brick = player.resource_brick + 1
        elif resource2 == 3:
            player.resource_wool = player.resource_wool + 1
        elif resource2 == 4:
            player.resource_grain = player.resource_grain + 1
        elif resource2 == 5:
            player.resource_ore = player.resource_ore + 1
        random_testing.successful_activate_yearofplenty_func += 1
        phase.statechange = 1
        return 1 
    return 0 

@count_calls
def activate_monopoly_func(resource):
    random_testing.activate_monopoly_func += 1
    player = players[game.cur_player]
    opponent = players[1-game.cur_player]
    if player.monopoly_cards_old > 0:
        player.monopoly_cards_old = player.monopoly_cards_old - 1
        if resource == 1:
            player.resource_lumber = player.resource_lumber + opponent.resource_lumber
            opponent.resource_lumber = 0
        elif resource == 2:
            player.resource_wool = player.resource_wool + opponent.resource_wool
            opponent.resource_wool = 0
        elif resource == 3:
            player.resource_grain = player.resource_grain + opponent.resource_grain
            opponent.resource_grain = 0
        elif resource == 4:
            player.resource_brick = player.resource_brick + opponent.resource_brick
            opponent.resource_brick = 0
        elif resource == 5:
            player.resource_ore = player.resource_ore + opponent.resource_ore
            opponent.resource_ore = 0
        random_testing.successful_activate_monopoly_func += 1
        phase.statechange = 1
        return 1
    return 0

@count_calls  
def activate_road_building_func(a1,b1,a2,b2):
    random_testing.activate_road_building_func += 1
    player = players[game.cur_player]
    if player.roadbuilding_cards_old > 0:
        possible1 = road_possible_check(a1,b1)
        if possible1 == 1:
            road_place(a1,b1)
            possible2 = road_possible_check(a2,b2)
            if possible2 == 1:
                road_place(a2,b2)
                player.roadbuilding_cards_old = player.roadbuilding_cards_old - 1
                random_testing.successful_activate_road_building_func += 1
                phase.statechange = 1
                return 1
            else: 
                player.roads[a1][b1] = 0
    return 0
    
@count_calls
def trade_resources(give, get):
    a = 0
    random_testing.trade_resources += 1
    player = players[game.cur_player]
    if give == 1 and (board.harbor_lumber * player.settlements + board.harbor_lumber * player.cities).any() != 0:
        if player.resource_lumber > 1:
            player.resource_lumber -= 2
            if get == 2:
                player.resource_wool += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 4:
                player.resource_brick += 1
            elif get == 5:
                player.resource_ore += 1
    elif give == 2 and (board.harbor_wool * player.settlements + board.harbor_wool * player.cities).any() != 0:
        if player.resource_wool > 1:
            player.resource_wool -= 2
            if get == 1:
                player.resource_lumber += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 4:
                player.resource_brick += 1
            elif get == 5:
                player.resource_ore += 1
    elif give == 3 and (board.harbor_grain * player.settlements + board.harbor_grain * player.cities).any() != 0:
        if player.resource_grain > 1:
            player.resource_grain -= 2
            if get == 1:
                player.resource_lumber += 1
            elif get == 2:
                player.resource_wool += 1
            elif get == 4:
                player.resource_brick += 1
            elif get == 5:
                player.resource_ore += 1
    elif give == 4 and (board.harbor_brick * player.settlements + board.harbor_brick * player.cities).any() != 0:
        if player.resource_brick > 1:
            player.resource_brick -= 2
            if get == 1:
                player.resource_lumber += 1
            elif get == 2:
                player.resource_wool += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 5:
                player.resource_ore += 1
    elif give == 5 and (board.harbor_ore * player.settlements + board.harbor_ore * player.cities).any() != 0:
        if player.resource_ore > 1:
            player.resource_ore -= 2
            if get == 1:
                player.resource_lumber += 1
            elif get == 2:
                player.resource_wool += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 4:
                player.resource_brick += 1 
    elif (board.harbor_three_one * player.settlements + board.harbor_three_one * player.cities).any() != 0:
        if give == 1 and player.resource_lumber > 2:
            player.resource_lumber -= 3
            if get == 2:
                player.resource_wool += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 4:
                player.resource_brick += 1
            elif get == 5:
                player.resource_ore += 1
        elif give == 2 and player.resource_wool > 2:
            player.resource_wool -= 3
            if get == 1:
                player.resource_lumber += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 4:
                player.resource_brick += 1
            elif get == 5:
                player.resource_ore += 1        
        elif give == 3 and player.resource_grain > 2:
            player.resource_grain -= 3
            if get == 1:
                player.resource_lumber += 1
            elif get == 2:
                player.resource_wool += 1
            elif get == 4:
                player.resource_brick += 1
            elif get == 5:
                player.resource_ore += 1
        elif give == 4 and player.resource_brick > 2:
            player.resource_brick -= 3
            if get == 1:
                player.resource_lumber += 1
            elif get == 2:
                player.resource_wool += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 5:
                player.resource_ore += 1
        elif give == 5 and player.resource_ore > 2:
            player.resource_ore -= 3
            if get == 1:
                player.resource_lumber += 1
            elif get == 2:
                player.resource_wool += 1
            elif get == 3:
                player.resource_grain += 1
            elif get == 4:
                player.resource_brick += 1
    elif give == 1 and player.resource_lumber > 3:
        player.resource_lumber -= 4
        if get == 2:
            player.resource_wool += 1
        elif get == 3:
            player.resource_grain += 1
        elif get == 4:
            player.resource_brick += 1
        elif get == 5:
            player.resource_ore += 1
    elif give == 2 and player.resource_wool > 3:
        player.resource_wool -= 4
        if get == 1:
            player.resource_lumber += 1
        elif get == 3:
            player.resource_grain += 1
        elif get == 4:
            player.resource_brick += 1
        elif get == 5:
            player.resource_ore += 1    
    elif give == 3 and player.resource_grain > 3:
        player.resource_grain -= 4
        if get == 1:
            player.resource_lumber += 1
        elif get == 2:
            player.resource_wool += 1
        elif get == 4:
            player.resource_brick += 1
        elif get == 5:
            player.resource_ore += 1
    elif give == 4 and player.resource_brick > 3:
        player.resource_brick -= 4
        if get == 1:
            player.resource_lumber += 1
        elif get == 2:
            player.resource_wool += 1
        elif get == 3:
            player.resource_grain += 1
        elif get == 5:
            player.resource_ore += 1
    elif give == 5 and player.resource_ore > 3:
        player.resource_ore -= 4
        if get == 1:
            player.resource_lumber += 1
        elif get == 2:
            player.resource_wool += 1
        elif get == 3:
            player.resource_grain += 1
        elif get == 4:
            player.resource_brick += 1
    else:
        a = 1


@count_calls
def discard_resources(lumber, wool, grain, brick, ore):
    
    random_testing.discard_resources += 1
    player = players[game.cur_player]
    if player.discard_first_time == 1:
        player.total_resources = player.resource_lumber + player.resource_brick + player.resource_grain + player.resource_ore + player.resource_wool 
        player.discard_resources_lumber = player.resource_lumber
        player.discard_resources_wool = player.resource_wool
        player.discard_resources_grain = player.resource_grain
        player.discard_resources_brick = player.resource_brick
        player.discard_resources_ore = player.resource_ore
        player.resource_lumber = 0
        player.resource_wool = 0
        player.resource_grain = 0
        player.resource_brick = 0
        player.resource_ore = 0
        player.discard_first_time = 0

    if lumber == 1:  
        if player.discard_resources_lumber != 0:
            player.resource_lumber += 1
            player.discard_resources_lumber -= 1 
            player.discard_resources_turn += 1
            phase.statechange = 1
    elif wool == 1:
        if player.discard_resources_wool != 0:
            player.resource_wool += 1
            player.discard_resources_wool -= 1
            player.discard_resources_turn += 1
            phase.statechange = 1
    elif grain == 1:
        if player.discard_resources_grain != 0:
            player.resource_grain += 1
            player.discard_resources_grain -= 1 
            player.discard_resources_turn += 1
            phase.statechange = 1
    elif brick == 1:
        if player.discard_resources_brick != 0:
            player.resource_brick += 1
            player.discard_resources_brick -= 1 
            player.discard_resources_turn += 1
            phase.statechange = 1
    elif ore == 1:
        if player.discard_resources_ore != 0:
            player.resource_ore += 1
            player.discard_resources_ore -= 1 
            player.discard_resources_turn += 1
            phase.statechange = 1

    if player.discard_resources_turn == math.ceil(player.total_resources/2):
        player.discard_resources_lumber = 0
        player.discard_resources_wool = 0
        player.discard_resources_grain = 0
        player.discard_resources_brick = 0
        player.discard_resources_ore = 0
        player.discard_resources_turn = 0
        player.discard_resources_started = 0
        random_testing.successful_discard_resources += 1
        steal_card()

@count_calls
def longest_road(i, j, prev_move):
    random_testing.trav += 1
    n = 11 
    m = 21
    if i < 0 or j < 0 or i >= n or j >= m or board.longest_road[i][j] == 0: 
        return 0
    board.longest_road[i][j] = 0
    moves = [(i+1, j+1), (i+1, j-1), (i, j+2), (i-1, j+1), (i-1, j-1), (i, j-2)]
    if prev_move == (i, j+2):
        moves.remove((i+1, j+1))
        moves.remove((i-1, j+1))
    elif prev_move == (i, j-2):
        moves.remove((i+1, j-1))
        moves.remove((i-1, j-1))
    elif prev_move == (i+1, j+1):
        moves.remove((i+1,j-1))
        moves.remove((i,j+2))
    elif prev_move == (i+1, j-1):
        moves.remove((i+1,j+1))
        moves.remove((i,j-2))
    elif prev_move == (i-1, j+1):
        moves.remove((i-1,j-1))
        moves.remove((i,j+2))
    elif prev_move == (i-1, j-1):
        moves.remove((i-1,j+1)) 
        moves.remove((i,j-2))

    max_length = max(longest_road(x, y, (i, j)) for x, y in moves)
    return 1 + max_length
    
@count_calls
def find_longest_road():
    random_testing.find_longest_road += 1
    player = players[game.cur_player]
    ans, n, m = 0, 11, 21
    for i in range(n):
        for j in range(m):
            board.longest_road = player.roads * (1-board.ZEROBOARD)
            c = longest_road(i, j, (0, 0))
            if c > 0: 
                ans = max(ans,c)
    return ans
 
@count_calls
def update_longest_road():
    player = players[game.cur_player]
    opponent = players[game.cur_player]
    player.roads_connected = find_longest_road()
    if player.roads_connected >= 5 and player.roads_connected > opponent.roads_connected:
        if opponent.longest_road == 1:
            opponent.longest_road = 0
            opponent.victorypoints -= 2
        player.longest_road = 1
        player.victorypoints += 2

@count_calls
def find_largest_army():
    random_testing.find_largest_army += 1
    player = players[game.cur_player]
    opponent = players[1 - game.cur_player]
    if player.knight_cards_played >= 3 and player.knight_cards_played > opponent.knight_cards_played and player.largest_army == 0:
        if opponent.largest_army == 1:
            opponent.largest_army = 0
            opponent.victorypoints -= 2 
        player.largest_army = 1
        player.victorypoints += 2
    
@count_calls
def move_finished():
    random_testing.move_finished += 1
    player = players[game.cur_player]
    phase.reward += 0.0001
    phase.statechange = 1

    print("move finished")

    player.knight_cards_old += player.knight_cards_new
    player.victorypoints_cards_old += player.victorypoints_cards_new
    player.yearofplenty_cards_old += player.yearofplenty_cards_new
    player.monopoly_cards_old += player.monopoly_cards_new
    player.roadbuilding_cards_old += player.roadbuilding_cards_new

    player.knight_cards_new = 0
    player.victorypoints_cards_new = 0 
    player.yearofplenty_cards_new = 0
    player.monopoly_cards_new = 0
    player.roadbuilding_cards_new = 0 

    phase.development_card_played = 0

    random_testing.numberofturns += 1

    phase.reward = ((player0.victorypoints - player1.victorypoints) - (player0.victorypoints_before - player1.victorypoints_before))*0.02
    player0.victorypoints_before = player0.victorypoints
    player1.victorypoints_before = player1.victorypoints
    
    if player.victorypoints >= 10:
        if game.cur_player == 0: 
            phase.reward = 1  
            player0.wins += 1
        else: 
            phase.reward = -1
            player1.wins += 1
        random_testing.numberofgames += 1
        game.is_finished = 1
        new_game()

    game.cur_player = 1 - game.cur_player
    if game.placement_phase_pending != 1:
        turn_starts()

@count_calls
def new_initial_state():

#_______________________input_________________________
    board.tiles_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_probability_1 = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_probability_2 = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_probability_3 = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_probability_4 = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_probability_5 = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.rober_position = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_three_one = np.zeros((_NUM_ROWS, _NUM_COLS))
    #_________________ game specific ________________
    #board
    board.ZEROBOARD = np.zeros((_NUM_ROWS, _NUM_COLS))
    #tiles
    board.TILES_POSSIBLE = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_dice = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_dice_probabilities = np.zeros((_NUM_ROWS, _NUM_COLS))

    #settlements
    board.settlements_free = np.ones((_NUM_ROWS, _NUM_COLS))
    board.settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.settlements_used = np.zeros((_NUM_ROWS, _NUM_COLS))

    #roads
    board.roads_available = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.roads_free = np.zeros((_NUM_ROWS, _NUM_COLS))

    #harbors
    board.harbors_possible = np.zeros((9, 2, 2))

    #longest road
    board.longest_road = np.zeros((_NUM_ROWS,_NUM_COLS))
    board.increasing_roads = np.zeros((_NUM_ROWS,_NUM_COLS))


    distribution.tile_numbers = [0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,5,5,5]
    distribution.tile_random_numbers = np.random.choice(distribution.tile_numbers,19,replace=False)
    distribution.harbor_numbers = [1,2,3,4,5,6,6,6,6]
    distribution.harbor_random_numbers = np.random.choice(distribution.harbor_numbers,9,replace=False)
    distribution.plate_numbers = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]
    distribution.plate_random_numbers = np.random.choice(distribution.plate_numbers, 18, replace=False)
    distribution.development_card_numbers = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,4,4,5,5]
    distribution.development_card_random_number = np.random.choice(distribution.development_card_numbers,25,replace=False)
    distribution.development_cards_bought = 0

    for player in players:
        #________________________input board__________________________
        player.settlements = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.roads = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.cities = np.zeros((_NUM_ROWS, _NUM_COLS))       
        #_______________________input vector__________________________
        player.resource_lumber = 0
        player.resource_wool = 0
        player.resource_grain = 0
        player.resource_brick = 0
        player.resource_ore = 0

        player.settlements_left = 5
        player.roads_left = 15
        player.cities_left = 4

        player.army_size = 0

        player.knight_cards_old = 0
        player.victorypoints_cards_old = 0
        player.yearofplenty_cards_old = 0
        player.monopoly_cards_old = 0
        player.roadbuilding_cards_old = 0

        player.knight_cards_new = 0
        player.victorypoints_cards_new = 0
        player.yearofplenty_cards_new = 0
        player.monopoly_cards_new = 0
        player.roadbuilding_cards_new = 0

        player.harbor_lumber = 0
        player.harbor_wool = 0
        player.harbor_grain = 0
        player.harbor_brick = 0
        player.harbor_ore = 0
        player.harbor_three_one = 0

        player.largest_army = 0

        player.roads_connected = 0
        player.longest_road = 0

        player.knight_cards_played = 0

        player.victorypoints = 0

        player.development_card_played = 0
        player.knight_move_pending = 0
        player.monopoly_move_pending = 0
        player.roadbuilding_move_pending = 0
        player.roadbuilding1 = 0
        player.yearofplenty_move_pending = 0

        player.yearofplenty_started = 0
        player.yearofplenty1 = 0
        player.yearofplenty2 = 0

        player.discard_resources_started = 0
        player.discard_resources_turn = 0
        player.discard_first_time = 0
        player.total_resources = 0

        player.discard_resources_lumber = 0
        player.discard_resources_wool = 0
        player.discard_resources_grain = 0
        player.discard_resources_brick = 0
        player.discard_resources_ore = 0

        #__________________game-specific resource_____________
        #roads
        player.roads_possible = np.zeros((_NUM_ROWS, _NUM_COLS))

        #rewards 
        player.rewards_possible = np.zeros((_NUM_ROWS,_NUM_COLS))

        player.roadbuilding_d = 0
        player.roadbuilding_e = 0

    game.is_finished = 0
    game.settlementplaced = 0
    game.placement_phase_pending = 0
    game.placement_phase_turns_made = 0
    game.placement_phase_settlement_turn = 0
    game.placement_phase_road_turn = 0
    game.seven_rolled = 0
    game.placement_phase_settlement_coordinate1 = 0
    game.placement_phase_settlement_coordinate2 = 0


def set_terminal_width(columns):
    if os.name == 'posix':  # For Unix-based systems (Linux, macOS)
        os.system(f'sty columns {columns}')
    elif os.name == 'nt':  # For Windows
        os.system(f'mode con cols={columns}')
def setup():
    harbors_building()
    tiles_buidling()
    settlements_building()
    roads_building()
    tile_distribution()
    harbor_distribution()
    plate_distribution()
def reset():
    new_initial_state()
    setup()

def start():
    set_terminal_width(200)
    np.set_printoptions(linewidth=200)

    
def turn_starts():
    player = players[game.cur_player]
    c = roll_dice()

    if player.resource_wool > 0 and player.resource_grain > 0 and player.resource_ore > 0:
        random_testing.resources_buy_dc += 1
    if player.resource_brick > 0 and player.resource_lumber > 0:
        random_testing.resources_buy_road += 1
    if player.resource_grain > 1 and player.resource_ore > 2:
        random_testing.resources_buy_city += 1
    if player.resource_brick > 0 and player.resource_lumber > 0 and player.resource_grain > 0 and player.resource_wool > 0:
        random_testing.resources_buy_settlement += 1
    if c == 7:
        game.seven_rolled = 1
        

def new_game():
    new_initial_state()
    setup()
    game.placement_phase_pending = 1
    game.placement_phase_settlement_turn = 1


def main():
    start()
    new_game()
#def random_main():
#    start()
#    new_game()
#    z = 0
#    for i in range (10000000):
#        player = players[game.cur_player]
#        opponent = players[1 - game.cur_player]
#        random_assignment()
#        action_executor()    
#
#        random_testing.resource_lumber_total += player0.resource_lumber
#        random_testing.resource_wool_total += player0.resource_wool
#        random_testing.resource_grain_total += player0.resource_grain
#        random_testing.resource_brick_total += player0.resource_brick
#        random_testing.resource_ore_total += player0.resource_ore
#
#
#
#        if i % 100000 == 0:
#
#
#            player = players[game.cur_player]
#            opponent = players[1-game.cur_player]
#            print(i)
#            print("number of games:",random_testing.numberofgames)
#            print("number of turns:",random_testing.numberofturns)
#
#            print("placement phase done", game.placement_phase_pending - 1)
#
#            print("number of victorypoints player0", player0.victorypoints)
#            print("number of victorypoints player1", player1.victorypoints)
#
#            print("lumber player 0:", player0.resource_lumber)
#            print("wool player 0:", player0.resource_wool)
#            print("grain player 0:", player0.resource_grain)
#            print("brick player 0:", player0.resource_brick)
#            print("ore player 0:", player0.resource_ore)
#
#            print("lumber player 1:", player1.resource_lumber)
#            print("wool player 1:", player1.resource_wool)
#            print("grain player 1:", player1.resource_grain)
#            print("brick player 1:", player1.resource_brick)
#            print("ore player 1:", player1.resource_ore)
#
#            print(random_testing.resource_lumber_total)
#            print(random_testing.resource_wool_total)
#            print(random_testing.resource_grain_total)
#            print(random_testing.resource_brick_total)
#            print(random_testing.resource_ore_total)
#        if i % 10000 == 0:
#            if player.knight_move_pending == 1:
#                print("knight move is pending")
#            if player.monopoly_move_pending == 1:
#                print("monopoly move is pending")
#            if player.roadbuilding_move_pending == 1:
#                print("roadbuilding move is pending")
#            if player.yearofplenty_move_pending == 1:
#                print("yearofplenty move is pending")
#            if player.discard_resources_started == 1:
#                print("discard_resources move is pending")
#            if opponent.knight_move_pending == 1:
#                print("knight move is pending")
#            if opponent.monopoly_move_pending == 1:
#                print("monopoly move is pending")
#            if opponent.roadbuilding_move_pending == 1:
#                print("roadbuilding move is pending")
#            if opponent.yearofplenty_move_pending == 1:
#                print("yearofplenty move is pending")
#            if opponent.discard_resources_started == 1:
#                print("discard_resources move is pending")
#        if i % 100000 == 0:
#
#            print(opponent.roadbuilding_move_pending,player.roadbuilding_move_pending)
#            print("development_card_choose:",random_testing.development_card_choose)
#            print("settlement_place:",random_testing.settlement_place)
#            print("settlement_place_placement:",random_testing.settlement_place_placement)
#            print("settlement_possible_check:",random_testing.settlement_possible_check)
#            print("buy_settlement:",random_testing.buy_settlement)
#            print("development_card_choose:",random_testing.development_card_choose)
#            print("buy_development_cards:",random_testing.buy_development_cards)
#            print("buy_city:",random_testing.buy_city)
#            print("city_place:",random_testing.city_place)
#            print("road_place:",random_testing.road_place)
#            print("road_place_placement:",random_testing.road_place_placement)
#            print("road_possible_check:",random_testing.road_possible_check)
#            print("buy_road:",random_testing.buy_road)
#            print("activate_road_building_func:",random_testing.activate_road_building_func)
#            print("activate_monopoly_func:",random_testing.activate_monopoly_func)
#            print("activate_yearofplenty_func:",random_testing.activate_yearofplenty_func)
#            print("tile_update_rewards:",random_testing.tile_update_rewards)
#            print("update_longest_road:",random_testing.update_longest_road)
#            print("find_longest_road:",random_testing.find_longest_road)
#            print("check_longest_road:",random_testing.check_longest_road)
#            print("find_largest_army:",random_testing.find_largest_army)
#            print("trav:",random_testing.trav)
#            print("move_finished:",random_testing.move_finished)
#            print("randomly_pick_resources:",random_testing.randomly_pick_resources)
#            print("discard_resources:",random_testing.discard_resources)
#            print("trade_resources:",random_testing.trade_resources)
#            print("move_rober:",random_testing.move_rober)
#            print("play_knight:",random_testing.play_knight)
#            print("steal_card:",random_testing.steal_card)
#            print("roll_dice:",random_testing.roll_dice)
#
#            print("successful settlement_place:",random_testing.successful_settlement_place)
#            print("successful buy_city:",random_testing.successful_buy_city)
#            print("successful city_place:",random_testing.successful_city_place)
#            print("successful road_place:",random_testing.successful_road_place)
#            print("successful road_place_placement:",random_testing.successful_road_place_placement)
#            print("successful road_possible_check:",random_testing.successful_road_possible_check)
#            print("successful buy_road:",random_testing.successful_buy_road)
#            print("successful activate_road_building_func:",random_testing.successful_activate_road_building_func)
#            print("successful activate_monopoly_func:",random_testing.successful_activate_monopoly_func)
#            print("successful activate_yearofplenty_func:",random_testing.successful_activate_yearofplenty_func)
#            print("successful tile_update_rewards:",random_testing.successful_tile_update_rewards)
#            print("successful update_longest_road:",random_testing.successful_update_longest_road)
#            print("successful find_longest_road:",random_testing.successful_find_longest_road)
#            print("successful check_longest_road:",random_testing.successful_check_longest_road)
#            print("successful find_largest_army:",random_testing.successful_find_largest_army)
#            print("successful trav:",random_testing.successful_trav)
#            print("successful move_finished:",random_testing.successful_move_finished)
#            print("successful randomly_pick_resources:",random_testing.successful_randomly_pick_resources)
#            print("successful discard_resources:",random_testing.successful_discard_resources)
#            print("successful trade_resources:",random_testing.successful_trade_resources)
#            print("successful move_rober:",random_testing.successful_move_rober)
#            print("successful play_knight:",random_testing.successful_play_knight)
#            print("successful steal_card:",random_testing.successful_steal_card)
#            print("successful roll_dice:",random_testing.successful_roll_dice)
#
#            print("This is the most important thing", random_testing.howmuchisthisaccsessed)
#            random_testing.howmuchisthisaccsessed = 0
#            print("Something is deinetly not working correctly", z)
#            print("")
#            print("player cities left", player.cities_left)
#            print("opponent cities left", opponent.cities_left)
#            print("")
#            print("player settlements left", player.settlements_left)
#            print("opponent settlements left", opponent.settlements_left)
#            print("")
#            print("player roads left", player.roads_left)
#            print("opponent roads left", opponent.roads_left)
#            print("")
#
#            print("development cards left", distribution.development_cards_bought)
#
#
#            print("possibility to buy settlement", random_testing.resources_buy_settlement)
#            print("possibility to buy city", random_testing.resources_buy_city)
#            print("possibility to buy road", random_testing.resources_buy_road)
#            print("possibility to buy dc", random_testing.resources_buy_dc)
#
#            print("board.tiles_dice",board.tiles_dice)
#            print("player.settlements",player.settlements)
#            print("opponent.settlements",opponent.settlements)
#            print("player.cities",player.cities)
#            print("opponent.cities",opponent.cities)
#            print("player.rewards_possible",player.rewards_possible)
#            print("opponent.rewards_possible",opponent.rewards_possible)
#            print("player.roads",player.roads)

        
main()

@count_calls
def action_executor():
    player = players[game.cur_player]
    action = player_action[game.cur_player]
    keepresources = player_keepresources[game.cur_player]
    trading = player_trading[game.cur_player]

    if game.seven_rolled == 1:
        if np.any(action.rober_move == 1): 
            b,c = np.where(action.rober_move == 1)
            d = int(b)
            e = int(c)
            action.rober_move[d][e] = 0
            if d < 11 and d >= 0 and e < 21 and e >= 0: 
                move_rober(d,e)
                if player.resource_lumber + player.resource_wool + player.resource_grain + player.resource_brick + player.resource_ore >= 7:
                    player.discard_first_time = 1
                    player.discard_resources_started = 1
                else:
                    steal_card()
                game.seven_rolled = 0
    if player.knight_move_pending == 1:
        if np.any(action.rober_move == 1): 
            b,c = np.where(action.rober_move == 1)
            d = int(b)
            e = int(c)
            play_knight(d,e)
            player.knight_move_pending = 0 
            
    
    if player.roadbuilding_move_pending == 1:
        if np.any(action.road_place == 1):
            if player.roadbuilding1 == 0:
                b,c = np.where(action.road_place == 1)
                player.roadbuilding_d = int(b)
                player.roadbuilding_e = int(c)
                player.roadbuilding1 = 1
            elif player.roadbuilding1 == 1:
                b,c = np.where(action.road_place == 1)
                d = int(b)
                e = int(c)
            
                if  d < 11 and d >= 0 and e < 21 and e >= 0 and player.roadbuilding_d < 11 and player.roadbuilding_d >= 0 and player.roadbuilding_e < 21 and player.roadbuilding_e >= 0: 
                    road_place_card(player.roadbuilding_d,player.roadbuilding_e,d,e)
                    player.roadbuilding_move_pending = 0
                    player.roadbuilding1 = 0
            

            
                
    if game.placement_phase_pending == 1:
        if game.placement_phase_road_turn == 1:
            if np.any(action.road_place == 1):
                b,c = np.where(action.road_place == 1)
                d = int(b)
                e = int(c)
                print(d,e)
                print("works")
                if d < 11 and d >= 0 and e < 21 and e >= 0: 
                    possible = road_place_placement(game.placement_phase_settlement_coordinate1,game.placement_phase_settlement_coordinate2,d,e)
                    if possible == 1:
                        print("road_place_placement")
                        game.placement_phase_road_turn = 0
                        game.placement_phase_settlement_turn = 1
                        game.placement_phase_turns_made += 1
                        print(game.placement_phase_turns_made)
                        print("player.settlements")
                        print(player.settlements)
                        print("player.roads")
                        print(player.roads)
                        print(random_testing.roll_dice)
                        if game.placement_phase_turns_made == 1:
                            move_finished()
                        if game.placement_phase_turns_made == 3:
                            move_finished()    
                        if game.placement_phase_turns_made == 4:
                            move_finished()
                            game.placement_phase_pending = 0
                            game.placement_phase_turns_made = 0
    
    if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
        random_testing.howmuchisthisaccsessed += 1
        if np.any(action.settlement_place == 1):
            b,c = np.where(action.settlement_place == 1)
            d = int(b)
            e = int(c)
            if d < 11 and d >= 0 and e < 21 and e >= 0: 
                buy_settlement(d,e)
        if np.any(action.city_place == 1):
            b,c = np.where(action.city_place == 1)
            d = int(b)
            e = int(c)
            if d < 11 and d >= 0 and e < 21 and e >= 0: 
                buy_city(d,e)
        if np.any(action.road_place == 1):
            b,c = np.where(action.road_place == 1)
            d = int(b)
            e = int(c)
            if d < 11 and d >= 0 and e < 21 and e >= 0: 
                buy_road(d,e)


    if game.placement_phase_pending == 1:
        if game.placement_phase_settlement_turn == 1:
            if np.any(action.settlement_place == 1):
                print("settlement_place")
                b,c = np.where(action.settlement_place == 1)
                d = int(b)
                e = int(c)
                if d < 11 and d >= 0 and e < 21 and e >= 0: 
                    possible = settlement_place_placement(d,e)
                    if possible == 1:
                        game.placement_phase_settlement_coordinate1 = d
                        game.placement_phase_settlement_coordinate2 = e
                        game.placement_phase_settlement_turn = 0
                        game.placement_phase_road_turn = 1
                    

            
    if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
        if action.end_turn == 1:
            move_finished() #need to take a look at this function too
    
    if player.discard_resources_started == 1:
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        if keepresources.keep_lumber == 1: 
            a = 1
        elif keepresources.keep_wool == 1: 
            b = 1
        elif keepresources.keep_grain == 1: 
            c = 1
        elif keepresources.keep_brick == 1: 
            d = 1
        elif keepresources.keep_ore == 1: 
            e = 1
        if a != 0 or b != 0 or c != 0 or d != 0 or e != 0:
            discard_resources(a,b,c,d,e)
    
                    
                


    if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
        if trading.give_lumber_get_wool == 1:
            trade_resources(1,2)
        if trading.give_lumber_get_grain == 1:
            trade_resources(1,3)
        if trading.give_lumber_get_brick == 1:
            trade_resources(1,4)
        if trading.give_lumber_get_ore  == 1:
            trade_resources(1,5)
        if trading.give_wool_get_lumber == 1:
            trade_resources(2,1)
        if trading.give_wool_get_grain == 1:
            trade_resources(2,3)
        if trading.give_wool_get_brick == 1:
            trade_resources(2,4)
        if trading.give_wool_get_ore == 1:
            trade_resources(2,5)
        if trading.give_grain_get_lumber == 1:
            trade_resources(3,1)
        if trading.give_grain_get_wool == 1:
            trade_resources(3,2)
        if trading.give_grain_get_brick == 1:
            trade_resources(3,4)
        if trading.give_grain_get_ore == 1:
            trade_resources(3,5)
        if trading.give_brick_get_lumber == 1:
            trade_resources(4,1)
        if trading.give_brick_get_wool == 1:
            trade_resources(4,2)
        if trading.give_brick_get_grain == 1:
            trade_resources(4,3)
        if trading.give_brick_get_ore == 1:
            trade_resources(4,5)
        if trading.give_ore_get_lumber == 1:
            trade_resources(5,1)
        if trading.give_ore_get_wool == 1:
            trade_resources(5,2)
        if trading.give_ore_get_grain == 1:
            trade_resources(5,3)
        if trading.give_ore_get_brick == 1:
            trade_resources(5,4)
        if action.development_card_buy == 1:
            buy_development_cards()

        if action.knight_cards_activate == 1:
            if player.knight_cards_old >= 1:
                player.knight_move_pending = 1
        if action.yearofplenty_cards_activate == 1:
            if player.yearofplenty_cards_old >= 1:
                player.yearofplenty_move_pending = 1
        if action.monopoly_cards_activate == 1:
            if player.monopoly_cards_old >= 1:
                player.monopoly_move_pending = 1
        if action.road_building_cards_activate == 1:
            if player.roadbuilding_cards_old >= 1:
                if player.roads_left == 0 or player.roads_left == 1:
                    player.roadbuilding_cards_old -= 1
                player.roadbuilding_move_pending = 1


    if player.yearofplenty_move_pending == 1:
        if player.yearofplenty_started == 1:
            if action.yearofplenty_lumber == 1:
                player.yearofplenty2 = 1
            if action.yearofplenty_wool == 1:
                player.yearofplenty2 = 2
            if action.yearofplenty_grain == 1:
                player.yearofplenty2 = 3
            if action.yearofplenty_brick == 1:
                player.yearofplenty2 = 4
            if action.yearofplenty_ore == 1:
                player.yearofplenty2 = 5
            activate_yearofplenty_func(player.yearofplenty1,player.yearofplenty2)
            player.yearofplenty_started = 0
            player.yearofplenty_move_pending = 0

        if player.yearofplenty_started == 0:
            if action.yearofplenty_lumber == 1:
                player.yearofplenty1 = 1
            if action.yearofplenty_wool == 1:
                player.yearofplenty1 = 2
            if action.yearofplenty_grain == 1:
                player.yearofplenty1 = 3
            if action.yearofplenty_brick == 1:
                player.yearofplenty1 = 4
            if action.yearofplenty_ore == 1:
                player.yearofplenty1 = 5
            player.yearofplenty_started = 1 

    if player.monopoly_move_pending == 1:
        a = 0
        if action.monopoly_lumber == 1:
            a = 1
        elif action.monopoly_wool == 1:
            a = 2
        elif action.monopoly_grain == 1:
            a = 3
        elif action.monopoly_brick == 1:
            a = 4
        elif action.monopoly_ore == 1:
            a = 5    
        if a != 0:
            activate_monopoly_func(a)
            player.monopoly_move_pending = 0
    
    action.rober_move = action.rober_move * board.ZEROBOARD
    action.road_place = action.road_place * board.ZEROBOARD
    action.settlement_place = action.settlement_place * board.ZEROBOARD
    action.city_place = action.city_place * board.ZEROBOARD



            
def random_assignment():
    random_agent.random_action = np.random.choice(np.arange(1,46), p=[1/14, 1/14, 1/14, 1/14, 1/14, 1/70, 1/70, 1/70, 1/70, 1/70, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/280, 1/14, 1/14, 1/14, 1/14, 1/14, 1/70, 1/70, 1/70, 1/70, 1/70, 1/70, 1/70, 1/70, 1/70, 1/70])
    random_agent.random_position_y = np.random.choice(np.arange(0,11))
    random_agent.random_position_x = np.random.choice(np.arange(0,21))
    action_selecter(random_agent.random_action, random_agent.random_position_x, random_agent.random_position_y)


def action_selecter(selected_action, selected_position_x = 0, selected_position_y = 0):

    player = players[game.cur_player]
    action = player_action[game.cur_player]
    keepresources = player_keepresources[game.cur_player]
    trading = player_trading[game.cur_player]

    action.rober_move = action.rober_move * board.ZEROBOARD
    action.road_place = action.road_place * board.ZEROBOARD
    action.settlement_place = action.settlement_place * board.ZEROBOARD
    action.city_place = action.city_place * board.ZEROBOARD
    action.end_turn = 0

    keepresources.keep_lumber = 0
    keepresources.keep_wool = 0
    keepresources.keep_grain = 0
    keepresources.keep_brick = 0
    keepresources.keep_ore = 0

    trading.give_lumber_get_wool = 0
    trading.give_lumber_get_grain = 0
    trading.give_lumber_get_brick = 0
    trading.give_lumber_get_ore  = 0
    trading.give_wool_get_lumber = 0
    trading.give_wool_get_grain = 0
    trading.give_wool_get_brick = 0
    trading.give_wool_get_ore = 0
    trading.give_grain_get_lumber = 0
    trading.give_grain_get_wool = 0
    trading.give_grain_get_brick = 0
    trading.give_grain_get_ore = 0
    trading.give_brick_get_lumber = 0
    trading.give_brick_get_wool = 0
    trading.give_brick_get_grain = 0
    trading.give_brick_get_ore = 0
    trading.give_ore_get_lumber = 0
    trading.give_ore_get_wool = 0
    trading.give_ore_get_grain = 0
    trading.give_ore_get_brick = 0
    
    action.development_card_buy = 0
    action.knight_cards_activate = 0
    action.road_building_cards_activate = 0
    action.yearofplenty_cards_activate = 0
    action.monopoly_cards_activate = 0
    action.yearofplenty_lumber = 0
    action.yearofplenty_wool = 0
    action.yearofplenty_grain = 0
    action.yearofplenty_brick = 0
    action.yearofplenty_ore = 0
    action.monopoly_lumber = 0
    action.monopoly_wool = 0
    action.monopoly_grain = 0
    action.monopoly_brick = 0
    action.monopoly_ore = 0

    if selected_action == 1:  
        action.rober_move[selected_position_y] [selected_position_x] = 1
    if selected_action == 2:
        action.road_place[selected_position_y] [selected_position_x] = 1
    if selected_action == 3:
        action.settlement_place[selected_position_y] [selected_position_x] = 1
    if selected_action == 4:
        action.city_place[selected_position_y] [selected_position_x] = 1
    if selected_action == 5:
        action.end_turn = 1
    if selected_action == 6:    
        keepresources.keep_lumber = 1
    if selected_action == 7:    
        keepresources.keep_wool = 1
    if selected_action == 8:    
        keepresources.keep_grain = 1
    if selected_action == 9:    
        keepresources.keep_brick = 1
    if selected_action == 10:    
        keepresources.keep_ore = 1   
    if selected_action == 11:  
        trading.give_lumber_get_wool = 1
    if selected_action == 12:
        trading.give_lumber_get_grain = 1
    if selected_action == 13:
        trading.give_lumber_get_brick = 1
    if selected_action == 14:
        trading.give_lumber_get_ore  = 1
    if selected_action == 15:
        trading.give_wool_get_lumber = 1
    if selected_action == 16:
        trading.give_wool_get_grain = 1
    if selected_action == 17:
        trading.give_wool_get_brick = 1
    if selected_action == 18:
        trading.give_wool_get_ore = 1
    if selected_action == 19:
        trading.give_grain_get_lumber = 1
    if selected_action == 20:
        trading.give_grain_get_wool = 1
    if selected_action == 21:
        trading.give_grain_get_brick = 1
    if selected_action == 22:
        trading.give_grain_get_ore = 1
    if selected_action == 23:
        trading.give_brick_get_lumber = 1
    if selected_action == 24:
        trading.give_brick_get_wool = 1
    if selected_action == 25:
        trading.give_brick_get_grain = 1
    if selected_action == 26:
        trading.give_brick_get_ore = 1
    if selected_action == 27:
        trading.give_ore_get_lumber = 1
    if selected_action == 28:
        trading.give_ore_get_wool = 1
    if selected_action == 29:
        trading.give_ore_get_grain = 1
    if selected_action == 30:
        trading.give_ore_get_brick = 1
    if selected_action == 31:
        action.development_card_buy = 1
    if selected_action == 32:
        action.knight_cards_activate = 1
    if selected_action == 33:
        action.road_building_cards_activate = 1
    if selected_action == 34:
        action.yearofplenty_cards_activate = 1
    if selected_action == 35:
        action.monopoly_cards_activate = 1
    if selected_action == 36:
        action.yearofplenty_lumber = 1
    if selected_action == 37:
        action.yearofplenty_wool = 1
    if selected_action == 38:
        action.yearofplenty_grain = 1
    if selected_action == 39:
        action.yearofplenty_brick = 1
    if selected_action == 40:
        action.yearofplenty_ore = 1
    if selected_action == 41:
        action.monopoly_lumber = 1
    if selected_action == 42:
        action.monopoly_wool = 1
    if selected_action == 43:
        action.monopoly_grain = 1
    if selected_action == 44:
        action.monopoly_brick = 1
    if selected_action == 45:
        action.monopoly_ore = 1      

    action_executor()
        
          


def state_changer():
    player = players[game.cur_player]
    opponent = players[1 - game.cur_player]
    #23
    np_board_tensor = np.stack((
        board.rober_position,
        board.tiles_lumber,
        board.tiles_wool,
        board.tiles_grain,
        board.tiles_brick,
        board.tiles_ore,
        board.tiles_probability_1,
        board.tiles_probability_2,
        board.tiles_probability_3,
        board.tiles_probability_4,
        board.tiles_probability_5,
        board.harbor_three_one,
        board.harbor_lumber,
        board.harbor_wool,
        board.harbor_grain,
        board.harbor_brick,
        board.harbor_ore,
        player0.settlements,
        player1.settlements,
        player0.cities,
        player1.cities,
        player0.roads,
        player1.roads,
    ))
    #35
    np_vector_tensor = np.stack((
        player.victorypoints,
        player.resource_lumber,
        player.resource_wool,
        player.resource_grain,
        player.resource_brick,
        player.resource_ore,
        player.roads_left,
        player.settlements_left,
        player.cities_left,
        player.army_size,
        player.roads_connected,
        player.knight_cards_old,
        player.yearofplenty_cards_old,
        player.monopoly_cards_old,
        player.roadbuilding_cards_old,
        player.victorypoints_cards_old,
        player.knight_cards_new,
        player.yearofplenty_cards_new,
        player.monopoly_cards_new,
        player.roadbuilding_cards_new,
        player.victorypoints_cards_new,
        opponent.resource_lumber,
        opponent.resource_wool,
        opponent.resource_grain,
        opponent.resource_brick,
        opponent.resource_ore,
        opponent.victorypoints_cards_old + opponent.victorypoints_cards_new + opponent.knight_cards_old + opponent.knight_cards_new + opponent.yearofplenty_cards_old + opponent.yearofplenty_cards_new + opponent.monopoly_cards_old + opponent.monopoly_cards_new + opponent.roadbuilding_cards_old + opponent.roadbuilding_cards_new,
        opponent.army_size,
        opponent.roads_connected,
        phase.development_card_played,
        player.knight_move_pending,
        player.monopoly_move_pending,
        player.roadbuilding_move_pending,
        player.yearofplenty_move_pending,
        player.discard_resources_started,
    ))
    torch_board_tensor = torch.from_numpy(np_board_tensor)
    torch_vector_tensor = torch.from_numpy(np_vector_tensor)
    return torch_board_tensor, torch_vector_tensor


#__________________________________Neural Network_______________________________________________________

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('cur_boardstate','cur_vectorstate', 'action', 'next_boardstate','next_vectorstate', 'reward'))

torch.set_printoptions(precision=5)


class ReplayMemory(object):
    """docstring for ReplayMemory"""
    def __init__(self, capacity):
        self.memory = deque([],maxlen = capacity)
    def push(self, *args):
        """Saves a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class PrintShape(nn.Module):
    def forward(self, x):
        print(x.shape)
        return x
    
class BIGDQN(nn.Module):
    def __init__(self, num_resBlocks = 12):
        super().__init__()

        self.denselayer = nn.Sequential(
            nn.Linear(35,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
        
        )
        self.denseFinal = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,41)
        )
        

        self.ConvScalar = nn.Sequential(
            nn.Conv2d(23,46,kernel_size=(5,3),padding=0,stride=(2,2)),
            nn.BatchNorm2d(46),
            nn.ReLU(),
            nn.Conv2d(46,20,kernel_size=3,padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20,10, kernel_size=3,padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.ConvConv = nn.ModuleList(
            [ResBlock() for i in range(num_resBlocks)]
        )

        #quite a lot of features, hope that this works
        self.ConvCombine = nn.Sequential(
            nn.Conv2d(23,46,kernel_size = 1, padding=0),
            nn.BatchNorm2d(46),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(46*11*21, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),            
            #combine the last layer of DenseConv with the last one of ConvCombine
        )

        self.ConvCombineFinal = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,11*21*4),
        )
        #That might be too much of an incline, but let's see how it goes
        self.DenseConv = nn.Sequential(
            nn.Linear(35,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
        )

        self.optimizer = optim.Adam(self.parameters(), lr = 0.001, amsgrad=True)
        self.loss = nn.MSELoss()



        # adding the outputs of self.denselayer and self.ConvScalar

        # I think I logaically need to combine them earlier
        # Let's think about that in school  

        # I probably need to add a conv layer before the res layer but let's see

class DQN(nn.Module):
    def __init__(self, num_resBlocks = 5):
        super().__init__()

        self.denselayer = nn.Sequential(
            nn.Linear(35,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
        
        )
        self.denseFinal = nn.Sequential(
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,41)
        )
        

        self.ConvScalar = nn.Sequential(
            nn.Conv2d(23,10,kernel_size=(5,3),padding=0,stride=(2,2)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.ConvConv = nn.ModuleList(
            [ResBlock() for i in range(num_resBlocks)]
        )

        #quite a lot of features, hope that this works
        self.ConvCombine = nn.Sequential(
            nn.Conv2d(23,10,kernel_size=(5,3),padding=0,stride=(2,2)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 256),           
            #combine the last layer of DenseConv with the last one of ConvCombine
        )

        self.ConvCombineFinal = nn.Sequential(
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,1024),
            nn.ReLU(),
            nn.Linear(1024,11*21*4),
        )
        #That might be too much of an incline, but let's see how it goes
        self.DenseConv = nn.Sequential(
            nn.Linear(35,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),
        )

        self.optimizer = optim.Adam(self.parameters(), lr = 0.001, amsgrad=True)
        self.loss = nn.MSELoss()



        # adding the outputs of self.denselayer and self.ConvScalar

        # I think I logaically need to combine them earlier
        # Let's think about that in school  

        # I probably need to add a conv layer before the res layer but let's see


    def forward(self, boardstate2, vectorstate2):
        x1 = self.denselayer(vectorstate2)
        x2 = self.ConvScalar(boardstate2)
        y1 = self.DenseConv(vectorstate2)
        for resblock in self.ConvConv:
            y2 = resblock(boardstate2)
        y2 = self.ConvCombine(y2)
        #is this the right dimension in which I concentate?
        y = torch.cat((y1,y2),1)
        x = torch.cat((x1,x2),1)
        vectoractions = self.denseFinal(x)
        boardactions = self.ConvCombineFinal(y)
        state = torch.cat((boardactions,vectoractions),1)
        return state
    
# might change the number of hidden layers later on 
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(23, 23, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(23)
        self.conv2 = nn.Conv2d(23, 23, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(23)
    def forward(self,x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        x = F.relu(x)
        return x


#a1 = np.random.randint(0,2, size = (17,21,21))
#print(a1.shape)
#print(a1)
#
#b1 = np.random.randint(0,2,size=(39))
#print(b1.shape)
#print(b1)
#
#model = DQN(num_hidden=32,num_resBlocks=12)
#
#actions = model(torch.tensor(b1,dtype=torch.float32).unsqueeze(0),torch.tensor(a1,dtype=torch.float32).unsqueeze(0))
#print(actions.shape)
#print(actions)
#
#actionprobabilities = F.softmax(actions,dim=1)


BATCH_SIZE = 32
GAMMA = 0.999
EPS_START = 1
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.001
LR = 0.005

total_actions = 21*11*4 + 41
action_counts = [0] * total_actions
random_action_counts = [0] * total_actions


cur_boardstate = state_changer()[0]
cur_vectorstate = state_changer()[1]

state, info = 0,0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

agent2_policy_net = DQN().to(device)
agent1_policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(agent1_policy_net.state_dict())

optimizer = optim.Adam(agent1_policy_net.parameters(), lr = LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def select_action(boardstate, vectorstate):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            if game.cur_player == 0:
                action = agent1_policy_net(boardstate, vectorstate).max(1).indices.view(1,1)
                if action >= 4*11*21:
                    final_action = action - 4*11*21 + 5
                    position_y = 0
                    position_x = 0
                else:
                    final_action = math.ceil((action/11/21)+1)
                    position_y = math.floor((action - ((final_action-1)*11*21)-1)/21)
                    position_x = action % 21 
                action_selecter(final_action, position_x, position_y)
                action_counts[action-1] += 1
                return action
            elif game.cur_player == 1:
                action =  agent2_policy_net(boardstate, vectorstate).max(1).indices.view(1,1) 
                if action >= 4*11*21:
                    final_action = action - 4*11*21 + 5
                    position_y = 0
                    position_x = 0
                else:
                    final_action = math.ceil(action/11/21)
                    position_y = math.floor((action - ((final_action-1)*11*21)-1)/21)
                    position_x = action % 21 
                action_selecter(final_action, position_x, position_y)
                action_counts[action-1] += 1
                return action
    else:
        action = torch.tensor([[random.randrange(1,4*21*11 + 41)]], device=device, dtype=torch.long)
        if action >= 4*11*21:
            final_action = action - 4*11*21 + 5
            position_y = 0
            position_x = 0
        else:
            final_action = math.ceil(action/11/21)
            position_y = math.floor((action - ((final_action-1)*11*21)-1)/21)
            position_x = action % 21 
        action_selecter(final_action, position_x, position_y)

        #final_action, position_x, position_y = torch.tensor([[random.randrange(1,1+n_actions)]], device=device, dtype=torch.long), torch.tensor([[random.randrange(21)]], device=device, dtype=torch.long), torch.tensor([[random.randrange(11)]], device=device, dtype=torch.long)
        #action_selecter(final_action, position_x, position_y)
        #if final_action <= 4:
        #    action = (final_action-1)*21*11 + position_y*21 + position_x
        #else:
        #    action = 4*21*11 + (final_action-5)
        
        random_action_counts[action-1] += 1
        return action
    
episode_durations = []

def plotting():
    print()
#def optimize_model():
#    if len(memory) < BATCH_SIZE:
#        return
#    transitions = memory.sample(BATCH_SIZE)
#    batch = Transition(*zip(*transitions))
#
#    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=device, dtype=torch.bool)
#    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
#
#    state_batch = torch.cat(batch.state)
#    action_batch = torch.cat(batch.action)
#    reward_batch = torch.cat(batch.reward)
#
#    state_action_values = agent1_policy_net(state_batch).gather(1,action_batch)
#
#    next_state_values = torch.zeros(BATCH_SIZE,device=device)
#
#    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
#
#    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
#
#    loss = F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))
#    
#    optimizer.zero_grad()
#
#    loss.backward()
#
#    #torch.nn.utils.clip_grad_norm_(agent1_policy_net.parameters(), 100)
#
#    optimizer.step()



log_called = 0
def log():
    wandb.watch(agent1_policy_net)
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done / EPS_DECAY)
    wandb.log({"eps_threshold": eps_threshold})

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(action_counts))), y=action_counts, mode='markers', name='Action Counts'))
    fig.add_trace(go.Scatter(x=list(range(len(random_action_counts))), y=random_action_counts, mode='markers', name='Random Action Counts'))
    wandb.log({"Player 0 Wins": player0.wins})
    wandb.log({"Player 1 Wins": player1.wins})
    wandb.log({"Episode Duration": episode_durations})
    wandb.log({"player0.victorypoints": player0.victorypoints})
    wandb.log({"player1.victorypoints": player1.victorypoints})
    wandb.log({"Action Counts": wandb.Plotly(fig)})
    wandb.log({"random_testing.move_finsihed":random_testing.move_finished})
    wandb.log({"phase.statechangecount": phase.statechangecount})
    # Create a scatter plot for function call counts
    fig = go.Figure(data=go.Scatter(
        x=list(call_counts.keys()), 
        y=list(call_counts.values()), 
        mode='markers', 
        marker=dict(
            size=10,
            color=list(call_counts.values()), # set color to an array/list of desired values
            colorscale='Viridis', # choose a colorscale
            showscale=True
        )
    ))
    wandb.log({"game.seven_rolled": game.seven_rolled})
    wandb.log({"game.cur_player": game.cur_player})
    wandb.log({"game.placement_phase_pending": game.placement_phase_pending})
    wandb.log({"player0.knight_move_pending": player0.knight_move_pending})
    wandb.log({"player0.monopoly_move_pending": player0.monopoly_move_pending})
    wandb.log({"player0.roadbuilding_move_pending": player0.roadbuilding_move_pending})
    wandb.log({"player0.yearofplenty_move_pending": player0.yearofplenty_move_pending})
    wandb.log({"player0.discard_resources_started": player0.discard_resources_started})
    wandb.log({"player0.monopoly_cards_old": player0.monopoly_cards_old})

    wandb.log({"player1.knight_move_pending": player1.knight_move_pending})
    wandb.log({"player1.monopoly_move_pending": player1.monopoly_move_pending})
    wandb.log({"player1.roadbuilding_move_pending": player1.roadbuilding_move_pending})
    wandb.log({"player1.yearofplenty_move_pending": player1.yearofplenty_move_pending})
    wandb.log({"player1.discard_resources_started": player1.discard_resources_started})
    wandb.log({"player1.monopoly_cards_old": player1.monopoly_cards_old})
    # Log to Weights & Biases
    wandb.log({"Function Call Counts": wandb.Plotly(fig)})
    
    #for i in range (len(action_counts)):
    #    wandb.log({f"Action {i-1}": action_counts[i-1]})
    #
    #for i in range (len(random_action_counts)):
    #    wandb.log({f"Random Action {i-1}": random_action_counts[i-1]})


@count_calls
def optimize_model():
    print(len(memory))
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

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    start_time = time.time()
    loss.backward()
    final_time = time.time() - start_time
    print(final_time)

    optimizer.step()

start_time = time.time()


num_episodes = 1000
for i_episode in range (num_episodes):
    print(i_episode)
    if i_episode % 100 == 99:
        if player0.wins > 50:
            torch.save(agent1_policy_net.state_dict(), 'agent{num_episodes}_policy_net.pth')
            agent2_policy_net.load_state_dict(torch.load('agent{num_episodes}_policy_net.pth'))
        player0.wins = 0
    

    for t in count():
        cur_boardstate =  state_changer()[0]
        cur_vectorstate = state_changer()[1]
        cur_boardstate = torch.tensor(cur_boardstate, device = device, dtype = torch.float).unsqueeze(0)
        cur_vectorstate = torch.tensor(cur_vectorstate, device = device, dtype = torch.float).unsqueeze(0)

        
        if game.cur_player == 1:
            action = select_action(cur_boardstate, cur_vectorstate)
            if phase.statechange == 1:
                #calculate reward and check done
                next_board_state, next_vector_state, reward, done = state_changer()[0], state_changer()[1], phase.reward, game.is_finished  #[this is were I need to perform an action and return the next state, reward, done
                reward = torch.tensor([reward], device = device)
                next_board_state = torch.tensor(next_board_state, device = device, dtype = torch.float).unsqueeze(0)
                next_vector_state = torch.tensor(next_vector_state, device = device, dtype = torch.float).unsqueeze(0)

                if done:
                    next_board_state = None
                    next_vector_state = None
                cur_boardstate = next_board_state
                cur_vector_state = next_vector_state
        elif game.cur_player == 0:
            action = select_action(cur_boardstate, cur_vectorstate)
            #calculate reward and check done
            if phase.statechange == 1:
                next_board_state, next_vector_state, reward, done = state_changer()[0], state_changer()[1], phase.reward, game.is_finished  #[this is were I need to perform an action and return the next state, reward, done
                reward = torch.tensor([reward], device = device)
                next_board_state = torch.tensor(next_board_state, device = device, dtype = torch.float).unsqueeze(0)
                next_vector_state = torch.tensor(next_vector_state, device = device, dtype = torch.float).unsqueeze(0)
                if done:
                    next_board_state = None
                    next_vector_state = None
                memory.push(cur_boardstate, cur_vectorstate,action,next_board_state, next_vector_state,reward)
                cur_boardstate = next_board_state
                cur_vectorstate = next_vector_state
                optimize_model()

                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = agent1_policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = TAU*policy_net_state_dict[key] + (1-TAU)*target_net_state_dict[key]
                target_net.load_state_dict(target_net_state_dict)
                if done:
                    episode_durations.append(t+1)
                    break
        steps_done += phase.statechange
        phase.statechangecount += phase.statechange
        phase.statechange = 0
            

        
        if t % 10000 == 0:
            a = int(t/100)
            log()
            elapsed_time = time.time() - start_time
            wandb.log({"Elapsed Time": elapsed_time}, step=t)
            wandb.log({"t": t})
            #print(t)
            #print(player0.victorypoints)
            #print(player1.victorypoints)
print('Complete')


#might add more than 1 training agent later on 
    





