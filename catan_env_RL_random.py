import numpy as np
import random
import math 

import time
from itertools import product

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
        self.prettyboard = np.zeros((_NUM_ROWS, _NUM_COLS))
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

        #harbors
        self.harbors_possible = np.zeros((9, 2, 2))

        #rewards
        self.rewards_possible_player0 = np.zeros((_NUM_ROWS,_NUM_COLS))
        self.rewards_possible_player1 = np.zeros((_NUM_ROWS,_NUM_COLS))

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
        self.ressource_lumber = 0
        self.ressource_wool = 0
        self.ressource_grain = 0
        self.ressource_brick = 0
        self.ressource_ore = 0

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

        self.victorypoints = 0

        self.development_card_played = 0
        self.knight_move_pending = 0
        self.monopoly_move_pending = 0
        self.roadbuilding_move_pending = 0
        self.yearofplenty_move_pending = 0

        #__________________game-specific ressource_____________
        #roads
        self.roads_possible = np.zeros((_NUM_ROWS, _NUM_COLS))

        #rewards 
        self.rewards_possible = np.zeros((_NUM_ROWS,_NUM_COLS))




    
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


            #Which ressources do you want to take (Chooses twice)
        
            self.yearofplenty_lumber = 0
            self.yearofplenty_wool = 0
            self.yearofplenty_grain = 0
            self.yearofplenty_brick = 0
            self.yearofplenty_ore = 0

            #Which ressource do you want to take when playing monopoly
            self.monopoly_lumber = 0
            self.monopoly_wool = 0
            self.monopoly_grain = 0
            self.monopoly_brick = 0
            self.monopoly_ore = 0
            
        class KeepRessources:
            def __init__(self):
                self.keep_4x_lumber = 0
                self.keep_4x_wool = 0
                self.keep_4x_grain = 0
                self.keep_4x_brick = 0
                self.keep_4x_ore = 0

                self.keep_3x_lumber_1x_wool = 0
                self.keep_3x_lumber_1x_grain = 0
                self.keep_3x_lumber_1x_brick = 0
                self.keep_3x_lumber_1x_ore = 0
                self.keep_3x_wool_1x_lumber = 0
                self.keep_3x_wool_1x_grain = 0
                self.keep_3x_wool_1x_brick = 0
                self.keep_3x_wool_1x_ore = 0
                self.keep_3x_grain_1x_lumber = 0
                self.keep_3x_grain_1x_grain = 0
                self.keep_3x_grain_1x_brick = 0
                self.keep_3x_grain_1x_ore = 0
                self.keep_3x_brick_1x_lumber = 0
                self.keep_3x_brick_1x_wool = 0
                self.keep_3x_brick_1x_grain = 0
                self.keep_3x_brick_1x_ore = 0
                self.keep_3x_ore_1x_lumber = 0
                self.keep_3x_ore_1x_wool = 0
                self.keep_3x_ore_1x_grain = 0
                self.keep_3x_ore_1x_brick = 0

                self.keep_2x_lumber_2x_wool = 0
                self.keep_2x_lumber_2x_grain = 0
                self.keep_2x_lumber_2x_brick = 0
                self.keep_2x_lumber_2x_ore = 0
                self.keep_2x_wool_2x_grain = 0
                self.keep_2x_wool_2x_brick = 0
                self.keep_2x_wool_2x_ore = 0
                self.keep_2x_grain_2x_brick = 0
                self.keep_2x_grain_2x_ore = 0
                self.keep_2x_brick_2x_ore = 0

                self.keep_2x_lumber_1x_wool_1x_grain = 0
                self.keep_2x_lumber_1x_wool_1x_brick = 0
                self.keep_2x_lumber_1x_wool_1x_ore = 0
                self.keep_2x_lumber_1x_grain_1x_brick = 0
                self.keep_2x_lumber_1x_grain_1x_ore = 0
                self.keep_2x_lumber_1x_brick_1x_ore = 0
                
                self.keep_2x_wool_1x_lumber_1x_grain = 0
                self.keep_2x_wool_1x_lumber_1x_brick = 0
                self.keep_2x_wool_1x_lumber_1x_ore = 0
                self.keep_2x_wool_1x_grain_1x_brick = 0
                self.keep_2x_wool_1x_grain_1x_ore = 0
                self.keep_2x_wool_1x_brick_1x_ore = 0
                
                self.keep_2x_grain_1x_lumber_1x_wool = 0
                self.keep_2x_grain_1x_lumber_1x_brick = 0
                self.keep_2x_grain_1x_lumber_1x_ore = 0
                self.keep_2x_grain_1x_wool_1x_brick = 0
                self.keep_2x_grain_1x_wool_1x_ore = 0
                self.keep_2x_grain_1x_brick_1x_ore = 0
                
                self.keep_2x_brick_1x_lumber_1x_wool = 0
                self.keep_2x_brick_1x_lumber_1x_grain = 0
                self.keep_2x_brick_1x_lumber_1x_ore = 0
                self.keep_2x_brick_1x_wool_1x_grain = 0
                self.keep_2x_brick_1x_wool_1x_ore = 0
                self.keep_2x_brick_1x_grain_1x_ore = 0
                
                self.keep_2x_ore_1x_lumber_1x_wool = 0
                self.keep_2x_ore_1x_lumber_1x_grain = 0
                self.keep_2x_ore_1x_lumber_1x_brick = 0
                self.keep_2x_ore_1x_wool_1x_grain = 0
                self.keep_2x_ore_1x_wool_1x_brick = 0
                self.keep_2x_ore_1x_grain_1x_brick = 0
                
                self.keep_1x_lumber_1x_wool_1x_grain_1x_brick = 0
                self.keep_1x_lumber_1x_wool_1x_grain_1x_ore = 0
                self.keep_1x_lumber_1x_wool_1x_brick_1x_ore = 0
                self.keep_1x_lumber_1x_grain_1x_brick_1x_ore = 0
                self.keep_1x_wool_1x_grain_1x_brick_1x_ore = 0    
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
    
#phase
class Phase:
    def __init__(self):
        #___________________input___________________
        self.development_card_placed = 0
        self.knight_move = 0
        self.using_yearofplenty = 0
        self.using_roadbuilding = 0

class Random: 
    def __init__(self):
        self.random_action = 0
        self.random_position_x = 0
        self.random_position_y = 0
        

#config Variables
class Game: 
    def __init__(self):
        self.cur_player = 0
        self.is_finished = False
        self.terminal = False
        self.testing = True
        self.settlementplaced = 0
        self.placement_phase_pending = 0
        self.seven_rolled = 0

board = Board()
distribution = Distribution()
player0 = Player()
player1 = Player()
player0_action = player0.Action()
player1_action = player1.Action()
players_action = [player0_action, player1_action]
players = [player0,player1]
game = Game()
phase = Phase()
action = Player.Action()
keepressources = Player.Action.KeepRessources()
trading = Player.Action.Trading()

random_agent = Random()


class Color:
    RED = "\u001b[31m"
    BLUE = "\u001b[34m"
    WHITE_CYAN = "\u001b[37;46m"
    RED_YELLOW = "\u001b[31;43m"
    PINK = "\u001b[38;5;201m"
    LAVENDER = "\u001b[38;5;147m"
    AQUA = "\u001b[38;2;145;231;255m"
    PENCIL = "\u001b[38;2;253;182;0m"
    RESET1 = "\u001b[48;2;0;0;0m"
    BEIGE = "\u001b[48;2;210;202;85m"
    GREY = "\u001b[48;2;145;145;145m"
    DARKGREEN = "\u001b[48;2;0;102;0m"
    LIGHTGREEN = "\u001b[48;2;51;255;51m"
    YELLOW = "\u001b[48;2;255;255;0"
    DARKBEIGE = "\u001b[48;2;102;102;0m"
    BROWN = "\u001b[48;2;102;51;0m"
    RESET2 = "\u001b[38;2;255;255;255m"

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
    
def tiles_buidling():
    for i in range(1,10,2):
        for j in range(2 + abs(5-i),20 - abs(5-i),4):
            board.TILES_POSSIBLE[i][j] = 1

def settlements_building():
    for i in range(0,11,2):
        for j in range(-1 + abs(5-i),23 - abs(5-i),2):
            board.settlements_available[i][j] = 1  
 

def roads_building():
    for i in range(0,10,1):
        for j in range(0,20,1):
            if board.settlements_available[i + 1][j] == 1 and board.settlements_available[i - 1][j] == 1:
                board.roads_available[i][j] = 1
            if board.settlements_available[i + 1][j + 1] == 1 and board.settlements_available[i - 1][j + 1] == 1:
                board.roads_available[i][j+1] = 1
            
            if board.settlements_available[i][j + 1] == 1 and board.settlements_available[i][j - 1] == 1:
                board.roads_available[i][j] = 1
            if board.settlements_available[i + 1][j + 1] == 1 and board.settlements_available[i + 1][j - 1] == 1:
                board.roads_available[i+1][j] = 1

    board.roads_available = board.roads_available*(1-board.TILES_POSSIBLE)
            
def tile_distribution(): #fix tile distribution
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
            
def plate_distribution():
    a = 0
    for i in range (1,11,1):
        for j in range (1,21,1):
            if board.TILES_POSSIBLE[i][j] == 1 and board.rober_position[i][j] != 0: #is there a desert here
                board.tiles_dice[i][j] = distribution.plate_random_numbers[a-1]
                board.tiles_dice_probabilities[i][j] = 6-abs(7-board.tiles_dice[i][j])
                if board.tiles_dice_probabilities == 1:
                    board.tiles_probability_1[i][j] = 1
                if board.tiles_dice_probabilities == 2:
                    board.tiles_probability_2[i][j] = 1
                if board.tiles_dice_probabilities == 3:
                    board.tiles_probability_3[i][j] = 1
                if board.tiles_dice_probabilities == 4:
                    board.tiles_probability_4[i][j] = 1
                if board.tiles_dice_probabilities == 5:
                    board.tiles_probability_5[i][j] = 1
                a += 1

def development_card_choose():
    
    player = players[game.cur_player]

    if distribution.development_card_random_number[distribution.development_cards_bought] == 1:
        player.knight_cards_new += 1 
    if distribution.development_card_random_number[distribution.development_cards_bought] == 2:
        player.victorypoints_cards_new += 1 
        player.victorypoints += 1
    if distribution.development_card_random_number[distribution.development_cards_bought] == 3:
        player.yearofplenty_cards_new += 1 
    if distribution.development_card_random_number[distribution.development_cards_bought] == 4:
        player.monopoly_cards_new += 1 
    if distribution.development_card_random_number[distribution.development_cards_bought] == 5:
        player.roadbuilding_cards_new += 1 
        
    distribution.development_cards_bought += 1

    return 1

def tile_update_rewards(a,b):
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
        if db < 0 and (b == 0 or b == 1):
            continue
        if db > 0 and (b == 20 or b == 19): 
            continue

        x = da + a
        y = db + b
        player.rewards_possible[x][y] += 1
    player.rewards_possible = player.rewards_possible * board.TILES_POSSIBLE

def settlement_place(a,b):
    player = players[game.cur_player]
    if settlement_possible_check(a,b,0) == 1:
        player.settlements[a][b] = 1
        tile_update_rewards(a,b)
        player.victorypoints += 1
        return 1 
    return 0

def settlement_place_placement(a,b):
    player = players[game.cur_player]
    board.settlements_used = (1-player0.settlements)*(1-player1.settlements)
    board.settlements_free = board.settlements_available * board.settlements_used
    if board.settlements_free[a,b] == 1 and settlement_possible_check(a,b,1) == 1:
        player.settlements[a][b] = 1
        tile_update_rewards(a,b)
        player.victorypoints += 1
        return 1 
    return 0

def settlement_possible_check(a,b,c):

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
        if board.settlements_free[x][y] == 0 and board.settlements_available[x][y] == 1:
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
                      
def road_place(a,b):
    player = players[game.cur_player]
    possible = 0
    possible = road_possible_check(a,b)
    if possible == 1:
        player.roads[a][b] = 1
        update_longest_road()
        return 1 
    return 0
def road_place_placement(settlement_a,settlement_b,road_a,road_b):
    player = players[game.cur_player]
    if ((((road_a + 1) == settlement_a or (road_a - 1)  == settlement_a) and road_b == settlement_b) or (((road_b + 1) == settlement_b or (road_b - 1)  == settlement_b) and road_a == settlement_a)):
        player.roads[road_a][road_b] = 1
        update_longest_road()
        return 1 
    return 0
def road_possible_check(a,b):
    board.roads_available = board.roads_available * (1 - player0.roads) * (1 - player1.roads)
    player = players[game.cur_player]
    opponent = players[1-game.cur_player]    
    #I could work with boards and multiply them at the end to check 
    player.roads_possible = (1-board.ZEROBOARD)
        

    if b != 20 and opponent.settlements[a][b + 1] == 1:
        if b != 19:
            player.roads_possible[a][b+2] = 0
        if a != 0:
            player.roads_possible[a-1][b+1] = 0
        if a != 10: 
            player.roads_possible[a+1][b+1] = 0
    if b != 0 and opponent.settlements[a][b - 1] == 1:
        if b != 1:
            player.roads_possible[a][b-2] = 0
        if a != 0:
            player.roads_possible[a-1][b-1] = 0
        if a != 10: 
            player.roads_possible[a+1][b-1] = 0
    if a != 10 and opponent.settlements[a+1][b] == 1:
        if a != 9:
            player.roads_possible[a+2][b] = 0
        if b != 0:
            player.roads_possible[a+1][b-1] = 0
        if b != 20: 
            player.roads_possible[a+1][b+1] = 0
    if a != 0 and opponent.settlements[a-1][b] == 1:
        if a != 1:
            player.roads_possible[a-2][b] = 0
        if b != 0:
            player.roads_possible[a-1][b-1] = 0
        if b != 20: 
            player.roads_possible[a-1][b+1] = 0

    neighboring_roads = [(0,2),(0,-2),(2,0),(-2,0),(1,1),(1,-1),(-1,1),(-1,-1)]

    for da,db in neighboring_roads:
        
        if a == 10 and da > 0:
            da = 0
        elif a == 9 and da > 1: 
            da = 0
        elif a == 0 and da < 0:
            da = 0
        elif a == 1 and da < -1: 
            da = 0
        
        if b == 20 and db > 0:
            db = 0
        elif b == 19 and db > 1: 
            db = 0
        elif b == 0 and db < 0:
            db = 0
        elif b == 1 and db < -1: 
            db = 0

        x = a + da
        y = b + db

        if player.roads[x][y] == 1 and player.roads_possible[x][y] == 1:
            return 1
    return 0 

def city_place(a,b):
    #still need to add a max cities check, the same comes to settlements
    player = players[game.cur_player]

    if player.settlements[a][b] == 1:
        player.cities[a][b] = 1
        player.settlements[a][b] = 0
        tile_update_rewards()
        player.victorypoints += 1
        return 1
    return 0 

def roll_dice(): 

    roll = np.random.choice(np.arange(2, 13), p=[1/36,2/36,3/36,4/36,5/36,6/36,5/36,4/36,3/36,2/36,1/36])

    for i in range (0,11,1):
        for j in range(0,21,1):
            if board.tiles_dice[i][j] == roll:
                #I could probably also multiply it with the ressource cards and add them up (would be much faster)
                if player0.rewards_possible[i][j] != 0:
                    if board.tiles_lumber[i][j] == 1:
                        player0.ressource_lumber += player0.rewards_possible[i][j]
                    
                    if board.tiles_wool[i][j] == 1:
                        player0.ressource_wool += player0.rewards_possible[i][j]
                    
                    if board.tiles_grain[i][j] == 1:
                        player0.ressource_grain += player0.rewards_possible[i][j]
                    
                    if board.tiles_brick[i][j] == 1:
                        player0.ressource_brick += player0.rewards_possible[i][j]
                    
                    if board.tiles_ore[i][j] == 1:
                        player0.ressource_ore += player0.rewards_possible[i][j]

                if player1.rewards_possible[i][j] != 0:

                    if board.tiles_lumber[i][j] == 1:
                        player1.ressource_lumber += player1.rewards_possible[i][j]
                    
                    if board.tiles_wool[i][j] == 1:
                        player1.ressource_wool += player1.rewards_possible[i][j]
                    
                    if board.tiles_grain[i][j] == 1:
                        player1.ressource_grain += player1.rewards_possible[i][j]
                    
                    if board.tiles_brick[i][j] == 1:
                        player1.ressource_brick += player1.rewards_possible[i][j]
                    
                    if board.tiles_ore[i][j] == 1:
                        player1.ressource_ore += player1.rewards_possible[i][j]
    return roll

def buy_development_cards():
    player = players[game.cur_player]
    possible = 0
    if player.ressource_wool > 0 and player.ressource_grain > 0 and player.ressource_ore > 0 and distribution.development_cards_bought != 25:
        possible = development_card_choose()
        if possible == 1:
            find_largest_army()
            player.ressource_wool -= 1
            player.ressource_grain -= 1 
            player.ressource_ore -= 1 
            return 1
    return 0 
        


def buy_road(a,b):
    possible = 0
    player = players[game.cur_player]
    if player.ressource_brick > 0 and player.ressource_lumber > 0:
            possible = road_place(a,b)
            if possible == 1:
                player.ressource_brick -= 1
                player.ressource_lumber -= 1
                return 1
    return 0 


def buy_settlement(a,b):
    player = players[game.cur_player]
    possible = 0

    if player.ressource_brick > 0 and player.ressource_lumber > 0 and player.ressource_grain > 0 and player.ressource_ore > 0:
        possible = settlement_place(a,b)
        if possible == 1:
            player.ressource_lumber -= 1
            player.ressource_brick -= 1
            player.ressource_brick -= 1 
            player.ressource_grain -= 1
            return 1 
    return 0 
            
def buy_city(a,b):
    player = players[game.cur_player]
    possible = 0

    if player.ressource_grain > 1 and player.ressource_ore > 2:
        possible = city_place(a,b)
        if possible == 1:
            player.ressource_grain -= 2
            player.ressource_ore -= 3  
            return 1
    return 0 

def steal_card():
    player = players[game.cur_player]
    opponent = players[1-game.cur_player]
    
    opponent_ressources_total = opponent.ressource_lumber + opponent.ressource_brick + opponent.ressource_wool + opponent.ressource_grain + opponent.ressource_ore
    if opponent_ressources_total!= 0:
        random_ressource = np.random.choice(np.arange(1, 6), p=[opponent.ressource_lumber/opponent_ressources_total, opponent.ressource_brick/opponent_ressources_total, opponent.ressource_wool/opponent_ressources_total, opponent.ressource_grain/opponent_ressources_total, opponent.ressource_ore/opponent_ressources_total])
        if random_ressource == 1:
            opponent.ressource_lumber = opponent.ressource_lumber - 1
            player.ressource_lumber = player.ressource_lumber + 1
        if random_ressource == 2:
            opponent.ressource_brick = opponent.ressource_brick - 1
            player.ressource_brick = player.ressource_brick + 1
        if random_ressource == 3:
            opponent.ressource_wool = opponent.ressource_wool - 1
            player.ressource_wool = player.ressource_wool + 1
        if random_ressource == 4:
            opponent.ressource_grain = opponent.ressource_grain - 1
            player.ressource_grain = player.ressource_grain + 1
        if random_ressource == 5:
            opponent.ressource_ore = opponent.ressource_ore - 1
            player.ressource_ore = player.ressource_ore + 1

def play_knight(a,b):
    player = players[game.cur_player]
    possible = 0
    if player.knight_cards_new > 0:
        possible = move_rober(a,b)
        if possible == 1:
            steal_card()
            player.knight_cards_new -= 1
            player.knight_cards_played += 1
            return 1
        return 0 
    return 0

def move_rober(a,b):
    if board.rober_position[a][b] != 1 and board.TILES_POSSIBLE[a][b] == 1:
        board.rober_position = board.rober_position * board.ZEROBOARD
        board.rober_position[a][b] = 1
        player0.rewards_possible = player0.rewards_possible * (1 - board.rober_position)
        player1.rewards_possible = player1.rewards_possible * (1 - board.rober_position)
        return 1
    return 0

def activate_yearofplenty_func(ressource1,ressource2):
    #need to take a look at this later. I'm not sure how to comvert those ressources. 
    player = players[game.cur_player]
    if player.yearofplenty_cards_old > 0:
        player.yearofplenty_cards_old = player.yearofplenty_cards_old - 1 
        if ressource1 == 1:
            player.ressource_lumber += 1
        if ressource1 == 1:
            player.ressource_lumber = player.ressource_lumber + 1
        if ressource1 == 2:
            player.ressource_brick = player.ressource_brick + 1
        if ressource1 == 3:
            player.ressource_wool = player.ressource_wool + 1
        if ressource1 == 4:
            player.ressource_grain = player.ressource_grain + 1
        if ressource1 == 5:
            player.ressource_ore = player.ressource_ore + 1
        if ressource2 == 1:
            player.ressource_lumber = player.ressource_lumber + 1
        if ressource2 == 2:
            player.ressource_brick = player.ressource_brick + 1
        if ressource2 == 3:
            player.ressource_wool = player.ressource_wool + 1
        if ressource2 == 4:
            player.ressource_grain = player.ressource_grain + 1
        if ressource2 == 5:
            player.ressource_ore = player.ressource_ore + 1
        return 1 
    return 0 

def activate_monopoly_func(ressource):
    player = players[game.cur_player]
    opponent = players[1-game.cur_player]
    if player.monopoly_cards_old > 0:
        player.monopoly_cards_old = player.monopoly_cards_old - 1
        if ressource == 1:
            player.ressource_lumber = player.ressource_lumber + opponent.ressource_lumber
            opponent.ressource_lumber = 0
        if ressource == 2:
            player.ressource_wool = player.ressource_wool + opponent.ressource_wool
            opponent.ressource_wool = 0
        if ressource == 3:
            player.ressource_grain = player.ressource_grain + opponent.ressource_grain
            opponent.ressource_grain = 0
        if ressource == 4:
            player.ressource_brick = player.ressource_brick + opponent.ressource_brick
            opponent.ressource_brick = 0
        if ressource == 5:
            player.ressource_ore = player.ressource_ore + opponent.ressource_ore
            opponent.ressource_ore = 0
    
def activate_road_building_func(a1,b1,a2,b2):
    player = players[game.cur_player]
    if player.roadbuilding_cards_old > 0:
        
        
        possible1 = road_possible_check(a1,b1)
        possible2 = road_possible_check(a2,b2)
        if possible1 == 1 and possible2 == 1:
            player.roadbuilding_cards_old = player.roadbuilding_cards_old - 1
            road_place(a1,b1)
            road_place(a2,b2)
            return 1 
        return 0 
    return 0 
    
def trade_ressources(give, get):
    player = players[game.cur_player]
    if give == player.ressource_brick and (board.harbor_brick * player.settlements + board.harbor_brick * player.cities).any() != 0:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player.ressource_lumber and (board.harbor_lumber * player.settlements + board.harbor_lumber * player.cities).any() != 0:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player.ressource_wool and (board.harbor_wool * player.settlements + board.harbor_wool * player.cities).any() != 0:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player.ressource_grain and (board.harbor_grain * player.settlements + board.harbor_grain * player.cities).any() != 0:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player.ressource_ore and (board.harbor_ore * player.settlements + board.harbor_ore * player.cities).any() != 0:
        if give < 1:
            give -= 2 
            get += 1 
    elif (board.harbor_three_one * player.settlements + board.harbor_three_one * player.cities).any() != 0:
        if give < 2:
            give -= 3 
            get += 1
    elif give < 3:
        give -= 4
        get += 1

def discard_ressources(lumber, wool, grain, brick, ore):
    player = players[game.cur_player]
    total_ressources = player.ressource_lumber + player.ressource_brick + player.ressource_grain + player.ressource_ore + player.ressource_wool 
    player.ressource_lumber = lumber
    player.ressource_wool = wool
    player.ressource_grain = grain
    player.ressource_brick = brick
    player.ressource_ore = ore

    #remove ressource
    for i in range(0,math.ceil(total_ressources/2)-4,1):
        randomly_pick_ressources()

def randomly_pick_ressources():

    #this is a mixture and not correct
    player = players[game.cur_player]
    possible_ressources_left = np.ones((5)) #if there are still one of those ressources available after picking the first four
    if player.ressource_lumber != 0:
        possible_ressources_left[0] = 1
    if player.ressource_wool != 0:
        possible_ressources_left[1] = 1
    if player.ressource_grain != 0:
        possible_ressources_left[2] = 1
    if player.ressource_brick != 0:
        possible_ressources_left[3] = 1
    if player.ressource_ore != 0:
        possible_ressources_left[4] = 1
    
    numbers = np.random.choice(np.arange(1,6),p = [possible_ressources_left[0]/possible_ressources_left.sum(),possible_ressources_left[1]/possible_ressources_left.sum(),possible_ressources_left[2]/possible_ressources_left.sum(),possible_ressources_left[3]/possible_ressources_left.sum(),possible_ressources_left[4]/possible_ressources_left.sum()])
    if numbers == 1:
        player.ressource_lumber += 1 
    if numbers == 2:
        player.ressource_brick += 1 
    if numbers == 3:
        player.ressource_wool += 1 
    if numbers == 4:
        player.ressource_grain += 1 
    if numbers == 5:
        player.ressource_ore += 1 

def trav(i,j, a):
    #it might be an idea to use this for other function too, but let's see
    n = 11 
    m = 21
    a += 1
    if i < 0 or j < 0 or i >= n or j >= m or board.longest_road[i][j] == 0: return 0
    board.longest_road[i][j] = 0
    board.increasing_roads[i][j] = a
    return 1 + trav(i+2, j,a) + trav(i+1, j+1,a) + trav(i+1, j-1,a) + trav(i, j+2,a) + trav(i-1, j+1,a) + trav(i-2, j,a) + trav(i-1, j-1,a) + trav(i, j-2,a) 
def check_longest_road(i,j,f,g):
    n = 11
    m = 21
    if i < 0 or j < 0 or i >= n or j >= m or board.increasing_roads [i][j] == 0:
        return 0
    if  board.increasing_roads[i][j] <= board.increasing_roads[f][g]:
        return 0
    f = i 
    g = j
    return 1 + check_longest_road(i+2, j,f,g) + check_longest_road(i+1, j+1,f,g) + check_longest_road(i+1, j-1,f,g) + check_longest_road(i, j+2,f,g) + check_longest_road(i-1, j+1,f,g) + check_longest_road(i-2, j,f,g) + check_longest_road(i-1, j-1,f,g) + check_longest_road(i, j-2,f,g) 

    

def find_longest_road():
    player = players[game.cur_player]
    ans, n, m = 0, 11, 21
    for i in range(n):
        for j in range(m):
            board.increasing_roads = board.ZEROBOARD * board.ZEROBOARD
            board.longest_road = player.roads * (1-board.ZEROBOARD)
            a = 0
            c = trav(i, j, a)
            if c > 0: 
                ans = max(ans,check_longest_road(i,j,0,0))
    return ans

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

def find_largest_army():
    player = players[game.cur_player]
    opponent = players[1 - game.cur_player]
    if player.knight_cards_played >= 3 and player.knight_cards_played > opponent.knight_cards_played and player.largest_army == 0:
        if opponent.largest_army == 1:
            opponent.largest_army = 0
            opponent.victorypoints -= 2 
    
def move_finished():

    player = players[game.cur_player]

    player.knight_cards_new += player.knight_cards_new
    player.victorypoints_cards_old += player.victorypoints_cards_new
    player.yearofplenty_cards_old += player.yearofplenty_cards_new
    player.monopoly_cards_old += player.monopoly_cards_new
    player.roadbuilding_cards_old += player.roadbuilding_cards_new

    player.knight_cards_new = 0
    player.victorypoints_cards_new = 0 
    player.yearofplenty_cards_new = 0
    player.monopoly_cards_new = 0
    player.roadbuilding_cards_new = 0 

    if player.victorypoints >= 10:
        game.is_finished = 1

    game.cur_player = 1 - game.cur_player

def new_initial_state():
    #board

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
    

    board.TILES_POSSIBLE = np.zeros((_NUM_ROWS, _NUM_COLS))

    board.tiles_dice = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_dice_probabilities = np.zeros((_NUM_ROWS, _NUM_COLS))

    board.settlements_free = np.ones((_NUM_ROWS, _NUM_COLS))
    board.settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.settlements_used = np.zeros((_NUM_ROWS, _NUM_COLS))

    board.roads_available = np.zeros((_NUM_ROWS, _NUM_COLS))

    board.harbors_possible = np.zeros((9, 2, 2))

    board.rewards_possible_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.rewards_possible_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))
    

    distribution.harbor_random_numbers = np.random.choice(distribution.harbor_numbers,9,replace=False)
    distribution.tile_random_numbers = np.random.choice(distribution.tile_numbers,19,replace=False)
    distribution.development_card_random_number = np.random.choice(distribution.development_card_numbers,25,replace=False)
    distribution.development_cards_bought = 0

    phase.development_card_placed = 0
    phase.knight_move = 0
    phase.using_yearofplenty = 0
    phase.using_roadbuilding = 0


    for player in players:
        player.settlements = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.settlements_left = 5

        player.roads = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.roads_left = 15

        player.cities = np.zeros((_NUM_ROWS, _NUM_COLS))       
        player.cities_left = 4


        player.ressource_lumber = 0
        player.ressource_wool = 0
        player.ressource_grain = 0
        player.ressource_brick = 0
        player.ressource_ore = 0

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

        player.rewards_possible = np.zeros((_NUM_ROWS,_NUM_COLS))

        player.roads_possible = np.zeros((_NUM_ROWS, _NUM_COLS))


def agent_place_placement_phase():
    player = players[game.cur_player]
    possible = 0
    a = 0
    b = 0
    while possible == 0:
        a = 0
        b = 0
        possible = settlement_place_placement(a-1,b-1)
        
    c = 0
    d = 0
    possible = 0
    while possible == 0:
        c = 0
        d = 0
        possible = road_place_placement(a-1,b-1,c-1,d-1)

def placement_phase():
    game.cur_player = 0
    agent_place_placement_phase()    
    game.cur_player = 1
    agent_place_placement_phase()
    agent_place_placement_phase()
    game.cur_player = 0
    agent_place_placement_phase()
    


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
    c = roll_dice()
    player = players[game.cur_player]
    opponent = players[1 - game.cur_player]
    

    if c == 7:
        d = 0
        e = 0
        discard_ressources()
        move_rober(d,e)
        steal_card()
            

def game_episode():
    while game.is_finished == 0:
        turn_starts()
        move_finished()

def main():
    start()
    new_initial_state()
    setup()
    placement_phase()
    game_episode()

def action_executor():
    player = players[game.cur_player]

    if player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and game.seven_rolled != 1:
        if np.any(action.rober_move == 1): 
                if player.knight_move_pending == 0:
                    b,c = np.where(action.rober_move == 1)
                    d = int(b)
                    e = int(c)
                    if d < 11 and d >= 0 and e < 21 and e >= 0: 
                        move_rober(d,e)
                else:
                    b,c = np.where(action.rober_move == 1)
                    d = int(b)
                    e = int(c)
                    if d < 11 and d >= 0 and e < 21 and e >= 0: 
                        play_knight(d,e)


    if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.seven_rolled != 1:
        if game.placement_phase_pending != 1:
            if np.any(action.road_place == 1):
                    b,c = np.where(action.road_place == 1)
                    d = int(b)
                    e = int(c)
                    if d < 11 and d >= 0 and e < 21 and e >= 0: 
                        possible = buy_road(d,e)
                        if possible == 1:
                            player.roadbuilding_move_pending = 0
        if game.placement_phase_pending != 1:
            if np.any(action.road_place == 1):
                        b,c = np.where(action.road_place == 1)
                        d = int(b)
                        e = int(c)
                        if d < 11 and d >= 0 and e < 21 and e >= 0: 
                            possible = road_place_placement(d,e)
                            if possible == 1:
                                player.roadbuilding_move_pending = 0
                


    if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.seven_rolled != 1:
        if game.placement_phase_pending != 1:
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
            if action.end_turn == 1:
                move_finished()
        if game.placement_phase_pending == 1:
            if np.any(action.settlement_place == 1):
                b,c = np.where(action.settlement_place == 1)
                d = int(b)
                e = int(c)
                if d < 11 and d >= 0 and e < 21 and e >= 0: 
                    settlement_place_placement(d,e)

    
    if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1:
        if keepressources.keep_4x_lumber == 1: 
            discard_ressources(4,0,0,0,0)
        if keepressources.keep_4x_wool == 1: 
            discard_ressources(0,4,0,0,0)
        if keepressources.keep_4x_grain == 1: 
            discard_ressources(0,0,4,0,0)
        if keepressources.keep_4x_brick == 1: 
            discard_ressources(0,0,0,4,0)
        if keepressources.keep_4x_ore == 1: 
            discard_ressources(0,0,0,0,4)
        if keepressources.keep_3x_lumber_1x_wool == 1: 
            discard_ressources(3,1,0,0,0)
        if keepressources.keep_3x_lumber_1x_grain == 1: 
            discard_ressources(3,0,1,0,0)
        if keepressources.keep_3x_lumber_1x_brick == 1: 
            discard_ressources(3,0,0,1,0)
        if keepressources.keep_3x_lumber_1x_ore == 1: 
            discard_ressources(3,0,0,0,1)
        if keepressources.keep_3x_wool_1x_lumber == 1: 
            discard_ressources(1,3,0,0,0)
        if keepressources.keep_3x_wool_1x_grain == 1: 
            discard_ressources(0,3,1,0,0)
        if keepressources.keep_3x_wool_1x_brick == 1: 
            discard_ressources(0,3,0,1,0)
        if keepressources.keep_3x_wool_1x_ore == 1: 
            discard_ressources(0,3,0,0,1)
        if keepressources.keep_3x_grain_1x_lumber == 1: 
            discard_ressources(1,0,3,0,0)
        if keepressources.keep_3x_grain_1x_grain == 1: 
            discard_ressources(0,1,3,0,0)
        if keepressources.keep_3x_grain_1x_brick == 1: 
            discard_ressources(0,0,3,1,0)
        if keepressources.keep_3x_grain_1x_ore == 1: 
            discard_ressources(0,0,3,0,1)
        if keepressources.keep_3x_brick_1x_lumber == 1: 
            discard_ressources(1,0,0,3,0)
        if keepressources.keep_3x_brick_1x_wool == 1: 
            discard_ressources(0,1,0,3,0)
        if keepressources.keep_3x_brick_1x_grain == 1: 
            discard_ressources(0,0,1,3,0)
        if keepressources.keep_3x_brick_1x_ore == 1: 
            discard_ressources(0,0,0,3,1)
        if keepressources.keep_3x_ore_1x_lumber == 1: 
            discard_ressources(1,0,0,0,3)
        if keepressources.keep_3x_ore_1x_wool == 1: 
            discard_ressources(0,1,0,0,3)
        if keepressources.keep_3x_ore_1x_grain == 1: 
            discard_ressources(0,0,1,0,3)
        if keepressources.keep_3x_ore_1x_brick == 1: 
            discard_ressources(0,0,0,1,3)
        if keepressources.keep_2x_lumber_2x_wool == 1: 
            discard_ressources(2,2,0,0,0)
        if keepressources.keep_2x_lumber_2x_grain == 1: 
            discard_ressources(2,0,2,0,0)
        if keepressources.keep_2x_lumber_2x_brick == 1: 
            discard_ressources(2,0,0,2,0)
        if keepressources.keep_2x_lumber_2x_ore == 1: 
            discard_ressources(2,0,0,0,2)
        if keepressources.keep_2x_wool_2x_grain == 1: 
            discard_ressources(0,2,2,0,0)
        if keepressources.keep_2x_wool_2x_brick == 1: 
            discard_ressources(0,2,0,2,0)
        if keepressources.keep_2x_wool_2x_ore == 1: 
            discard_ressources(0,2,0,0,2)
        if keepressources.keep_2x_grain_2x_brick == 1: 
            discard_ressources(0,0,2,2,0)
        if keepressources.keep_2x_grain_2x_ore == 1: 
            discard_ressources(0,0,2,0,2)
        if keepressources.keep_2x_brick_2x_ore == 1: 
            discard_ressources(0,0,0,2,2)
        if keepressources.keep_2x_lumber_1x_wool_1x_grain == 1: 
            discard_ressources(2,1,1,0,0)
        if keepressources.keep_2x_lumber_1x_wool_1x_brick == 1: 
            discard_ressources(2,1,0,1,0)
        if keepressources.keep_2x_lumber_1x_wool_1x_ore == 1: 
            discard_ressources(2,1,0,0,1)
        if keepressources.keep_2x_lumber_1x_grain_1x_brick == 1: 
            discard_ressources(2,0,1,1,0)
        if keepressources.keep_2x_lumber_1x_grain_1x_ore == 1: 
            discard_ressources(2,0,1,0,1)
        if keepressources.keep_2x_lumber_1x_brick_1x_ore == 1:
            discard_ressources(2,0,0,1,1)
        if keepressources.keep_2x_wool_1x_lumber_1x_grain == 1: 
            discard_ressources(1,2,1,0,0)
        if keepressources.keep_2x_wool_1x_lumber_1x_brick == 1: 
            discard_ressources(1,2,0,1,0)
        if keepressources.keep_2x_wool_1x_lumber_1x_ore == 1: 
            discard_ressources(1,2,0,0,1)
        if keepressources.keep_2x_wool_1x_grain_1x_brick == 1: 
            discard_ressources(0,2,1,1,0)
        if keepressources.keep_2x_wool_1x_grain_1x_ore == 1: 
            discard_ressources(0,2,1,0,1)
        if keepressources.keep_2x_wool_1x_brick_1x_ore == 1: 
            discard_ressources(0,2,0,1,1)
        if keepressources.keep_2x_grain_1x_lumber_1x_wool == 1: 
            discard_ressources(1,1,2,0,0)
        if keepressources.keep_2x_grain_1x_lumber_1x_brick == 1: 
            discard_ressources(1,0,2,1,0)
        if keepressources.keep_2x_grain_1x_lumber_1x_ore == 1: 
            discard_ressources(1,0,2,0,1)
        if keepressources.keep_2x_grain_1x_wool_1x_brick == 1: 
            discard_ressources(0,1,2,1,0)
        if keepressources.keep_2x_grain_1x_wool_1x_ore == 1: 
            discard_ressources(0,1,2,0,1)
        if keepressources.keep_2x_grain_1x_brick_1x_ore == 1: 
            discard_ressources(0,0,2,1,1)
        if keepressources.keep_2x_brick_1x_lumber_1x_wool == 1: 
            discard_ressources(1,1,0,2,0)
        if keepressources.keep_2x_brick_1x_lumber_1x_grain == 1: 
            discard_ressources(1,0,1,2,0)
        if keepressources.keep_2x_brick_1x_lumber_1x_ore == 1: 
            discard_ressources(1,0,0,2,1)
        if keepressources.keep_2x_brick_1x_wool_1x_grain == 1: 
            discard_ressources(0,1,1,2,0)
        if keepressources.keep_2x_brick_1x_wool_1x_ore == 1: 
            discard_ressources(0,1,0,2,1)
        if keepressources.keep_2x_brick_1x_grain_1x_ore == 1: 
            discard_ressources(0,0,1,2,1)
        if keepressources.keep_2x_ore_1x_lumber_1x_wool == 1: 
            discard_ressources(1,1,0,0,2)
        if keepressources.keep_2x_ore_1x_lumber_1x_grain == 1: 
            discard_ressources(1,0,1,0,2)
        if keepressources.keep_2x_ore_1x_lumber_1x_brick == 1: 
            discard_ressources(1,0,0,1,2)
        if keepressources.keep_2x_ore_1x_wool_1x_grain == 1: 
            discard_ressources(0,1,1,0,2)
        if keepressources.keep_2x_ore_1x_wool_1x_brick == 1: 
            discard_ressources(0,1,0,1,2)
        if keepressources.keep_2x_ore_1x_grain_1x_brick == 1: 
            discard_ressources(0,0,1,1,2)
        if keepressources.keep_1x_lumber_1x_wool_1x_grain_1x_brick == 1: 
            discard_ressources(1,1,1,1,0)
        if keepressources.keep_1x_lumber_1x_wool_1x_grain_1x_ore == 1: 
            discard_ressources(1,1,1,0,1)
        if keepressources.keep_1x_lumber_1x_wool_1x_brick_1x_ore == 1: 
            discard_ressources(1,1,0,1,1)
        if keepressources.keep_1x_lumber_1x_grain_1x_brick_1x_ore == 1: 
            discard_ressources(1,0,1,1,1)
        if keepressources.keep_1x_wool_1x_grain_1x_brick_1x_ore == 1: 
            discard_ressources(0,1,1,1,1)

    if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and game.seven_rolled != 1:
        if trading.give_lumber_get_wool == 1:
            trade_ressources(1,2)
        if trading.give_lumber_get_grain == 1:
            trade_ressources(1,3)
        if trading.give_lumber_get_brick == 1:
            trade_ressources(1,4)
        if trading.give_lumber_get_ore  == 1:
            trade_ressources(1,5)
        if trading.give_wool_get_lumber == 1:
            trade_ressources(2,1)
        if trading.give_wool_get_grain == 1:
            trade_ressources(2,3)
        if trading.give_wool_get_brick == 1:
            trade_ressources(2,4)
        if trading.give_wool_get_ore == 1:
            trade_ressources(2,5)
        if trading.give_grain_get_lumber == 1:
            trade_ressources(3,1)
        if trading.give_grain_get_wool == 1:
            trade_ressources(3,2)
        if trading.give_grain_get_brick == 1:
            trade_ressources(3,4)
        if trading.give_grain_get_ore == 1:
            trade_ressources(3,5)
        if trading.give_brick_get_lumber == 1:
            trade_ressources(4,1)
        if trading.give_brick_get_wool == 1:
            trade_ressources(4,2)
        if trading.give_brick_get_grain == 1:
            trade_ressources(4,3)
        if trading.give_brick_get_ore == 1:
            trade_ressources(4,5)
        if trading.give_ore_get_lumber == 1:
            trade_ressources(5,1)
        if trading.give_ore_get_wool == 1:
            trade_ressources(5,2)
        if trading.give_ore_get_grain == 1:
            trade_ressources(5,3)
        if trading.give_ore_get_brick == 1:
            trade_ressources(5,4)
    if action.development_card_buy == 1:
        buy_development_cards()
    if action.knight_cards_activate == 1:
        player.knight_move_pending = 1
    if action.yearofplenty_cards_activate == 1:
        player.yearofplenty_move_pending = 1
    if action.monopoly_cards_activate == 1:
        player.monopoly_move_pending = 1
    if action.road_building_cards_activate == 1:
        player.roadbuilding_move_pending = 1
    if action.yearofplenty_lumber_wool == 1:
        activate_yearofplenty_func(1,2)

        
        
def random_assignment():
    #after this assignment I'll try to copy all of them again and say if keepresources.keep_4x_lumber == 1, do this action with these parameters
    player = players[game.cur_player]
    action = players_action[game.cur_player]
    
    if random_agent.random_action == 1:  
        action.rober_move[random_agent.random_position_y] [random_agent.random_position_x] = 1
    if random_agent.random_action == 2:
        action.road_place[random_agent.random_position_y] [random_agent.random_position_x] = 1
    if random_agent.random_action == 3:
        action.settlement_place[random_agent.random_position_y] [random_agent.random_position_x] = 1
    if random_agent.random_action == 4:
        action.city_place[random_agent.random_position_y] [random_agent.random_position_x] = 1
    if random_agent.random_action == 5:
        action.end_turn = 1
    if random_agent.random_action == 6:    
        keepressources.keep_4x_lumber = 1
    if random_agent.random_action == 7:    
        keepressources.keep_4x_wool = 1
    if random_agent.random_action == 8:    
        keepressources.keep_4x_grain = 1
    if random_agent.random_action == 9:    
        keepressources.keep_4x_brick = 1
    if random_agent.random_action == 10:    
        keepressources.keep_4x_ore = 1
    if random_agent.random_action == 11:    
        keepressources.keep_3x_lumber_1x_wool = 1
    if random_agent.random_action == 12:    
        keepressources.keep_3x_lumber_1x_grain = 1
    if random_agent.random_action == 1:    
        keepressources.keep_3x_lumber_1x_brick = 1
    if random_agent.random_action == 13:    
        keepressources.keep_3x_lumber_1x_ore = 1
    if random_agent.random_action == 14:    
        keepressources.keep_3x_wool_1x_lumber = 1
    if random_agent.random_action == 15:    
        keepressources.keep_3x_wool_1x_grain = 1
    if random_agent.random_action == 16:    
        keepressources.keep_3x_wool_1x_brick = 1
    if random_agent.random_action == 17:    
        keepressources.keep_3x_wool_1x_ore = 1
    if random_agent.random_action == 18:    
        keepressources.keep_3x_grain_1x_lumber = 1
    if random_agent.random_action == 19:    
        keepressources.keep_3x_grain_1x_grain = 1
    if random_agent.random_action == 20:    
        keepressources.keep_3x_grain_1x_brick = 1
    if random_agent.random_action == 21:    
        keepressources.keep_3x_grain_1x_ore = 1
    if random_agent.random_action == 22:    
        keepressources.keep_3x_brick_1x_lumber = 1
    if random_agent.random_action == 23:    
        keepressources.keep_3x_brick_1x_wool = 1
    if random_agent.random_action == 24:    
        keepressources.keep_3x_brick_1x_grain = 1
    if random_agent.random_action == 25:    
        keepressources.keep_3x_brick_1x_ore = 1
    if random_agent.random_action == 26:    
        keepressources.keep_3x_ore_1x_lumber = 1
    if random_agent.random_action == 27:    
        keepressources.keep_3x_ore_1x_wool = 1
    if random_agent.random_action == 28:    
        keepressources.keep_3x_ore_1x_grain = 1
    if random_agent.random_action == 29:    
        keepressources.keep_3x_ore_1x_brick = 1
    if random_agent.random_action == 30:    
        keepressources.keep_2x_lumber_2x_wool = 1
    if random_agent.random_action == 31:    
        keepressources.keep_2x_lumber_2x_grain = 1
    if random_agent.random_action == 32:    
        keepressources.keep_2x_lumber_2x_brick = 1
    if random_agent.random_action == 33:    
        keepressources.keep_2x_lumber_2x_ore = 1
    if random_agent.random_action == 34:    
        keepressources.keep_2x_wool_2x_grain = 1
    if random_agent.random_action == 35:    
        keepressources.keep_2x_wool_2x_brick = 1
    if random_agent.random_action == 36:    
        keepressources.keep_2x_wool_2x_ore = 1
    if random_agent.random_action == 37:    
        keepressources.keep_2x_grain_2x_brick = 1
    if random_agent.random_action == 38:    
        keepressources.keep_2x_grain_2x_ore = 1
    if random_agent.random_action == 39:    
        keepressources.keep_2x_brick_2x_ore = 1
    if random_agent.random_action == 40:    
        keepressources.keep_2x_lumber_1x_wool_1x_grain = 1
    if random_agent.random_action == 41:    
        keepressources.keep_2x_lumber_1x_wool_1x_brick = 1
    if random_agent.random_action == 42:    
        keepressources.keep_2x_lumber_1x_wool_1x_ore = 1
    if random_agent.random_action == 43:    
        keepressources.keep_2x_lumber_1x_grain_1x_brick = 1
    if random_agent.random_action == 44:    
        keepressources.keep_2x_lumber_1x_grain_1x_ore = 1
    if random_agent.random_action == 45:    
        keepressources.keep_2x_lumber_1x_brick_1x_ore = 1    
    if random_agent.random_action == 46:    
        keepressources.keep_2x_wool_1x_lumber_1x_grain = 1
    if random_agent.random_action == 47:    
        keepressources.keep_2x_wool_1x_lumber_1x_brick = 1
    if random_agent.random_action == 48:    
        keepressources.keep_2x_wool_1x_lumber_1x_ore = 1
    if random_agent.random_action == 49:    
        keepressources.keep_2x_wool_1x_grain_1x_brick = 1
    if random_agent.random_action == 50:    
        keepressources.keep_2x_wool_1x_grain_1x_ore = 1
    if random_agent.random_action == 51:    
        keepressources.keep_2x_wool_1x_brick_1x_ore = 1   
    if random_agent.random_action == 52:    
        keepressources.keep_2x_grain_1x_lumber_1x_wool = 1
    if random_agent.random_action == 53:    
        keepressources.keep_2x_grain_1x_lumber_1x_brick = 1
    if random_agent.random_action == 54:    
        keepressources.keep_2x_grain_1x_lumber_1x_ore = 1
    if random_agent.random_action == 55:    
        keepressources.keep_2x_grain_1x_wool_1x_brick = 1
    if random_agent.random_action == 56:    
        keepressources.keep_2x_grain_1x_wool_1x_ore = 1
    if random_agent.random_action == 57:    
        keepressources.keep_2x_grain_1x_brick_1x_ore = 1    
    if random_agent.random_action == 58:    
        keepressources.keep_2x_brick_1x_lumber_1x_wool = 1
    if random_agent.random_action == 59:    
        keepressources.keep_2x_brick_1x_lumber_1x_grain = 1
    if random_agent.random_action == 60:    
        keepressources.keep_2x_brick_1x_lumber_1x_ore = 1
    if random_agent.random_action == 61:    
        keepressources.keep_2x_brick_1x_wool_1x_grain = 1
    if random_agent.random_action == 62:    
        keepressources.keep_2x_brick_1x_wool_1x_ore = 1
    if random_agent.random_action == 63:    
        keepressources.keep_2x_brick_1x_grain_1x_ore = 1 
    if random_agent.random_action == 64:    
        keepressources.keep_2x_ore_1x_lumber_1x_wool = 1
    if random_agent.random_action == 65:    
        keepressources.keep_2x_ore_1x_lumber_1x_grain = 1
    if random_agent.random_action == 66:    
        keepressources.keep_2x_ore_1x_lumber_1x_brick = 1
    if random_agent.random_action == 67:    
        keepressources.keep_2x_ore_1x_wool_1x_grain = 1
    if random_agent.random_action == 68:    
        keepressources.keep_2x_ore_1x_wool_1x_brick = 1
    if random_agent.random_action == 69:    
        keepressources.keep_2x_ore_1x_grain_1x_brick = 1  
    if random_agent.random_action == 70:    
        keepressources.keep_1x_lumber_1x_wool_1x_grain_1x_brick = 1
    if random_agent.random_action == 71:    
        keepressources.keep_1x_lumber_1x_wool_1x_grain_1x_ore = 1
    if random_agent.random_action == 72:    
        keepressources.keep_1x_lumber_1x_wool_1x_brick_1x_ore = 1
    if random_agent.random_action == 73:    
        keepressources.keep_1x_lumber_1x_grain_1x_brick_1x_ore = 1
    if random_agent.random_action == 74:
        keepressources.keep_1x_wool_1x_grain_1x_brick_1x_ore = 1       
    if random_agent.random_action == 75:  
        trading.give_lumber_get_wool = 1
    if random_agent.random_action == 76:
        trading.give_lumber_get_grain = 1
    if random_agent.random_action == 77:
        trading.give_lumber_get_brick = 1
    if random_agent.random_action == 78:
        trading.give_lumber_get_ore  = 1
    if random_agent.random_action == 79:
        trading.give_wool_get_lumber = 1
    if random_agent.random_action == 80:
        trading.give_wool_get_grain = 1
    if random_agent.random_action == 81:
        trading.give_wool_get_brick = 1
    if random_agent.random_action == 82:
        trading.give_wool_get_ore = 1
    if random_agent.random_action == 83:
        trading.give_grain_get_lumber = 1
    if random_agent.random_action == 84:
        trading.give_grain_get_wool = 1
    if random_agent.random_action == 85:
        trading.give_grain_get_brick = 1
    if random_agent.random_action == 86:
        trading.give_grain_get_ore = 1
    if random_agent.random_action == 87:
        trading.give_brick_get_lumber = 1
    if random_agent.random_action == 88:
        trading.give_brick_get_wool = 1
    if random_agent.random_action == 89:
        trading.give_brick_get_grain = 1
    if random_agent.random_action == 90:
        trading.give_brick_get_ore = 1
    if random_agent.random_action == 91:
        trading.give_ore_get_lumber = 1
    if random_agent.random_action == 92:
        trading.give_ore_get_wool = 1
    if random_agent.random_action == 93:
        trading.give_ore_get_grain = 1
    if random_agent.random_action == 94:
        trading.give_ore_get_brick = 1
    if random_agent.random_action == 95:
        action.development_card_buy = 1
    if random_agent.random_action == 96:
        action.knight_cards_activate = 1
    if random_agent.random_action == 97:
        action.road_building_cards_activate = 1
    if random_agent.random_action == 98:
        action.yearofplenty_cards_activate = 1
    if random_agent.random_action == 99:
        action.monopoly_cards_activate = 1
    if random_agent.random_action == 100:
        action.yearofplenty_lumber_wool = 1
    if random_agent.random_action == 101:
        action.yearofplenty_lumber_grain = 1
    if random_agent.random_action == 102:
        action.yearofplenty_lumber_brick = 1
    if random_agent.random_action == 103:
        action.yearofplenty_lumber_ore = 1
    if random_agent.random_action == 104:
        action.yearofplenty_wool_grain = 1
    if random_agent.random_action == 105:
        action.yearofplenty_wool_brick = 1
    if random_agent.random_action == 106:
        action.yearofplenty_wool_ore = 1
    if random_agent.random_action == 107:
        action.yearofplenty_grain_brick = 1
    if random_agent.random_action == 108:
        action.yearofplenty_grain_ore = 1
    if random_agent.random_action == 109:
        action.yearofplenty_brick_ore = 1
    if random_agent.random_action == 110:
        action.monopoly_lumber = 1
    if random_agent.random_action == 111:
        action.monopoly_wool = 1
    if random_agent.random_action == 112:
        action.monopoly_grain = 1
    if random_agent.random_action == 113:
        action.monopoly_brick = 1
    if random_agent.random_action == 114:
        action.monopoly_ore = 1        
        
def selection():
    if random_agent.random_action == 1:
        move_rober(random_agent.random_position_x, random_agent.random_position_y)
    if random_agent.random_action == 2:
        buy_road(random_agent.random_position_x, random_agent.random_position_y)
    if random_agent.random_action == 3:
        buy_settlement(random_agent.random_position_x, random_agent.random_position_y)
    if random_agent.random_action == 4:
        buy_city(random_agent.random_position_x, random_agent.random_position_y)
    if random_agent.random_action == 5:
        move_finished()
    if random_agent.random_action == 1:
        discard_ressources()
    if random_agent.random_action == 1:
        discard_ressources()
    if random_agent.random_action == 1:
        move_finished()
    if random_agent.random_action == 1:
        move_finished()
    if random_agent.random_action == 1:
        move_finished()
    if random_agent.random_action == 1:
        move_finished()
    if random_agent.random_action == 1:
        move_finished()
    if random_agent.random_action == 1:
        move_finished()
    if random_agent.random_action == 1:
        move_finished()
    if random_agent.random_action == 1:
        move_finished()
    

def random_choice():
    random_agent.random_action = np.random.choice(np.arange(1,116))
    random_agent.random_position_y = np.random.choice(np.arange(0,11))
    random_agent.random_position_x = np.random.choice(np.arange(0,21))

main()

    


