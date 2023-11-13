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
        #board
        self.ZEROBOARD = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.prettyboard = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.ressouceboard = np.zeros((_NUM_ROWS,_NUM_COLS))
        #tiles
        self.tiles_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_dice = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_dice_probabilities = np.zeros((_NUM_ROWS, _NUM_COLS))
        
        self.tiles_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.tiles_pretty = np.zeros((_NUM_ROWS, _NUM_COLS))

        #settlements
        self.settlements_free = np.ones((_NUM_ROWS, _NUM_COLS))
        self.settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.settlements_used = np.zeros((_NUM_ROWS, _NUM_COLS))

        #roads
        self.roads_available = np.zeros((_NUM_ROWS, _NUM_COLS))
        
        #robers 
        self.rober_position = np.zeros((_NUM_ROWS, _NUM_COLS))

        #harbors
        self.harbors_possible = np.zeros((9, 2, 2))
        self.harbor_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.harbor_three_one = np.zeros((_NUM_ROWS, _NUM_COLS))

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

        self.tile_numbers = [0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,5,5,5]
        self.tile_random_numbers = np.random.choice(self.tile_numbers,19,replace=False)

        self.development_card_numbers = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,4,4,5,5]
        self.development_card_random_number = np.random.choice(self.development_card_numbers,25,replace=False)
        self.development_cards_bought = 0

class Player:
    def __init__(self):
        #settlements
        self.settlements = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.settlements_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.settlements_left = 5

        #roads
        self.roads = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.roads_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.roads_left = 15

        #cities
        self.cities = np.zeros((_NUM_ROWS, _NUM_COLS))       
        self.cities_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.cities_left = 4

        #rewards 
        self.rewards_possible = np.zeros((_NUM_ROWS,_NUM_COLS))

        #number of ressources
        self.ressource_lumber = 0
        self.ressource_wool = 0
        self.ressource_grain = 0
        self.ressource_brick = 0
        self.ressource_ore = 0
        
        #army
        self.army_size = 0
        
        #number of development cards per type
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

        self.knight_cards_played = 0

        #access to harbors (0 = no, 1 = yes)
        self.harbor_lumber = 0
        self.harbor_wool = 0
        self.harbor_grain = 0
        self.harbor_brick = 0
        self.harbor_ore = 0
        self.harbor_three_one = 0

        #largest army / longest road
        self.largest_army = 0

        self.roads_connected = 0
        self.longest_road = 0

          #Victory Points
        self.victorypoints = 0.0




    
    class Action: 
        def __init__(self):
            self.thief_move = np.zeros((_NUM_ROWS,_NUM_COLS))

            self.road_place = np.zeros((_NUM_ROWS,_NUM_COLS))
            self.settlement_place = np.zeros((_NUM_ROWS,_NUM_COLS))
            self.city_place = np.zeros((_NUM_ROWS,_NUM_COLS))

            #how many development cards the agent wants to buy 
            self.development_card_buy_1 = 0
            self.development_card_buy_2 = 0
            self.development_card_buy_3 = 0
            self.development_card_buy_4 = 0
            self.development_card_buy_5 = 0

            #Play a development card
            self.knight_cards_activate = 0 
            self.road_building_cards_activate = 0
            self.monopoly_cards_activate = 0
            self.yearofplenty_cards_activate = 0

            #Which ressource do you want to take when playing monopoly
            self.monopoly_lumber = 0
            self.monopoly_wool = 0
            self.monopoly_grain = 0
            self.monopoly_brick = 0
            self.monopoly_ore = 0

            #Which 2 ressources do you want to take 
            self.yearofplenty1_lumber = 0
            self.yearofplenty1_wool = 0
            self.yearofplenty1_grain = 0
            self.yearofplenty1_brick = 0
            self.yearofplenty1_ore = 0
            self.yearofplenty2_lumber = 0
            self.yearofplenty2_wool = 0
            self.yearofplenty2_grain = 0
            self.yearofplenty2_brick = 0
            self.yearofplenty2_ore = 0

    
#phase
class Phase:
    def __init__(self):
        self.rolled = 0
        self.developmentcard_placed = 0

#phase_roadbuilding = 0
#phase_yearofplenty = 0

#testing

#config Variables
class Game: 
    def __init__(self):
        self.cur_player = 0
        self.is_finished = False
        self.terminal = False
        self.testing = True
        self.settlementplaced = 0

board = Board()
distribution = Distribution()
player0 = Player()
player1 = Player()
player0_action = player0.Action()
player1_action = player1.Action()
players = [player0,player1]
game = Game()

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
            board.tiles_possible[i][j] = 1

def settlements_building():
    for i in range(0,11,2):
        for j in range(-1 + abs(5-i),23 - abs(5-i),2):
            board.settlements_available[i][j] = 1  
            print(board.settlements_available)
            player0.settlements_possible[i][j] = 1
            player1.settlements_possible[i][j] = 1

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

    board.roads_available = board.roads_available*(1-board.tiles_possible)
            
def pretty_board_update():
    print("pretty board roads and settlements: first number (1 = settlements,3 = roads), second number(0 = none 1 = player0, 2 = player1), third number(1 = lumber harbor, 2 = wool harbor, 3 = grain harbor, 4 = brick harbor, 5 = ore harbor, 6 = three/one harbor)")
    print("pretty board tiles: fist number(tiles = 2) second number(lumber = 1, wool = 2, grain = 3, brick = 4, ore = 5) third number(probability of dice landing)")

    
    for i in range (0,11,1):
        for j in range(0,21,1):

            if (board.settlements_available[i][j] == 1 and player0.settlements[i][j] == 0 and player0.settlements[i][j] == 0):
                if board.harbor_lumber[i][j] == 1:
                    board.prettyboard[i][j] = 101
                elif board.harbor_wool[i][j] == 1:
                    board.prettyboard[i][j] = 102
                elif board.harbor_grain[i][j] == 1:
                    board.prettyboard[i][j] = 103
                elif board.harbor_brick[i][j] == 1:
                    board.prettyboard[i][j] = 104
                elif board.harbor_ore[i][j] == 1:
                    board.prettyboard[i][j] = 105
                elif board.harbor_three_one[i][j] == 1:
                    board.prettyboard[i][j] = 106
                else: 
                    board.prettyboard[i][j] = 10

            if player0.roads[i][j] == 1:
                board.prettyboard[i][j] = 31        
            if player1.roads[i][j] == 1:
                board.prettyboard[i][j] = 32

            if (board.roads_available[i][j] == 1 and player0.roads[i][j] == 0 and player1.roads[i][j] == 0):
                board.prettyboard[i][j] = 30




            if player0.settlements[i][j] == 1:
                if board.harbor_lumber[i][j] == 1:
                    board.prettyboard[i][j] = 111
                elif board.harbor_wool[i][j] == 1:
                    board.prettyboard[i][j] = 112
                elif board.harbor_grain[i][j] == 1:
                    board.prettyboard[i][j] = 113
                elif board.harbor_brick[i][j] == 1:
                    board.prettyboard[i][j] = 114
                elif board.harbor_ore[i][j] == 1:
                    board.prettyboard[i][j] = 115
                elif board.harbor_three_one[i][j] == 1:
                    board.prettyboard[i][j] = 116
                else: 
                    board.prettyboard[i][j] = 11

            if player1.settlements[i][j] == 1:
                if board.harbor_lumber[i][j] == 1:
                    board.prettyboard[i][j] = 121
                elif board.harbor_wool[i][j] == 1:
                    board.prettyboard[i][j] = 122
                elif board.harbor_grain[i][j] == 1:
                    board.prettyboard[i][j] = 123
                elif board.harbor_brick[i][j] == 1:
                    board.prettyboard[i][j] = 124
                elif board.harbor_ore[i][j] == 1:
                    board.prettyboard[i][j] = 125
                elif board.harbor_three_one[i][j] == 1:
                    board.prettyboard[i][j] = 126
                else: 
                    board.prettyboard[i][j] = 12
            
            if (board.tiles_possible[i][j] == 1 and board.tiles_lumber[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 1):
                board.prettyboard[i][j] = 211
            if (board.tiles_possible[i][j] == 1 and board.tiles_lumber[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 2):
                board.prettyboard[i][j] = 212
            if (board.tiles_possible[i][j] == 1 and board.tiles_lumber[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 3):
                board.prettyboard[i][j] = 213
            if (board.tiles_possible[i][j] == 1 and board.tiles_lumber[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 4):
                board.prettyboard[i][j] = 214
            if (board.tiles_possible[i][j] == 1 and board.tiles_lumber[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 5):
                board.prettyboard[i][j] = 215
            
            if (board.tiles_possible[i][j] == 1 and board.tiles_wool[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 1):
                board.prettyboard[i][j] = 221
            if (board.tiles_possible[i][j] == 1 and board.tiles_wool[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 2):
                board.prettyboard[i][j] = 222
            if (board.tiles_possible[i][j] == 1 and board.tiles_wool[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 3):
                board.prettyboard[i][j] = 223
            if (board.tiles_possible[i][j] == 1 and board.tiles_wool[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 4):
                board.prettyboard[i][j] = 224
            if (board.tiles_possible[i][j] == 1 and board.tiles_wool[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 5):
                board.prettyboard[i][j] = 225
            
            if (board.tiles_possible[i][j] == 1 and board.tiles_grain[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 1):
                board.prettyboard[i][j] = 231
            if (board.tiles_possible[i][j] == 1 and board.tiles_grain[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 2):
                board.prettyboard[i][j] = 232
            if (board.tiles_possible[i][j] == 1 and board.tiles_grain[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 3):
                board.prettyboard[i][j] = 233
            if (board.tiles_possible[i][j] == 1 and board.tiles_grain[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 4):
                board.prettyboard[i][j] = 234
            if (board.tiles_possible[i][j] == 1 and board.tiles_grain[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 5):
                board.prettyboard[i][j] = 235
            
            if (board.tiles_possible[i][j] == 1 and board.tiles_brick[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 1):
                board.prettyboard[i][j] = 241
            if (board.tiles_possible[i][j] == 1 and board.tiles_brick[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 2):
                board.prettyboard[i][j] = 242
            if (board.tiles_possible[i][j] == 1 and board.tiles_brick[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 3):
                board.prettyboard[i][j] = 243
            if (board.tiles_possible[i][j] == 1 and board.tiles_brick[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 4):
                board.prettyboard[i][j] = 244
            if (board.tiles_possible[i][j] == 1 and board.tiles_brick[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 5):
                board.prettyboard[i][j] = 245
            
            if (board.tiles_possible[i][j] == 1 and board.tiles_ore[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 1):
                board.prettyboard[i][j] = 251
            if (board.tiles_possible[i][j] == 1 and board.tiles_ore[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 2):
                board.prettyboard[i][j] = 252
            if (board.tiles_possible[i][j] == 1 and board.tiles_ore[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 3):
                board.prettyboard[i][j] = 253
            if (board.tiles_possible[i][j] == 1 and board.tiles_ore[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 4):
                board.prettyboard[i][j] = 254
            if (board.tiles_possible[i][j] == 1 and board.tiles_ore[i][j] == 1 and board.tiles_dice_probabilities[i][j] == 5):
                board.prettyboard[i][j] = 255
            

    print("prettyboard:\n",board.prettyboard)


def ressource_board():
    print("ressource board (1 = lumber, 2 = wool, 3 = grain, 4 = brick, 5 = ore)")
    for i in range (0,11,1):
        for j in range(0,21,1):
            print()
    


def tile_distribution(): #fix tile distribution
    a = 0
    for i in range (1,11,1):
        for j in range(1,21,1):
            if board.tiles_possible[i][j] == 1:
                if distribution.tile_random_numbers[a-1] == 0:
                    board.rober_position[i][j] = 1
                elif distribution.tile_random_numbers[a-1] == 1:
                    board.tiles_lumber[i][j] = 1
                    board.tiles_pretty[i][j] = 1
                elif distribution.tile_random_numbers[a-1] == 2:
                    board.tiles_wool[i][j] = 1
                    board.tiles_pretty[i][j] = 2
                elif distribution.tile_random_numbers[a-1] == 3:
                    board.tiles_grain[i][j] = 1
                    board.tiles_pretty[i][j] = 3
                elif distribution.tile_random_numbers[a-1] == 4:
                    board.tiles_brick[i][j] = 1
                    board.tiles_pretty[i][j] = 4
                elif distribution.tile_random_numbers[a-1] == 5:
                    board.tiles_ore[i][j] = 1
                    board.tiles_pretty[i][j] = 5
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
    plate_numbers = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]
    plate_random_numbers = np.random.choice(plate_numbers, 18, replace=False)
    a = 0
    for i in range (1,11,1):
        for j in range (1,21,1):
            if board.tiles_possible[i][j] == 1 and board.tiles_pretty[i][j] != 0: #is there a desert here
                board.tiles_dice[i][j] = plate_random_numbers[a-1]
                board.tiles_dice_probabilities[i][j] = 6-abs(7-board.tiles_dice[i][j])
                a += 1

def development_card_buy():
    
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
    player.rewards_possible = player.rewards_possible * board.tiles_possible

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
    print("board.settlements_free")
    print(board.settlements_free)
    if board.settlements_free[a,b] == 1:
        print("at least it gets to this stage")
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
        print(x)
        print(y)
        if board.settlements_free[x][y] == 0 and board.settlements_available[x][y] == 1:
            return 0
    
    #Might find an easier way later
    neighboring_roads1 = [
        (0, 1), (0, -1),
        (1, 0), (-1, 0),
    ]
    neighboring_roads2 = [
        (0, 2), (0, -2),
        (2, 0), (-2, 0),
    ]
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
    opponent = players[1 - game.cur_player]
    possible = 0

    possible = road_possible_check(a,b)
    if possible == 1:
        player.roads[a][b] = 1
        update_longest_road()
        return 1 
    return 0
def road_place_placement(settlement_a,settlement_b,road_a,road_b):
    player = players[game.cur_player]
    print(road_a + 1)
    print(settlement_a)
    if ((((road_a + 1) == settlement_a or (road_a - 1)  == settlement_a) and road_b == settlement_b) or (((road_b + 1) == settlement_b or (road_b - 1)  == settlement_b) and road_a == settlement_a)):
        player.roads[road_a][road_b] = 1
        print("player",player.roads)
        print("player",player0.roads)
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
        possible = development_card_buy()
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
    if board.rober_position[a][b] != 1 and board.tiles_possible[a][b] == 1:
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
    
def trading(give, get):
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

def discard_ressources():
    player = players[game.cur_player]
    total_ressources = player.ressource_lumber + player.ressource_brick + player.ressource_grain + player.ressource_ore + player.ressource_wool 
    ressources_keeping = np.zeros((4))
    for i in range (1,4):
        ressources_keeping[i] = 0 #number between 1 and 5 for ressource type

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
    print("gets here")
    board.longest_road[i][j] = 0
    board.increasing_roads[i][j] = a
    return 1 + trav(i+2, j,a) + trav(i+1, j+1,a) + trav(i+1, j-1,a) + trav(i, j+2,a) + trav(i-1, j+1,a) + trav(i-2, j,a) + trav(i-1, j-1,a) + trav(i, j-2,a) 
def check_longest_road(i,j,f,g):
    print("accsessed")
    n = 11
    m = 21
    if i < 0 or j < 0 or i >= n or j >= m or board.increasing_roads [i][j] == 0:
        print("failed failed faileeeeed")
        return 0
    if  board.increasing_roads[i][j] <= board.increasing_roads[f][g]:
        print(f,g,"failed")
        return 0
    f = i 
    g = j
    return 1 + check_longest_road(i+2, j,f,g) + check_longest_road(i+1, j+1,f,g) + check_longest_road(i+1, j-1,f,g) + check_longest_road(i, j+2,f,g) + check_longest_road(i-1, j+1,f,g) + check_longest_road(i-2, j,f,g) + check_longest_road(i-1, j-1,f,g) + check_longest_road(i, j-2,f,g) 

    

def find_longest_road():
    print("accsessed")
    player = players[game.cur_player]
    ans, n, m = 0, 11, 21
    for i, j in product(range(n), range(m)):
        board.increasing_roads = board.ZEROBOARD * board.ZEROBOARD
        board.longest_road = player.roads * (1-board.ZEROBOARD)
        a = 0
        c = trav(i, j, a)
        if c > 0:
            print("increasing roads: ")
            print(board.increasing_roads)
        if c > 0: 
            ans = max(ans,check_longest_road(i,j,0,0))

    print ("test",player.roads)
    return ans

def update_longest_road():
    player = players[game.cur_player]
    opponent = players[game.cur_player]
    player.roads_connected = find_longest_road()
    if player.roads_connected >= 5 and player.roads_connected > opponent.roads_connected and player.largest_army == 0:
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

    player.knight_cards_new += player0.knight_cards_new
    player.victorypoints_cards_old += player0.victorypoints_cards_new
    player.yearofplenty_cards_old += player0.yearofplenty_cards_new
    player.monopoly_cards_old += player0.monopoly_cards_new
    player.roadbuilding_cards_old += player0.roadbuilding_cards_new

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
    board.prettyboard = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_dice = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_dice_probabilities = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.tiles_pretty = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.settlements_free = np.ones((_NUM_ROWS, _NUM_COLS))
    board.settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.roads_available = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.rober_position = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbors_possible = np.zeros((9, 2, 2))
    board.harbor_lumber = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_wool = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_grain = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_brick = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_ore = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.harbor_three_one = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.rewards_possible_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
    board.rewards_possible_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))

    distribution.harbor_random_numbers = np.random.choice(distribution.harbor_numbers,9,replace=False)
    distribution.tile_random_numbers = np.random.choice(distribution.tile_numbers,19,replace=False)
    distribution.development_card_random_number = np.random.choice(distribution.development_card_numbers,25,replace=False)
    distribution.development_cards_bought = 0

    for player in players:
        player.settlements = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.settlements_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.settlements_left = 5

        player.roads = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.roads_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.roads_left = 15

        player.cities = np.zeros((_NUM_ROWS, _NUM_COLS))       
        player.cities_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
        player.cities_left = 4

        player.rewards_possible = np.zeros((_NUM_ROWS,_NUM_COLS))

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
        player.knight_cards_played = 0
        player.harbor_lumber = 0
        player.harbor_wool = 0
        player.harbor_grain = 0
        player.harbor_brick = 0
        player.harbor_ore = 0
        player.harbor_three_one = 0

def player_place_placement_phase():
    player = players[game.cur_player]
    possible = 0
    a = 0
    b = 0
    while possible == 0:
        a = 0
        b = 0
        while a not in range(1,12) or b not in range(1,22):
            print("Please choose your first settlement")
            print("1 = settlement, 2 = tile, 3 = road")
            print(board.prettyboard)
            a = int(input("which row?\n"))
            b = int(input("which collumn?\n"))
        possible = settlement_place_placement(a-1,b-1)
        
    c = 0
    d = 0
    possible = 0
    while possible == 0:
        c = 0
        d = 0
        while c not in range(1,12) or d not in range(1,22):
            print("Please choose a surrounding road")
            print("3 = settlement, 2 = tile, 1 = road")
            print(board.prettyboard)
            c = int(input("which row?\n"))
            d = int(input("which collumn?\n"))
        print(a-1,b-1,c-1,d-1)
        possible = road_place_placement(a-1,b-1,c-1,d-1)

def placement_phase():
    print(board.roads_available)
    print("The game begins")
    print("Placement Phase:")
    game.cur_player = 0
    print("Player 0 is starting")
    pretty_board_update()
    player_place_placement_phase()
    print(player0.settlements)
    print(player0.roads)
    
    game.cur_player = 1
    print("Player 1's turn")
    print(player0.roads)
    pretty_board_update()
    player_place_placement_phase()
    pretty_board_update()
    player_place_placement_phase()
    game.cur_player = 0
    print("Player 0's turn")
    pretty_board_update()
    player_place_placement_phase()
    


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
    pretty_board_update()
    tile_distribution()
    harbor_distribution()
    plate_distribution()
def reset():
    new_initial_state()
    setup()

def start():
    set_terminal_width(200)
    np.set_printoptions(linewidth=200)

def print_information():
    print("Information:")
    print("print Information (i)")
    print("my ressources (mr)")
    print("opponent ressources (or)")
    print("pretty board (pb)")
    print("development cards (dc)")
    print("longest road / largest army (l)")
    print("victory points (vp)")

    print("Actions:")
    print("Activate knight (ak)")
    print("Activate road building (ar)")
    print("Activate year of plenty (ay)")
    print("Activate monopoly (am)")
    
    print("Buy road (br)")
    print("Buy settlement (bs)")
    print("Buy city (bc)")
    print("Buy development card (bdc)")

    print("End turn (et)")
def turn_starts():
    c = roll_dice()
    player = players[game.cur_player]
    opponent = players[1 - game.cur_player]
    print("a",c,"has been rolled, now you own these ressources:")
    print("lumber:",player.ressource_lumber)
    print("wool:",player.ressource_wool)
    print("grain:",player.ressource_grain)
    print("brick:",player.ressource_brick)
    print("ore:",player.ressource_ore)

    if c == 7:
        print(board.prettyboard)
        print("You need to move the rober")
        d = int(input("Which row?"))
        e = int(input("Which column?"))
        move_rober(d,e)
        discard_ressources()
        steal_card()

    a = "a"

    print("It's Player",game.cur_player,"'s turn")
    print("Enter 1 of these commands to get information about the state or play an action")
    print_information()
    while a != "et":
        a = input("Which things would you like to look at?")
        if a == "i":
            print_information()
        if a == "ak":
            print(board.prettyboard)
            d = 0 
            e = 0
            print("Where do you want to place the thief?")
            d = int(input("Which row?"))
            e = int(input("Which column?"))
            possible = play_knight(d,e)
            if possible == 0:
                "Invalid move"
        if a == "ar":
            print(board.prettyboard)
            d = 0 
            e = 0
            f = 0
            g = 0   
            print("where do you want to place your first road?")
            d = int(input("Which row?"))
            e = int(input("Which column?"))
            print("where do you want to place your second road?")
            f = int(input("Which row?"))
            g = int(input("Which column?"))
            possible = activate_road_building_func(d,e,f,g)
            if possible == 0:
                print("invalid positions or no road building card available")
        if a == "ay":
            e = input("Which ressource do you want to take? (l = lumber, w = wool, g = grain, b = brick, o = ore)")
            f = input("Which ressource do you want to take? (l = lumber, w = wool, g = grain, b = brick, o = ore)")
            g = 0
            h = 0
            if e == "l":
                g = 1 
            if e == "w":
                g = 2
            if e == "g":
                g = 3
            if e == "b":
                g = 4
            if e == "o":
                g = 5
            if f == "l":
                h = 1 
            if f == "w":
                h = 2
            if f == "g":
                h = 3
            if f == "b":
                h = 4
            if f == "o":
                h = 5
            possible = activate_yearofplenty_func(g,h)
            if possible == 0:
                print("You don't have a year of plenty card")
        if a == "am":
            print("Opponenet's ressources")
            print("lumber:",opponent.ressource_lumber)
            print("wool:",opponent.ressource_wool)
            print("grain:",opponent.ressource_grain)
            print("brick:",opponent.ressource_brick)
            print("ore:",opponent.ressource_ore)

            e = input("Which ressource do you want to have from your opponent")

            possible = activate_monopoly_func(e)
            if possible == 0:
                print("You don't have a monopoly card")            
        if a  == "br":
            print("Where do you want to buy a road?")
            b = input("Which row?")
            c = input("Which column?")
            possible = buy_road(b,c)
            if possible == 0:
                "Invalid position or not enough ressources"
        if a == "bs":
            print("Where do you want to buy a settlement?")
            b = input("Which row?")
            c = input("Which column?")
            possible = buy_settlement(b,c)
            if possible == 0:
                "Invalid position or not enough ressources"
        if a == "bc":
            print("Where do you want to buy a city?")
            b = input("Which row?")
            c = input("Which column?")
            possible = buy_city(b,c)
            if possible == 0:
                "Invalid position or not enough ressources"
        if a == "bdc":
            possible = buy_development_cards()
            if possible == 1:
                print("You now have these new development cards")
                print("new knights:",player.knight_cards_new)
                print("new victory point card:",player.victorypoints_cards_new)
                print("new monopoly cards:",player.monopoly_cards_new)
                print("new year of plenty cards:", player.yearofplenty_cards_new)
                print("new road building cards:", player.roadbuilding_cards_new)
            if possible == 0:
                print("not enough ressources or no development cards left")
            
        
        if a == "mr":
            print("Your ressources")
            print("lumber:",player.ressource_lumber)
            print("wool:",player.ressource_wool)
            print("grain:",player.ressource_grain)
            print("brick:",player.ressource_brick)
            print("ore:",player.ressource_ore)
        if a == "or":
            print("Opponenet's ressources")
            print("lumber:",opponent.ressource_lumber)
            print("wool:",opponent.ressource_wool)
            print("grain:",opponent.ressource_grain)
            print("brick:",opponent.ressource_brick)
            print("ore:",opponent.ressource_ore)
        if a == "pb":
            print("pretty board")
            pretty_board_update()
            print(board.prettyboard)
        if a == "dc":
            print("development cards")
            print("old knights/new knights/played knights:",player.knight_cards_old,",", player.knight_cards_new,",", player.knight_cards_played)
            print("old victory point cards / new victory point card:",player.victorypoints_cards_old,",", player.victorypoints_cards_new)
            print("old monopoly cards / new monopoly cards:", player.monopoly_cards_old,",",player.monopoly_cards_new)
            print("old year of plenty cards / new year of plenty cards:",player.yearofplenty_cards_old, ",", player.yearofplenty_cards_new)
            print("old road building cards / new road building cards:", player.roadbuilding_cards_old, ",",player.roadbuilding_cards_new)
        if a == "l":
            print("Your longest road", player.roads_connected)
            print("Longest road oponnent",opponent.roads_connected)
            print("Your knights played",player.knight_cards_played)
            print("Opponent knights played", opponent.knight_cards_played)
        if a  == "vp":
            print("victory points", player.victorypoints)
            



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
main()

    


