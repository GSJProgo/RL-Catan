import numpy as np
import random
import math 

import time


_NUM_ROWS = 11
_NUM_COLS = 21

class Board: 
    def __init__(self):
        #board
        self.ZEROBOARD = np.zeros((_NUM_ROWS, _NUM_COLS))
        self.board = np.zeros((_NUM_ROWS, _NUM_COLS))
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

            #Victory Points
            self.victorypoints = 0.0
    
#bank ressources (Might take this out as probability of reaching this state is so low. Except bank_development_cards) 
class Bank:
    def __init__(self):
        self.lumber = 19
        self.wool = 19
        self.grain = 19
        self.brick = 19
        self.ore = 19
        self.development_cards = 25

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
    TEST = "\u001b[48;2;0;0;0m"
    RESET = "\u001b[38;2;255;255;255m"

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
            
def pretty_board_building():
    for i in range (0,11,1):
        for j in range(0,21,1):
            if (board.roads_available[i][j] == 1):
                board[i][j] = 3
            if (board.tiles_possible[i][j] == 1):
                board[i][j] = 2
            if (board.settlements_available[i][j] == 1):
                board[i][j] = 1

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
    if settlement_possible_check(a,b) == 1:
        tile_update_rewards()
        return 1 
    return 0

def settlement_possible_check(a,b):

    #Only update for a and b instead of doing it with i for everything

    player = players[game.cur_player]    
    board.settlements_free = board.settlements_available * board.settlements_free

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
        if board.settlements_free[x][y] == 0:
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
                      
def road_place(a,b):
    player = players[game.cur_player]
    if player.roads_possible[a][b] == 1:
        #roadplaced(a,b)
        road_possible_check(a,b)
        return 1 
    return 0
def road_possible_check(a,b):
    board.roads_available = board.roads_available * (1 - player0.roads) * (1 - player1.roads)
    player = players[game.cur_player]
    opponent = players[1-game.cur_player]    
    #I could work with boards and multiply them at the end to check 
    player.roads_possible = board.ZEROBOARD
        

    if b != 20 and opponent.settlements[a][b + 1] != 1:
        if b != 19:
            player.roads_possible[a][b+2] = 0
        if a != 0:
            player.roads_possible[a-1][b+1] = 0
        if a != 10: 
            player.roads_possible[a+1][b+1] = 0
    if b != 0 and opponent.settlements[a][b - 1] != 1:
        if b != 1:
            player.roads_possible[a][b-2] = 0
        if a != 0:
            player.roads_possible[a-1][b-1] = 0
        if a != 10: 
            player.roads_possible[a+1][b-1] = 0
    if a != 10 and opponent.settlements[a+1][b] != 1:
        if a != 9:
            player.roads_possible[a+2][b] = 0
        if b != 0:
            player.roads_possible[a+1][b-1] = 0
        if b != 20: 
            player.roads_possible[a+1][b+1] = 0
    if a != 0 and opponent.settlements[a-1][b] != 1:
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

        if player.roads[x][y] == 1 and player.roads_possible[x][y] == 0:
            return 1
    return 0 

def city_place(a,b):
    #still need to add a max cities check, the same comes to settlements
    player = players[game.cur_player]

    if player.settlements[a][b] == 1:
        player.cities[a][b] = 1
        player.settlements[a][b] = 0
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

def buy_development_cards():
    player = players[game.cur_player]
    possible = 0
    if player.ressource_wool > 0 and player.ressource_grain > 0 and player.ressource_ore > 0 and distribution.development_cards_bought != 25:
        possible = development_card_buy()
        if possible == 1:
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

def play_knight():
    player = players[game.cur_player]
    if player.knight_cards_new > 0:
        move_rober()
        steal_card()
        player.knight_cards_new -= 1
        player.knight_cards_played += 1

def move_rober(a,b):
    if board.rober_position[a][b] != 1:
        board.rober_position = board.rober_position * board.ZEROBOARD
        board.rober_position[a][b] = 1
    player0.rewards_possible = player0.rewards_possible * (1 - board.rober_position)
    player1.rewards_possible = player1.rewards_possible * (1 - board.rober_position)

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
        player.roadbuilding_cards_old = player.roadbuilding_cards_old - 1
        road_place(a1,b1)
        road_place(a2,b2)
    
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

    game.cur_player = 1 - game.cur_player

def reset1():
    if distribution.development_cards_bought == 24:
        distribution.development_cards_bought = 0

st = time.time()
for i in range (0,100000):
    move_finished()
    

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')