import numpy as np
import random
import math 

_NUM_ROWS = 11
_NUM_COLS = 21

zeroboard = np.zeros((_NUM_ROWS, _NUM_COLS))
# Game board setup
board = np.zeros((_NUM_ROWS, _NUM_COLS))
tiles_possible = np.zeros((_NUM_ROWS, _NUM_COLS))
harbors_possible = np.zeros((9, 2, 2))
dice_tiles = np.zeros((_NUM_ROWS, _NUM_COLS))
dice_probabilities = np.zeros((_NUM_ROWS, _NUM_COLS))
rober_position = np.zeros((_NUM_ROWS, _NUM_COLS))

# Resource tiles setup
lumber_tiles = np.zeros((_NUM_ROWS, _NUM_COLS))
wool_tiles = np.zeros((_NUM_ROWS, _NUM_COLS))
grain_tiles = np.zeros((_NUM_ROWS, _NUM_COLS))
brick_tiles = np.zeros((_NUM_ROWS, _NUM_COLS))
ore_tiles = np.zeros((_NUM_ROWS, _NUM_COLS))
pretty_tiles = np.zeros((_NUM_ROWS, _NUM_COLS))

# Port setups
lumber_harbor = np.zeros((_NUM_ROWS, _NUM_COLS))
wool_harbor = np.zeros((_NUM_ROWS, _NUM_COLS))
grain_harbor = np.zeros((_NUM_ROWS, _NUM_COLS))
brick_harbor = np.zeros((_NUM_ROWS, _NUM_COLS))
ore_harbor = np.zeros((_NUM_ROWS, _NUM_COLS))
three_one_harbor = np.zeros((_NUM_ROWS, _NUM_COLS))

# Settlement and road setups
settlements_free = np.ones((_NUM_ROWS, _NUM_COLS))
settlements_available = np.zeros((_NUM_ROWS, _NUM_COLS))
settlements_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
settlements_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))
possible_settlements_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
possible_settlements_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))

# cities 
cities_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
cities_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))
possible_cities_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
possible_cities_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))

roads_available = np.zeros((_NUM_ROWS, _NUM_COLS))
roads_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
roads_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))
possible_roads_player0 = np.zeros((_NUM_ROWS, _NUM_COLS))
possible_roads_player1 = np.zeros((_NUM_ROWS, _NUM_COLS))




#sum of the possible rewards for each player 
possible_rewards_player0 = np.zeros((_NUM_ROWS,_NUM_COLS))

possible_rewards_player1 = np.zeros((_NUM_ROWS,_NUM_COLS))
#ressources 
player0_lumber = 0
player0_wool = 0
player0_grain = 0
player0_brick = 0
player0_ore = 0

player1_lumber = 0
player1_wool = 0
player1_grain = 0
player1_brick = 0
player1_ore = 0

#buildings left
player0_roads_left = 15
player0_settlements_left = 5
player0_cities_left = 4

player1_roads_left = 15
player1_settlements_left = 5
player1_cities_left = 4

#army size
player0_army_size = 0

player1_army_size = 0
#development cards per type

player0_knight_cards_old = 0
player0_victorypoints_cards_old = 0
player0_yearofplenty_cards_old = 0
player0_monopoly_cards_old = 0
player0_roadbuilding_cards_old = 0

player0_knight_cards_new = 0
player0_victorypoints_cards_new = 0
player0_yearofplenty_cards_new = 0
player0_monopoly_cards_new = 0
player0_roadbuilding_cards_new = 0

player1_knight_cards_old = 0
player1_victorypoints_cards_old = 0
player1_yearofplenty_cards_old = 0
player1_monopoly_cards_old = 0
player1_roadbuilding_cards_old = 0

player1_knight_cards_new = 0
player1_victorypoints_cards_new = 0
player1_yearofplenty_cards_new = 0
player1_monopoly_cards_new = 0
player1_roadbuilding_cards_new = 0

player0_knight_cards_played = 0
player1_knight_cards_played = 1

#access to harbors (0 = no, 1 = yes)
player0_lumber_harbor = 0.0
player0_wool_harbor = 0
player0_grain_harbor = 0
player0_brick_harbor = 0
player0_ore_harbor = 0
player0_three_one_harbor = 0

player1_lumber_harbor = 0
player1_wool_harbor = 0
player1_grain_harbor = 0
player1_brick_harbor = 0
player1_ore_harbor = 0
player1_three_one_harbor = 0

#largest army or longest road
player0_longest_road = 0
player0_largest_army = 0

player1_longest_road = 0
player1_largest_army = 0

#bank ressources (Might take this out as probability of reaching this state is so low. Except bank_development_cards) 
bank_lumber = 19
bank_wool = 19
bank_grain = 19
bank_brick = 19
bank_ore = 19
bank_development_cards = 25

#phase
phase_rolled = 0
phase_developmentcard_placed = 0
phase_roadbuilding = 0
phase_yearofplenty = 0

#action moves 
thief_move = np.zeros((_NUM_ROWS,_NUM_COLS))
place_road = np.zeros((_NUM_ROWS,_NUM_COLS))
place_settlement = np.zeros((_NUM_ROWS,_NUM_COLS))
place_city = np.zeros((_NUM_ROWS,_NUM_COLS))

#testing
dice_prob_count = np.zeros((13,1))
print("dice_prob_count", dice_prob_count)

#development cards
development_card_buy_1 = 0
development_card_buy_2 = 0
development_card_buy_3 = 0
development_card_buy_4 = 0
development_card_buy_5 = 0
activate_knight = 0 
activate_road_building = 0
activate_monopoly = 0
activate_yearofplenty = 0
monopoly_lumber = 0
monopoly_wool = 0
monopoly_grain = 0
monopoly_brick = 0
monopoly_ore = 0
yearofplenty1_lumber = 0
yearofplenty1_wool = 0
yearofplenty1_grain = 0
yearofplenty1_brick = 0
yearofplenty1_ore = 0
yearofplenty2_lumber = 0
yearofplenty2_wool = 0
yearofplenty2_grain = 0
yearofplenty2_brick = 0
yearofplenty2_ore = 0

#give up cards

# Game state variables
player0_score = 0.0
cur_player = 0
is_finished = False
terminal = False
testing = True
settlementplaced = 0

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

print(RESET +"test")
def harbors_building():
    # Define harbor locations

    harbors_possible[0] = [[0,4],[0,6]]
    harbors_possible[1] = [[0,10],[0,12]]
    harbors_possible[2] = [[2,16],[2,18]]
    harbors_possible[3] = [[2,2],[4,2]]
    harbors_possible[4] = [[6,2],[8,2]]
    harbors_possible[5] = [[10,4],[10,6]]
    harbors_possible[6] = [[10,10],[10,12]]
    harbors_possible[7] = [[8,16],[8,18]]
    harbors_possible[8] = [[4,20],[6,20]]
    print(harbors_possible)

def reset():
    # Reset various game state variables

    # Reset settlement placement count
    settlementplaced = 0

    for i in range(1,10,2):
        print(i)
        for j in range(2 + abs(5-i),20 - abs(5-i),4):
            tiles_possible[i][j] = 1

    harbors_building()

    
    for i in range(0,11,2):
        print(i)
        for j in range(-1 + abs(5-i),23 - abs(5-i),2):
            settlements_available[i][j] = 1  
            possible_settlements_player0[i][j] = 1
            possible_settlements_player1[i][j] = 1  

    for i in range(0,10,1):
        for j in range(0,20,1):
            if settlements_available[i + 1][j] == 1 and settlements_available[i - 1][j] == 1:
                roads_available[i][j] = 1
            if settlements_available[i + 1][j + 1] == 1 and settlements_available[i - 1][j + 1] == 1:
                roads_available[i][j+1] = 1
            
            if settlements_available[i][j + 1] == 1 and settlements_available[i][j - 1] == 1:
                roads_available[i][j] = 1
            if settlements_available[i + 1][j + 1] == 1 and settlements_available[i + 1][j - 1] == 1:
                roads_available[i+1][j] = 1


    for i in range (0,11,1):
        for j in range(0,21,1):
            if (roads_available[i][j] == 1):
                board[i][j] = 3
            if (tiles_possible[i][j] == 1):
                board[i][j] = 2
            if (settlements_available[i][j] == 1):
                board[i][j] = 1

def tile_distribution(): #fix tile distribution
    tile_numbers = [0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,5,5,5]
    tile_random_numbers = np.random.choice(tile_numbers,19,replace=False)
    print("tile_random_numbers",tile_random_numbers)
    a = 0
    for i in range (1,11,1):
        for j in range(1,21,1):
            if tiles_possible[i][j] == 1:
                if tile_random_numbers[a-1] == 0:
                    rober_position[i][j] = 1
                    print("rober_position",rober_position)
                if tile_random_numbers[a-1] == 1:
                    print("first")
                    lumber_tiles[i][j] = 1
                    pretty_tiles[i][j] = 1
                    print("first")
            
                if tile_random_numbers[a-1] == 2:
                    print("second")
                    wool_tiles[i][j] = 1
                    pretty_tiles[i][j] = 2
                    print("second")
            
                if tile_random_numbers[a-1] == 3:
                    print("third")
                    grain_tiles[i][j] = 1
                    pretty_tiles[i][j] = 3
                    print("third")

            
                if tile_random_numbers[a-1] == 4:
                    print("forth")
                    brick_tiles[i][j] = 1
                    pretty_tiles[i][j] = 4
                    print("forth")
            
                if tile_random_numbers[a-1] == 5:
                    print("fifth")
                    ore_tiles[i][j] = 1
                    pretty_tiles[i][j] = 5
                    print("fifth")
                a += 1 
                print(a-1)
            
def harbor_distribution():
    harbor_numbers = [1,2,3,4,5,6,6,6,6]
    harbor_random_numbers = np.random.choice(harbor_numbers,9,replace=False)
    for i in range(0,9,1):
        x1 = int(harbors_possible[i][0][0])
        y1 = int(harbors_possible[i][0][1])
        x2 = int(harbors_possible[i][1][0])
        y2 = int(harbors_possible[i][1][1])

        print(x1,x2,y1,y2)

        if harbor_random_numbers[i] == 1:
            lumber_harbor[x1][y1] = 1
            lumber_harbor[x2][y2] = 1
        if harbor_random_numbers[i] == 2:
            wool_harbor[x1][y1] = 1
            wool_harbor[x2][y2] = 1
        if harbor_random_numbers[i] == 3:
            grain_harbor[x1][y1] = 1
            grain_harbor[x2][y2] = 1
        if harbor_random_numbers[i] == 4:
            brick_harbor[x1][y1] = 1
            brick_harbor[x2][y2] = 1
        if harbor_random_numbers[i] == 5:
            ore_harbor[x1][y1] = 1
            ore_harbor[x2][y2] = 1
        if harbor_random_numbers[i] == 6:
            three_one_harbor[x1][y1] = 1
            three_one_harbor[x2][y2] = 1
    print("lumber_harbor \n",lumber_harbor,"wool_harbor \n",wool_harbor,"grain_harbor\n",grain_harbor,"brick_harbor\n",brick_harbor,"ore_harbor\n",ore_harbor,"three_one_harbor\n",three_one_harbor)   
            
def plate_distribution():

    plate_numbers = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]
    plate_random_numbers = np.random.choice(plate_numbers, 18, replace=False)
    a = 0
    for i in range (1,11,1):
        for j in range (1,21,1):
            if tiles_possible[i][j] == 1 and pretty_tiles[i][j] != 0: #is there a desert here
                dice_tiles[i][j] = plate_random_numbers[a-1]
                a += 1
    
    for i in range (1,11,1):
        for j in range (1,21,1):
            if dice_tiles[i][j] != 0:
                dice_probabilities[i][j] = 6-abs(7-dice_tiles[i][j])

development_card_numbers = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,4,4,5,5]
development_card_random_number = np.random.choice(development_card_numbers,25,replace=False)
development_cards_bought = 0


def development_card_distribution():
    global development_cards_bought

    global player0_knight_cards_new 
    global player0_victorypoints_cards_new 
    global player0_yearofplenty_cards_new 
    global player0_monopoly_cards_new 
    global player0_roadbuilding_cards_new 

    global player1_knight_cards_new 
    global player1_victorypoints_cards_new 
    global player1_yearofplenty_cards_new 
    global player1_monopoly_cards_new 
    global player1_roadbuilding_cards_new 
    
    if cur_player == 0:
        if development_card_random_number[development_cards_bought] == 1:
            player0_knight_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 2:
            player0_victorypoints_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 3:
            player0_yearofplenty_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 4:
            player0_monopoly_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 5:
            player0_roadbuilding_cards_new += 1 

    if cur_player == 1:
        if development_card_random_number[development_cards_bought] == 1:
            player1_knight_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 2:
            player1_victorypoints_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 3:
            player1_yearofplenty_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 4:
            player1_monopoly_cards_new += 1 
        if development_card_random_number[development_cards_bought] == 5:
            player1_roadbuilding_cards_new += 1 
        
    development_cards_bought += 1






def tile_update_rewards(a,b):
    if cur_player == 0:
        if tiles_possible[a-1][b] == 1:
            if a != 0:
                possible_rewards_player0[a-1][b] += 1
            if a != 10:
                if b != 20 or b != 19:
                    possible_rewards_player0[a+1][b+2] += 1
                if b != 1 or b != 0:
                    possible_rewards_player0[a+1][b-2] += 1
        elif tiles_possible[a+1][b] == 1:
            if a != 10:
                possible_rewards_player0[a+1][b] += 1
            if a != 0:
                if b != 20 or b != 19:
                    possible_rewards_player0[a-1][b+2] += 1
                if b != 1 or b != 0:
                    possible_rewards_player0[a-1][b-2] += 1
        else:
            if b != 0 or b != 1:
                if tiles_possible[a+1][b+2] == 1:
                    possible_rewards_player0[a+1][b+2] += 1
                if tiles_possible[a-1][b+2] == 1:
                    possible_rewards_player0[a-1][b+2] += 1
            if b != 19 or b != 20:
                if tiles_possible[a+1][b-2] == 1:
                    possible_rewards_player0[a+1][b-2] += 1
                if tiles_possible[a-1][b-2] == 1:
                    possible_rewards_player0[a-1][b-2] += 1
            
  

    if cur_player == 1:
        if tiles_possible[a-1][b] == 1:
            if a != 0:
                possible_rewards_player1[a-1][b] += 1
            if a != 10:
                if b != 20 or b != 19:
                    possible_rewards_player1[a+1][b+2] += 1
                if b != 1 or b != 0:
                    possible_rewards_player1[a+1][b-2] += 1

        if tiles_possible[a+1][b] == 1:
            if a != 10:
                possible_rewards_player1[a+1][b] += 1
            if a != 0:
                if b != 20 or b != 19:
                    possible_rewards_player1[a-1][b+2] += 1
                if b != 1 or b != 0:
                    possible_rewards_player1[a-1][b-2] += 1    

def settlement_place(a,b):
    if cur_player == 0:
        if possible_settlements_player0[a][b] == 1:
            settlements_player0[a][b] = 1 
            settlementplaced(a,b)
            settlement_possible_update(a,b)
            return 1 
    if cur_player == 1:
        if possible_settlements_player1[a][b] == 1:
            settlements_player1[a][b] = 1
            settlementplaced(a,b)
            settlement_possible_update(a,b)
            return 1 
    return 0
rightfree = 0
leftfree = 0
topfree = 0
bottomfree = 0 
tworoads = 0
def settlement_possible_update():
    global rightfree
    global leftfree 
    global topfree
    global bottomfree
    global tworoads
    global cur_player
    
    settlements_free = settlements_available * settlements_free
    for i in range (0,11,21):
        for j in range(0,21,1):
            if (i == 0):
                topfree = 1
            if (i == 10):
                bottomfree = 1
            if j == 0: 
                leftfree = 1
            if j == 20:
                rightfree = 1

            if j != 20 and j != 19 and settlements_free[i][j + 2] == 1:
                rightfree = 1
            if j != 0 and j != 1 and settlements_free[i][j - 2] == 1:
                leftfree = 1
            if i != 0 and i != 1 and settlements_free[i - 2][j] == 1:
                topfree = 1
            if j != 10 and j != 9 and settlements_free[i + 2][j] == 1:
                bottomfree = 1
            if cur_player == 0:
            
                if j!= 20 and roads_player0[i][j + 1] == 1:
                    if j != 18 and j != 19 and roads_player0[i][j+3] == 1:
                        tworoads = 1
                    elif i != 10 and j != 19 and roads_player0[i+1][j+2] == 1:
                        tworoads = 1
                    elif i != 0 and j != 19 and roads_player0[i-1][j+2] == 1:
                        tworoads = 1
                if j != 0 and roads_player0[i][j - 1] == 1:
                    if j != 2 and j != 1 and roads_player0[i][j - 3] == 1:
                        tworoads = 1
                    elif i != 10 and j != 1 and roads_player0[i + 1][j - 2] == 1:
                        tworoads = 1
                    elif i != 0 and j != 1 and roads_player0[i - 1][j - 2] == 1:
                        tworoads = 1

                if i != 0 and roads_player0[i - 1][j] == 1:
                    if i != 2 and i != 1 and roads_player0[i - 3][j] == 1:
                        tworoads = 1
                    elif j != 20 and i != 1 and roads_player0[i - 2][j + 1] == 1:
                        tworoads = 1
                    elif j != 0 and i != 1 and roads_player0[i - 2][j - 1] == 1:
                        tworoads = 1

                if i != 10 and roads_player0[i + 1][j] == 1:
                    if i != 8 and i != 9 and roads_player0[i + 3][j] == 1:
                        tworoads = 1
                    elif j != 20 and i != 9 and roads_player0[i + 2][j + 1] == 1:
                        tworoads = 1
                    elif j != 0 and i != 9 and roads_player0[i + 2][j - 1] == 1:
                        tworoads = 1

                if tworoads == 1 and bottomfree == 1 and topfree == 1 and rightfree == 1 and leftfree == 1:
                    settlements_player0[i][j] = 1
            if cur_player == 1:
                
                if j!= 20 and roads_player1[i][j + 1] == 1:
                    if j != 18 and j != 19 and roads_player1[i][j+3] == 1:
                        tworoads = 1
                    elif i != 10 and j != 19 and roads_player1[i+1][j+2] == 1:
                        tworoads = 1
                    elif i != 0 and j != 19 and roads_player1[i-1][j+2] == 1:
                        tworoads = 1
                
                if j != 0 and roads_player1[i][j - 1] == 1:
                    if j != 2 and j != 1 and roads_player1[i][j - 3] == 1:
                        tworoads = 1
                    elif i != 10 and j != 1 and roads_player1[i + 1][j - 2] == 1:
                        tworoads = 1
                    elif i != 0 and j != 1 and roads_player1[i - 1][j - 2] == 1:
                        tworoads = 1

                if i != 0 and roads_player1[i - 1][j] == 1:
                    if i != 2 and i != 1 and roads_player1[i - 3][j] == 1:
                        tworoads = 1
                    elif j != 20 and i != 1 and roads_player1[i - 2][j + 1] == 1:
                        tworoads = 1
                    elif j != 0 and i != 1 and roads_player1[i - 2][j - 1] == 1:
                        tworoads = 1

                if i != 10 and roads_player1[i + 1][j] == 1:
                    if i != 8 and i != 9 and roads_player1[i + 3][j] == 1:
                        tworoads = 1
                    elif j != 20 and i != 9 and roads_player1[i + 2][j + 1] == 1:
                        tworoads = 1
                    elif j != 0 and i != 9 and roads_player1[i + 2][j - 1] == 1:
                        tworoads = 1

                if tworoads == 1 and bottomfree == 1 and topfree == 1 and rightfree == 1 and leftfree == 1:
                    settlements_player1[i][j] = 1
                    
def road_place(a,b):
    if cur_player == 0:
        if possible_roads_player0[a][b] == 1:
            roads_player0[a][b] = 1
            #roadplaced(a,b)
            road_possible_update(a,b)
            settlement_possible_update()
            return 1 
    if cur_player == 1:
        if possible_roads_player1[a][b] == 1:
            roads_player1[a][b] = 1
            #roadplaced(a,b)
            road_possible_update(a,b)
            settlement_possible_update()

            return 1 
    return 0
def road_possible_update(a,b):
    roads_available = roads_available * (1-roads_player0)(1-roads_player1)
    if cur_player == 0:
        possible_roads_player0[a][b + 2] = 1
        possible_roads_player0[a][b - 2] = 1
        possible_roads_player0[a + 2][b] = 1
        possible_roads_player0[a - 2][b] = 1
        possible_roads_player0[a + 1][b + 1] = 1
        possible_roads_player0[a + 1][b - 1] = 1
        possible_roads_player0[a - 1][b + 1] = 1
        possible_roads_player0[a - 1][b - 1] = 1

        if roads_player1[a][b + 1] == 1:
            possible_roads_player0[a][b + 2] = 0
            possible_roads_player0[a + 1][b + 1] = 0
            possible_roads_player0[a - 1][b + 1] = 0
        
        if roads_player1[a][b - 1] == 1:
            possible_roads_player0[a][b - 2] = 0
            possible_roads_player0[a + 1][b - 1] = 0
            possible_roads_player0[a - 1][b - 1] = 0
        
        if roads_player1[a + 1][b] == 1:
            possible_roads_player0[a + 2][b] = 0
            possible_roads_player0[a + 1][b + 1] = 0
            possible_roads_player0[a + 1][b - 1] = 0
        
        if roads_player1[a - 1][b] == 1:
            possible_roads_player0[a - 2][b] = 0
            possible_roads_player0[a - 1][b + 1] = 0
            possible_roads_player0[a - 1][b - 1] = 0
        



    if cur_player == 1:
        possible_roads_player1[a][b + 2] = 1
        possible_roads_player1[a][b - 2] = 1
        possible_roads_player1[a + 2][b] = 1
        possible_roads_player1[a + 2][b] = 1
        possible_roads_player1[a + 1][b + 1] = 1
        possible_roads_player1[a + 1][b - 1] = 1
        possible_roads_player1[a - 1][b + 1] = 1
        possible_roads_player1[a - 1][b - 1] = 1

        if roads_player0[a][b + 1] == 1:
            possible_roads_player1[a][b + 2] = 0
            possible_roads_player1[a + 1][b + 1] = 0
            possible_roads_player1[a - 1][b + 1] = 0
        
        if roads_player0[a][b - 1] == 1:
            possible_roads_player1[a][b - 2] = 0
            possible_roads_player1[a + 1][b - 1] = 0
            possible_roads_player1[a - 1][b - 1] = 0
        
        if roads_player0[a + 1][b] == 1:
            possible_roads_player1[a + 2][b] = 0
            possible_roads_player1[a + 1][b + 1] = 0
            possible_roads_player1[a + 1][b - 1] = 0
        
        if roads_player0[a - 1][b] == 1:
            possible_roads_player1[a - 2][b] = 0
            possible_roads_player1[a - 1][b + 1] = 0
            possible_roads_player1[a - 1][b - 1] = 0

    possible_roads_player0 = roads_available * possible_roads_player0
    possible_roads_player1 = roads_available * possible_roads_player1

def city_place(a,b):
    #still need to add a max cities check, the same comes to settlements
    if cur_player == 0:
        if possible_cities_player0[a][b] == 1:
            cities_player0[a][b] = 1
            settlements_player0[a][b] = 0
    if cur_player == 1:
        if possible_cities_player1[a][b] == 1:
            cities_player1[a][b] = 1
            settlements_player1[a][b] = 0

def city_possible_update():
    global possible_cities_player0
    global possible_cities_player1
    possible_cities_player0 = settlements_player0
    possible_cities_player1 = settlements_player1
    


def settlementplaced(a,b):
    settlements_available[a][b] = 0
    possible_settlements_player1[a][b] = 0
    possible_settlements_player0[a][b] = 0
    if a != 1 and a != 0:
        possible_settlements_player0[a-2][b] = 0
        possible_settlements_player1[a-2][b] = 0    
    if a != 9 and a != 10:
        possible_settlements_player0[a+2][b] = 0
        possible_settlements_player1[a+2][b] = 0
    
    if b != 20 and b != 21:
        possible_settlements_player0[a][b+2] = 0
        possible_settlements_player1[a][b+2] = 0
    if b != 1 and b != 0:
        possible_settlements_player1[a][b-2] = 0
        possible_settlements_player0[a][b-2] = 0

def roadplaced(a,b):
    print()

def player_places_road_placement(a,b):
    print(a,a)
    print(b,b,b)
    road_direction = 0
    print("roads available \n", roads_available)
    print("In which direction do you want to place the road") #would be good to later add it in the board, less searching
    while not int(road_direction) in range(1, 5):
        road_direction = int(input("   1   \n2     3\n   4   \n"))
        print (int(road_direction))
        if road_direction  == 1 and a != 0 and roads_available[a-1][b] != 0:
            print("first")
            roads_available[a-1][b] = 0
            if cur_player == 0:
                roads_player0[a-1][b] = 1            
            if cur_player == 1:
                roads_player1[a-1][b] = 1        
        elif road_direction == 2 and b != 0 and roads_available[a][b-1] != 0:
            print("second")
            roads_available[a][b-1] = 0
            if cur_player == 0:
                roads_player0[a][b-1] = 1            
            if cur_player == 1:
                roads_player1[a][b-1] = 1        
        elif road_direction  == 3 and b != 20 and roads_available[a][b+1] != 0:
            print("third")
            roads_available[a][b+1] = 0
            if cur_player == 0:
                roads_player0[a][b+1] = 1           
            if cur_player == 1:
                roads_player1[a][b+1] = 1        
        elif road_direction  == 4 and a != 10 and roads_available[a+1][b] != 0:
            print("fourth")
            roads_available[a+1][b] = 0
            if cur_player == 0:
                roads_player0[a+1][b] = 1            
            if cur_player == 1:
                roads_player1[a+1][b] = 1      
        else:
            road_direction = 0  
            print("This is not a possible direction")
        
        print("You have chosen the direction",road_direction)
        print(int(road_direction))

def agent_places_road_placement(a,b):
    road_direction = 0
    while road_direction == 0:
        road_direction = random.randint(1,4)
        if road_direction  == 1 and a != 0 and roads_available[a-1][b] != 0:
            roads_available[a-1][b] = 0
            if cur_player == 0:
                roads_player0[a-1][b] = 1            
            if cur_player == 1:
                roads_player1[a-1][b] = 1        
        elif road_direction == 2 and b != 0 and roads_available[a][b-1] != 0:
            roads_available[a][b-1] = 0
            if cur_player == 0:
                roads_player0[a][b-1] = 1            
            if cur_player == 1:
                roads_player1[a][b-1] = 1        
        elif road_direction  == 1 and b != 20 and roads_available[a][b+1] != 0:
            roads_available[a][b+1] = 0
            if cur_player == 0:
                roads_player0[a][b+1] = 1            
            if cur_player == 1:
                roads_player1[a][b+1] = 1        
        elif road_direction  == 1 and a != 10 and roads_available[a+1][b] != 0:
            roads_available[a+1][b] = 0
            if cur_player == 0:
                roads_player0[a+1][b] = 1            
            if cur_player == 1:
                roads_player1[a+1][b] = 1      
        else:
            road_direction = 0  
        
        print("The RL agent has chosen the in direction",road_direction)

def agent_places_settlement():
    result = 0
    print("The RL agent is choosing first settlement placement")
    while result == 0:
        input_row = random.randint(0,10)
        input_column = random.randint(0,20)
        print("input column", input_column)
        print("input_row",input_row)
        result = settlement_place(input_row, input_column)
    print("The RL agent has chosen the settlement in column ",input_column + 1," and row ",input_row + 1)

    agent_places_road_placement(input_row,input_column)

    result = 0

def player_places_settlement():
    result = 0  # Use a different variable name
    while result == 0:
        input_row = 0
        input_column = 0
        print("These are your possible settlements \n", possible_settlements_player0)
        while not int(input_row) in range(1, 12):
            input_row = int(input("In which row do you want to set your settlement (Number between 1-11) \n"))
        while not int(input_column) in range(1, 22):
            input_column = int(input("In which column do you want to set your settlement (Number between 1-21) \n"))
        result = settlement_place(input_row - 1, input_column - 1)
        if result == 0:
            print("This is not a possible settlement place. Please try again")
    tile_update_rewards(input_row - 1, input_column - 1)
    player_places_road_placement(input_row - 1, input_column - 1)
    
    result = 0

def placement_phase():
    if testing:    
        reset()
        harbor_distribution()
        reset()
        harbor_distribution()
        reset()
        harbor_distribution()


        print("Welcome to Settlers of Catan.")
        player = 5
        
        while not int(player) in range(0, 4):
            player = input("Choose the player that you want to play(0 = player-agent,1 = agent-player,2 = player-player,3 = agent-agent)\n")
            player = int(player)
        print("Let's start with the placement phase")
        print("You're playing as", player)

        print("Randomly building board......")
        print("lumber = 1, wool = 2, ")
        tile_distribution()
        print(pretty_tiles)
        print(lumber_tiles)
        print("Randomly assigning dice numbers......")
        plate_distribution()
        print(dice_tiles)
        print("player", player)
        print(board)
        if (player == 0):
            print("You're starting, place your first settlement")
            player_places_settlement()
            move_finished()
            agent_places_settlement()
            agent_places_settlement()
            move_finished()
            player_places_settlement()
        if (player == 1):
            print("The RL agent is startingm, you're place the second and third settlement")
            agent_places_settlement()
            player_places_settlement()
            player_places_settlement()
            agent_places_settlement()
        if (player == 2):
            cur_player2 = 0
            while not int(cur_player2) in range(1, 3):
                cur_player2 = input("Which player wants to start?")
            cur_player = cur_player2
            player_places_settlement()
            cur_player = 1 - cur_player
            player_places_settlement()
            player_places_settlement()
            cur_player = 1 - cur_player
            player_places_settlement()
        if (player == 3):
            for i in range(0,20):
                agent_places_settlement()
        
def roll_dice(): 
    cur_player = 0
    global player0_lumber
    global player0_wool
    global player0_grain
    global player0_brick
    global player0_ore

    global player1_lumber
    global player1_wool
    global player1_grain
    global player1_brick
    global player1_ore

    roll = np.random.choice(np.arange(2, 13), p=[1/36,2/36,3/36,4/36,5/36,6/36,5/36,4/36,3/36,2/36,1/36])
    dice_prob_count[roll][0] += 1
    if testing:
        print("A",roll, "has been rolled") 
    for i in range (0,11,1):
        for j in range(0,21,1):
            if dice_tiles[i][j] == roll:
                if possible_rewards_player0[i][j] != 0:
                    
                    if lumber_tiles[i][j] == 1:
                        player0_lumber += possible_rewards_player0[i][j]
                    
                    if wool_tiles[i][j] == 1:
                        player0_wool += possible_rewards_player0[i][j]
                    
                    if grain_tiles[i][j] == 1:
                        player0_grain += possible_rewards_player0[i][j]
                    
                    if brick_tiles[i][j] == 1:
                        player0_brick += possible_rewards_player0[i][j]
                    
                    if ore_tiles[i][j] == 1:
                        player0_ore += possible_rewards_player0[i][j]
                
                if possible_rewards_player1[i][j] != 0:
                    
                    if lumber_tiles[i][j] == 1:
                        player1_lumber += possible_rewards_player1[i][j]
                    
                    if wool_tiles[i][j] == 1:
                        player1_wool += possible_rewards_player1[i][j]
                    
                    if grain_tiles[i][j] == 1:
                        player1_grain += possible_rewards_player1[i][j]
                    
                    if brick_tiles[i][j] == 1:
                        player1_brick += possible_rewards_player1[i][j]
                    
                    if ore_tiles[i][j] == 1:
                        player1_ore += possible_rewards_player1[i][j]
    
    if testing: 
        print("You have",player0_lumber,"lumber")
        print("You have",player0_wool,"wool")
        print("You have",player0_grain,"grain")
        print("You have",player0_brick,"brick")
        print("You have",player0_ore,"ore")

        print("Opponent has",player1_lumber,"lumber")
        print("Opponent has",player1_wool,"wool")
        print("Opponent has",player1_grain,"grain")
        print("Opponent has",player1_brick,"brick")
        print("Opponent has",player1_ore,"ore")

    

def buy_development_cards():
    global player0_wool
    global player0_grain
    global player0_ore
    if player0_wool > 0 and player0_grain > 0 and player0_ore > 0 and development_cards_bought != 25:
        development_card_distribution()
        player0_wool -= 1
        player0_grain -= 1 
        player0_ore -= 1 
    print()
def buy_road(a,b):
    global player0_brick
    global player0_lumber



    if player0_brick > 0 and player0_lumber > 0:
            road_place(a,b)
            player0_brick -= 1
            player0_lumber -= 1


def buy_settlement(a,b):
    global player0_brick
    global player0_lumber
    global player0_wool
    global player0_grain

    possible = 0

    if player0_brick > 0 and player0_lumber > 0 and player0_grain > 0 and player0_ore > 0:
            possible = settlement_place(a,b)
            if possible == 1:
                player0_lumber -= 1
                player0_brick -= 1
                player0_brick -= 1 
                player0_grain -= 1
                possible
            else:
                print("This is not a possible action")


            
def buy_city(a,b):
    global player0_ore
    global player0_grain

    possible = 0
    if player0_grain < 1 and player0_ore < 2:
        possible = city_place()
        if possible == 1:
            player0_grain -= 2
            player0_ore -= 3   
    print()
def steal_card():
    player1_ressources_total = player1_lumber + player1_brick + player1_wool + player1_grain + player1_ore
    random_ressource = np.random.choice(np.arange(1, 6), p=[player1_lumber/player1_ressources_total, player1_brick/player1_ressources_total, player1_wool/player1_ressources_total, player1_grain/player1_ressources_total, player1_ore/player1_ressources_total])
    if random_ressource == 1:
        player1_lumber = player1_lumber - 1
        player0_lumber = player0_lumber + 1
    if random_ressource == 2:
        player1_brick = player1_brick - 1
        player0_brick = player0_brick + 1
    if random_ressource == 3:
        player1_wool = player1_wool - 1
        player0_wool = player0_wool + 1
    if random_ressource == 4:
        player1_grain = player1_grain - 1
        player0_grain = player0_grain + 1
    if random_ressource == 5:
        player1_ore = player1_ore - 1
        player0_ore = player0_ore + 1

def play_knight():
    if player0_knight_cards_old > 0:
        move_rober()
        steal_card()
        player0_knight_cards_old = player0_knight_cards_old - 1
        player0_knight_cards_played = player0_knight_cards_played + 1
    print()
def move_rober(a,b):
    if rober_position[a][b] != 1:
        rober_position = rober_position*zeroboard
        rober_position[a][b] = 1
    possible_rewards_player0 = possible_rewards_player0 * (1 - rober_position)
    possible_rewards_player1 = possible_rewards_player1 * (1- rober_position)

def activate_yearofplenty_func(ressource1,ressource2):
    if player0_yearofplenty_cards_old > 0:
        player0_yearofplenty_cards_old = player0_yearofplenty_cards_old - 1 
        if ressource1 == 1:
            player0_lumber += 1
        if ressource1 == 1:
            player0_lumber = player0_lumber + 1
        if ressource1 == 2:
            player0_brick = player0_brick + 1
        if ressource1 == 3:
            player0_wool = player0_wool + 1
        if ressource1 == 4:
            player0_grain = player0_grain + 1
        if ressource1 == 5:
            player0_ore = player0_ore + 1
        if ressource2 == 1:
            player0_lumber = player0_lumber + 1
        if ressource2 == 2:
            player0_brick = player0_brick + 1
        if ressource2 == 3:
            player0_wool = player0_wool + 1
        if ressource2 == 4:
            player0_grain = player0_grain + 1
        if ressource2 == 5:
            player0_ore = player0_ore + 1
    print()
def activate_monopoly_func(ressource):
    if player0_monopoly_cards_old > 0:
        player0_monopoly_cards_old = player0_monopoly_cards_old - 1
        if ressource == 1:
            player0_lumber = player0_lumber + player1_lumber
            player1_lumber = 0
        if ressource == 2:
            player0_wool = player0_wool + player1_wool
            player1_wool = 0
        if ressource == 3:
            player0_grain = player0_grain + player1_grain
            player1_grain = 0
        if ressource == 4:
            player0_brick = player0_brick + player1_brick
            player1_brick = 0
        if ressource == 5:
            player0_ore = player0_ore + player1_ore
            player1_ore = 0
    

def activate_road_building_func(a1,b1,a2,b2):
    if player0_roadbuilding_cards_old > 0:
        player0_roadbuilding_cards_old = player0_roadbuilding_cards_old - 1
        road_place(a1,b1)
        road_place(a2,b2)
    
    print()
def trading(give, get):
    if give == player0_brick and (brick_harbor * settlements_player0 + brick_harbor * cities_player0) != zeroboard:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player0_lumber and (lumber_harbor * settlements_player0 + lumber_harbor * cities_player0) != zeroboard:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player0_wool and (wool_harbor * settlements_player0 + wool_harbor * cities_player0) != zeroboard:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player0_grain and (grain_harbor * settlements_player0 + grain_harbor * cities_player0) != zeroboard:
        if give < 1:
            give -= 2 
            get += 1 
    elif give == player0_ore and (ore_harbor * settlements_player0 + ore_harbor * cities_player0) != zeroboard:
        if give < 1:
            give -= 2 
            get += 1 
    elif (three_one_harbor * settlements_player0 + three_one_harbor * cities_player0) != zeroboard:
        if give < 2:
            give -= 3 
            get += 1
    elif give < 3:
        give -= 4
        get += 1
def discard_ressources():
    total_ressources = player0_lumber + player0_brick + player0_grain + player0_ore + player0_wool 
    ressources_keeping = np.zeros = ((1,4))
    for i in range (1,5):
        ressources_keeping[i] = 0 #number between 1 and 5 for ressource type


    #remove ressource
    for i in range(0,math.ceil(total_ressources/2)-4,1):
        randomly_pick_ressources()


    print()    
def randomly_pick_ressources():
    possible_ressources_left = np.zeros((5)) #if there are still one of those ressources available after picking the first four
    if player1_lumber != 0:
        possible_ressources_left[0] = 1
    if player1_wool != 0:
        possible_ressources_left[1] = 1
    if player1_grain != 0:
        possible_ressources_left[2] = 1
    if player1_brick != 0:
        possible_ressources_left[3] = 1
    if player1_ore != 0:
        possible_ressources_left[4] = 1

    numbers = np.random.choice(np.arange(1,6),possible_ressources_left[0]/possible_ressources_left.sum(),possible_ressources_left[1]/possible_ressources_left.sum(),possible_ressources_left[2]/possible_ressources_left.sum(),possible_ressources_left[3]/possible_ressources_left.sum(),possible_ressources_left[4]/possible_ressources_left.sum())
    if numbers == 1:
        player0_lumber += 1 
    if numbers == 2:
        player0_brick += 1 
    if numbers == 3:
        player0_wool += 1 
    if numbers == 4:
        player0_grain += 1 
    if numbers == 5:
        player0_ore += 1 
    
def move_finished():
    global cur_player
    cur_player = 1 - cur_player

    global player0_knight_cards_new 
    global player0_victorypoints_cards_new 
    global player0_yearofplenty_cards_new 
    global player0_monopoly_cards_new 
    global player0_roadbuilding_cards_new 

    global player1_knight_cards_new 
    global player1_victorypoints_cards_new 
    global player1_yearofplenty_cards_new 
    global player1_monopoly_cards_new 
    global player1_roadbuilding_cards_new

    global player0_knight_cards_old 
    global player0_victorypoints_cards_old 
    global player0_yearofplenty_cards_old 
    global player0_monopoly_cards_old 
    global player0_roadbuilding_cards_old 

    global player1_knight_cards_old 
    global player1_victorypoints_cards_old 
    global player1_yearofplenty_cards_old 
    global player1_monopoly_cards_old 
    global player1_roadbuilding_cards_old

    player0_knight_cards_old += player0_knight_cards_new
    player0_victorypoints_cards_old += player0_victorypoints_cards_new
    player0_yearofplenty_cards_old += player0_yearofplenty_cards_new
    player0_monopoly_cards_old += player0_monopoly_cards_new
    player0_roadbuilding_cards_old += player0_roadbuilding_cards_new

    player1_knight_cards_old += player1_knight_cards_new
    player1_victorypoints_cards_old += player1_victorypoints_cards_new
    player1_yearofplenty_cards_old += player1_yearofplenty_cards_new
    player1_monopoly_cards_old += player1_monopoly_cards_new
    player1_roadbuilding_cards_old += player1_roadbuilding_cards_new

    player0_knight_cards_new = 0
    player0_victorypoints_cards_new = 0 
    player0_yearofplenty_cards_new = 0
    player0_monopoly_cards_new = 0
    player0_roadbuilding_cards_new = 0 
    player1_knight_cards_new = 0
    player1_victorypoints_cards_new = 0 
    player1_yearofplenty_cards_new = 0
    player1_monopoly_cards_new = 0
    player1_roadbuilding_cards_new = 0 


placement_phase()
for i in range (0,50,1):
    print(i)
    roll_dice()
print(dice_probabilities)
print(dice_tiles)
print(settlements_player0)
print(possible_rewards_player0)

print(pretty_tiles)
print("lumber",lumber_tiles)
print(wool_tiles)
print(grain_tiles)
print(brick_tiles)
print(ore_tiles)
print(dice_prob_count)

