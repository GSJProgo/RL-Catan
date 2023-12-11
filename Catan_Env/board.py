import numpy as np


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

board = Board()
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
            print(i,j)
            print(board.settlements_available)


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