import numpy as np


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

    def tile_distribution(self, board):
        a = 0
        for i in range (1,11,1):
            for j in range(1,21,1):
                if board.TILES_POSSIBLE[i][j] == 1:
                    if self.tile_random_numbers[a-1] == 0:
                        board.rober_position[i][j] = 1
                    elif self.tile_random_numbers[a-1] == 1:
                        board.tiles_lumber[i][j] = 1
                    elif self.tile_random_numbers[a-1] == 2:
                        board.tiles_wool[i][j] = 1
                    elif self.tile_random_numbers[a-1] == 3:
                        board.tiles_grain[i][j] = 1
                    elif self.tile_random_numbers[a-1] == 4:
                        board.tiles_brick[i][j] = 1
                    elif self.tile_random_numbers[a-1] == 5:
                        board.tiles_ore[i][j] = 1
                    a += 1 

            
    def harbor_distribution(self, board):
        for i in range(0,9,1):
            x1 = int(board.harbors_possible[i][0][0])
            y1 = int(board.harbors_possible[i][0][1])
            x2 = int(board.harbors_possible[i][1][0])
            y2 = int(board.harbors_possible[i][1][1])

            if self.harbor_random_numbers[i] == 1:
                board.harbor_lumber[x1][y1] = 1
                board.harbor_lumber[x2][y2] = 1
            elif self.harbor_random_numbers[i] == 2:
                board.harbor_wool[x1][y1] = 1
                board.harbor_wool[x2][y2] = 1
            elif self.harbor_random_numbers[i] == 3:
                board.harbor_grain[x1][y1] = 1
                board.harbor_grain[x2][y2] = 1
            elif self.harbor_random_numbers[i] == 4:
                board.harbor_brick[x1][y1] = 1
                board.harbor_brick[x2][y2] = 1
            elif self.harbor_random_numbers[i] == 5:
                board.harbor_ore[x1][y1] = 1
                board.harbor_ore[x2][y2] = 1
            elif self.harbor_random_numbers[i] == 6:
                board.harbor_three_one[x1][y1] = 1
                board.harbor_three_one[x2][y2] = 1

            
    def plate_distribution(self, board):
        a = 0
        for i in range (1,11,1):
            for j in range (1,21,1):
                if board.TILES_POSSIBLE[i][j] == 1 and board.rober_position[i][j] == 0:
                    board.tiles_dice[i][j] = self.plate_random_numbers[a-1]
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

