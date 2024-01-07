import numpy as np
import math 
import random

from Catan_Env.player import Player
from Catan_Env.randomtesting import Random_Testing
from Catan_Env.distribution import Distribution
from Catan_Env.game import Game
from Catan_Env.phase import Phase
from Catan_Env.board import Board
from Catan_Env.distribution import Distribution

_NUM_ROWS = 11
_NUM_COLS = 21

def create_env():
    """
    Create the Catan environment.

    Returns:
        Catan_Env: The Catan environment.
    """
    return Catan_Env()

class Catan_Env:
    def __init__(self):

        self.board = Board()

        self.player0 = Player()
        self.player1 = Player()

        self.player0_log = self.player0.Log()
        self.player1_log = self.player1.Log()

        self.player_log = [self.player0_log, self.player1_log]

        self.player0_action = self.player0.Action()
        self.player1_action = self.player1.Action()

        self.player_action = [self.player0_action, self.player1_action]


        self.player0_keepresources = self.player0.Keepresources()
        self.player1_keepresources = self.player1.Keepresources()

        self.player_keepresources = [self.player0_keepresources, self.player1_keepresources]

        self.player0_trading = self.player0.Trading()
        self.player1_trading = self.player1.Trading()

        self.player_trading = [self.player0_trading, self.player1_trading]

        self.players = [self.player0, self.player1]

        self.game = Game()

        self.random_testing = Random_Testing()
        self.distribution = Distribution()

        self.phase = Phase()

        self.total_step = 0
        

        self.legal_actions = np.zeros((1,965))


    def development_card_choose(self):
        """
        This function represents the action of choosing a development card.
        It updates the player's card counts based on the randomly chosen card.
        It also updates the game phase and returns 1.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        distribution = self.distribution
        phase = self.phase

        random_testing.development_card_choose += 1


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

    def tile_update_rewards(self,a, b):
        """
        Update the rewards for the tiles adjacent to the given coordinates (a, b).

        Args:
            a (int): The row index of the placed settlement or city.
            b (int): The column index of the placed settlement or city.

        Returns:
            None
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        random_testing.tile_update_rewards += 1
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


    def settlement_place(self,a,b):
        """
        Places a settlement on the Catan board at the specified coordinates (a, b).

        Args:
            a (int): The row index of the settlement.
            b (int): The column index of the settlement.

        Returns:
            int: 1 if the settlement was successfully placed, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        board = self.board
        player0 = self.player0
        player1 = self.player1
        random_testing.settlement_place += 1

        board.settlements_used = (1-player0.settlements)*(1-player1.settlements)
        board.settlements_free = board.settlements_available * board.settlements_used
        if player.settlements_left > 0:
            if board.settlements_free[a][b] == 1 and self.settlement_possible_check(a,b,0) == 1:
                player.settlements[a][b] = 1
                player.settlements_left -= 1
                self.tile_update_rewards(a,b)
                player.victorypoints += 1
                random_testing.successful_settlement_place += 1
                
                phase.statechange = 1
                return 1 
            return 0


    def settlement_place_placement(self,a, b):
        """
        Places a settlement on the Catan board in the placement phase at the specified coordinates (a, b).

        Args:
            a (int): The row index of the settlement.
            b (int): The column index of the settlement.

        Returns:
            int: 1 if the settlement placement is successful, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.settlement_place_placement += 1
        player = self.players[self.game.cur_player]  # Define "player" using "self.players"
        board = self.board  # Define "board"
        player0 = self.player0  # Define "player0"
        player1 = self.player1  # Define "player1"
        board.settlements_used = (1 - player0.settlements) * (1 - player1.settlements)
        board.settlements_free = board.settlements_available * board.settlements_used

        if board.settlements_free[a][b] == 1 and self.settlement_possible_check(a, b, 1) == 1:  # Use "self.settlement_possible_check" instead of "settlement_possible_check"
            player.settlements[a][b] = 1
            self.tile_update_rewards(a, b)  # Use "self.tile_update_rewards" instead of "tile_update_rewards"
            player.victorypoints += 1
            
            phase.statechange = 1
            return 1
        return 0


    def settlement_possible_check(self,a, b, c):
        """
        Check if it is possible to build a settlement at the given coordinates.

        Args:
            a (int): The row index of the settlement.
            b (int): The column index of the settlement.
            c (int): A flag indicating if the settlement is being placed in the placement phase (1) or not (0).

        Returns:
            int: 1 if it is possible to build a settlement, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
 
        random_testing.settlement_possible_check += 1

        player0 = self.player0
        player1 = self.player1

        tworoads = 0

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
            if b != 20 and player.roads[a][b + 1] == 1:
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

                        
    def road_place(self,a, b):
        """
        Place a road on the game board.

        Args:
            a (int): The row index of the road.
            b (int): The column index of the road.

        Returns:
            int: 1 if the road placement was successful, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.road_place += 1
        possible = 0
        possible = self.road_possible_check(a, b)
        if player.roads_left > 0:
            if possible == 1:
                player.roads[a][b] = 1
                player.roads_left -= 1
                self.update_longest_road()
                random_testing.successful_road_place += 1
                phase.statechange = 1
                
                return 1
        return 0


    def road_place_card(self,a, b, c, d):
        """
        Use a road building development card to build 2 roads.

        Args:
            a (int): The row index of the first road.
            b (int): The column index of the first road.
            a (int): The row index of the second road.
            b (int): The column index of the second settlement.

        Returns:
            int: 1 if both of the road placement were successful, 0 otherwise.
        """
        

        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        if a == c and b == d:
            return 0 
        possible = 0
        possible = self.road_possible_check(a, b)
        possible2 = 0
        possible2 = self.road_possible_check(c, d)
        if player.roads_left > 1:
            if possible == 1 and possible2 == 1:
                player.roads[a][b] = 1
                player.roads[a][b] = 1
                player.roads_left -= 2
                self.update_longest_road()
                player.roadbuilding_cards_old -= 1
                random_testing.successful_road_place += 1
                phase.statechange = 1
                
                return 1 
        return 0


    def road_place_placement(self,settlement_a, settlement_b, road_a, road_b):
        """
        Places a road on the board in the placement phase.

        Args:
            settlement_a (int): Index of the first settlement.
            settlement_b (int): Index of the second settlement.
            road_a (int): Index of the first road.
            road_b (int): Index of the second road.

        Returns:
            int: 1 if the road placement is valid and successful, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.road_place_placement += 1
        if ((((road_a + 1) == settlement_a or (road_a - 1)  == settlement_a) and road_b == settlement_b) or (((road_b + 1) == settlement_b or (road_b - 1)  == settlement_b) and road_a == settlement_a)):
            player.roads[road_a][road_b] = 1
            player.roads_left -= 1
            self.update_longest_road()
            phase.statechange = 1
            
            return 1 
        return 0


    def road_possible_check(self,a, b):
        """
        Check if a road is possible to be built at the given coordinates (a, b).

        Args:
        a (int): The row index of the coordinate.
        b (int): The column index of the coordinate.

        Returns:
        int: 1 if a road is possible, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]

        random_testing.road_possible_check += 1
        board = self.board
        player0 = self.player0
        player1 = self.player1
        game = self.game
        players = self.players

        board.roads_free = board.roads_available * (1 - player0.roads) * (1 - player1.roads)
        player = players[game.cur_player]
        opponent = players[1 - game.cur_player]

        player.roads_possible = (1 - board.ZEROBOARD)

        if board.roads_free[a][b] == 0:
            return 0
        if b != 1 and b != 0:
            if player.roads[a][b - 2] == 1:
                if opponent.settlements[a][b - 1] == 0:
                    return 1
        if b != 20 and b != 19:
            if player.roads[a][b + 2] == 1:
                if opponent.settlements[a][b + 1] == 0:
                    return 1
        if b != 20 and a != 20:
            if player.roads[a + 1][b + 1] == 1:
                if opponent.settlements[a + 1][b] == 0 and opponent.settlements[a][b + 1] == 0:
                    return 1
        if b != 0 and a != 20:
            if player.roads[a + 1][b - 1] == 1:
                if opponent.settlements[a + 1][b] == 0 and opponent.settlements[a][b - 1] == 0:
                    return 1
        if b != 0 and a != 0:
            if player.roads[a - 1][b - 1] == 1:
                if opponent.settlements[a + 1][b] == 0 and opponent.settlements[a][b - 1] == 0:
                    return 1
        if b != 20 and a != 0:
            if player.roads[a - 1][b + 1] == 1:
                if opponent.settlements[a - 1][b] == 0 and opponent.settlements[a][b + 1] == 0:
                    return 1
        return 0


    def city_place(self,a, b):
        """
        Place a city on the specified coordinates (a, b) on the game board.

        Args:
            a (int): The row index of the city.
            b (int): The column index of the city.

        Returns:
            int: 1 if the city placement is successful, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.city_place += 1
        # still need to add a max cities check, the same comes to settlements
        players = self.players
        game = self.game
        player = players[game.cur_player]
        if player.cities_left > 0:    
            if player.settlements[a][b] == 1:
                player.cities[a][b] = 1
                player.cities_left -= 1
                player.settlements[a][b] = 0
                player.settlements_left += 1
                self.tile_update_rewards(a, b)
                player.victorypoints += 1
                random_testing.successful_city_place += 1
                phase.statechange = 1
                
                return 1
            return 0


    def roll_dice(self):
        """
        Simulates rolling a pair of dice and updates player resources based on the roll.

        Returns:
            int: The sum of the pair of dice.

        """
        board = self.board
        random_testing = self.random_testing
        player0 = self.player0
        player1 = self.player1
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
                        #phase.reward += player0.rewards_possible[i][j] * 0.0002

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
                        #phase.reward += player0.rewards_possible[i][j] * 0.0002
                    self.player_log[self.game.cur_player].total_resources_found += player0.rewards_possible[i][j]
        
        return roll


    def buy_development_cards(self):
        """
        Function to simulate the action of a player buying development cards in the game of Catan.

        Returns:
            int: 1 if the action was successful, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        distribution = self.distribution
        phase = self.phase
        random_testing.buy_development_cards += 1
        player_log = self.player_log
        game = self.game
        possible = 0
        if player.resource_wool > 0 and player.resource_grain > 0 and player.resource_ore > 0 and distribution.development_cards_bought < 25:
            possible = self.development_card_choose()
            if possible == 1:
                self.find_largest_army()
                player.resource_wool -= 1
                player.resource_grain -= 1 
                player.resource_ore -= 1 
                phase.statechange = 1
                
                player_log[game.cur_player].total_development_cards_bought += 1
                return 1
        return 0
            


    def buy_road(self,a,b):
        """
        Buys a road in the game of Catan.

        Args:
        a (int): The row index of the road.
        b (int): The column index of the road.

        Returns:
        int: 1 if the road was successfully bought, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.buy_road += 1
        possible = 0
        game = self.game
        player_log = self.player_log
        if player.resource_brick > 0 and player.resource_lumber > 0:
                possible = self.road_place(a,b)
                if possible == 1:
                    player.resource_brick -= 1
                    player.resource_lumber -= 1
                    phase.statechange = 1
                    
                    player_log[game.cur_player].total_roads_built += 1
                    return 1
        return 0


    def buy_settlement(self,a,b):
        """
        Function to buy a settlement in the game of Catan.

        Args:
        a (int): The row index of the settlement.
        b (int): The column index of the settlement.

        Returns:
        int: 1 if the settlement was successfully bought, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.buy_settlement += 1
        possible = 0

        player_log = self.player_log
        game = self.game

        if player.resource_brick > 0 and player.resource_lumber > 0 and player.resource_grain > 0 and player.resource_wool > 0:
            possible = self.settlement_place(a,b)
            if possible == 1:
                player.resource_lumber -= 1
                player.resource_brick -= 1
                player.resource_wool -= 1 
                player.resource_grain -= 1
                phase.statechange = 1
                
                player_log[game.cur_player].total_settlements_built += 1
                return 1 
        return 0

            
    def buy_city(self,a, b):
        """
        Buys a city in the Catan game.

        Args:
            a (int): The row index of the city.
            b (int): The column index of the city.

        Returns:
            int: 1 if the city was successfully bought, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.buy_city += 1
        player_log = self.player_log
        game = self.game
        possible = 0
        if player.resource_grain > 1 and player.resource_ore > 2:
            possible = self.city_place(a, b)
            if possible == 1:
                player.resource_grain -= 2
                player.resource_ore -= 3  
                phase.statechange = 1
                
                player_log[game.cur_player].total_cities_built += 1
                return 1
        return 0


    def steal_card(self):
        """
        Function to simulate stealing a card from an opponent in the game of Catan.

        This function randomly selects a resource card from the opponent's resources and transfers it to the current player.
        The selection is based on the proportion of each resource card in the opponent's resources.

        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        random_testing.steal_card += 1
        game = self.game
        player_log = self.player_log
        opponent = self.players[1-game.cur_player]
        #phase.reward += 0.0004
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

            player_log[game.cur_player].total_resources_found += 1
            random_testing.steal_card += 1
        



    def play_knight(self,a,b):
        """
        Play a knight card in the game of Catan.

        Parameters:
        a (int): The row index on which the rober should be placed.
        b (int): The column index on which the settlement should be placed.

        Returns:
        int: 1 if the knight card was played successfully, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.play_knight += 1
        player_log = self.player_log
        game = self.game
        possible = 0
        if player.knight_cards_old > 0: #this is wrong, need to update that
            possible = self.move_rober(a,b)
            if possible == 1:
                player_log[game.cur_player].total_knights_played += 1
                self.steal_card()
                player.knight_cards_old -= 1
                player.knight_cards_played += 1
                phase.statechange = 1
                
                player_log[game.cur_player].total_development_cards_used += 1
                return 1
        return 0


    def move_rober(self,a, b):
        """
        Move the robber to the specified position on the board.

        Args:
            a (int): The row index to which the rober should be moved.
            b (int): The column index to which the rober should be moved.

        Returns:
            int: 1 if the move was successful, 0 otherwise.
        """
        random_testing = self.random_testing
        phase = self.phase
        board = self.board
        random_testing.move_rober += 1
        if board.rober_position[a][b] != 1 and board.TILES_POSSIBLE[a][b] == 1:
            board.rober_position = board.rober_position * board.ZEROBOARD
            board.rober_position[a][b] = 1
            random_testing.successful_move_rober += 1
            phase.statechange = 1
            
            return 1
        return 0


    def activate_yearofplenty_func(self,resource1, resource2):
        """
        Activates the Year of Plenty development card for the current player and adds the specified resources.

        Args:
            resource1 (int): The first resource to add (1 for lumber, 2 for wool, 3 for grain, 4 for brick, 5 for ore).
            resource2 (int): The second resource to add (1 for lumber, 2 for wool, 3 for grain, 4 for brick, 5 for ore).

        Returns:
            int: 1 if the Year of Plenty card was successfully activated, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.activate_yearofplenty_func += 1
        # need to take a look at this later. I'm not sure how to convert those resources.
        player_log = self.player_log
        game = self.game
        if player.yearofplenty_cards_old > 0:
            player.yearofplenty_cards_old -= 1
            if resource1 == 1:
                player.resource_lumber += 1
            if resource1 == 1:
                player.resource_lumber += 1
            elif resource1 == 2:
                player.resource_brick += 1
            elif resource1 == 3:
                player.resource_wool += 1
            elif resource1 == 4:
                player.resource_grain += 1
            elif resource1 == 5:
                player.resource_ore += 1
            if resource2 == 1:
                player.resource_lumber += 1
            elif resource2 == 2:
                player.resource_brick += 1
            elif resource2 == 3:
                player.resource_wool += 1
            elif resource2 == 4:
                player.resource_grain += 1
            elif resource2 == 5:
                player.resource_ore += 1
            random_testing.successful_activate_yearofplenty_func += 1
            # phase.reward += 0.0008
            player_log[game.cur_player].total_resources_found += 2
            phase.statechange = 1
            
            player_log[game.cur_player].total_development_cards_used += 1
            return 1
        return 0


    def activate_monopoly_func(self,resource):
        """
        Uses the Monopoly development card.

        Args:
            resource (int): The resource to steal from the opponents (1 for lumber, 2 for wool, 3 for grain, 4 for brick, 5 for ore).

        Returns:
            int: 1 if the monopoly function was successfully activated, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.activate_monopoly_func += 1
        player_log = self.player_log
        game = self.game
        opponent = self.players[1-game.cur_player]
        if player.monopoly_cards_old > 0:
            player.monopoly_cards_old = player.monopoly_cards_old - 1
            if resource == 1:
                player.resource_lumber = player.resource_lumber + opponent.resource_lumber
                opponent.resource_lumber = 0
                #phase.reward += 0.0004 * opponent.resource_lumber
                player_log[game.cur_player].total_resources_found += opponent.resource_lumber
            elif resource == 2:
                player.resource_wool = player.resource_wool + opponent.resource_wool
                opponent.resource_wool = 0
                #phase.reward += 0.0004 * opponent.resource_wool
                player_log[game.cur_player].total_resources_found += opponent.resource_wool
            elif resource == 3:
                player.resource_grain = player.resource_grain + opponent.resource_grain
                opponent.resource_grain = 0
                #phase.reward += 0.0004 * opponent.resource_grain
                player_log[game.cur_player].total_resources_found += opponent.resource_grain
            elif resource == 4:
                player.resource_brick = player.resource_brick + opponent.resource_brick
                opponent.resource_brick = 0
                #phase.reward += 0.0004 * opponent.resource_brick
                player_log[game.cur_player].total_resources_found += opponent.resource_brick
            elif resource == 5:
                player.resource_ore = player.resource_ore + opponent.resource_ore
                opponent.resource_ore = 0
                #phase.reward += 0.0004 * opponent.resource_ore
                player_log[game.cur_player].total_resources_found += opponent.resource_ore
            
            random_testing.successful_activate_monopoly_func += 1
            phase.statechange = 1
            
            player_log[game.cur_player].total_development_cards_used += 1
            return 1
        return 0

    
    def activate_road_building_func(self,a1, b1, a2, b2):
        """
        Activates the road building function for the current player.

        Args:
            a1 (int): The row index of the first road placement.
            b1 (int): The column index of the first road placement.
            a2 (int): The row index of the second road placement.
            b2 (int): The column index of the second road placement.

        Returns:
            int: 1 if the road building function was successfully activated, 0 otherwise.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.activate_road_building_func += 1
        player_log = self.player_log
        game = self.game
        if player.roadbuilding_cards_old > 0:
            possible1 = self.road_possible_check(a1, b1)
            if possible1 == 1:
                self.road_place(a1, b1)
                possible2 = self.road_possible_check(a2, b2)
                if possible2 == 1:
                    self.road_place(a2, b2)
                    player.roadbuilding_cards_old = player.roadbuilding_cards_old - 1
                    random_testing.successful_activate_road_building_func += 1
                    phase.statechange = 1
                    
                    player_log[game.cur_player].total_development_cards_used += 1
                    return 1
                else:
                    player.roads[a1][b1] = 0
        return 0
        

    def trade_resources(self, give, get):
        """
        Trades resources with the bank or at harbors.

        Args:
            give (int): The resource to give.
            get (int): The resource to receive.

        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        player_log = self.player_log
        game = self.game
        a = 0
        random_testing.trade_resources += 1
        board = self.board
        if give == 1 and (board.harbor_lumber * player.settlements + board.harbor_lumber * player.cities).any() != 0:
            if player.resource_lumber > 1:
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
                phase.statechange = 1
                
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
            phase.statechange = 1
            
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
            phase.statechange = 1
            
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
            phase.statechange = 1
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
            phase.statechange = 1
            
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
            phase.statechange = 1
            
            
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
        if phase.statechange == 1:
            player_log[game.cur_player].total_resources_traded += 1



    def discard_resources(self,lumber, wool, grain, brick, ore):
        """
        Discard resources from the player's inventory until half of the ressources have been discarded. 

        There is always only 1 argument "1" and the rest "0"
        
        Args:
            lumber (int): The number of lumber resources to discard.
            wool (int): The number of wool resources to discard.
            grain (int): The number of grain resources to discard.
            brick (int): The number of brick resources to discard.
            ore (int): The number of ore resources to discard.
        """

        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.discard_resources += 1
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
        
        if player.discard_resources_turn > math.ceil(player.total_resources/2):
            print("ERROR: Discard resources turn is greater than half of total resources")
            print("Discard resources turn: ", player.discard_resources_turn)
            print("Total resources: ", player.total_resources)
        if player.discard_resources_lumber == 0 and player.discard_resources_wool == 0 and player.discard_resources_grain == 0 and player.discard_resources_brick == 0 and player.discard_resources_ore == 0:
            print("ERROR: All discard resources are 0")
        if player.discard_resources_turn >= math.ceil(player.total_resources/2) or (player.discard_resources_lumber == 0 and player.discard_resources_wool == 0 and player.discard_resources_grain == 0 and player.discard_resources_brick == 0 and player.discard_resources_ore == 0):
            player.discard_resources_lumber = 0
            player.discard_resources_wool = 0
            player.discard_resources_grain = 0
            player.discard_resources_brick = 0
            player.discard_resources_ore = 0
            player.discard_resources_turn = 0
            player.discard_resources_started = 0
            random_testing.successful_discard_resources += 1
            self.steal_card()

    def longest_road(self,i, j, prev_move):
        
        """
        Calculates the length of the longest road starting from position (i, j).

        Args:
            i (int): The row index of the starting position.
            j (int): The column index of the starting position.
            prev_move (tuple): The previous move made, represented as a tuple (i, j).

        Returns:
            int: The length of the longest road starting from position (i, j).
        """

        random_testing = self.random_testing
        board = self.board
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

        max_length = max(self.longest_road(x, y, (i, j)) for x, y in moves)
        return 1 + max_length
        

    def find_longest_road(self):
        """
        Finds the length of the longest road for the current player on the Catan board.

        Returns:
            int: The length of the longest road.
        """

        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        random_testing.find_longest_road += 1
        board = self.board

        ans, n, m = 0, 11, 21
        for i in range(n):
            for j in range(m):
                board.longest_road = player.roads * (1-board.ZEROBOARD)
                c = self.longest_road(i, j, (0, 0))
                if c > 0: 
                    ans = max(ans,c)
        return ans
    

    def update_longest_road(self):
        """
        Updates the longest road for the current player.
        If the current player has a longer road than the opponent and it is at least 5 roads long,
        the current player gains 2 victory points and the opponent loses 2 victory points if he held that longest road card until now.
        """
        player = self.players[self.game.cur_player]
        game = self.game
        opponent = self.players[1 - game.cur_player]
        player.roads_connected = self.find_longest_road()
        if player.roads_connected >= 5 and player.roads_connected > opponent.roads_connected and player.longest_road == 0:
            if opponent.longest_road == 1:
                opponent.longest_road = 0
                opponent.victorypoints -= 2
            player.longest_road = 1
            player.victorypoints += 2


    def find_largest_army(self):
        """
        Finds the player with the largest army. 
        If the player has activated more than 3 knight cards and has more knight cards than the opponent, he gains 2 victory points.
        """

        random_testing = self.random_testing
        player = self.players[self.game.cur_player]

        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        random_testing.find_largest_army += 1
        game = self.game
        opponent = self.players[1 - game.cur_player]
        if player.knight_cards_played >= 3 and player.knight_cards_played > opponent.knight_cards_played and player.largest_army == 0:
            if opponent.largest_army == 1:
                opponent.largest_army = 0
                opponent.victorypoints -= 2 
            player.largest_army = 1
            player.victorypoints += 2
        

    def move_finished(self):
        """
        Function to handle the completion of a player's move.
        This function updates various game state variables and rewards based on the player's actions.
        It also checks if the game is finished and starts a new game if necessary.
        """
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        game = self.game
        players = self.players

        player0_log = self.player0_log
        player1_log = self.player1_log

        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        phase = self.phase
        random_testing.move_finished += 1
        phase.statechange = 1
        

        player0 = self.player0
        player1 = self.player1

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

        #phase.reward = ((player0.victorypoints - player1.victorypoints) - (player0.victorypoints_before - player1.victorypoints_before))*0.02
        #if game.cur_player == 0:
            #phase.reward += (player0.victorypoints - player0.victorypoints_before) * 0.02
        #if game.cur_player == 1:
            #phase.reward += (player1.victorypoints - player1.victorypoints_before) * 0.02
        
        player0.victorypoints_before = player0.victorypoints
        player1.victorypoints_before = player1.victorypoints   
        if player.victorypoints >= 10:
            print("achievement unlocked")
            if game.cur_player == 0: 
                #phase.reward += (1 + (players[game.cur_player].victorypoints - players[1-game.cur_player].victorypoints) * 0.02 + (phase.statechangecount - phase.statechangecountafter) * 0.0001 - phase.gamemoves * 0.00002)
                phase.reward += 0.75 + (players[game.cur_player].victorypoints - players[1-game.cur_player].victorypoints) * 0.02
                print(phase.reward)
                phase.victoryreward = 1
                phase.victorypointreward = (players[game.cur_player].victorypoints - players[1-game.cur_player].victorypoints) * 0.02
                phase.legalmovesreward = (phase.statechangecount - phase.statechangecountafter) * 0.0002
                phase.illegalmovesreward = -phase.gamemoves * 0.00002
                player0.wins += 1
            else: 
                #phase.reward -= (1 + (players[game.cur_player].victorypoints - players[1-game.cur_player].victorypoints) * 0.02 - (phase.statechangecount - phase.statechangecountafter) * 0.0001 + phase.gamemoves * 0.00002)
                phase.reward -= (0.75 + (players[game.cur_player].victorypoints - players[1-game.cur_player].victorypoints) * 0.02)
                print(phase.reward)
                player1.wins += 1
                phase.victoryreward = -1
                phase.victorypointreward = (players[game.cur_player].victorypoints - players[1-game.cur_player].victorypoints) * 0.02
                phase.legalmovesreward = (phase.statechangecount - phase.statechangecountafter) * 0.0001
                phase.illegalmovesreward = -phase.gamemoves * 0.00002
            phase.statechangecountafter = phase.statechangecount
            random_testing.numberofgames += 1
            game.is_finished = 1
            player0_log.average_victory_points.insert(0, player0.victorypoints)
            if len(player0_log.average_victory_points) > 10:
                player0_log.average_victory_points.pop(10)
            player1_log.average_victory_points.insert(0, player1.victorypoints)
            if len(player1_log.average_victory_points) > 10:
                player1_log.average_victory_points.pop(10)
            player0_log.average_resources_found.insert(0, player0_log.total_resources_found)
            if len(player0_log.average_resources_found) > 10:
                player0_log.average_resources_found.pop(10)
            player1_log.average_resources_found.insert(0, player1_log.total_resources_found)
            if len(player1_log.average_resources_found) > 10:
                player1_log.average_resources_found.pop(10)
            player0_log.average_development_cards_bought.insert(0, player0_log.total_development_cards_bought)
            if len(player0_log.average_development_cards_bought) > 10:
                player0_log.average_development_cards_bought.pop(10)
            player1_log.average_development_cards_bought.insert(0, player1_log.total_development_cards_bought)
            if len(player1_log.average_development_cards_bought) > 10:
                player1_log.average_development_cards_bought.pop(10)
            player0_log.average_development_cards_used.insert(0, player0_log.total_development_cards_used)
            if len(player0_log.average_development_cards_used) > 10:
                player0_log.average_development_cards_used.pop(10)
            player1_log.average_development_cards_used.insert(0, player1_log.total_development_cards_used)
            if len(player1_log.average_development_cards_used) > 10:
                player1_log.average_development_cards_used.pop(10)
            player0_log.average_settlements_built.insert(0, player0_log.total_settlements_built)
            if len(player0_log.average_settlements_built) > 10:
                player0_log.average_settlements_built.pop(10)
            player1_log.average_settlements_built.insert(0, player1_log.total_settlements_built)
            if len(player1_log.average_settlements_built) > 10:
                player1_log.average_settlements_built.pop(10)
            player0_log.average_cities_built.insert(0, player0_log.total_cities_built)
            if len(player0_log.average_cities_built) > 10:
                player0_log.average_cities_built.pop(10)
            player1_log.average_cities_built.insert(0, player1_log.total_cities_built)
            if len(player1_log.average_cities_built) > 10:
                player1_log.average_cities_built.pop(10)
            player0_log.average_roads_built.insert(0, player0_log.total_roads_built)
            if len(player0_log.average_roads_built) > 10:
                player0_log.average_roads_built.pop(10)
            player1_log.average_roads_built.insert(0, player1_log.total_roads_built)
            if len(player1_log.average_roads_built) > 10:
                player1_log.average_roads_built.pop(10)
            player0_log.average_resources_traded.insert(0, player0_log.total_resources_traded)
            if len(player0_log.average_resources_traded) > 10:
                player0_log.average_resources_traded.pop(10)
            player1_log.average_resources_traded.insert(0, player1_log.total_resources_traded)
            if len(player1_log.average_resources_traded) > 10:
                player1_log.average_resources_traded.pop(10)
            player0_log.average_longest_road.insert(0,player0.roads_connected)
            if len(player0_log.average_longest_road) > 10:
                player0_log.average_longest_road.pop(10)
            player1_log.average_longest_road.insert(0,player1.roads_connected)
            if len(player1_log.average_longest_road) > 10:
                player1_log.average_longest_road.pop(10)

            player0_log.average_knights_played.insert(0, player0_log.total_knights_played)
            if len(player0_log.average_knights_played) > 10:
                player0_log.average_knights_played.pop(10)
            player1_log.average_knights_played.insert(0, player1_log.total_knights_played)
            if len(player1_log.average_knights_played) > 10:
                player1_log.average_knights_played.pop(10)

            game.average_win_ratio.insert(0, 1-game.cur_player)
            if len(game.average_win_ratio) > 20:
                game.average_win_ratio.pop(20)

            player0_log.total_knights_played = 0
            player1_log.total_knights_played = 0	
            player0_log.total_resources_found = 0
            player1_log.total_resources_found = 0
            player0_log.total_development_cards_bought = 0
            player1_log.total_development_cards_bought = 0
            player0_log.total_development_cards_used = 0
            player1_log.total_development_cards_used = 0
            player0_log.total_settlements_built = 0
            player1_log.total_settlements_built = 0
            player0_log.total_cities_built = 0
            player1_log.total_cities_built = 0
            player0_log.total_roads_built = 0
            player1_log.total_roads_built = 0
            player0_log.total_resources_traded = 0
            player1_log.total_resources_traded = 0


            

        game.cur_player = 1 - game.cur_player
        if game.placement_phase_pending != 1:
            self.turn_starts()

    def new_initial_state(self):
        player = self.players[self.game.cur_player]
        distribution = self.distribution
        board = self.board

        player0_log = self.player0_log
        player1_log = self.player1_log

        players = self.players

        game = self.game

        self.total_step = 0
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
        
        
        player0_log.total_resources_found = 0
        player1_log.total_resources_found = 0
        
        
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


        game.settlementplaced = 0
        game.placement_phase_pending = 0
        game.placement_phase_turns_made = 0
        game.placement_phase_settlement_turn = 0
        game.placement_phase_road_turn = 0
        game.seven_rolled = 0
        game.placement_phase_settlement_coordinate1 = 0
        game.placement_phase_settlement_coordinate2 = 0

    def setup(self):
        board = self.board
        board.harbors_building()
        board.tiles_buidling()
        board.settlements_building()
        board.roads_building()
        self.distribution.tile_distribution(self.board)
        self.distribution.harbor_distribution(self.board)
        self.distribution.plate_distribution(self.board)

    def turn_starts(self):
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        c = self.roll_dice()

        if player.resource_wool > 0 and player.resource_grain > 0 and player.resource_ore > 0:
            random_testing.resources_buy_dc += 1
        if player.resource_brick > 0 and player.resource_lumber > 0:
            random_testing.resources_buy_road += 1
        if player.resource_grain > 1 and player.resource_ore > 2:
            random_testing.resources_buy_city += 1
        if player.resource_brick > 0 and player.resource_lumber > 0 and player.resource_grain > 0 and player.resource_wool > 0:
            random_testing.resources_buy_settlement += 1
        if c == 7:
            total_resources = player.resource_lumber + player.resource_wool + player.resource_grain + player.resource_brick + player.resource_ore
            #if total_resources >= 7:
                #phase.reward = -0.0002*total_resources/2
            self.game.seven_rolled = 1
            

    def new_game(self):
        game = self.game    
        print("new game")
        self.new_initial_state()
        self.setup()
        game.placement_phase_pending = 1
        game.placement_phase_settlement_turn = 1


    def action_executor(self):
        phase = self.phase
        random_testing = self.random_testing
        player = self.players[self.game.cur_player]
        players = self.players
        game = self.game
        board = self.board
        player_action = self.player_action
        player_keepresources = self.player_keepresources
        player_trading = self.player_trading
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
                    self.move_rober(d,e)
                    if player.resource_lumber + player.resource_wool + player.resource_grain + player.resource_brick + player.resource_ore >= 7:
                        player.discard_first_time = 1
                        player.discard_resources_started = 1
                    else:
                        self.steal_card()
                    game.seven_rolled = 0
        if player.knight_move_pending == 1:
            if np.any(action.rober_move == 1): 
                b,c = np.where(action.rober_move == 1)
                d = int(b)
                e = int(c)
                self.play_knight(d,e)
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
                        self.road_place_card(player.roadbuilding_d,player.roadbuilding_e,d,e)
                        player.roadbuilding_move_pending = 0
                        player.roadbuilding1 = 0
                

                
        
        if game.placement_phase_pending == 1:
            if game.placement_phase_road_turn == 1:
                if np.any(action.road_place == 1):
                    b,c = np.where(action.road_place == 1)
                    d = int(b)
                    e = int(c)
                    #print(d,e)
                    #print("works")
                    if d < 11 and d >= 0 and e < 21 and e >= 0: 
                        possible = self.road_place_placement(game.placement_phase_settlement_coordinate1,game.placement_phase_settlement_coordinate2,d,e)
                        if possible == 1:
                            #print("road_place_placement")
                            game.placement_phase_road_turn = 0
                            game.placement_phase_settlement_turn = 1
                            game.placement_phase_turns_made += 1
                            #print(game.placement_phase_turns_made)
                            #print("player.settlements")
                            #print(player.settlements)
                            #print("player.roads")
                            #print(player.roads)
                            #print(random_testing.roll_dice)
                            if game.placement_phase_turns_made == 1:
                                self.move_finished()
                            if game.placement_phase_turns_made == 3:
                                self.move_finished()    
                            if game.placement_phase_turns_made == 4:
                                self.move_finished()
                                game.placement_phase_pending = 0
                                game.placement_phase_turns_made = 0
        
        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            random_testing.howmuchisthisaccsessed += 1
            if np.any(action.settlement_place == 1):
                b,c = np.where(action.settlement_place == 1)
                d = int(b)
                e = int(c)
                if d < 11 and d >= 0 and e < 21 and e >= 0: 
                    self.buy_settlement(d,e)
            if np.any(action.city_place == 1):
                b,c = np.where(action.city_place == 1)
                d = int(b)
                e = int(c)
                if d < 11 and d >= 0 and e < 21 and e >= 0: 
                    self.buy_city(d,e)
            if np.any(action.road_place == 1):
                b,c = np.where(action.road_place == 1)
                d = int(b)
                e = int(c)
                if d < 11 and d >= 0 and e < 21 and e >= 0: 
                    self.buy_road(d,e)

        if game.placement_phase_pending == 1:
            if game.placement_phase_settlement_turn == 1:
                if np.any(action.settlement_place == 1):
                    b,c = np.where(action.settlement_place == 1)
                    d = int(b)
                    e = int(c)
                    if d < 11 and d >= 0 and e < 21 and e >= 0: 
                        possible = self.settlement_place_placement(d,e)
                        if possible == 1:
                            game.placement_phase_settlement_coordinate1 = d
                            game.placement_phase_settlement_coordinate2 = e
                            game.placement_phase_settlement_turn = 0
                            game.placement_phase_road_turn = 1
                        

                
        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            if action.end_turn == 1:
                self.move_finished() #need to take a look at this function too
        
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
                self.discard_resources(a,b,c,d,e)
        

        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            if trading.give_lumber_get_wool == 1:
                self.trade_resources(1,2)
            if trading.give_lumber_get_grain == 1:
                self.trade_resources(1,3)
            if trading.give_lumber_get_brick == 1:
                self.trade_resources(1,4)
            if trading.give_lumber_get_ore  == 1:
                self.trade_resources(1,5)
            if trading.give_wool_get_lumber == 1:
                self.trade_resources(2,1)
            if trading.give_wool_get_grain == 1:
                self.trade_resources(2,3)
            if trading.give_wool_get_brick == 1:
                self.trade_resources(2,4)
            if trading.give_wool_get_ore == 1:
                self.trade_resources(2,5)
            if trading.give_grain_get_lumber == 1:
                self.trade_resources(3,1)
            if trading.give_grain_get_wool == 1:
                self.trade_resources(3,2)
            if trading.give_grain_get_brick == 1:
                self.trade_resources(3,4)
            if trading.give_grain_get_ore == 1:
                self.trade_resources(3,5)
            if trading.give_brick_get_lumber == 1:
                self.trade_resources(4,1)
            if trading.give_brick_get_wool == 1:
                self.trade_resources(4,2)
            if trading.give_brick_get_grain == 1:
                self.trade_resources(4,3)
            if trading.give_brick_get_ore == 1:
                self.trade_resources(4,5)
            if trading.give_ore_get_lumber == 1:
                self.trade_resources(5,1)
            if trading.give_ore_get_wool == 1:
                self.trade_resources(5,2)
            if trading.give_ore_get_grain == 1:
                self.trade_resources(5,3)
            if trading.give_ore_get_brick == 1:
                self.trade_resources(5,4)
            if action.development_card_buy == 1:
                self.buy_development_cards()

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
                self.activate_yearofplenty_func(player.yearofplenty1,player.yearofplenty2)
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
                self.activate_monopoly_func(a)
                player.monopoly_move_pending = 0

        if phase.statechange == 1:
            phase.statechangecount += 1
            phase.statechange = 0
        
        action.rober_move = action.rober_move * board.ZEROBOARD
        action.road_place = action.road_place * board.ZEROBOARD
        action.settlement_place = action.settlement_place * board.ZEROBOARD
        action.city_place = action.city_place * board.ZEROBOARD

    def checklegalmoves(self):
        
        player = self.players[self.game.cur_player]
        players = self.players
        game = self.game
        board = self.board
        player = players[game.cur_player]
        player0 = players[0]
        player1 = players[1]
        distribution = self.distribution

        self.legal_actions = np.zeros((1,(4*21*11+41)))
        a = 0
        if player.roadbuilding_move_pending == 1:
            if player.roads_left > 0:
                for i in range(0,11):
                    for j in range(0,21):
                        if self.road_possible_check(i,j) == 1:
                            self.legal_actions[0][21*11 + i*21+j] = 1
                            a = 1
            if a == 0:
                player.roadbuilding_move_pending = 0

        b = 0
        if player.discard_resources_started == 1:
            if player.discard_resources_lumber != 0 or player.resource_lumber != 0:
                self.legal_actions[0][4*21*11 + 1] = 1
                b = 1
            if player.discard_resources_wool != 0 or player.resource_wool != 0:
                self.legal_actions[0][4*21*11 + 2] = 1
                b = 1
            if player.discard_resources_grain != 0 or player.resource_grain != 0:
                self.legal_actions[0][4*21*11 + 3] = 1
                b = 1
            if player.discard_resources_brick != 0 or player.resource_brick != 0:
                self.legal_actions[0][4*21*11 + 4] = 1
                b = 1
            if player.discard_resources_ore != 0 or player.resource_ore != 0:
                self.legal_actions[0][4*21*11 + 5] = 1
                b = 1
            if b == 0:
                print("something is wrong with the discard resources")
                player.discard_resources_started = 0

        
        if game.seven_rolled == 1:
            for i in range(0,11):
                for j in range(0,21):
                    if board.rober_position[i][j] != 1 and board.TILES_POSSIBLE[i][j] == 1:
                        self.legal_actions[0][i*21+j] = 1

        if player.knight_move_pending == 1:
            for i in range(0,11):
                for j in range(0,21):
                    if board.rober_position[i][j] != 1 and board.TILES_POSSIBLE[i][j] == 1:
                        self.legal_actions[0][i*21+j] = 1
    
        if game.placement_phase_pending == 1:
            for i in range(0,11):
                for j in range(0,21):
                    if ((((i + 1) == game.placement_phase_settlement_coordinate1 or (i - 1)  == game.placement_phase_settlement_coordinate1) and j == game.placement_phase_settlement_coordinate2) or (((j + 1) == game.placement_phase_settlement_coordinate2 or (j - 1)  == game.placement_phase_settlement_coordinate2) and i == game.placement_phase_settlement_coordinate1)):
                        self.legal_actions[0][21*11 + i*21+j] = 1

        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            if player.resource_brick > 0 and player.resource_lumber > 0:
                if player.roads_left > 0:
                    for i in range(0,11):
                        for j in range(0,21):
                            if self.road_possible_check(i,j) == 1:
                                self.legal_actions[0][21*11 + i*21+j] = 1

        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            if player.resource_brick > 0 and player.resource_lumber > 0 and player.resource_grain > 0 and player.resource_wool > 0:
                board.settlements_used = (1-player0.settlements)*(1-player1.settlements)
                board.settlements_free = board.settlements_available * board.settlements_used
                if player.settlements_left > 0:
                    for i in range(0,11):
                        for j in range(0,21):
                            if board.settlements_free[i][j] == 1 and self.settlement_possible_check(i,j, 0) == 1:
                                self.legal_actions[0][21*11 + 21*11 + i*21+j] = 1

        if game.placement_phase_pending == 1:
            if game.placement_phase_settlement_turn == 1:
                board.settlements_used = (1-player0.settlements)*(1-player1.settlements)
                board.settlements_free = board.settlements_available * board.settlements_used
                if player.settlements_left > 0:
                    for i in range(0,11):
                        for j in range(0,21):
                            if board.settlements_free[i][j] == 1 and self.settlement_possible_check(i,j, 1) == 1:
                                self.legal_actions[0][21*11 + 21*11 + i*21+j] = 1

    
        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            if player.resource_grain > 1 and player.resource_ore > 2:
                if player.cities_left > 0:
                    for i in range(0,11):
                        for j in range(0,21):
                            if player.settlements[i][j] == 1:
                                self.legal_actions[0][21*11 + 21*11 + 21*11 + i*21+j] = 1
    
        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            self.legal_actions[0][4*21*11] = 1
        
        if player.knight_move_pending != 1 and player.monopoly_move_pending != 1 and player.roadbuilding_move_pending != 1 and player.yearofplenty_move_pending != 1 and game.placement_phase_pending != 1 and player.discard_resources_started != 1:
            if (board.harbor_lumber * player.settlements + board.harbor_lumber * player.cities).any() != 0:
                if player.resource_lumber > 1:
                    self.legal_actions[0][4*21*11 + 6] = 1
                    self.legal_actions[0][4*21*11 + 7] = 1
                    self.legal_actions[0][4*21*11 + 8] = 1
                    self.legal_actions[0][4*21*11 + 9] = 1
            if (board.harbor_wool * player.settlements + board.harbor_wool * player.cities).any() != 0:
                if player.resource_wool > 1:
                    self.legal_actions[0][4*21*11 + 10] = 1
                    self.legal_actions[0][4*21*11 + 11] = 1
                    self.legal_actions[0][4*21*11 + 12] = 1
                    self.legal_actions[0][4*21*11 + 13] = 1
            if (board.harbor_grain * player.settlements + board.harbor_grain * player.cities).any() != 0:
                if player.resource_grain > 1:
                    self.legal_actions[0][4*21*11 + 14] = 1
                    self.legal_actions[0][4*21*11 + 15] = 1
                    self.legal_actions[0][4*21*11 + 16] = 1
                    self.legal_actions[0][4*21*11 + 17] = 1
            if (board.harbor_brick * player.settlements + board.harbor_brick * player.cities).any() != 0:
                if player.resource_brick > 1:
                    self.legal_actions[0][4*21*11 + 18] = 1
                    self.legal_actions[0][4*21*11 + 19] = 1
                    self.legal_actions[0][4*21*11 + 20] = 1
                    self.legal_actions[0][4*21*11 + 21] = 1
            if (board.harbor_ore * player.settlements + board.harbor_ore * player.cities).any() != 0:
                if player.resource_ore > 1:
                    self.legal_actions[0][4*21*11 + 22] = 1
                    self.legal_actions[0][4*21*11 + 23] = 1
                    self.legal_actions[0][4*21*11 + 24] = 1
                    self.legal_actions[0][4*21*11 + 25] = 1
            if (board.harbor_three_one * player.settlements + board.harbor_three_one * player.cities).any() != 0:
                if player.resource_lumber > 2:
                    self.legal_actions[0][4*21*11 + 6] = 1
                    self.legal_actions[0][4*21*11 + 7] = 1
                    self.legal_actions[0][4*21*11 + 8] = 1
                    self.legal_actions[0][4*21*11 + 9] = 1
                if player.resource_wool > 2:
                    self.legal_actions[0][4*21*11 + 10] = 1
                    self.legal_actions[0][4*21*11 + 11] = 1
                    self.legal_actions[0][4*21*11 + 12] = 1
                    self.legal_actions[0][4*21*11 + 13] = 1
                if player.resource_grain > 2:
                    self.legal_actions[0][4*21*11 + 14] = 1
                    self.legal_actions[0][4*21*11 + 15] = 1
                    self.legal_actions[0][4*21*11 + 16] = 1
                    self.legal_actions[0][4*21*11 + 17] = 1
                if player.resource_brick > 2:
                    self.legal_actions[0][4*21*11 + 18] = 1
                    self.legal_actions[0][4*21*11 + 19] = 1
                    self.legal_actions[0][4*21*11 + 20] = 1
                    self.legal_actions[0][4*21*11 + 21] = 1
                if player.resource_ore > 2:
                    self.legal_actions[0][4*21*11 + 22] = 1
                    self.legal_actions[0][4*21*11 + 23] = 1
                    self.legal_actions[0][4*21*11 + 24] = 1
                    self.legal_actions[0][4*21*11 + 25] = 1
            if player.resource_lumber > 3:
                self.legal_actions[0][4*21*11 + 6] = 1
                self.legal_actions[0][4*21*11 + 7] = 1
                self.legal_actions[0][4*21*11 + 8] = 1
                self.legal_actions[0][4*21*11 + 9] = 1
            if player.resource_wool > 3:
                self.legal_actions[0][4*21*11 + 10] = 1
                self.legal_actions[0][4*21*11 + 11] = 1
                self.legal_actions[0][4*21*11 + 12] = 1
                self.legal_actions[0][4*21*11 + 13] = 1
            if player.resource_grain > 3:
                self.legal_actions[0][4*21*11 + 14] = 1
                self.legal_actions[0][4*21*11 + 15] = 1
                self.legal_actions[0][4*21*11 + 16] = 1
                self.legal_actions[0][4*21*11 + 17] = 1
            if player.resource_brick > 3:
                self.legal_actions[0][4*21*11 + 18] = 1
                self.legal_actions[0][4*21*11 + 19] = 1
                self.legal_actions[0][4*21*11 + 20] = 1
                self.legal_actions[0][4*21*11 + 21] = 1
            if player.resource_ore > 3:
                self.legal_actions[0][4*21*11 + 22] = 1
                self.legal_actions[0][4*21*11 + 23] = 1
                self.legal_actions[0][4*21*11 + 24] = 1
                self.legal_actions[0][4*21*11 + 25] = 1
            if player.resource_wool > 0 and player.resource_grain > 0 and player.resource_ore > 0 and distribution.development_cards_bought < 25:
                self.legal_actions[0][4*21*11 + 26] = 1
        
            if player.knight_cards_old >= 1:
                self.legal_actions[0][4*21*11 + 27] = 1

            if player.roadbuilding_cards_old >= 1:
                self.legal_actions[0][4*21*11 + 28] = 1
        
            if player.yearofplenty_cards_old >= 1:
                self.legal_actions[0][4*21*11 + 29] = 1

            if player.monopoly_cards_old >= 1:
                self.legal_actions[0][4*21*11 + 30] = 1

        if player.yearofplenty_move_pending == 1:
            self.legal_actions[0][4*21*11 + 31] = 1
            self.legal_actions[0][4*21*11 + 32] = 1
            self.legal_actions[0][4*21*11 + 33] = 1
            self.legal_actions[0][4*21*11 + 34] = 1
            self.legal_actions[0][4*21*11 + 35] = 1

        if player.monopoly_move_pending == 1:
            self.legal_actions[0][4*21*11 + 36] = 1
            self.legal_actions[0][4*21*11 + 37] = 1
            self.legal_actions[0][4*21*11 + 38] = 1
            self.legal_actions[0][4*21*11 + 39] = 1
            self.legal_actions[0][4*21*11 + 40] = 1

        if self.legal_actions.any() != 1:
            print("something is blocking it")
            print("player.knight_move_pending", player.knight_move_pending)
            print("player.monopoly_move_pending", player.monopoly_move_pending)
            print("player.roadbuilding_move_pending", player.roadbuilding_move_pending)
            print("player.yearofplenty_move_pending", player.yearofplenty_move_pending)
            print("game.placement_phase_pending", game.placement_phase_pending)
            print("player.discard_resources_started", player.discard_resources_started)

            

        return self.legal_actions

