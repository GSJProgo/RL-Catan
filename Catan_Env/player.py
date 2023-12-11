import numpy as np

_NUM_ROWS = 11
_NUM_COLS = 21

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
    class Keepresources:
        def __init__(self):
            self.keep_lumber = 0
            self.keep_wool = 0
            self.keep_grain = 0
            self.keep_brick = 0
            self.keep_ore = 0

    