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

        self.average_time = []
        self.average_moves = []
        self.average_q_value_loss = []
        self.average_highest_q_value = []

        self.average_reward_per_move = []
        self.average_expected_state_action_value = []

        self.average_win_ratio = []
        self.average_legal_moves_ratio = []
        
        self.random_action_made = 0