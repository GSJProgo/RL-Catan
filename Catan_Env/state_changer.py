from Catan_Env.catan_env import Catan_Env
import numpy as np
import torch 

def state_changer(env):
    players = env.players
    game = env.game
    board = env.board
    player0 = players[0]
    player1 = players[1]

    phase = env.phase

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
    torch_board_tensor = torch.from_numpy(np_board_tensor[None,:]).float()
    torch_vector_tensor = torch.from_numpy(np_vector_tensor[None,:]).float()
    return torch_board_tensor, torch_vector_tensor