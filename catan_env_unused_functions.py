#def player_places_road_placement(a,b):
#    print(a,a)
#    print(b,b,b)
#    road_direction = 0
#    print("roads available \n", board.roads_available)
#    print("In which direction do you want to place the road") #would be good to later add it in the board, less searching
#    while not int(road_direction) in range(1, 5):
#        road_direction = int(input("   1   \n2     3\n   4   \n"))
#        print (int(road_direction))
#        if road_direction  == 1 and a != 0 and board.roads_available[a-1][b] != 0:
#            print("first")
#            board.roads_available[a-1][b] = 0
#            if game.cur_player == 0:
#                player0.roads[a-1][b] = 1            
#            if game.cur_player == 1:
#                player1.roads[a-1][b] = 1        
#        elif road_direction == 2 and b != 0 and board.roads_available[a][b-1] != 0:
#            print("second")
#            board.roads_available[a][b-1] = 0
#            if game.cur_player == 0:
#                player0.roads[a][b-1] = 1            
#            if game.cur_player == 1:
#                player1.roads[a][b-1] = 1        
#        elif road_direction  == 3 and b != 20 and board.roads_available[a][b+1] != 0:
#            print("third")
#            board.roads_available[a][b+1] = 0
#            if game.cur_player == 0:
#                player0.roads[a][b+1] = 1           
#            if game.cur_player == 1:
#                player1.roads[a][b+1] = 1        
#        elif road_direction  == 4 and a != 10 and board.roads_available[a+1][b] != 0:
#            print("fourth")
#            board.roads_available[a+1][b] = 0
#            if game.cur_player == 0:
#                player0.roads[a+1][b] = 1            
#            if game.cur_player == 1:
#                player1.roads[a+1][b] = 1      
#        else:
#            road_direction = 0  
#            print("This is not a possible direction")
#        
#        print("You have chosen the direction",road_direction)
#        print(int(road_direction)
#              
#
#def settlementplaced(a,b):
#    board.settlements_available[a][b] = 0
#    player1.settlements_possible[a][b] = 0
#    player0.settlements_possible[a][b] = 0
#    if a != 1 and a != 0:
#        player0.settlements_possible[a-2][b] = 0
#        player1.settlements_possible[a-2][b] = 0    
#    if a != 9 and a != 10:
#        player0.settlements_possible[a+2][b] = 0
#        player1.settlements_possible[a+2][b] = 0
#    
#    if b != 20 and b != 21:
#        player0.settlements_possible[a][b+2] = 0
#        player1.settlements_possible[a][b+2] = 0
#    if b != 1 and b != 0:
#        player1.settlements_possible[a][b-2] = 0
#        player0.settlements_possible[a][b-2] = 0
#
#def agent_places_road_placement(a,b):
#    road_direction = 0
#    while road_direction == 0:
#        road_direction = random.randint(1,4)
#        if road_direction  == 1 and a != 0 and board.roads_available[a-1][b] != 0:
#            board.roads_available[a-1][b] = 0
#            if game.cur_player == 0:
#                player0.roads[a-1][b] = 1            
#            if game.cur_player == 1:
#                player1.roads[a-1][b] = 1        
#        elif road_direction == 2 and b != 0 and board.roads_available[a][b-1] != 0:
#            board.roads_available[a][b-1] = 0
#            if game.cur_player == 0:
#                player0.roads[a][b-1] = 1            
#            if game.cur_player == 1:
#                player1.roads[a][b-1] = 1        
#        elif road_direction  == 1 and b != 20 and board.roads_available[a][b+1] != 0:
#            board.roads_available[a][b+1] = 0
#            if game.cur_player == 0:
#                player0.roads[a][b+1] = 1            
#            if game.cur_player == 1:
#                player1.roads[a][b+1] = 1        
#        elif road_direction  == 1 and a != 10 and board.roads_available[a+1][b] != 0:
#            board.roads_available[a+1][b] = 0
#            if game.cur_player == 0:
#                player0.roads[a+1][b] = 1            
#            if game.cur_player == 1:
#                player1.roads[a+1][b] = 1      
#        else:
#            road_direction = 0  
#        
#        print("The RL agent has chosen the in direction",road_direction)
#
#def player_places_settlement():
#    result = 0  # Use a different variable name
#    while result == 0:
#        input_row = 0
#        input_column = 0
#        print("These are your possible settlements \n", player0.settlements_possible)
#        while not int(input_row) in range(1, 12):
#            input_row = int(input("In which row do you want to set your settlement (Number between 1-11) \n"))
#        while not int(input_column) in range(1, 22):
#            input_column = int(input("In which column do you want to set your settlement (Number between 1-21) \n"))
#        result = settlement_place(input_row - 1, input_column - 1)
#        if result == 0:
#            print("This is not a possible settlement place. Please try again")
#    tile_update_rewards(input_row - 1, input_column - 1)
#    player_places_road_placement(input_row - 1, input_column - 1)
#    
#    result = 0def agent_places_settlement():
#    result = 0
#    print("The RL agent is choosing first settlement placement")
#    while result == 0:
#        input_row = random.randint(0,10)
#        input_column = random.randint(0,20)
#        print("input column", input_column)
#        print("input_row",input_row)
#        result = settlement_place(input_row, input_column)
#    print("The RL agent has chosen the settlement in column ",input_column + 1," and row ",input_row + 1)
#
#    agent_places_road_placement(input_row,input_column)
#
#    result = 0
#
#def placement_phase():
#    if game.testing:    
#        reset()
#        harbor_distribution()
#        reset()
#        harbor_distribution()
#        reset()
#        harbor_distribution()
#
#
#        print("Welcome to Settlers of Catan.")
#        player = 5
#        
#        while not int(player) in range(0, 4):
#            player = input("Choose the player that you want to play(0 = player-agent,1 = agent-player,2 = player-player,3 = agent-agent)\n")
#            player = int(player)
#        print("Let's start with the placement phase")
#        print("You're playing as", player)
#
#        print("Randomly building board......")
#        print("lumber = 1, wool = 2, ")
#        tile_distribution()
#        print(board.tiles_pretty)
#        print(board.tiles_lumber)
#        print("Randomly assigning dice numbers......")
#        plate_distribution()
#        print(board.tiles_dice)
#        print("player", player)
#        print(board)
#        if (player == 0):
#            print("You're starting, place your first settlement")
#            player_places_settlement()
#            move_finished()
#            agent_places_settlement()
#            agent_places_settlement()
#            move_finished()
#            player_places_settlement()
#        if (player == 1):
#            print("The RL agent is startingm, you're place the second and third settlement")
#            agent_places_settlement()
#            player_places_settlement()
#            player_places_settlement()
#            agent_places_settlement()
#        if (player == 2):
#            game.cur_player2 = 0
#            while not int(game.cur_player2) in range(1, 3):
#                game.cur_player2 = input("Which player wants to start?")
#            game.cur_player = game.cur_player2
#            player_places_settlement()
#            game.cur_player = 1 - game.cur_player
#            player_places_settlement()
#            player_places_settlement()
#            game.cur_player = 1 - game.cur_player
#            player_places_settlement()
#        if (player == 3):
#            for i in range(0,20):
#                agent_places_settlement()