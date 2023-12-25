import numpy as np
import math
import random

from Catan_Env.action_selection import action_selecter

class Random: 
    def __init__(self):
        self.random_action = 0
        self.random_position_x = 0
        self.random_position_y = 0

random_agent = Random()

def random_assignment(env):
    legal_actions = env.checklegalmoves()

    legal_indices = np.where(legal_actions == 1)[1]

    randomaction = np.random.choice(legal_indices)
    if randomaction >= 4*11*21:
            final_action = randomaction - 4*11*21 + 5
            position_y = 0
            position_x = 0
    else:
        final_action = math.ceil(((randomaction + 1)/11/21))
        position_y = math.floor((randomaction- ((final_action-1)*11*21))/21)
        position_x = randomaction % 21 

    action_selecter(env, final_action, position_x, position_y)



    #print(randomaction)

    #random_agent.random_action = np.random.choice(np.arange(1,46), p=[1/14, 2/14, 2/14, 2/14, 1/14, 1/35, 1/35, 1/35, 1/35, 1/35, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/14, 1/28, 1/28, 1/28, 1/28, 1/140, 1/140, 1/140, 1/140, 1/140, 1/700, 1/700, 1/700, 1/700, 1/700])    
    #random_agent.random_position_y = np.random.choice(np.arange(0,11))
    #random_agent.random_position_x = np.random.choice(np.arange(0,21))
    #action_selecter(env,random_agent.random_action, random_agent.random_position_x, random_agent.random_position_y)
    return randomaction