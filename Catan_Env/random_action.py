import numpy as np

from Catan_Env.action_selection import action_selecter

class Random: 
    def __init__(self):
        self.random_action = 0
        self.random_position_x = 0
        self.random_position_y = 0

random_agent = Random()

def random_assignment():
    random_agent.random_action = np.random.choice(np.arange(1,46), p=[1/14, 2/14, 1/14, 1/14, 3/14, 1/35, 1/35, 1/35, 1/35, 1/35, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/700, 1/14, 1/28, 1/28, 1/28, 1/28, 1/140, 1/140, 1/140, 1/140, 1/140, 1/700, 1/700, 1/700, 1/700, 1/700])    
    random_agent.random_position_y = np.random.choice(np.arange(0,11))
    random_agent.random_position_x = np.random.choice(np.arange(0,21))
    action_selecter(random_agent.random_action, random_agent.random_position_x, random_agent.random_position_y)
    return random_agent.random_action, random_agent.random_position_x, random_agent.random_position_y