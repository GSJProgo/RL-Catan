from collections import namedtuple
import random
from collections import deque

Transition = namedtuple('Transition', ('cur_boardstate','cur_vectorstate', 'action', 'next_boardstate','next_vectorstate', 'reward'))

class ReplayMemory(object):
    """docstring for ReplayMemory"""
    def __init__(self, capacity):
        self.memory = deque([],maxlen = capacity)
    def push(self, *args):
        """Saves a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)