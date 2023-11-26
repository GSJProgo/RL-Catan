import math
import random
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

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
    


class DQN(nn.Module):
    def __init__(self, num_hidden = 32, num_resBlocks = 5):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(23, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyhead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*21*21, 4*21*21),
        )
        self.valuehead = nn.Sequential(
            nn.Conv2d(num_hidden, 21, kernel_size=3, padding = 1),
            nn.BatchNorm2d(21),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(21*21*21, 1),
            nn.Tanh(),
        )

        self.denselayer = nn.Sequential(
            nn.Linear(39,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,41)
        )
        

        self.ConvScalar = nn.Sequential(
            nn.Conv2d(17,32,kernel_size=(5,3),padding=0,stride=(2,2)),
            nn.Conv2d(32,17,kernel_size=3,padding=1),
            nn.Conv2d(17,10, kernel_size=3,padding=1),
            nn.Flatten(),
            nn.Linear(10*9*10, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 41)
        )

        # adding the outputs of self.denselayer and self.ConvScalar

        # I think I logaically need to combine them earlier



    def forward(self, x):
        x = self.startBlock(x)
        for resblock in self.backBone:
            x = resblock(x)
        policy = self.policyhead(x)
        value = self.valuehead(x)
        return policy, value
    


class ResBlock(nn.Module):
    def __init__(self, num_hidden = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        x = F.relu(x) 
        return x 

