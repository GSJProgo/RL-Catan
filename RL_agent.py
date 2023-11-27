import math
import random
from collections import namedtuple, deque
from itertools import count

import numpy as np

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
    def __init__(self, num_hidden = 32, num_resBlocks = 12):
        super().__init__()
        #self.startBlock = nn.Sequential(
        #    nn.Conv2d(23, num_hidden, kernel_size=3, padding=1),
        #    nn.BatchNorm2d(num_hidden),
        #    nn.ReLU(),
        #)
        #self.backBone = nn.ModuleList(
        #    [ResBlock(num_hidden) for i in range(num_resBlocks)]
        #)
        #self.policyhead2 = nn.Sequential(
        #    nn.Conv2d(num_hidden, 32, kernel_size=3, padding = 1),
        #    nn.BatchNorm2d(32),
        #    nn.ReLU(),
        #    nn.Flatten(),
        #    nn.Linear(32*21*21, 4*21*21),
        #)
        #self.valuehead = nn.Sequential(
        #    nn.Conv2d(num_hidden, 21, kernel_size=3, padding = 1),
        #    nn.BatchNorm2d(21),
        #    nn.ReLU(),
        #    nn.Flatten(),
        #    nn.Linear(21*21*21, 1),
        #    nn.Tanh(),
        #)

        self.denselayer = nn.Sequential(
            nn.Linear(39,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
        
        )
        self.denseFinal = nn.Sequential(
            nn.Linear(128,128),
            nn.Linear(128,64),
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
        )

        self.ConvConv = nn.ModuleList(
            [ResBlock() for i in range(num_resBlocks)]
        )

        #quite a lot of features, hope that this works
        self.ConvCombine = nn.Sequential(
            nn.Conv2d(17,34,kernel_size = 1, padding=0),
            nn.BatchNorm2d(34),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(34*21*21, 2048),
            nn.Linear(2048, 1024),
            nn.Linear(1024, 1024),            
            #combine the last layer of DenseConv with the last one of ConvCombine
        )

        self.ConvCombineFinal = nn.Sequential(
            nn.Linear(2048,2048),
            nn.Linear(2048,2048),
            nn.Linear(2048,21*21*4),
        )
        #That might be too much of an incline, but let's see how it goes
        self.DenseConv = nn.Sequential(
            nn.Linear(39,1024),
            nn.Linear(1024,1024),
            nn.Linear(1024,1024),
        )






        # adding the outputs of self.denselayer and self.ConvScalar

        # I think I logaically need to combine them earlier
        # Let's think about that in school  

        # I probably need to add a conv layer before the res layer but let's see


    def forward(self, x , y):
        print(x.shape)
        print(y.shape)
        x1 = self.denselayer(x)
        print(x1.shape)
        x2 = self.ConvScalar(y)
        print(x2.shape)
        y1 = self.DenseConv(x)
        print("how?",y1.shape)
        for resblock in self.ConvConv:
            y2 = resblock(y)
        y2 = self.ConvCombine(y2)
        print(y2.shape)
        #is this the right dimension in which I concentate?
        y = torch.cat((y1,y2),1)
        x = torch.cat((x1,x2),1)
        print(x.shape)
        print(y.shape)
        vectoractions = self.denseFinal(x)
        boardactions = self.ConvCombineFinal(y)
        print(vectoractions.shape)
        print(boardactions.shape)

        actions = torch.cat((vectoractions,boardactions),1)

        return actions
    
# might change the number of hidden layers later on 
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(17, 17, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(17)
        self.conv2 = nn.Conv2d(17, 17, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(17)
    def forward(self,x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        x = F.relu(x)
        return x
    



class ResBlock2(nn.Module):
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


a1 = np.random.randint(0,2, size = (17,21,21))
print(a1.shape)
print(a1)

b1 = np.random.randint(0,2,size=(39))
print(b1.shape)
print(b1)

model = DQN(num_hidden=32,num_resBlocks=12)

actions = model(torch.tensor(b1,dtype=torch.float32).unsqueeze(0),torch.tensor(a1,dtype=torch.float32).unsqueeze(0))
print(actions.shape)
print(actions)

actionprobabilities = F.softmax(actions,dim=1)
print(actionprobabilities.shape)
print(actionprobabilities)
print(torch.sum(actionprobabilities,dim=1))
print(torch.max(actionprobabilities,dim=1))
max_indices = torch.argmax(actionprobabilities,dim=1)
print(max_indices)
print(max_indices.shape)
random_max_index = np.random.choice(max_indices.numpy())
max_indices_where = torch.where(torch.argmax(actionprobabilities,dim=1))
max_values = actionprobabilities[0, max_indices]
print(max_values)

print(max_indices_where)
print(random_max_index)
