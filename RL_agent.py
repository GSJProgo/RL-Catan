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

torch.set_printoptions(precision=5)


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

        self.denselayer = nn.Sequential(
            nn.Linear(39,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
        
        )
        self.denseFinal = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,41)
        )
        

        self.ConvScalar = nn.Sequential(
            nn.Conv2d(17,34,kernel_size=(5,3),padding=0,stride=(2,2)),
            nn.BatchNorm2d(34),
            nn.ReLU(),
            nn.Conv2d(34,17,kernel_size=3,padding=1),
            nn.BatchNorm2d(17),
            nn.ReLU(),
            nn.Conv2d(17,10, kernel_size=3,padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10*9*10, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
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
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),            
            #combine the last layer of DenseConv with the last one of ConvCombine
        )

        self.ConvCombineFinal = nn.Sequential(
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,2048),
            nn.ReLU(),
            nn.Linear(2048,21*21*4),
        )
        #That might be too much of an incline, but let's see how it goes
        self.DenseConv = nn.Sequential(
            nn.Linear(39,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
        )

        self.optimizer = optim.Adam(self.parameters(), lr = 0.001, amsgrad=True)
        self.loss = nn.MSELoss()
        self.to(self.device)


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
    

class Agent():
    def __init__(self,gamma,epsilon,lr,input_dims,batch_size,n_actions,max_mem_size = 100000,eps_end = 0.01,eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.batch_size = batch_size
        self.max_mem_size = max_mem_size
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.mem_cntr = 0

        self.Q_eval = DQN()

        self.state_memory = np.zeros((self.max_mem_size,*input_dims),dtype = np.float32)

        self.new_state_memory = np.zeros((self.max_mem_size,*input_dims),dtype = np.float32)

        self.action_memory = np.zeros(self.max_mem_size,dtype = np.int32)
        self.reward_memory = np.zeros(self.max_mem_size,dtype = np.float32)
        self.terminal_memory = np.zeros(self.max_mem_size,dtype = np.bool)  

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


BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TAU = 0.001
LR = 0.001

n_actions = 0

state, info = 0,0
n_obs = len(state)

agent1_policy_net = DQN().to(device)
target_net = DQN().to(device)
target_net.load_state_dict(agent1_policy_net.state_dict())

optimizer = optim.Adam(agent1_policy_net.parameters(), lr = LR, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def Agent1_select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return agent1_policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
def Agent2_select_action(state):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END)*math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return agent2_policy_net(state).max(1).indices.view(1,1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
    
episode_durations = []

def plotting():
    print()
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = agent1_policy_net(state_batch).gather(1,action_batch)

    next_state_values = torch.zeros(BATCH_SIZE,device=device)

    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values,expected_state_action_values.unsqueeze(1))
    

    optimizer.zero_grad()

    loss.backward()

    #torch.nn.utils.clip_grad_norm_(agent1_policy_net.parameters(), 100)

    optimizer.step()

num_episodes = 1000
for i_episode in range (num_episodes):
    if i_episode % 100 == 99:
        if wins > 50:
            torch.save(agent1_policy_net.state_dict(), 'agent{num_episodes}_policy_net.pth')
            agent2_policy_net = DQN()
            agent2_policy_net.load_state_dict(torch.load('agent{num_episodes}_policy_net.pth'))
            agent2_policy_net.to(device)
        wins = 0
    
    state = 0 #info = 0
    state = torch.tensor(state,device=device,dtype=torch.float).unsqueeze(0)
    for t in count():
        if curagent == 1:
            action = Agent2_select_action(state)
            ext_state, reward, done = 0,0,0 #this is were I need to perform an action and return the next state, reward, done
            next_state = torch.tensor(next_state,device=device,dtype=torch.float).unsqueeze(0)
            if done:
                next_state = None
            state = next_state

        if curagent == 0:
            action = Agent1_select_action(state)
            next_state, reward, done = 0,0,0 #this is were I need to perform an action and return the next state, reward, done
            reward = torch.tensor([reward],device=device)
            next_state = torch.tensor(next_state,device=device,dtype=torch.float).unsqueeze(0)
            if done:
                next_state = None
            memory.push(state,action,next_state,reward)
            state = next_state
            optimize_model()

            target_net.load_state_dict(agent1_policy_net.state_dict())

        if done:
            episode_durations.append(t+1)
            #if player1 won, update win counter
            break

print('Complete')


#might add more than 1 training agent later on 