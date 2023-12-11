import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class ActorCritic(nn.Module):
    def __init__(self, gamma = 0.99, num_resBlocks = 4):
        super().__init__()

        self.gamma = gamma

        self.denselayer = nn.Sequential(
            nn.Linear(35,64),
            nn.ReLU(),
            nn.Linear(64,64),

        
        )
        self.denseFinal = nn.Sequential(
            nn.Linear(128,41),
        )
        

        self.ConvScalar = nn.Sequential(
            nn.Conv2d(23,5,kernel_size=(3,5),padding=0,stride=(2,2)),
            nn.BatchNorm2d(5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(225, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.ConvConv = nn.ModuleList(
            [ResBlock() for i in range(num_resBlocks)]
        )

        #quite a lot of features, hope that this works
        self.ConvCombine = nn.Sequential(
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(450, 128),          
            #combine the last layer of DenseConv with the last one of ConvCombine
        )

        self.ConvCombineFinal = nn.Sequential(
            nn.Linear(256,11*21*4),
        )
        #That might be too much of an incline, but let's see how it goes
        self.DenseConv = nn.Sequential(
            nn.Linear(35,64),
            nn.ReLU(),
            nn.Linear(64,128),
        )

        self.ResnetChange = nn.Sequential(
            nn.Conv2d(23,10,kernel_size=(3,5),padding=0,stride=(2,2)),
            nn.BatchNorm2d(10),
            nn.ReLU(),
        )

        self.CombineValue = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

        
        self.states = []
        self.actions = []
        self.rewards = []

    def remember(self, state,action,reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    
    
    def calc_R(self,done):
        states = torch.tensor(self.states, dtype=torch.float)
        _, value = self.forward(states)
        R = value[-1]*(1-int(done))
        batch_return = []
        for reward in self.rewards[::-1]:
            R = reward + self.gamma*R
            batch_return.append(R)
        batch_return.reverse()
        batch_return = torch.tensor(batch_return,dtype=torch.float)
        return batch_return
    
    def calc_loss(self,done):
        states = torch.tensor(self.states, dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.float)
        returns = self.calc_R(done)

        pi, values = self.forward(states)
        values = values.squeeze()
        critic_loss = (returns-values)**2

        probs = torch.softmax(pi, dim=1)
        dist = Categorical(probs)
        log_probs = distorch.log_prob(actions)
        actor_loss = -log_probs*(returns-values)

        total_loss = (critic_loss + actor_loss).mean()

        return total_loss

    def chosse_action(self,observation):
        state = torch.tensor([observation],dtype=torch.float)
        pi, v = self.forward(state)
        probs = torch.softmax(pi,dim=1)
        dist = Categorical(probs)
        action = distorch.sample().numpy()[0]

        return action
    
    
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(10)
    def forward(self,x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        x = F.relu(x)
        return x
