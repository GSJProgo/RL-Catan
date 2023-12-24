import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_resBlocks = 12):
        super().__init__()

        self.denselayer = nn.Sequential(
            nn.Linear(35,256),
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
            nn.Conv2d(23,46,kernel_size=(5,3),padding=0,stride=(2,2)),
            nn.BatchNorm2d(46),
            nn.ReLU(),
            nn.Conv2d(46,20,kernel_size=3,padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.Conv2d(20,10, kernel_size=3,padding=1),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(400, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        self.ConvConv = nn.ModuleList(
            [ResBlock() for i in range(num_resBlocks)]
        )

        
        self.ConvCombine = nn.Sequential(
            nn.Conv2d(23,46,kernel_size = 1, padding=0),
            nn.BatchNorm2d(46),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(46*11*21, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),            
            
        )

        self.ConvCombineFinal = nn.Sequential(
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,11*21*4),
        )
        
        self.DenseConv = nn.Sequential(
            nn.Linear(35,1024),
            nn.ReLU(),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
        )

    def forward(self, boardstate2, vectorstate2):
        x1 = self.denselayer(vectorstate2)
        x2 = self.ConvScalar(boardstate2)
        y1 = self.DenseConv(vectorstate2)
        for resblock in self.ConvConv:
            y2 = resblock(boardstate2)
        y2 = self.ConvCombine(y2)

        y = torch.cat((y1,y2),1)
        x = torch.cat((x1,x2),1)
        vectoractions = self.denseFinal(x)
        boardactions = self.ConvCombineFinal(y)
        state = torch.cat((boardactions,vectoractions),1)
        return state
    
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(23, 23, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(23)
        self.conv2 = nn.Conv2d(23, 23, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(23)
    def forward(self,x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        x = F.relu(x)
        return x