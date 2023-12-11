import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_resBlocks = 4):
        super().__init__()

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


        # adding the outputs of self.denselayer and self.ConvScalar

        # I think I logaically need to combine them earlier
        # Let's think about that in school  

        # I probably need to add a conv layer before the res layer but let's see


    def forward(self, boardstate2, vectorstate2):
        x1 = self.denselayer(vectorstate2)
        x2 = self.ConvScalar(boardstate2)
        y1 = self.DenseConv(vectorstate2)
        y2 = self.ResnetChange(boardstate2)
        for resblock in self.ConvConv:
            y2 = resblock(y2)
        y2 = self.ConvCombine(y2)
        #is this the right dimension in which I concentate?
        y = torch.cat((y1,y2),1)
        x = torch.cat((x1,x2),1)
        vectoractions = self.denseFinal(x)
        boardactions = self.ConvCombineFinal(y)
        state = torch.cat((boardactions,vectoractions),1)
        return state
    
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