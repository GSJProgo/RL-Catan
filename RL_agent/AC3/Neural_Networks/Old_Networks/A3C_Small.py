import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import init_weights

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

        self.distribution = torch.distributions.Categorical
        self.apply(init_weights)
    
    
    
    def loss_func(self, boardstate, vectorstate, a, v_t, device, total_step):
        torch.set_num_threads(1)
        self.train()
        boardstate = boardstate.to(device)
        vectorstate = vectorstate.to(device)
        logits, values = self.forward(boardstate, vectorstate)
        logits = logits.cpu()
        values = values.cpu()
        td = v_t - values
        c_loss = td.pow(2)
        a = a.cpu()
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        entropy = m.entropy().cpu()
        print(entropy.mean())
        exp_v = m.log_prob(a).cpu() * td.detach().squeeze().cpu()
        a_loss = -exp_v
        l2_activity_loss = torch.sum(logits**2).cpu()

        if total_step % 2000 == 0:
            print("c_loss: ", c_loss.mean()* 10**3)
            print("a_loss: ", a_loss.mean())
            print("entropy: ", entropy * 10**-3)
            print("l2_activity_loss: ", l2_activity_loss * 10**-8)
            print("total_loss: ", (c_loss * 10**3 + a_loss + entropy *10**-3 + l2_activity_loss * 10**-6).mean())


        total_loss = (c_loss * 10**3 + a_loss + entropy * 10**-4 + l2_activity_loss* 5*10**-7).mean()
        return total_loss

    def choose_action(self,boardstate,vectorstate, env, total_step):
        torch.set_num_threads(1)
        self.eval()
        logits, _ = self.forward(boardstate, vectorstate)
        logits = logits.cpu()
        mean = logits.mean()
        prob = F.softmax(logits, dim=1).data
        prob2 = prob * env.checklegalmoves()
        prob2 /= prob2.sum()
        if prob2.sum() == 0:
            print("no legal moves")
        if total_step % 5000 == 0:
            print(logits.sum())
            print(logits)
            print(prob)
            
        try: 
            m = self.distribution(prob2)
        except ValueError:
            print(prob)
            print(prob2)
            print(prob.sum())
            print(prob2.sum())
            print("no legal moves")
            m = self.distribution(torch.tensor([1/len(prob)]*len(prob)))
            print(logits)
            print(mean)
            print(env.checklegalmoves())

        return m.sample().cpu().numpy()[0], mean
    
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

        z = torch.cat((x,y),1)
        value = self.CombineValue(z)
        return state, value
        
    
    
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
