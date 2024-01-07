import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import init_weights

class ActorCritic(nn.Module):
    def __init__(self, gamma = 0.99, num_resBlocks = 8):
        super().__init__()

        self.gamma = gamma

        self.denselayer = nn.Sequential(
            nn.Linear(35,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,64), 
        )
        self.denseFinal = nn.Sequential(
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
            nn.LeakyReLU(),
            nn.Linear(64,41)
        )

        self.ResTransfrom = nn.Sequential(
            nn.Conv2d(23,64,kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )
        

        self.ConvScalar = nn.Sequential(
            nn.Conv2d(23,5,kernel_size=(3,5),padding=0,stride=(2,2)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(225, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
        )

        self.ConvConv = nn.ModuleList(
            [ResBlock() for i in range(num_resBlocks)]
        )

        self.ConvCombine = nn.Sequential(
            nn.Conv2d(64,4,kernel_size = 1, padding=0), 
            nn.BatchNorm2d(4),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(4*11*21, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),             
        )

        self.ConvCombineFinal = nn.Sequential(
            nn.Linear(768,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,1024),
            nn.LeakyReLU(),
            nn.Linear(1024,11*21*4),
        )

        self.DenseConv = nn.Sequential(
            nn.Linear(35,256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
        )

        self.CombineValue = nn.Sequential(
            nn.Linear(896, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
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
        entropy = -m.entropy().cpu()
        exp_v = m.log_prob(a).cpu() * td.detach().squeeze().cpu()
        a_loss = exp_v
        l2_activity_loss = torch.sum(logits**2).cpu()

        #if total_step % 2000 == 0:
        #    print("c_loss: ", c_loss.mean()* 10**3)
        #    print("a_loss: ", a_loss.mean())
        #    print("entropy: ", entropy * 10**-3)
        #    print("l2_activity_loss: ", l2_activity_loss * 10**-6)
        #    print("total_loss: ", (c_loss * 10**3 + a_loss + entropy * 5 * 10**-3 + l2_activity_loss * 5 * 10**-6).mean())


        total_loss = (c_loss * 10**3 + a_loss * 4 + entropy * 2 * 10 ** -3 + l2_activity_loss * 5 * 10 ** -4).mean()
        return values, total_loss, c_loss.mean() * 10 ** 3, a_loss.mean() * 4, entropy.mean() * 2 * 10 ** -3, l2_activity_loss.mean() * 2 * 10 ** -4

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
        #if total_step % 5000 == 0:
        #    print(logits.sum())
        #    print(logits)
        #    print(prob)
            
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
        y2 = self.ResTransfrom(boardstate2)
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
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.LeakyReLU = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self,x):
        residual = x
        x = self.LeakyReLU(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        x = self.LeakyReLU(x)
        return x
