import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from utils import init_weights
import gc

class ActorCritic(nn.Module):
    def __init__(self, gamma = 0.99, num_resBlocks = 6):
        super().__init__()

        self.gamma = gamma
        self.last_values = []

        self.denselayer = nn.Sequential(
            nn.Linear(35,128),
            nn.LeakyReLU(),
            nn.Linear(128,128),
            nn.LeakyReLU(),
            nn.Linear(128,64), 
            nn.LeakyReLU(),
            
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
            nn.LeakyReLU(),
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
            nn.LeakyReLU(), 
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
            nn.LeakyReLU(),
        )

        self.DenseValue = nn.Sequential(    
            nn.Linear(35, 256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,128),
            nn.LeakyReLU(),
            nn.Linear(128,64),
        )


        self.ResValue = nn.ModuleList(
            [ResBlock2() for i in range(4)]
        )
        self.CombineValue = nn.Sequential(
            nn.Linear(64,1),
        )
        self.ConvValue = nn.Sequential(

            nn.Conv2d(23,10,kernel_size=(3,5),padding=0,stride=(2,2)),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(450, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64), 
        )

        self.distribution = torch.distributions.Categorical
        self.apply(init_weights)

    @property
    def actor_parameters(self):
        # Returns parameters for the actor network
        return list(self.denseFinal.parameters()) + list(self.ConvCombineFinal.parameters()) + list(self.DenseConv.parameters()) + list(self.ConvCombine.parameters()) + list(self.denselayer.parameters()) + list(self.ConvScalar.parameters()) + list(self.ResTransfrom.parameters()) + list(self.ConvConv.parameters())

    @property
    def critic_parameters(self):
        # Returns parameters for the critic network
        return list(self.CombineValue.parameters()) + list(self.DenseValue.parameters()) + list(self.ConvValue.parameters())
    
    def loss_func(self, boardstate, vectorstate, a, v_t, device):
        torch.set_num_threads(1)
        self.train()
        boardstate = boardstate.to(device)
        vectorstate = vectorstate.to(device)
        logits, values = self.forward(boardstate, vectorstate)
        logits2 = logits.cpu().detach()
        td = v_t - values
        c_loss = td.pow(2)
        a = a.to(device)  # move a to the GPU
        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        entropy = -m.entropy().cpu()
        exp_v = m.log_prob(a).cpu() * td.detach().squeeze().cpu()
        a_loss = -exp_v
        l2_activity_loss = torch.sum(logits2**2)
        boardstate = boardstate.cpu()
        vectorstate = vectorstate.cpu()
        c_loss = c_loss.cpu()
        a_loss = a_loss.cpu()
        entropy = entropy.cpu()
        l2_activity_loss = l2_activity_loss.cpu()
        
        #del boardstate, vectorstate, logits, td, probs, m, exp_v, a, v_t

        #if total_step % 2000 == 0:
        #    print("c_loss: ", c_loss.mean()* 10**3)
        #    print("a_loss: ", a_loss.mean())
        #    print("entropy: ", entropy * 10**-3)
        #    print("l2_activity_loss: ", l2_activity_loss * 10**-6)
        #    print("total_loss: ", (c_loss * 10**3 + a_loss + entropy * 5 * 10**-3 + l2_activity_loss * 5 * 10**-6).mean())


        total_loss = (c_loss * 10 ** 3 + a_loss * 10 + entropy * 10 ** -3 + l2_activity_loss *5 * 10 ** -4).mean()
        return values, total_loss, c_loss.mean(), a_loss.mean(), entropy.mean() * 10 ** -4, l2_activity_loss.mean()* 5 * 10 ** -5

    
    def get_value(self, boardstate, vectorstate, device):
        torch.set_num_threads(1)
        self.eval()
        boardstate = boardstate.to(device)
        vectorstate = vectorstate.to(device)
        _, values = self.forward(boardstate, vectorstate)
        values2 = values.cpu().detach()
        del boardstate, vectorstate
        return values2.data.numpy()
    
    def choose_action(self,boardstate,vectorstate, env, total_step):
        torch.set_num_threads(1)
        self.eval()
        logits, values = self.forward(boardstate, vectorstate)
        prob = F.softmax(logits, dim=1).data

        prob = prob.cpu()  # move prob to the CPU

        prob2 = prob * torch.from_numpy(env.checklegalmoves()).float()  # convert env.checklegalmoves() to a tensor

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

        logits2 = logits.cpu().detach()
        values2 = values.cpu().detach()

        mean = logits2.mean()

        return m.sample().numpy()[0], mean, logits2, values2  # m.sample() is already on the CPU, so you can convert it to a numpy array directly   
            
    def forward(self, boardstate2, vectorstate2):
        vectorstate2 = F.normalize(vectorstate2)
        boardstate2 = F.normalize(boardstate2)
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
        #for resblock in self.ResValue:
        #    value2 = resblock(boardstate2)
        value1 = self.DenseValue(vectorstate2)
        value2 = self.ConvValue(boardstate2)
        value = value1 + value2
        value = self.CombineValue(value)
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
    
class ResBlock2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(23, 23, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(23)
        self.LeakyReLU = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(23, 23, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(23)
    def forward(self,x):
        residual = x
        x = self.LeakyReLU(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))

        x += residual
        x = self.LeakyReLU(x)
        return x
