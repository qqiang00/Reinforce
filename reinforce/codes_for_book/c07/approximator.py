import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
    '''一种较合理的初始化网络参数，参考：https://arxiv.org/abs/1502.01852
    '''
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)  
    x = torch.Tensor(size).uniform_(-v, v) # 从-v到v的均匀分布
    return x.type(torch.FloatTensor)

class Critic(nn.Module):

    def __init__(self, state_dim, action_dim):
        '''构建一个评判家模型
        Args:
            state_dim: 状态的特征的数量 (int)
            action_dim: 行为作为输入的特征的数量 (int)
        '''
        super(Critic, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.fcs1 = nn.Linear(state_dim, 256) #状态第一次线性变换
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256,128) # 状态第二次线性变换
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

        self.fca1 = nn.Linear(action_dim, 128) # 行为第一次线性变换
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fc2 = nn.Linear(256, 128) # (状态+行为)联合的线性变换，注意参数值
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 1) # (状态+行为)联合的线性变换
        self.fc3.weight.data.uniform_(-EPS,EPS)

    def forward(self, state, action):
        '''前向运算，根据状态和行为的特征得到评判家给出的价值
        Args:
            state 状态的特征表示 torch Tensor [n, state_dim]
            action 行为的特征表示 torch Tensor [n, action_dim]
        Returns:
            Q(s,a) Torch Tensor [n, 1]
        '''
        # 该网络属于价值函数近似的第二种类型，根据状态和行为输出一个价值
        #print("first action type:{}".format(action.shape))
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)

        action = action.type(torch.FloatTensor)
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))

        a1 = F.relu(self.fca1(action))
        # 将状态和行为连接起来，使用第二种近似函数架构(s,a)-> Q(s,a)
        x = torch.cat((s2,a1), dim=1) 

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_lim):
        '''构建一个演员模型
        Args:
            state_dim: 状态的特征的数量 (int)
            action_dim: 行为作为输入的特征的数量 (int)
            action_lim: 行为值的限定范围 [-action_lim, action_lim]
        '''
        super(Actor, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_lim = action_lim
        
        self.fc1 = nn.Linear(self.state_dim, 256)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

        self.fc3 = nn.Linear(128, 64)
        self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

        self.fc4 = nn.Linear(64, self.action_dim)
        self.fc4.weight.data.uniform_(-EPS,EPS)

    def forward(self, state):
        '''前向运算，根据状态的特征表示得到具体的行为值
        Args:
            state 状态的特征表示 torch Tensor [n, state_dim]
        Returns:
            action 行为的特征表示 torch Tensor [n, action_dim]
        '''
        state = torch.from_numpy(state)
        state = state.type(torch.FloatTensor)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action = F.tanh(self.fc4(x)) # 输出范围-1,1
        action = action * self.action_lim # 更改输出范围
        return action
