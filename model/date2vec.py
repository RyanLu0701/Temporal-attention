import torch
from torch import nn



class Date2Vec(nn.Module):
    def __init__(self,input_dim=2, k=32, act="sin"):
        super(Date2Vec, self).__init__()

        if k % 2 == 0:
            k1 = k // 2
            k2 = k // 2
        else:
            k1 = k // 2
            k2 = k // 2 + 1

        self.fc1 = nn.Linear(input_dim, k1)

        self.fc2 = nn.Linear(input_dim, k2)

        if act == 'sin':
            self.activation = torch.sin
        else:
            self.activation = torch.cos


    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.activation(self.fc2(x))
        out = torch.cat([out1, out2], 1)
        return out


