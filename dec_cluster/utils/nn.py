import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class Decoder(nn.Module):
    def __init__(self, in_dim, out_dim, use_act=True):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        if use_act:
            self.act = nn.ReLU()
            
        self.use_act = use_act
         
    def forward(self, x):
        x = self.fc(x)
        if self.use_act:
            x = self.act(x) 
        return x
    

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1280, 1000),
            #nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(1000, 600),
            #nn.Dropout(0.3),
            nn.Tanh()
        )
        
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(600, 1000),
            #nn.Dropout(0.3),
            nn.Tanh(),
            nn.Linear(1000, 1280),
            #nn.Dropout(0.3),
            nn.Tanh()
        )
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.LeakyReLU = nn.LeakyReLU(0.01)

    def forward(self, x):
        # 通过编码器
        x = x.float()
        encode_data = self.encoder(x)
        return x,encode_data


class Cluster(nn.Module):
    def __init__(self, center, alpha):
        super().__init__()
        self.center = center
        self.alpha = alpha

    def forward(self, x):
        square_dist = torch.pow(x[:, None, :] - self.center, 2).sum(dim=2)
        nom = torch.pow(1 + square_dist / self.alpha, -(self.alpha + 1) / 2)
        denom = nom.sum(dim=1, keepdim=True)
        return nom / denom


def get_p(q):
    with torch.no_grad():
        f = q.sum(dim=0, keepdim=True)
        nom = q ** 2 / f
        denom = nom.sum(dim=1, keepdim=True)
    return nom / denom           #距离距离中心越近，权重越大
    

class DEC(nn.Module):
    def __init__(self, encoder, center, alpha=1):
        super().__init__()
        self.encoder = encoder
        
        self.cluster = Cluster(center, alpha)

    def forward(self, x):
        x = self.encoder(x)  #1708*600   ae的中间层编码数据
        #print(x.shape)
        #exit()
        x = self.cluster(x)
        #print(x.shape)
        #exit()
        return x

















