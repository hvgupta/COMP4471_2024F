import torch
from torch import nn
from eval import *


class Discriminator(nn.Module):
    def __init__(self,numChannels:int):
        super(Discriminator,self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(numChannels,64,4,2,1),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Conv2d(256,512,4,2,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2,inplace=True),
            
            nn.Flatten(),
            nn.Linear(512*4*4,1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        return self.model(x)
    
class Generator(nn.Module):
    def __init__(self,latentDim:int,numChannels:int):
        super(Generator,self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latentDim,512*4*4),
            nn.ReLU(inplace=True),
            nn.Reshape(512,4,4),
            
            nn.ConvTranspose2d(512,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64,numChannels,4,2,1),
            nn.Tanh()
        )
    
    def forward(self,x):
        return self.model(x)
    
class GAN(nn.Module):
    def __init__(self,latentDim:int,numChannels:int):
        super(GAN,self).__init__()
        self.generator = Generator(latentDim,numChannels)
        self.discriminator = Discriminator(numChannels)
    
    def forward(self,x):
        return self.discriminator(x)
    
    def generate(self,x):
        return self.generator(x)
    
    
class GANLoss(nn.Module):
    def __init__(self,
                 window_size: int = 11,
                 sigma: float = 1.5,
                 n_channels: int = 3,
                 weights: Tensor = None,
                 reduction: str = 'mean',
                 padding: bool = False,
                 value_range: float = 1.0,
                 k1: float = 0.01,
                 k2: float = 0.03):
        super(GANLoss,self).__init__()
        self.discriminatorLoss = nn.BCELoss()
        self.generatorLoss = MSSSIM_Loss(window_size,sigma,n_channels,weights,reduction,padding,value_range,k1,k2)
    
    def forward(self,real,fake):
        realLoss = self.discriminatorLoss(real,torch.ones(real.size(0),1))
        fakeLoss = self.discriminatorLoss(fake,torch.zeros(fake.size(0),1))
        discriminatorLoss = (realLoss+fakeLoss)/2
        generatorLoss = (self.generatorLoss(real,fake))/2
        return discriminatorLoss,generatorLoss