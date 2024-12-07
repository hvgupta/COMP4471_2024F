import torch
from torch import nn
from eval import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Discriminator(nn.Module):
    def __init__(self, numChannels: int):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
        nn.Conv2d(numChannels, 32, 5, 1),
        #N,32,124,124
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        #N,32,62,62
        nn.Conv2d(32, 64, 5, 1),
        #N,64,58,58
        nn.LeakyReLU(0.01),
        nn.MaxPool2d(2, 2),
        #N,64,29,29
        nn.Flatten(),
        nn.Linear(29*29*64, 4*4*64),
        nn.Linear(4*4*64, 1),
        nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self, latentDim: int, numChannels: int):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(latentDim, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            nn.Linear(2048, 16*16*128),
            nn.ReLU(),
            nn.BatchNorm1d(16*16*128),
            nn.Unflatten(1, (128, 16, 16)),
            #N,128,16,16
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            #N,64,32,32
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            #N,32,64,64
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, numChannels, 4, 2, 1),
            #N,3,128,128
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN(nn.Module):
    def __init__(self, latentDim: int, numChannels: int):
        super(GAN, self).__init__()
        self.generator = Generator(latentDim, numChannels)
        self.discriminator = Discriminator(numChannels)

    def forward(self, x):
        return self.discriminator(x)

    def generate(self, x):
        return self.generator(x)

class GANLoss(nn.Module):
    def __init__(self,
                 window_size: int = 11,
                 sigma: float = 1.5,
                 n_channels: int = 3,
                 weights: torch.Tensor = None,
                 reduction: str = 'mean',
                 padding: bool = False,
                 value_range: float = 1.0,
                 k1: float = 0.01,
                 k2: float = 0.03):
        super(GANLoss, self).__init__()
        self.discriminatorLoss = nn.BCELoss()
        self.generatorLoss = MSSSIM_Loss(window_size, sigma, n_channels, weights, reduction, padding, value_range, k1, k2)

    def forward(self, real, fake, cleanImg, fakeImg):
        realLoss = self.discriminatorLoss(real, torch.ones(real.size(0), 1, device=device))
        fakeLoss = self.discriminatorLoss(fake, torch.zeros(fake.size(0), 1, device=device))
        discriminatorLoss = (realLoss + fakeLoss) / 2

        generatorLoss = self.generatorLoss(cleanImg, fakeImg) / 2
        return discriminatorLoss, generatorLoss