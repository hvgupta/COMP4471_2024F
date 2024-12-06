import torch
from torch import nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = nn.functional.adaptive_avg_pool2d(x, 1).view(batch_size, channels) # Squeeze
        y = nn.functional.relu(self.fc1(y)) # Excitation
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)  # Input scaling

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x  # Save the input for the skip connection
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity  # Add the skip connection
        out = self.relu(out)
        return out

# DnCNN Model
class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_layers=8):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True)] # input layer

        for _ in range(num_layers - 2):
            layers.append(ResidualBlock(64)) # Add Residual Block
            layers.append(SEBlock(64)) # Add SE block after each Residual Block

        layers.append(nn.Conv2d(64, out_channels, kernel_size=3, padding=1)) # output layer
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.model(x)
        x = nn.Sigmoid()(x)
        return x
