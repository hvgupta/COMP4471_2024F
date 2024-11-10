import torch

torch.set_default_device("cuda:0" if torch.cuda.is_available() else "cpu")