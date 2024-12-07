import torch
import torchvision.transforms as transforms
from torchvision.transforms import Resize, ToTensor
from PIL import Image
import os


cleanImage = []
transformer = transforms.Compose(
    [Resize((220,220)),
    ToTensor()
    ]
)

imgDir = "./SIDD_Small_sRGB_Only/Data"
for folder in os.listdir(imgDir):
    imgfolderPath = imgDir + "/" + folder
    for img in os.listdir(imgfolderPath):
        if "NOISY" in img:
            continue
        imgPath = imgfolderPath + "/" + img
        img = Image.open(imgPath)
        img = transformer(img)
        cleanImage.append(img)

cleanImage = torch.stack(cleanImage)
print(cleanImage.shape)

torch.save(cleanImage,"cleanImage.pt")