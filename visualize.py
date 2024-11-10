import matplotlib.pyplot as plt
import torch
from eval import MSSSIM_Loss
from torchvision import transforms
from PIL import Image

def scale(x):
    return (x - x.min()) / (x.max() - x.min())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

testImg = Image.open('testImages/flowerField.jpg')
testImg = transforms.ToTensor()(testImg).unsqueeze(0).to(device)

msssim = MSSSIM_Loss().to(device)

means = torch.linspace(-1, 1, 10)
sds = torch.linspace(0, 1, 20)

plt.figure()

for mean in means:
    losses = []
    for sd in sds:
        noisy_img = scale(testImg + mean + torch.randn_like(testImg) * sd)
        loss = msssim(noisy_img, testImg)
        losses.append(loss.item())
    plt.plot(sds.cpu().numpy(), losses, label=f'Mean = {mean.item()}')

plt.xlabel('Standard Deviation')
plt.ylabel('MSSSIM Loss')
plt.title('MSSSIM Loss vs. Noise Standard Deviation for Different Means')
plt.legend()
plt.show()