import matplotlib.pyplot as plt
import torch
from eval import MSSSIM_Loss
from torchvision import transforms
from PIL import Image
import random
import imageio.v2 as imageio  # Updated import statement
import os

def scale(x):
    return (x - x.min()) / (x.max() - x.min())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
plt.savefig('msssim_loss_plot.png')  # Save the plot

# Randomly select a mean and standard deviation
random_mean = random.choice(means).item()
random_sd = random.choice(sds).item()

# Create a list to store the noisy images
noisy_images = []

# Generate noisy images with increasing standard deviation
for sd in sds:
    noisy_img = scale(testImg + random_mean + torch.randn_like(testImg) * sd)
    noisy_img_pil = transforms.ToPILImage()(noisy_img.squeeze().cpu())
    # Add text to the image to show mean and sd
    plt.figure()
    plt.imshow(noisy_img_pil)
    plt.title(f'Mean = {random_mean}, SD = {sd.item()}')
    plt.axis('off')
    temp_file = 'temp_noisy_image.png'
    plt.savefig(temp_file)
    plt.close()
    noisy_images.append(imageio.imread(temp_file))
    os.remove(temp_file)  # Delete the temporary file

# Save the images as a GIF
gif_path = 'noisy_images.gif'
imageio.mimsave(gif_path, noisy_images, duration=0.5)

print(f'GIF saved at {gif_path}')