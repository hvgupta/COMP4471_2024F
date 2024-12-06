from GAN import *
from eval import *

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def add_noise(images, noise_factor=0.1):
    noisy_images = images + noise_factor * torch.randn_like(images)
    noisy_images = torch.clamp(noisy_images, 0., 1.)
    return noisy_images

# Training function
def train(model, dataloader, optimizer, criterion, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        for data in dataloader:
            optimizer.zero_grad()
            clean_images, _ = data
            clean_images = clean_images.to(device)
            noisy_images = add_noise(clean_images)
            output = model(noisy_images)
            loss = criterion(output, clean_images)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Visualization function
def visualize_results(model, num_images=3):  # Reduced to visualize 3 images
    model.eval()
    # can be replaced with the actual dataset
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    plt.figure(figsize=(15, 5))
    for i, (clean_image, _) in enumerate(test_loader):
        clean_image = clean_image.to(device)
        if i >= num_images:
            break
        noisy_image = add_noise(clean_image)
        with torch.no_grad():
            denoised_image = model(noisy_image)

        # Plotting
        plt.subplot(3, num_images, i + 1)
        plt.imshow(clean_image[0].permute(1, 2, 0).to("cpu").numpy())
        plt.title("Clean Image")
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_image[0].permute(1, 2, 0).to("cpu").numpy())
        plt.title("Noisy Image")
        plt.axis('off')

        plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(denoised_image[0].permute(1, 2, 0).to("cpu").numpy())
        plt.title("Denoised Image")
        plt.axis('off')

    plt.show(block=True)