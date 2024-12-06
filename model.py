import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

from DnCNN import *
from eval import MSSSIM_Loss

# Squeeze-and-Excitation Block

# Function to add noise to images(temp), can be replaced with the actual dataset
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

# Main function to run everything
if __name__ == "__main__":
    # Hyperparameters(require further tuning afterwards)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    num_epochs = 5 
    learning_rate = 0.001

    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    
    # can be replaced with the actual dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # Use a subset of the training dataset for faster training
    indices = list(range(0, 1000))  # Use only the first 1000 samples
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)

    # Model, criterion, and optimizer
    model = DnCNN().to(device)
    criterion = MSSSIM_Loss(window_size=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train(model, train_loader, optimizer, criterion, num_epochs=num_epochs)

    # Visualize results
    visualize_results(model)