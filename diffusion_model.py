import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import math
import argparse

# Hyperparameters
NB_STEPS = 5
NOISE_LEVELS = torch.linspace(0.05, 0.1, NB_STEPS)
ALPHA_VALUES = 1 - NOISE_LEVELS
ALPHA_CUMULATIVE = torch.cumprod(ALPHA_VALUES, 0)

# Noise addition function
def apply_noise(data, step_idx, alpha_cum):
    alpha_factor = alpha_cum[step_idx].view(-1, 1, 1, 1)
    gaussian_noise = torch.randn_like(data)
    return torch.sqrt(alpha_factor) * data + torch.sqrt(1 - alpha_factor) * gaussian_noise, gaussian_noise

# Dataset loader
def fetch_mnist_data(batch=128):
    data_transform = transforms.ToTensor()
    train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=data_transform)
    test_data = torchvision.datasets.MNIST('./data', train=False, download=True, transform=data_transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch, shuffle=False)
    return train_loader, test_loader


# Minimal UNet Model
class SimpleUNet(nn.Module):
    def __init__(self, time_steps):
        super().__init__()
        self.time_layer = nn.Embedding(time_steps, 28 * 28)
        self.conv1 = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU())
        self.downsample = nn.MaxPool2d(2)
        self.conv2 = nn.Sequential(nn.Conv2d(8, 16, 3, padding=1), nn.ReLU())
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Sequential(nn.Conv2d(24, 8, 3, padding=1), nn.ReLU())
        self.output_layer = nn.Conv2d(8, 1, 1)

    def forward(self, data, step):
        step_emb = self.time_layer(step).view(-1, 1, 28, 28)
        conditioned_input = data + step_emb
        c1 = self.conv1(conditioned_input)
        p1 = self.downsample(c1)
        c2 = self.conv2(p1)
        up = self.upsample(c2)
        x = self.conv3(torch.cat([up, c1], 1))
        return self.output_layer(x)

# Training function
def train_model(model, data_loader, num_epochs, device, alpha_cum):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            batch_size = batch_data.size(0)
            time_idx = torch.randint(0, NB_STEPS, (batch_size,), device=device)
            noisy_data, noise = apply_noise(batch_data, time_idx, alpha_cum)
            optimizer.zero_grad()
            prediction = model(noisy_data, time_idx)
            loss = loss_fn(prediction, noise)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_size
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(data_loader.dataset):.4f}")
    
    torch.save(model.state_dict(), f"trained_model_epoch_{num_epochs}.pth")
    return total_loss / len(data_loader.dataset)

# Inference function (denoising process)
def denoise_image(model, device, alpha_cum, total_steps, img_shape=(1, 1, 28, 28)):
    noisy_input = torch.randn(img_shape, device=device)
    for step in reversed(range(total_steps)):
        step_tensor = torch.full((noisy_input.size(0),), step, dtype=torch.long, device=device)
        with torch.no_grad():
            predicted_noise = model(noisy_input, step_tensor)
        alpha_factor = alpha_cum[step].view(-1, 1, 1, 1).to(device)
        noisy_input = (noisy_input - torch.sqrt(1 - alpha_factor) * predicted_noise) / torch.sqrt(alpha_factor)
    return noisy_input

# Main function
def main():
    parser = argparse.ArgumentParser(description="Diffusion Model Trainer")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--model_type", type=str, default="minimal", choices=["minimal", "sinus"],
                        help="Choose between 'minimal' or 'sinus'")
    parser.add_argument("--embed_size", type=int, default=128, help="Embedding size for sinusoidal model")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")
    
    train_loader, test_loader = fetch_mnist_data(batch=args.batch_size)
    
    if args.model_type == "minimal":
        net = SimpleUNet(NB_STEPS).to(device)

    alpha_cum_device = ALPHA_CUMULATIVE.to(device)
    
    final_loss = train_model(net, train_loader, num_epochs=args.epochs, device=device, alpha_cum=alpha_cum_device)
    
    net.eval()
    generated_img = denoise_image(net, device, alpha_cum_device, NB_STEPS, img_shape=(1, 1, 28, 28))

    plt.figure(figsize=(4, 4))
    plt.imshow(generated_img.cpu().squeeze().detach().numpy(), cmap='gray')
    plt.title("Generated Sample")
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
