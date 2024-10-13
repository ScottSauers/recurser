import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Configuration Class
class Config:
    latent_dim = 100
    img_channels = 3
    img_size = 256  # Desired resolution
    output_dir = os.path.join(os.getcwd(), 'output')
    num_iterations = 10000
    save_interval = 500
    print_interval = 100
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moving_average_window = 10  # For stabilizing loss history
    gradient_clipping = 1.0  # Maximum norm for gradient clipping

# Ensure the output directory exists
os.makedirs(Config.output_dir, exist_ok=True)

# Residual Block to enhance generator capability
class ResidualBlock(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels_out, affine=True),
            nn.ReLU(True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(channels_out, affine=True)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

# Generator Model - Enhanced for Higher Resolution and Quality
class Generator(nn.Module):
    def __init__(self, latent_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.init_size = img_size // 8  # For 256x256, init_size=32
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.init_size * self.init_size),
            nn.LayerNorm(512 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        self.conv_blocks = nn.Sequential(
            # Upsample to 64x64
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32 -> 64
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.ReLU(True),
            
            # Residual Block
            ResidualBlock(256, 256),
            
            # Upsample to 128x128
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64 -> 128
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
            
            # Residual Block
            ResidualBlock(128, 128),
            
            # Upsample to 256x256
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128 -> 256
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            
            # Residual Block
            ResidualBlock(64, 64),
            
            # Final layer to get desired channels
            nn.Conv2d(64, img_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)  # Reshape to (batch_size, 512, 32, 32)
        img = self.conv_blocks(x)  # Output: (batch_size, img_channels, 256, 256)
        return img

# Function to generate and save the loss graph
def generate_loss_graph(iteration, loss_history, output_dir, img_size, moving_average_window=10):
    plt.figure(figsize=(8, 6), facecolor='black')  # Increased figure size for better resolution
    if len(loss_history) > 0:
        # Calculate moving average to stabilize loss history
        if len(loss_history) >= moving_average_window:
            smoothed_loss = np.convolve(loss_history, np.ones(moving_average_window)/moving_average_window, mode='valid')
        else:
            smoothed_loss = loss_history.copy()
        
        plt.plot(range(1, len(smoothed_loss) + 1), smoothed_loss, label='Loss (MA)', color='cyan', linewidth=5)  # Extremely thick line
    
    plt.title('Training Loss', color='white', fontsize=20)
    plt.xlabel('Iteration', color='white', fontsize=16)
    plt.ylabel('MSE Loss (Log Scale)', color='white', fontsize=16)
    plt.yscale('log')  # Apply log scale to y-axis
    plt.legend(facecolor='black', edgecolor='white', labelcolor='white', fontsize=14)
    plt.grid(True, which="both", ls="--", linewidth=0.5, color='gray')
    plt.tick_params(axis='x', colors='white', labelsize=14)
    plt.tick_params(axis='y', colors='white', labelsize=14)
    plt.tight_layout()
    filename = os.path.join(output_dir, f'loss_graph_{iteration}.png')
    plt.savefig(filename, facecolor='black')
    plt.close()
    
    # Resize to match generator output resolution
    loss_image = Image.open(filename).convert('RGB')
    
    # Handle different Pillow versions
    try:
        # For Pillow >= 10.0.0
        resample_filter = Image.Resampling.LANCZOS
    except AttributeError:
        # For Pillow < 10.0.0
        resample_filter = Image.LANCZOS
    
    loss_image = loss_image.resize((img_size, img_size), resample=resample_filter)
    loss_image.save(filename)
    return filename

# Function to save generated images
def save_generated_image(fake_image, iteration, output_dir):
    fake_image = fake_image.squeeze(0).detach().cpu()
    # Scale from [-1, 1] to [0, 255]
    fake_image = (fake_image * 0.5 + 0.5) * 255
    fake_image = fake_image.clamp(0, 255).byte()
    fake_image = fake_image.permute(1, 2, 0).numpy()
    fake_image_pil = Image.fromarray(fake_image)
    filename = os.path.join(output_dir, f'generated_image_{iteration}.png')
    fake_image_pil.save(filename)
    return filename

# Training loop
def train(generator, criterion, optimizer, config):
    generator.train()
    loss_history = []
    iteration = 0

    while iteration < config.num_iterations:
        iteration += 1

        # Generate random noise
        z = torch.randn(1, config.latent_dim, device=config.device)

        # Generate fake image
        fake_image = generator(z)

        # Create loss graph image
        loss_image_filename = generate_loss_graph(
            iteration, 
            loss_history, 
            config.output_dir, 
            config.img_size, 
            config.moving_average_window
        )

        # Save generated image
        generated_image_filename = save_generated_image(fake_image, iteration, config.output_dir)

        # Load loss function image as target
        loss_image = Image.open(loss_image_filename).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor()
        ])
        loss_image = transform(loss_image).unsqueeze(0).to(config.device)

        # Calculate loss
        loss = criterion(fake_image, loss_image)
        loss_history.append(loss.item())

        # Update generator
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=config.gradient_clipping)
        
        optimizer.step()

        # Print loss periodically
        if iteration % config.print_interval == 0:
            print(f'Iteration [{iteration}/{config.num_iterations}], Loss: {loss.item():.6f}')

        # Keep one in every 5 images, delete others to save space
        if iteration % 5 != 0:
            try:
                os.remove(loss_image_filename)
                os.remove(generated_image_filename)
            except Exception as e:
                print(f"Error deleting files at iteration {iteration}: {e}")

        # Save checkpoints and images at intervals
        if iteration % config.save_interval == 0:
            print(f"Saving progress at iteration {iteration}")
            generate_loss_graph(iteration, loss_history, config.output_dir, config.img_size, config.moving_average_window)
            save_generated_image(fake_image, iteration, config.output_dir)
            # Save model checkpoint
            checkpoint = {
                'iteration': iteration,
                'model_state_dict': generator.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss_history': loss_history
            }
            torch.save(checkpoint, os.path.join(config.output_dir, f'checkpoint_{iteration}.pth'))

    # Save final loss graph and image
    generate_loss_graph(config.num_iterations, loss_history, config.output_dir, config.img_size, config.moving_average_window)
    save_generated_image(fake_image, config.num_iterations, config.output_dir)
    # Save final checkpoint
    checkpoint = {
        'iteration': config.num_iterations,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history
    }
    torch.save(checkpoint, os.path.join(config.output_dir, f'checkpoint_final.pth'))
    print("Training completed.")

# Initialize model, criterion, and optimizer
def main():
    config = Config()

    generator = Generator(
        latent_dim=config.latent_dim,
        img_channels=config.img_channels,
        img_size=config.img_size
    ).to(config.device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=config.betas)

    # Train the model
    train(generator, criterion, optimizer, config)

if __name__ == '__main__':
    main()
