import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import cv2  # For camera access

# Configuration Class
class Config:
    latent_dim = 100
    img_channels = 3
    img_size = 256  # Desired resolution
    output_dir = os.path.join(os.getcwd(), 'output')
    num_iterations = 10000
    save_interval = 5
    print_interval = 5
    learning_rate = 0.0002
    betas = (0.5, 0.999)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    moving_average_window = 10  # For stabilizing loss history
    gradient_clipping = 1.0  # Maximum norm for gradient clipping

    @classmethod
    def print_config(cls):
        print("\n=== Configuration ===")
        for attr, value in cls.__dict__.items():
            if not attr.startswith('__') and not callable(value):
                print(f"{attr}: {value}")
        print("======================\n")

os.makedirs(Config.output_dir, exist_ok=True)
print(f"[INFO] Output directory set at: {Config.output_dir}")

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
        print(f"[INFO] Initialized ResidualBlock with in_channels={channels_in}, out_channels={channels_out}")

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
        print(f"[INFO] Generator fully connected layer initialized with output size={512 * self.init_size * self.init_size}")

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
        print(f"[INFO] Generator convolutional blocks initialized.")

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)  # Reshape to (batch_size, 512, 32, 32)
        img = self.conv_blocks(x)  # Output: (batch_size, img_channels, 256, 256)
        return img

# Function to save generated images
def save_generated_image(fake_image, iteration, output_dir):
    try:
        fake_image = fake_image.squeeze(0).detach().cpu()
        # Scale from [-1, 1] to [0, 255]
        fake_image = (fake_image * 0.5 + 0.5) * 255
        fake_image = fake_image.clamp(0, 255).byte()
        fake_image = fake_image.permute(1, 2, 0).numpy()
        fake_image_pil = Image.fromarray(fake_image)
        filename = os.path.join(output_dir, f'generated_image_{iteration}.png')
        fake_image_pil.save(filename)
        print(f"[INFO] Generated image saved at iteration {iteration}: {filename}")
        return filename
    except Exception as e:
        print(f"[ERROR] Failed to save generated image at iteration {iteration}: {e}")

# Training loop
def train(generator, criterion, optimizer, config, camera):
    generator.train()
    loss_history = []
    iteration = 0

    transform = transforms.Compose([
        transforms.Resize((config.img_size, config.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*config.img_channels, [0.5]*config.img_channels)  # Normalize to [-1, 1]
    ])
    print("[INFO] Image transformations for target images set.")

    print("[INFO] Starting training loop.")
    while iteration < config.num_iterations:
        iteration += 1
        print(f"\n[ITERATION] Starting iteration {iteration}/{config.num_iterations}")

        # Generate random noise
        z = torch.randn(1, config.latent_dim, device=config.device)
        print(f"[DEBUG] Generated random noise vector of shape {z.shape}")

        # Generate fake image
        fake_image = generator(z)
        print(f"[DEBUG] Generated fake image with shape {fake_image.shape}")

        # Capture image from camera
        ret, frame = camera.read()
        if not ret:
            print(f"[WARNING] Failed to capture image from camera at iteration {iteration}. Skipping this iteration.")
            continue
        print(f"[DEBUG] Captured frame from camera at iteration {iteration}")

        try:
            # Convert BGR (OpenCV format) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            print(f"[DEBUG] Converted captured frame to RGB and PIL Image.")
        except Exception as e:
            print(f"[ERROR] Failed to process captured frame at iteration {iteration}: {e}")
            continue

        # Apply transformations
        try:
            target_image = transform(frame_pil).unsqueeze(0).to(config.device)
            print(f"[DEBUG] Applied transformations to target image.")
        except Exception as e:
            print(f"[ERROR] Failed to transform target image at iteration {iteration}: {e}")
            continue

        # Calculate loss
        try:
            loss = criterion(fake_image, target_image)
            loss_value = loss.item()
            loss_history.append(loss_value)
            print(f"[DEBUG] Calculated loss: {loss_value:.6f}")
        except Exception as e:
            print(f"[ERROR] Failed to calculate loss at iteration {iteration}: {e}")
            continue

        # Update generator
        try:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=config.gradient_clipping)
            optimizer.step()
            print(f"[DEBUG] Updated generator parameters.")
        except Exception as e:
            print(f"[ERROR] Failed to update generator at iteration {iteration}: {e}")
            continue

        # Print loss periodically
        if iteration % config.print_interval == 0:
            print(f"[INFO] Iteration [{iteration}/{config.num_iterations}], Loss: {loss_value:.6f}")

        # Save generated images at intervals
        if iteration % config.save_interval == 0:
            print(f"[INFO] Saving generated image at iteration {iteration}")
            save_generated_image(fake_image, iteration, config.output_dir)

    # Save final generated image and checkpoint
    print("\n[INFO] Saving final generated image and checkpoint.")
    save_generated_image(fake_image, config.num_iterations, config.output_dir)
    checkpoint = {
        'iteration': config.num_iterations,
        'model_state_dict': generator.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history
    }
    final_checkpoint_path = os.path.join(config.output_dir, f'checkpoint_final.pth')
    try:
        torch.save(checkpoint, final_checkpoint_path)
        print(f"[INFO] Final model checkpoint saved: {final_checkpoint_path}")
    except Exception as e:
        print(f"[ERROR] Failed to save final checkpoint: {e}")

    print("\n[INFO] Training completed successfully.")

# Initialize model, criterion, and optimizer
def main():
    print("[INFO] Initializing configuration.")
    config = Config()
    Config.print_config()

    print("[INFO] Initializing Generator model.")
    generator = Generator(
        latent_dim=config.latent_dim,
        img_channels=config.img_channels,
        img_size=config.img_size
    ).to(config.device)
    print(f"[INFO] Generator model moved to device: {config.device}")

    print("[INFO] Setting up loss criterion and optimizer.")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(generator.parameters(), lr=config.learning_rate, betas=config.betas)
    print("[INFO] Loss criterion and optimizer initialized.")

    # Initialize camera
    print("[INFO] Attempting to access the laptop's camera.")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("[ERROR] Could not open the camera.")
        sys.exit(1)
    else:
        print("[INFO] Camera successfully accessed.")

    try:
        # Train the model
        print("[INFO] Starting training process.")
        train(generator, criterion, optimizer, config, camera)
    except KeyboardInterrupt:
        print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during training: {e}")
    finally:
        # Release the camera resource
        camera.release()
        print("[INFO] Camera resource released.")
        cv2.destroyAllWindows()
        print("[INFO] All OpenCV windows closed.")

if __name__ == '__main__':
    main()
