import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import os

# Ensure the directory exists
output_dir = '/your/fav/path/recur'
os.makedirs(output_dir, exist_ok=True)

# Generator Model
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 3, 32, 32)
        return x

# Loss function graph generator
def generate_loss_graph(iteration, loss):
    plt.figure(figsize=(2, 2))
    if len(loss) > 0:
        plt.plot(range(len(loss)), loss, label='Loss')
    plt.legend()
    filename = os.path.join(output_dir, f'loss_graph_{iteration}.png')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()
    return filename

# Training loop
def train(generator, criterion, optimizer):
    z = Variable(torch.randn(1, 100))
    loss_history = []
    iteration = 0

    while True:
        # Generate image
        fake_image = generator(z)
        
        # Create loss function image
        loss_image_filename = generate_loss_graph(iteration, loss_history)

        # Save generated image
        fake_image_np = fake_image.squeeze(0).detach().numpy().transpose(1, 2, 0)
        fake_image_pil = Image.fromarray(((fake_image_np + 1) * 127.5).astype(np.uint8))
        generated_image_filename = os.path.join(output_dir, f'generated_image_{iteration}.png')
        fake_image_pil.save(generated_image_filename)

        # Load loss function image
        loss_image = Image.open(loss_image_filename).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        loss_image = transform(loss_image)
        loss_image = loss_image.unsqueeze(0)

        # Calculate loss
        loss = criterion(fake_image, loss_image)
        loss_history.append(loss.item())

        # Update generator
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if iteration % 100 == 0:
            print(f'Iteration [{iteration}], Loss: {loss.item():.4f}')

        # Keep one in every 5 images, delete others
        if iteration % 5 != 0:
            os.remove(loss_image_filename)
            os.remove(generated_image_filename)

        iteration += 1

# Initialize model, criterion and optimizer
generator = Generator()
criterion = nn.MSELoss()
optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# Train the model
train(generator, criterion, optimizer)
