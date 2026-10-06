import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

class PetDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(128, 128)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # Old
        #self.image_filenames = sorted(os.listdir(image_dir))
        #New
        self.image_filenames = sorted([
            f for f in os.listdir(image_dir) if f.endswith(".jpg")
        ])

        self.mask_filenames = sorted(os.listdir(mask_dir))

        # Transform for images
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        # Transform for masks (no normalization!)
        self.mask_transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=InterpolationMode.NEAREST),
            #transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        # Get file paths
        img_name = self.image_filenames[idx]

        # Replace .jpg with .png to find matching mask
        mask_name = img_name.replace(".jpg", ".png")

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Transform mask (resizes)
        mask = self.mask_transform(mask)

        # Convert mask to binary (cat vs background)
        mask = np.array(mask)
        mask = (mask == 1).astype(np.float32)

        # Convert mask to tensor
        mask = torch.from_numpy(mask).unsqueeze(0)

        # Apply transforms
        image = self.image_transform(image)

        return image, mask

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Left side: downsampling
        self.down1 = DoubleConv(3, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottom
        self.bottleneck = DoubleConv(128, 256)

        # Right side: upsampling
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        # Final output: 1 channel mask
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        # Down path
        x1 = self.down1(x)
        x2 = self.pool1(x1)

        x3 = self.down2(x2)
        x4 = self.pool2(x3)

        # Bottom
        x5 = self.bottleneck(x4)

        # Up path
        x6 = self.up1(x5)
        x6 = torch.cat([x6, x3], dim=1)
        x6 = self.conv_up1(x6)

        x7 = self.up2(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.conv_up2(x7)

        # Final prediction
        out = self.final(x7)

        return out

dataset = PetDataset(
    image_dir="../images/pet_images",
    mask_dir="../images/annotations/trimaps"
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

model = UNet()

print("Is cuda available: ", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Compares predicted mask with true mask and outputs loss as a single number
criterion = nn.BCEWithLogitsLoss()

# Updates model weights to reduce loss
# lr = learning rate (step-size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training loop
#-------------
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")

    for images, masks in dataloader:
        # Move to device (CPU/GPU)
        images = images.to(device)
        masks = masks.to(device)

        # Forward pass (prediction)
        outputs = model(images)

        # Compute loss
        loss = criterion(outputs, masks)

        # Reset gradients
        optimizer.zero_grad()

        # Backward pass (compute gradients)
        loss.backward()

        # Update weights
        optimizer.step()

    print(f"Loss: {loss.item():.4f}")

# Put model into evaluation mode
model.eval()

# Turn off gradient tracking since we are only inspecting results
with torch.no_grad():
    # Get one batch from the dataloader
    images, masks = next(iter(dataloader))

    # Move to same device as model
    images = images.to(device)
    masks = masks.to(device)

    # Get model output
    outputs = model(images)

    # Convert raw output scores into values between 0 and 1
    preds = torch.sigmoid(outputs)

    # Pick the first item in the batch
    image = images[0].cpu()
    true_mask = masks[0].cpu()
    pred_mask = preds[0].cpu()


# Convert image from [C, H, W] to [H, W, C] for matplotlib
image = image.permute(1, 2, 0)

# Remove the single channel dimension from masks: [1, H, W] -> [H, W]
true_mask = true_mask.squeeze(0)
pred_mask = pred_mask.squeeze(0)

# Show the three images side by side
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(true_mask, cmap="gray")
plt.title("Ground Truth Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap="gray")
plt.title("Predicted Mask")
plt.axis("off")

plt.show()