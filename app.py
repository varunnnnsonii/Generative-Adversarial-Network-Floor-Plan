# import os
# import random
# import torch
# from torch import nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.utils import save_image
# from PIL import Image
# from tqdm import tqdm

# # ----- CONFIG -----
# INPUT_DIR = 'image_gan_in_final'
# OUTPUT_DIR = 'image_gan_out_final'

# IMG_SIZE = 128
# BATCH_SIZE = 16
# EPOCHS = 10
# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# SAMPLE_SIZE = 10000

# # ----- DATASET -----
# class PairedImageDataset(Dataset):
#     def __init__(self, input_dir, output_dir, file_list, transform):
#         self.input_dir = input_dir
#         self.output_dir = output_dir
#         self.file_list = file_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         fname = self.file_list[idx]
#         input_path = os.path.join(self.input_dir, fname)
#         output_path = os.path.join(self.output_dir, fname)

#         input_img = Image.open(input_path).convert("L")  # Grayscale input
#         output_img = Image.open(output_path).convert("RGB")  # RGB target

#         return self.transform(input_img), self.transform(output_img)

# transform = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor()
# ])

# file_names = list(set(os.listdir(INPUT_DIR)) & set(os.listdir(OUTPUT_DIR)))
# file_names = random.sample(file_names, min(SAMPLE_SIZE, len(file_names)))
# dataset = PairedImageDataset(INPUT_DIR, OUTPUT_DIR, file_names, transform)
# dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# # ----- MODELS -----
# def weights_init(m):
#     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
#         nn.init.normal_(m.weight.data, 0.0, 0.02)

# class Generator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
#             nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
#         )

#     def forward(self, x):
#         return self.model(x)

# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(4, 64, 4, 2, 1), nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid()
#         )

#     def forward(self, x, y):
#         return self.model(torch.cat([x, y], dim=1))

# # ----- TRAIN -----
# generator = Generator().to(DEVICE)
# discriminator = Discriminator().to(DEVICE)
# generator.apply(weights_init)
# discriminator.apply(weights_init)

# criterion = nn.BCELoss()
# optim_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
# optim_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# print("🚀 Training Started...")

# for epoch in range(EPOCHS):
#     pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
#     for i, (x, y) in enumerate(pbar):
#         x, y = x.to(DEVICE), y.to(DEVICE)

#         # Forward once to get shape for dynamic labels
#         with torch.no_grad():
#             fake_y = generator(x)
#             pred_shape = discriminator(x, fake_y).shape

#         real_labels = torch.ones(pred_shape).to(DEVICE)
#         fake_labels = torch.zeros(pred_shape).to(DEVICE)

#         # ----- Train Discriminator -----
#         with torch.no_grad():
#             fake_y = generator(x)
#         pred_real = discriminator(x, y)
#         pred_fake = discriminator(x, fake_y.detach())

#         loss_D_real = criterion(pred_real, real_labels)
#         loss_D_fake = criterion(pred_fake, fake_labels)
#         loss_D = (loss_D_real + loss_D_fake) * 0.5

#         discriminator.zero_grad()
#         loss_D.backward()
#         optim_D.step()

#         # ----- Train Generator -----
#         fake_y = generator(x)
#         pred_fake = discriminator(x, fake_y)
#         loss_G = criterion(pred_fake, real_labels)

#         generator.zero_grad()
#         loss_G.backward()
#         optim_G.step()

#         pbar.set_postfix({"Loss_D": loss_D.item(), "Loss_G": loss_G.item()})

#     save_image(fake_y[:8], f"generated_epoch_{epoch+1}.png", nrow=4, normalize=True)

# print("✅ Training Complete.")
import os
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm

# ----- CONFIG -----
INPUT_DIR = 'image_gan_in_final'
OUTPUT_DIR = 'image_gan_out_final'
IMG_SIZE = 128
BATCH_SIZE = 16
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAMPLE_SIZE = 10000

# ----- DATASET -----
class PairedImageDataset(Dataset):
    def __init__(self, input_dir, output_dir, file_list, transform):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        input_path = os.path.join(self.input_dir, fname)
        output_path = os.path.join(self.output_dir, fname)

        input_img = Image.open(input_path).convert("L")  # Grayscale input
        output_img = Image.open(output_path).convert("RGB")  # RGB target

        return self.transform(input_img), self.transform(output_img)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

file_names = list(set(os.listdir(INPUT_DIR)) & set(os.listdir(OUTPUT_DIR)))
file_names = random.sample(file_names, min(SAMPLE_SIZE, len(file_names)))
dataset = PairedImageDataset(INPUT_DIR, OUTPUT_DIR, file_names, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ----- MODELS -----
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
        nn.init.normal_(m.weight.data, 0.0, 0.02)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1), nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, 4, 1, 1), nn.Sigmoid()
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))

# ----- TRAIN -----
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)
generator.apply(weights_init)
discriminator.apply(weights_init)

criterion = nn.BCELoss()
optim_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

print("🚀 Training Started... Press Ctrl+C to stop and save model.")

epoch = 0
try:
    while True:
        epoch += 1
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for x, y in pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)

            # Get dynamic label shape
            with torch.no_grad():
                fake_y = generator(x)
                pred_shape = discriminator(x, fake_y).shape

            real_labels = torch.ones(pred_shape).to(DEVICE)
            fake_labels = torch.zeros(pred_shape).to(DEVICE)

            # --- Train Discriminator ---
            with torch.no_grad():
                fake_y = generator(x)
            pred_real = discriminator(x, y)
            pred_fake = discriminator(x, fake_y.detach())

            loss_D_real = criterion(pred_real, real_labels)
            loss_D_fake = criterion(pred_fake, fake_labels)
            loss_D = 0.5 * (loss_D_real + loss_D_fake)

            discriminator.zero_grad()
            loss_D.backward()
            optim_D.step()

            # --- Train Generator ---
            fake_y = generator(x)
            pred_fake = discriminator(x, fake_y)
            loss_G = criterion(pred_fake, real_labels)

            generator.zero_grad()
            loss_G.backward()
            optim_G.step()

            pbar.set_postfix({"Loss_D": loss_D.item(), "Loss_G": loss_G.item()})

        # Save generated images every epoch
        save_image(fake_y[:8], f"generated_epoch_{epoch}.png", nrow=4, normalize=True)

except KeyboardInterrupt:
    print("\n🛑 Training interrupted. Saving models...")
    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")
    print("✅ Models saved successfully.")
