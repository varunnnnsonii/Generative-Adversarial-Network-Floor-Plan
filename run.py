import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import os

# ----- CONFIG -----
MODEL_PATH = "generator_final.pth"
INPUT_IMAGE_PATH = "image_gan_in_final/0.png"  # <-- Change this to your input
OUTPUT_IMAGE_PATH = "output/output.png"
IMG_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----- MODEL DEFINITION -----
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 4, 2, 1), torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 4, 2, 1), torch.nn.BatchNorm2d(128), torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1), torch.nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# ----- LOAD MODEL -----
generator = Generator().to(DEVICE)
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()

# ----- PREPROCESS INPUT -----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

img = Image.open(INPUT_IMAGE_PATH).convert("L")  # Convert to grayscale
img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension

# ----- GENERATE OUTPUT -----
with torch.no_grad():
    output = generator(img_tensor)

# ----- SAVE OUTPUT -----
save_image(output, OUTPUT_IMAGE_PATH, normalize=True)
print(f"✅ Output saved as {OUTPUT_IMAGE_PATH}")
