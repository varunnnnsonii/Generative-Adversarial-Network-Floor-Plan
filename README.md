# 🧠 Generative Adversarial Network for Floor Plan Generation 🏠

A PyTorch-based project that trains a **Generative Adversarial Network (GAN)** to generate architectural floor plans from grayscale or black-and-white input sketches. Designed to be lightweight and trainable on systems with limited resources (like 8GB RAM and GTX 1650 GPU), it uses a carefully selected 10K paired dataset from a larger collection of 80K images.

---

## 🚧 Project Highlights

- 🔁 **Paired Dataset**: Input and output images matched by filename in two separate folders
- 🧠 **Custom GAN**: Includes both Generator (U-Net style) and PatchGAN Discriminator
- 💾 **Model Checkpointing**: Automatically saves generator and discriminator
- 🧪 **Inference-Ready**: Easily generate new floor plans using your trained model
- 🛠️ **Optimized Training**: Configurable for shorter training cycles


---

## ✨ Features

- ✅ Trainable GAN using paired input/output images
- ✅ Automatically preprocesses images (resized, B/W conversion)
- ✅ Save and reuse trained generator for inference
- ✅ Lightweight and runs comfortably on mid-range systems
- ✅ Custom dataset support — just match filenames!

---

## 🗂️ Folder Structure

```
project_root/ 
├── saved_models/ # Contains generator.pth, discriminator.pth │ ├── train.py # Script to train GAN ├── run.py # Script to generate output from new input ├── requirements.txt # List of dependencies └── README.md # You're here!

```
---

## 🖼️ Sample Workflow

### Input:
* [Input Example](assets/99_input.png)

### Generated Output:  
*  [Output Example](assets/99_output.png)

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/varunnnnsonii/Generative-Adversarial-Network-Floor-Plan.git
cd Generative-Adversarial-Network-Floor-Plan

pip install -r requirements.txt

python train.py

python run.py
```
---
## 💾 Save & Load Models
### Models are saved /:
* generator_final.pth
* discriminator_final.pth

### Use these with run.py to generate outputs for new input images.
---

## 🤖 Model Architecture

- **Generator:** U-Net style convolutional network
- **Discriminator:** PatchGAN (CNN-based classifier)

---

## 🛠️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- Pillow
- tqdm

---

## ✍️ Author

**Varun D Soni**  
🔗 [GitHub](https://github.com/varunnnnsonii)  
📫 varun271203@gmail.com

---

## 🌟 Star this repo if it helped you!
### If this project helped you or inspired you, don’t forget to:

* ⭐ Star the repo

* 🍴 Fork it

* 
🐛 Raise an issue or contribute!




### ✅ Next Steps

- Add `requirements.txt` using:
  ```bash
  pip freeze > requirements.txt
  ```
- Add your actual **input/output example images** in the repo (or upload them via GitHub → "Upload files")
- Replace the `![Input]` and `![Output]` image URLs with GitHub image links (once uploaded)
- Commit and push the updated `README.md`:
  ```bash
  git add README.md
  git commit -m "Added beautiful README 💫"
  git push
  ```

---
### That’s it! When you go to your GitHub repo, you'll see your new professional README automatically rendered ✨
