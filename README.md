# Artattack GAN 🎨

This project uses a Deep Convolutional GAN (DCGAN) to generate artistic human portraits. Trained on a dataset of 4,117 curated art images.

## 📌 Tech Stack

* Python 3
* TensorFlow 2.x
* Keras
* Matplotlib, Seaborn

## 🎯 Goal

The primary objective of this project is to generate high-quality human portrait artworks using a Deep Convolutional GAN. This model learns to mimic artistic patterns and styles by analyzing thousands of real paintings, enabling creative AI applications such as art synthesis and AI-assisted design.

## 🧠 GAN Training Logic

This GAN implementation follows a custom training loop using Keras' subclassed `Model` class. Key features:

* Generator: Converts random noise vectors (latent space) into 64×64 RGB portrait images using multiple upsampling + Conv2D + BatchNorm layers.
* Discriminator: A CNN that classifies 64×64 images as real or fake using stacked Conv2D layers with LeakyReLU and Dropout.
* GAN Model: Trained using Binary Crossentropy loss and alternate optimization of Generator and Discriminator.
* Training loop includes label smoothing and noise injection for better convergence.

## Features

* Custom DCGAN with Keras subclassed training loop
* 200-epoch training on 64x64 RGB portraits
* Generator & Discriminator visualized and summarized
* Sample outputs from random noise inputs
* Custom loss tracking with plotted learning curves

## Dataset

[Art Portraits Dataset](https://www.kaggle.com/datasets/karnikakapoor/art-portraits)

## Sample Outputs

![image](https://github.com/user-attachments/assets/35af771f-cd7e-4841-a62f-233936ed8d2b)


## 📉 Training Curves

![image](https://github.com/user-attachments/assets/a56e41e3-6816-4191-8764-107a5479308b)


## 🛠️ How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/vedhadla/art-by-gan.git
cd art-by-gan
```

2. Set up a Python environment with TensorFlow 2.x:

```bash
pip install -r requirements.txt
```

3. Download the dataset and place it at:

```
../input/art-portraits/Portraits/
```

4. Run the notebook:

```bash
jupyter notebook art_by_gan.ipynb
```

## Model Architecture

Detailed model summaries:

* [`generator_summary.txt`](./generator_summary.txt)
* [`discriminator_summary.txt`](./discriminator_summary.txt)
