# Transfer Learning - CIFAR-10 Classification

## 📌 Overview

This project implements **transfer learning** using a pre-trained convolutional neural network from Keras Applications to classify images from the CIFAR-10 dataset.

The model leverages features learned from large-scale datasets (ImageNet) and adapts them to a smaller dataset (CIFAR-10), improving performance and reducing training time.

---
## ⚙️ Requirements

* Python 3.9
* Ubuntu 20.04 LTS
* TensorFlow 2.15
* NumPy 1.25.2
* pycodestyle 2.11.1

---

## 🧠 Approach

1. **Data Preprocessing**

   * Normalize input images using MobileNetV2 preprocessing
   * Convert labels to one-hot encoding

2. **Model Architecture**

   * Input images resized from **32×32 → 96×96**
   * Pre-trained model: MobileNetV2 (without top layers)
   * Custom classification head added

3. **Training Strategy**

   * Freeze base model (feature extraction)
   * Train custom classifier
   * Unfreeze top layers for fine-tuning

4. **Optimization**

   * Early stopping
   * Learning rate reduction
   * Dropout and batch normalization

---

## 🚀 Usage

Run the training script:

```bash
./0-transfer.py
```

Evaluate the model:

```bash
./0-main.py
```

---

## 📊 Performance

The trained model achieves **≥ 87% validation accuracy** on CIFAR-10.

---

## 💡 Key Concept

Transfer learning allows reusing knowledge from large pre-trained models to solve new tasks efficiently, reducing both training time and data requirements.
