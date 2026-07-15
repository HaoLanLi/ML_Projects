# MNIST Handwritten Digit Classification 🧠✍️

## Introduction

This is my second Machine Learning project while learning from **"Zero-Basic Machine Learning" by Jia Huang**.

In this project, I built my first **Convolutional Neural Network (CNN)** to classify handwritten digits from the MNIST dataset.

Compared with my first project (California Housing Price Prediction), this project introduced me to **Deep Learning** and image classification using TensorFlow and Keras.

---

## Project Goal

The goal is to train a CNN model that can recognize handwritten digits (0–9) from grayscale images.

This project helped me understand the complete workflow of a simple image classification task.

---

## Dataset

This project uses the **MNIST Handwritten Digits Dataset** provided by TensorFlow/Keras.

### Dataset Information

* Training images: **60,000**
* Testing images: **10,000**
* Image size: **28 × 28 pixels**
* Color channel: **Grayscale (1 channel)**
* Number of classes: **10 (Digits 0–9)**

---

## Data Preprocessing

Before training the model, I completed the following preprocessing steps:

* Loaded the MNIST dataset
* Reshaped images from `(28, 28)` to `(28, 28, 1)`
* Converted labels into one-hot encoded vectors using `to_categorical()`
* Prepared the data for CNN input

---

## CNN Architecture

The model consists of:

* Conv2D (32 filters)
* MaxPooling2D
* Conv2D (64 filters)
* MaxPooling2D
* Dropout (0.25)
* Flatten
* Dense (128 neurons)
* Dropout (0.5)
* Dense (10 neurons, Softmax)

---

## Model Configuration

* Optimizer: **RMSprop**
* Loss Function: **Categorical Crossentropy**
* Evaluation Metric: **Accuracy**

Training settings:

* Epochs: **5**
* Batch Size: **128**
* Validation Split: **30%**

---

## Prediction

After training, the model predicts the digit shown in the test image.

The project also displays the handwritten image using Matplotlib and compares it with the model's prediction.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Jupyter Notebook

---

## What I Learned

Through this project, I learned:

* How to use the MNIST dataset
* Basic image preprocessing for deep learning
* What a Convolutional Neural Network (CNN) is
* How convolution and pooling layers work
* Why Dropout is used to reduce overfitting
* How to train and evaluate a neural network using TensorFlow/Keras

This project is my first step into Deep Learning and computer vision.

---

## Future Improvements

In future versions, I would like to:

* Train for more epochs
* Tune hyperparameters
* Visualize training and validation accuracy
* Display prediction results for multiple test images
* Try more advanced CNN architectures

---

## Project Status

✅ Completed

This project is my second Machine Learning project and my first Deep Learning project.

It helped me understand the basic CNN workflow:

```text
Image
   ↓
Preprocessing
   ↓
CNN
   ↓
Training
   ↓
Prediction
```

I'm looking forward to building more computer vision projects as I continue learning Machine Learning and Deep Learning.
