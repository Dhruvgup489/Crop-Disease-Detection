# 🌱 AgroZyen AI

## AI-Based Crop Disease Detection System

AgroZyen AI is a web-based crop disease detection system that uses a Convolutional Neural Network (CNN) to identify diseases from crop leaf images.

The system supports three crops:

- Corn
- Potato
- Tomato

It can classify uploaded leaf images into **9 different categories**.

---

## 🎯 Project Objective

The main objective of AgroZyen AI is to provide a simple AI-based system that can help identify common crop diseases from leaf images.

The user uploads an image of a crop leaf, and the trained deep learning model predicts the most likely disease along with its confidence score.

---

## 🌿 Supported Classes

The model can classify the following 9 categories:

### Corn

1. Corn Common Rust
2. Corn Northern Leaf Blight
3. Corn Healthy

### Potato

4. Potato Early Blight
5. Potato Late Blight
6. Potato Healthy

### Tomato

7. Tomato Early Blight
8. Tomato Late Blight
9. Tomato Healthy

---

## 🤖 Machine Learning Model

A Convolutional Neural Network (CNN) was developed using TensorFlow and Keras.

### Input Image Size

```text
224 × 224 × 3

