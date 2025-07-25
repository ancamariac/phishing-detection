# 🧠 Phishing Detection – Neural Network Classifier

This repository contains the code and configuration used to train a binary classification model for phishing website detection using a **Feedforward Neural Network** implemented in **Keras**. The trained model is also integrated into the [Web Sentinel](https://github.com/yourusername/web-sentinel) browser extension for real-time phishing detection.

---

## 📌 Overview

The classification problem is addressed using a **feedforward (sequential) neural network**, where information flows in one direction—from the input layer to the output—without cycles or loops. The dataset is split into training, validation, and testing sets using `train_test_split()` in the following ratio:

- **Training:** 70%
- **Validation:** 10%
- **Testing:** 20%

---

## 🔧 Hyperparameter Tuning – Grid Search

To find the best-performing configuration, we used **Grid Search**, an exhaustive search technique for hyperparameter optimization. Grid Search systematically evaluates all possible combinations of hyperparameters to identify the optimal setup.

### Parameters Explored:

| Hyperparameter        | Values Tested                          |
|-----------------------|----------------------------------------|
| Hidden layers (1–3)   | 512, 1024                              |
| Dropout rates         | 0.1, 0.3                               |
| Optimizers            | `sgd`, `adam`                          |
| Learning rates        | 0.001, 0.01                            |

For each combination, a model was built using `Sequential()` with the defined layers. The final output layer uses a sigmoid activation function:

```python
Dense(1, activation='sigmoid')
```

## 🧱 Best Model Architecture

The best configuration identified via Grid Search is as follows:

Layer 1: 512 neurons
Dropout 1: 0.1
Layer 2: 1024 neurons
Dropout 2: 0.3
Layer 3: 512 neurons
Output Layer: 1 neuron with sigmoid activation
Optimizer: adam
Learning rate: 0.01
Loss Function: binary_crossentropy
Batch Size: 512
Epochs: 100

Training was monitored using a ModelCheckpoint callback to save the best weights during training.

<img width="540" height="390" alt="image" src="https://github.com/user-attachments/assets/9fbbe80a-30d7-4a55-bad9-bce241ea0675" />

## 📊 Evaluation Metrics

The final model was evaluated on the test set with the following performance:

- **Accuracy:** 91.5%
- **Precision:** 91.53%
- **Recall (TPR):** 91.31%
- **F1 Score:** 91.48%

### 📈 Plots

<img width="370" height="300" alt="image" src="https://github.com/user-attachments/assets/33330ebe-d793-48df-a5ac-c5d10eef1a72" />

<img width="370" height="300" alt="image" src="https://github.com/user-attachments/assets/f4d30fe3-67b8-4802-aa2c-6b930a1b80d3" />

<img width="370" height="300" alt="image" src="https://github.com/user-attachments/assets/7ba3fefd-7471-4f68-9ac6-d002bf0fd298" />

<img width="370" height="300" alt="image" src="https://github.com/user-attachments/assets/bf19d48f-2ea7-423e-985c-1d9697ec108d" />

---

## 🔗 Related Repository

The trained model is integrated into the **Web Sentinel** browser extension:

👉 [Web Sentinel – Anti-Phishing Browser Extension](https://github.com/ancamariac/web-sentinel)

---
