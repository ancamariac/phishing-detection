import pickle
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

best_config = [512, 1024, 512, 0.1, 0.3, 'adam', 0.01]

data = pd.read_csv('../dataset/dataset.csv')
X = data.iloc[:, 1:-1]
Y = data.iloc[:, -1]

print("Primele 5 rânduri ale datelor X:")
print(X.head())

print("\nPrimele 5 rânduri ale datelor Y:")
print(Y.head())