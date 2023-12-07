import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from tensorflow import keras
from feature_extraction import feature_extraction

NN_model = keras.models.load_model('model.h5')

print("Your input:")
text = input()
final = feature_extraction(text)
X_final = np.array([final])
val_pred = NN_model.predict(X_final)
print(val_pred)