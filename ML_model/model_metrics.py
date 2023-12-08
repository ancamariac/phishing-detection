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
   
X = np.array(X)
Y = np.array(Y)

# split into train, val, test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=42) # 0.125 x 0.8 = 0.1

checkpoint_filepath = os.path.join(os.getcwd(), 'tmp', 'checkpoint')

metrics = [
         tf.keras.metrics.BinaryAccuracy(name='accuracy'),
         tf.keras.metrics.Precision(),
         tf.keras.metrics.Recall(),
         tf.keras.metrics.AUC()
         ]

def create_model(l1, l2, l3, d1, d2, opt):
   model = Sequential()
   model.add(l1)
   model.add(d1)
   model.add(l2)
   model.add(d2)
   model.add(l3)
   model.add(Dense(1, activation='sigmoid'))
   model.compile(loss='binary_crossentropy', metrics=metrics, optimizer=opt, run_eagerly=True)
   return model


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
   filepath=checkpoint_filepath,
   save_weights_only=True,
   monitor='val_accuracy',
   mode='max',
   save_best_only=True)

# combination = [l1_num, l2_num, d1, act1, act2,  opt, lr]
l1 = Dense(best_config[0], input_shape=(24, ), activation='relu')
l2 = Dense(best_config[1], activation='relu')
l3 = Dense(best_config[2], activation='relu')
d1 = Dropout(best_config[3])
d2 = Dropout(best_config[4])
if best_config[5] == 'adam':
   opt = keras.optimizers.Adam(learning_rate=best_config[6])
else:
   opt = keras.optimizers.SGD(learning_rate=best_config[6])

model = create_model(l1, l2, l3, d1, d2, opt)
history = model.fit(X_train, Y_train, batch_size=512, epochs=100, validation_data=(X_val, Y_val), callbacks=[model_checkpoint_callback])
model.load_weights(checkpoint_filepath)
scores = model.evaluate(X_test, Y_test)
Y_pred_prob = model.predict(X_test)

Y_pred = [1 if x[0] >= 0.5 else 0 for x in Y_pred_prob]

# save the model
model.save('model.h5')

# PLOTS 
history_dict = history.history
history_len = len(history_dict['loss'])
epochs = range(1, (history_len + 1))

# VALIDATION DATA

# validation loss
plt.clf()
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# validation accuracy
plt.clf()
acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']
plt.plot(epochs, acc_values, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc_values, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# validation precision
plt.clf()
prec_values = history_dict['precision']
val_prec_values = history_dict['val_precision']
plt.plot(epochs, prec_values, 'bo', label='Training precision')
plt.plot(epochs, val_prec_values, 'b', label='Validation precision')
plt.title('Training and validation precision')
plt.xlabel('Epochs')
plt.ylabel('Precision')
plt.legend()
plt.show()

# validation recall
plt.clf()
rec_values = history_dict['recall']
val_rec_values = history_dict['val_recall']
plt.plot(epochs, rec_values, 'bo', label='Training recall')
plt.plot(epochs, val_rec_values, 'b', label='Validation recall')
plt.title('Training and validation recall')
plt.xlabel('Epochs')
plt.ylabel('Recall')
plt.legend()
plt.show()

# validation f1 score
# f1_score = 2 * (precision * recall) / (precision + recall)

# small hack to avoid division by zero error that happens at first epoch
# the result will stay the same and the f1_score for precision=0 and recall=0 will be 0
for i in range(history_len):
   if prec_values[i] == 0 and rec_values[i] == 0:
      prec_values[i] = 1
   if val_prec_values[i] == 0 and val_rec_values[i] == 0:
      val_prec_values[i] = 1

plt.clf()
f1_values = [f1_score(Y_test, Y_pred, zero_division=1)]
val_f1_values = [f1_score(Y_val, model.predict(X_val) > 0.5, zero_division=1)] 

plt.plot(epochs, acc_values, 'bo', label='Training F1 score')
plt.plot(epochs, val_acc_values, 'b', label='Validation F1 score')
plt.title('Training and validation F1 score')
plt.xlabel('Epochs')
plt.ylabel('F1 score')
plt.legend()
plt.show()

# validation area under ROC curve
plt.clf()
auc_values = history_dict['auc']
val_auc_values = history_dict['val_auc']
plt.plot(epochs, auc_values, 'bo', label='Training AUC')
plt.plot(epochs, val_auc_values, 'b', label='Validation AUC')
plt.title('Training and validation AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.show()

# TEST DATA

# confusion matrix
labels = ['phishing', 'non-phishing']
cm = confusion_matrix(Y_test, Y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.show()

# metrics barplot
x_val = range(1, 6)
labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
values = [
    round(accuracy_score(Y_test, Y_pred) * 100, 2),
    round(precision_score(Y_test, Y_pred, average='macro') * 100, 2),
    round(recall_score(Y_test, Y_pred, average='macro') * 100, 2),
    round(f1_score(Y_test, Y_pred, average='macro') * 100, 2),
    round(roc_auc_score(Y_test, Y_pred_prob, multi_class='ovr') * 100, 2)
]

print(values)
plt.clf()
plt.bar(labels, values, color='dodgerblue', edgecolor='dimgrey')
for i, v in enumerate(values):
    plt.text(i - 0.25 , v + .25, str(v) + '%', color='dodgerblue', fontweight='bold')

plt.title('Test data metrics')
plt.xlabel('Metric')
plt.ylabel('Value')
plt.show()