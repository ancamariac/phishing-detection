from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
import numpy as np
import itertools 
from tensorflow import keras
import matplotlib.pyplot as plt
import tensorflow as tf
import os 
import pandas as pd

# load dataset
# X = features
# Y = labels


data = pd.read_csv('../dataset/dataset.csv')
X = data.iloc[:, 1:-1]
Y = data.iloc[:, -1]
   
X = np.array(X)
Y = np.array(Y)

# split into train, val, test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.125, random_state=42) # 0.125 x 0.8 = 0.1

checkpoint_filepath = os.path.join(os.getcwd(), 'tmp', 'checkpoint')

# grid search dict
dct_grid_space = {
   'layer1' : [
      1024,
      512
   ],
   'layer2' : [
      1024,
      512
   ],
   'layer3' : [
      1024,
      512
   ],
   'dropout1' : [
      0.1,
      0.3
   ],
   'dropout2' : [
      0.1,
      0.3
   ],
   'opt_class' : [
      'sgd',
      'adam'
   ],
   'lr' : [
      0.001,
      0.01,
   ]
}

# generate all combinations for grid search
grid_params = []
grid_values = []

for k in dct_grid_space:
   grid_params.append(k)
   grid_values.append(dct_grid_space[k])
# se face produsul cu toate elementele din grid_values
grid_combs = list(itertools.product(*grid_values))

#TODO schimba linia 98, 99 ca sa fie pentru clasificare binara (1 neuron final, schimbi loss si activation)
def create_model(l1, l2, l3, d1, d2, opt):
   model = Sequential()
   model.add(l1)
   model.add(d1)
   model.add(l2)
   model.add(d2)
   model.add(l3)
   model.add(Dense(1, activation='sigmoid'))
   model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=opt, run_eagerly=True)
   return model


best_acc = 0
best_comb = grid_combs[0]

# pentru fiecare combinare cream cate un model ca sa vedem
# care are scorul cel mai bun

counter = 0

for combination in grid_combs:
   ### Modifica aici parametrii
   print("Combination ", counter)
   print(combination)
   counter += 1

   # callbacks to save the best epoch
   model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

   #TODO schimbi input_shape cu nr de coloane de features
   # combination = [l1_num, l2_num, d1, act1, act2,  opt, lr]
   l1 = Dense(combination[0], input_shape=(24, ), activation='relu')
   l2 = Dense(combination[1], activation='relu')
   l3 = Dense(combination[2], activation='relu')
   d1 = Dropout(combination[3])
   d2 = Dropout(combination[4])
   if combination[5] == 'adam':
      opt = keras.optimizers.Adam(learning_rate=combination[6])
   else:
      opt = keras.optimizers.SGD(learning_rate=combination[6])

   model = create_model(l1, l2, l3, d1, d2, opt)
   model.fit(X_train, Y_train, batch_size=512, epochs=50, validation_data=(X_val, Y_val), callbacks=[model_checkpoint_callback])
   model.load_weights(checkpoint_filepath)
   scores = model.evaluate(X_val, Y_val)

   acc = scores[1]
   print(f'{model.metrics_names[1]} of {acc*100}%;')
   
   if acc > best_acc:
      best_comb = combination
      best_acc = acc



with open('best.txt', 'w') as file:
   file.write(str(best_comb))
   file.write('\n')
   file.write(str(best_acc))
