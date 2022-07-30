import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import Sequence
import tensorflow as tf

class Data_Generator(Sequence):
    def __init__(self, input_data, labels, batch_size):
        self.input_data, self.labels = input_data, labels
        self.batch_size = batch_size
    
    def __len__(self):
        return int(np.ceil(len(self.input_data) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        x = self.input_data[idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_x, batch_y = np.array(x), np.array(y)
        
        return [batch_x, batch_x], batch_y

# Creates X, y one-hot encoding for entire smiles set
def vectorize(smiles, max_pad, npc, char_to_int):
    one_hot = np.zeros((smiles.shape[0], max_pad, npc), dtype = np.int8)
    for mol_index in range(smiles.shape[0]):
        one_hot[mol_index, 0, 0] = 1
        for char_index in range(len(smiles[mol_index])):
            one_hot[mol_index, char_index+1, char_to_int[smiles[mol_index][char_index]]] = 1
        for char_index in range(len(smiles[mol_index])+1, max_pad):
            one_hot[mol_index, char_index, 1] = 1
    return one_hot[:,0:-1,:], one_hot[:,1:,:]

# RNN-LSTM model w/ layers easily accessible to create future encoder, latent-space, and decoder models
def lstm(X, y):
  enc_inp = Input(shape = (X.shape[1:]))
  _, state_h, state_c = LSTM(256, return_state = True)(enc_inp)
  states = Concatenate(axis = -1)([state_h, state_c])
  bottle_neck = Dense(128, activation = "relu")(states)

  state_h_decoded = Dense(256, activation = "relu")(bottle_neck)
  state_c_decoded = Dense(256, activation = "relu")(bottle_neck)
  enc_states = [state_h_decoded, state_c_decoded]
  dec_inp = Input(shape = (X.shape[1:]))
  dec1 = LSTM(256, return_sequences = True)(dec_inp, initial_state = enc_states)
  output = Dense(y.shape[2], activation = "softmax")(dec1)

  model = Model(inputs = [enc_inp, dec_inp], outputs = output)
  return model

# Data: Collection, Vectorization, Partition
smiles = [r.rstrip() for r in open("relevant_smiles.txt")]
charset = list(sorted(set("".join(smiles))))
smiles = np.array(smiles)
char_to_int = dict((c, i+2) for i, c in enumerate(charset))
int_to_char = dict((i+2, c) for i, c in enumerate(charset))
char_to_int["!"] = 0
char_to_int["E"] = 1
int_to_char[0] = "!"
int_to_char[1] = "E"
n_vocab = len(char_to_int)
max_pad = max([len(smile) for smile in smiles])+2
# Split data into X and y for train, test and val
ftrain, test = train_test_split(smiles, test_size = 0.2, random_state = 432)
train, val = train_test_split(ftrain, test_size = 0.1, random_state = 432)
X_train, y_train = vectorize(train, max_pad, len(char_to_int), char_to_int)
X_test, y_test = vectorize(test, max_pad, len(char_to_int), char_to_int)
X_val, y_val = vectorize(val, max_pad, len(char_to_int), char_to_int)

# Create, train, and save model
model = lstm(X_train, y_train)
model.summary()
batch = 356
steps_per_epoch = len(X_train) // batch
lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, decay_steps = steps_per_epoch*50, decay_rate = 1.0, staircase = False)
opt = Adam()
model.compile(optimizer = opt, loss = "categorical_crossentropy", metrics = ["acc"])
training_gen = Data_Generator(X_train, y_train, batch)
val_gen = Data_Generator(X_val, y_val, batch)
nb_epochs = 200
validation_steps = len(X_val) // batch
history = model.fit(training_gen, steps_per_epoch=steps_per_epoch, epochs=nb_epochs, verbose=1, 
                              validation_data=val_gen, validation_steps=validation_steps, 
                             use_multiprocessing=False, shuffle=True, callbacks=[])
score, acc = model.evaluate([X_test, X_test], y_test, batch_size = batch, verbose = 0)
print(score, acc)
filename = "lstm_model.h5"
model.save(filename)