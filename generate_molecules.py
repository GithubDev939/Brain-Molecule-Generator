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

def vectorize(smiles, max_pad, npc, char_to_int):
    one_hot = np.zeros((smiles.shape[0], max_pad, npc), dtype = np.int8)
    for mol_index in range(smiles.shape[0]):
        one_hot[mol_index, 0, 0] = 1
        for char_index in range(len(smiles[mol_index])):
            one_hot[mol_index, char_index+1, char_to_int[smiles[mol_index][char_index]]] = 1
        for char_index in range(len(smiles[mol_index])+1, max_pad):
            one_hot[mol_index, char_index, 1] = 1
    return one_hot[:,0:-1,:], one_hot[:,1:,:]

def sample_w_temp(preds, sampling_temp):
  stretched = np.log(preds) / sampling_temp
  stretched_probs = np.exp(stretched) / np.sum(np.exp(stretched))
  return np.random.choice(range(len(stretched)), p = stretched_probs)

def sample_smiles(latent, n_vocab, sampling_temp):
  states = latent_to_states_model.predict(latent)
  gen_model.layers[1].reset_states(states=[states[0], states[1]])
  startidx = char_to_int["!"]
  samplevec = np.zeros((1, 1, n_vocab))
  samplevec[0, 0, startidx] = 1
  sequence = ""
  for i in range(101):
    preds = gen_model.predict(samplevec)[0][-1]
    if sampling_temp == 1.0:
      sampleidx = np.argmax(preds)
    else:
      sampleidx = sample_w_temp(preds, sampling_temp)
    samplechar = int_to_char[int(sampleidx)]
    if samplechar != "E":
      sequence+=samplechar
      samplevec = np.zeros((1, 1, n_vocab))
      samplevec[0, 0, sampleidx] = 1
    else:
      break
  return sequence

def generate(latent_seed, sampling_temp, scale, quant):
    samples, mols = [], []
    for i in range(quant):
        latent_vec = latent_seed + scale*(np.random.randn(latent_seed.shape[1]))
        out = sample_smiles(latent_vec, n_vocab, sampling_temp)
        samples.append(out)
    return samples

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
ftrain, test = train_test_split(smiles, test_size = 0.2, random_state = 432)
train, val = train_test_split(ftrain, test_size = 0.1, random_state = 432)

X_train, y_train = vectorize(train, max_pad, len(char_to_int), char_to_int)
X_test, y_test = vectorize(test, max_pad, len(char_to_int), char_to_int)
X_val, y_val = vectorize(val, max_pad, len(char_to_int), char_to_int)

model = load_model("lstm_model.h5")
model.summary()
encoder_model = Model(inputs = model.layers[0].input, outputs = model.layers[4].output)
encoder_model.summary()
latent_input = Input(shape=(128,))
state_h = model.layers[6](latent_input)
state_c = model.layers[7](latent_input)
latent_to_states_model = Model(latent_input, [state_h, state_c])
latent_to_states_model.summary()
decoder_inputs = Input(batch_shape = (1, 1, 41))
decoder_lstm = LSTM(256, return_sequences = True, stateful = True)(decoder_inputs)
decoder_outputs = Dense(41, activation = "softmax")(decoder_lstm)
gen_model = Model(decoder_inputs, decoder_outputs)
for i in range(1, 3):
  gen_model.layers[i].set_weights(model.layers[i+7].get_weights())
gen_model.save("gen_model.h5")
gen_model.summary() 

latent_space = encoder_model.predict(X_train)
latent_seed = latent_space[60:61]
sampling_temp = 0.75
scale = 0.5
quantity = 50
t_smiles = generate(latent_seed, sampling_temp, scale, quantity)
print(latent_space.shape)
for i in t_smiles:
    fout = open("output_smiles.txt", "a")
    fout.write(i+"\n")
fout.close()