# -*- coding: utf-8 -*-
"""
Created on Tues Aug 16 14:14:53 2022

@author: ricky
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from keras import losses
from keras.optimizers import Adam
from brewster import reflectivity_curve
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout

# create NN, train with brewster's data
# return array of ne, no
l = 100
pow = 3
n_pts = int(10**pow)
# change for each crystal type
a_c = 29.2*(np.pi/180)
lamb = 853e-9
inputs = np.zeros((n_pts, n_pts, l))
outputs = np.zeros((n_pts, n_pts, 2))
load = True
train = True

if load: # load previously stored data
    inputs = np.load("./Training data Brewster/rp_curve_"+str(n_pts**2)+"_input.npy").reshape(n_pts, n_pts, l)
    outputs = np.load("./Training data Brewster/rp_curve_"+str(n_pts**2)+"_output.npy").reshape(n_pts, n_pts, 2)
else: # save new data
    nos = np.random.uniform(1.6, 1.7, n_pts)
    nes = np.random.uniform(1.5, 1.6, n_pts)
    for i in range(n_pts):
        for j in range(n_pts):
            outputs[i,j,:] = np.array([nos[i], nes[j]])
            inputs[i,j,:] = reflectivity_curve(nos[i], nes[j], a_c, lamb)[2]
    np.save(os.curdir+"/Training data Brewster/rp_curve_"+str(n_pts**2)+"_input", inputs.flatten())
    np.save(os.curdir+"/Training data Brewster/rp_curve_"+str(n_pts**2)+"_output", outputs.flatten())

# split train/val/test
train_ratio = 0.75
val_ratio = 0.15
test_ratio = 0.10

inputs = inputs.reshape(inputs.shape[0]*inputs.shape[1], inputs.shape[2])
outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1], outputs.shape[2])

x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=1 - train_ratio)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/
                                                                         (test_ratio + val_ratio))

# feed-forward RNN with batch norm and dropout
input_rnn = Input(shape=(l,))
rnn = BatchNormalization(input_shape=(l,))(input_rnn)
rnn = Dense(64, activation="elu")(rnn)
rnn = BatchNormalization()(rnn)
rnn = Activation('elu')(rnn)
rnn = Dropout(0.3)(rnn)
rnn = Dense(64, activation="elu")(rnn)
rnn = BatchNormalization()(rnn)
rnn = Activation('elu')(rnn)
rnn = Dropout(0.3)(rnn)
rnn = Dense(64, activation="elu")(rnn)
rnn = BatchNormalization()(rnn)
rnn = Activation('elu')(rnn)
rnn = Dropout(0.3)(rnn)
rnn = Dense(32, activation="elu")(rnn)
rnn = BatchNormalization()(rnn)
rnn = Activation('elu')(rnn)
rnn = Dropout(0.3)(rnn)
rnn = BatchNormalization()(rnn)
rnn = Activation('elu')(rnn)
output_rnn = Dense(2)(rnn)

model = Model(inputs=input_rnn, outputs=output_rnn)

adam = Adam(lr=0.1,decay=0.01)
model.compile(loss=losses.logcosh, optimizer=adam, metrics=['mean_squared_error'])
model.summary()

if train:
    model_checkpoint_callback = ModelCheckpoint(filepath=os.curdir+'/weights_Brewster_'+str(n_pts**2),
                                                monitor='val_mean_squared_error', mode='min',
                                                save_best_only=True)
    history = model.fit(x_train, y_train, epochs=30, batch_size=16, verbose=2, validation_data=(x_val, y_val), 
                        callbacks=[model_checkpoint_callback])
    history = history.history
    json.dump(history, open(os.curdir+'/weights_Brewster_'+str(n_pts**2)+'/model_history.json', 'w'))
    np.save(os.curdir+'/weights_Brewster_'+str(n_pts**2)+'/test_preds.npy', model.predict(x_test))
    np.save(os.curdir+'/weights_Brewster_'+str(n_pts**2)+'/test_outputs.npy', y_test)
else:
    model = load_model(os.curdir+'/weights_Brewster_'+str(n_pts**2))
    history = json.load(open(os.curdir+'/weights_Brewster_'+str(n_pts**2)+'/model_history.json', 'r'))
    
train_losses = history['loss']
val_losses = history['val_loss']

plt.plot(train_losses, label='train loss')
plt.plot(val_losses, '--', label='validation loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('losses')
plt.show()