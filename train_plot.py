# -*- coding: utf-8 -*-
"""
Created on Tues Aug 16 15:56:02 2022

@author: ricky
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from brewster import reflectivity_curve
from keras.models import load_model

l = 100
arr_n_pts = [3, 10, 31, 100, 316, 1000]
# brewster indices of refraction
no = 1.6599
ne = 1.5452
a_c = 29.2*(180/np.pi)
lamb = 853e-9
_, _, curve = reflectivity_curve(no, ne, a_c, lamb)
delta_nos = []
delta_nes = []

for i in range(len(arr_n_pts)):
    model = load_model(os.curdir + '/weights_Brewster_'+str(arr_n_pts[i]**2))
    history = json.load(open(os.curdir+'/weights_Brewster_'+str(arr_n_pts[i]**2)+'/model_history.json', 'r'))

    train_losses = history['loss']
    val_losses = history['val_loss']

    preds = model.predict(curve.reshape(1, l))
    print(preds.shape)
    delta_nos.append(preds[0,0] - no)
    delta_nes.append(preds[0,1] - ne)

plt.plot(delta_nes, color='blue')
plt.xlabel(r'$\log_{10} [\text{data points}]$')
plt.ylabel(r'$\delta n_e$')
plt.show()
plt.plot(delta_nos, color='red')
plt.xlabel(r'$\log_{10} [\text{data points}]$')
plt.ylabel(r'$\delta n_o$')
plt.show()