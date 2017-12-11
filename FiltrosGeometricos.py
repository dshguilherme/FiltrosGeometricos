# -*- coding: utf-8 -*-
"""
Spyder Editor
Inverse characterization of geometric filters using a Deep Neural Network
author: Guilherme H. da Silva
coauthor: Paulo Vitor Ribeiro Martins
"""

import numpy as np
from scipy.io import loadmat
import pandas as pd

import theano
from neupy import layers, algorithms, environment, plots
from sklearn import model_selection
environment.reproducible()
environment.speedup()

data = loadmat('resultsOver9000.mat')
data = data['results'].astype(np.float32)
target = data[[0,1,2,3],:] # Get the L, R, Theta_x, Theta_y of each individual
inputs = data[5:505,:] # Get the Power for each frequency of each individual

# Data manipulation
scalers = np.array([np.max(target[0,:]), np.max(target[1,:]), 180, 180])
target[0,:] = target[0,:]/scalers[0] # Scaling between 0 and 1
target[1,:] = target[1,:]/scalers[1] # Scaling between 0 and 1
target[[2,3],:] = target[[2,3],:]/180 # Regularizing angles to be between 0 and 1

# Shaping Data for train_test_split
inputs = inputs.reshape(inputs.shape[1], 1, inputs.shape[0])
target = target.reshape(target.shape[1],1,target.shape[0])

# Separating data
x_train, x_test, y_train, y_test = model_selection.train_test_split(
        inputs.astype(np.float32),
        target.astype(np.float32),
        test_size=(1 / 7.)
)
mean = x_train.mean(axis=(0, 2))
std = x_train.std(axis=(0, 2))

x_train -= mean
x_train /= std
x_test -= mean
x_test /= std
x_train = x_train.reshape(x_train.shape[0], x_train.shape[2])
y_train = y_train.reshape(y_train.shape[0], y_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[2])
y_test = y_test.reshape(y_test.shape[0], y_test.shape[2])

# Creating the network
network = algorithms.MinibatchGradientDescent(
        [
                layers.Input(500),
                layers.Relu(252),
                layers.Relu(128), 
                layers.Softplus(4),
        ],
        batch_size=128,
        step=0.1,
        # Using Mean Squared Error as the Loss Function
        error='mse',
        # Learning Rate
        #step=1.0,
        # Display network data in console
        verbose=True,
        # shuffle training data random before each epoch
        shuffle_data=True,
        show_epoch=1
)
# Show network architecture in console
network.architecture()
network.train(x_train, y_train, x_test, y_test, epochs=70)
plots.error_plot(network)

# Making test filters
MIN_POWER = np.min(x_train)
MAX_POWER = np.max(x_train)

FrequencyTestArray = np.empty([500,500], dtype='float32')
FrequencyTestArray[:,:] = MAX_POWER
np.fill_diagonal(FrequencyTestArray[0:500,:], MIN_POWER)

DoubleTestArray = np.empty([250,500], dtype='float32')
DoubleTestArray[:,:] = MAX_POWER
for i in range (0,250):
    DoubleTestArray[i,i] = MIN_POWER
    DoubleTestArray[i,-(i+1)] = MIN_POWER

PredictArray = network.predict(FrequencyTestArray)
PredictArray2 = network.predict(DoubleTestArray,)

# Deregularization
PredictArray *= std
PredictArray += mean
PredictArray2 *= std
PredictArray2 += mean
for i in range (0, len(scalers)):
    PredictArray[:,i] *= scalers[i]
    PredictArray2[:,i] *= scalers[i]

d = {'R' : PredictArray[:,0],
     'L' : PredictArray[:,1],
     'Tx': PredictArray[:,2],
     'Ty': PredictArray[:,3]}

Predictions = pd.DataFrame(d)

d = {'R' : PredictArray2[:,0],
     'L' : PredictArray2[:,1],
     'Tx': PredictArray2[:,2],
     'Ty': PredictArray2[:,3]}

Predictions2 = pd.DataFrame(d)
Predictions = Predictions.append(Predictions2)
Predictions.to_csv('Predictions.csv')