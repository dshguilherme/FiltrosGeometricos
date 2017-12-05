# -*- coding: utf-8 -*-
"""
Spyder Editor
Inverse characterization of geometric filters using a Deep Neural Network
author: Guilherme H. da Silva
coauthor: Paulo Vitor Ribeiro Martins
"""

import numpy as np
from scipy.io import loadmat

import theano
from neupy import layers, algorithms, environment
from sklearn import model_selection, metrics, datasets

environment.reproducible()
environment.speedup()

raw_data = loadmat('data.mat')
data = raw_data['results']
target = data[[0,1,2,3],:] # Get the R, L, Theta_x, Theta_y of each individual
inputs = data[4:,:] # Get the Power for each frequency of each individual

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
network = algorithms.RPROP(
        [
                layers.Input(500),
                layers.Relu(252),
                layers.Relu(128), 
                layers.Softplus(4),
        ],
        increase_factor = 1.1,
        decrease_factor = 0.9,
        # Using Mean Squared Error as the Loss Function
        error='binary_hinge',
        # A batch size of 32 will give us possibly 175 epochs for training
#       batch_size=25,
        # Learning Rate
        step=1.0,
        # Display network data in console
        verbose=True,
        # shuffle training data random before each epoch
        shuffle_data=True,

)
# Show network architecture in console
network.architecture()

