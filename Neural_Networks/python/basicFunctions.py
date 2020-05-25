"""
Neural Network prediction using feedforward propagation algorithm

We are using 1 input layer, 1 hidden layer and 1 output layer for the
handwritten prediction problem

Version: 25.05.2020
Author: Supriya Sudarshan
"""
import numpy as np

def sigmoid(z):
    """
    This function returns the hypothesis value
    """
    return 1.0 / (1 + np.exp(-z))

def predict(theta1, theta2, X):
    """
    This function calculates the hidden layer activation values from weights
    and features.

    a1 = feature values (X) with bias of 1
    a2 = activation unit in hidden layer
    a3 = hypothesis from the output layer
    """

    [m, n] = X.shape # m = size of training set
                     # n = number of features
    
    a1 = np.array(np.column_stack(((np.ones((m,1))), X))) # add bias + 1
    z1 = np.dot(a1, np.transpose(theta1))
    a2 = sigmoid(z1) 

    a2 = np.array(np.column_stack(((np.ones((m,1))), a2))) # add bias + 1
    z2 = np.dot(a2, np.transpose(theta2))
    a3 = sigmoid(z2)

    return a3 
