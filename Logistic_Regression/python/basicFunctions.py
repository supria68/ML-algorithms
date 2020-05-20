"""
Binary Classification using Logistic Regression

Author: Supriya Sudarshan
Version: 20.05.2020

The script contains functions to calculate:
    1. hypothesis, h(x) = sigmoid(theta * x)
    2. Cost function, J(theta) =  costFunction(X, y, theta)
    3. Gradient, partialderivative of J
    4. Predictions for test inputs
"""
import numpy as np

def sigmoid(z):
    # This function returns the hypothesis
    return 1.0/(1 + np.exp(-z))

def costFunction(theta, X, y):
    # This function should return J
    theta = np.matrix(theta) # (1,3)
    X = np.matrix(X) #(100,3)
    y = np.matrix(y) #(100,1)
      
    term1 = np.multiply(-y , np.log(sigmoid(X * theta.T)))
    term2 = np.multiply((1-y) , np.log(1 - sigmoid(X * theta.T))) 
    J = (np.sum(term1 - term2)) / len(X)
    return J

def gradient(theta, X, y):
    # This function computes the single step!
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    length = int(theta.shape[1])
    grad = np.zeros(length)

    error = sigmoid(X * theta.T) - y # (100,1)
    
    for i in range(length):
        grad[i] = np.sum(np.multiply(error, X[:,i])) / len(X)

    return grad 

def prediction(theta, X):
    #Predict whether the label is 0 or 1 using learned logistic regression parameters
    m, n = X.shape
    p = np.zeros(shape=(m, 1))
    h = sigmoid(X.dot(theta.T))

    for i in range(0, h.shape[0]):
        if h[i] >= 0.5:
            p[i, 0] = 1
        else:
            p[i, 0] = 0

    return p

def costFunctionReg(theta, X, y, regParam):
    # This function computes cost by maintaining tradeoff between decision boundary fit for
    # dataset and reducing magnitudes of theta
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    term1 = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    term2 = np.multiply((1 - y), np.log(1- sigmoid(X * theta.T)))
    regTerm = (regParam / 2 *len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]],2))
    J = np.sum(term1 - term2) / (len(X)) + regTerm
    return J

def gradientReg(theta, X, y, regParam):
    # This function computes the single step with a regularization parameter
    # "lambda or regParam"
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    length = int(theta.shape[1])
    grad = np.zeros(length)

    error = sigmoid(X * theta.T) - y 
    
    for i in range(length):
        val = np.multiply(error, X[:,i])
        
        if i == 0:
            grad[i] = np.sum(val) / len(X) # find only for theta = 1,2,3..n
        else:
            grad[i] = (np.sum(val) / len(X)) + ((regParam / len(X)) * theta[:,i])

    return grad 


