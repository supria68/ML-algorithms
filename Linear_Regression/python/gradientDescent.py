"""
Version: 14.05.2020
Author: Supriya Sudarshan

Gradient descent for Linear Regression


#######################################################################
Function : gradientDescent 
Description: Updates theta by taking num_iters gradient steps with learning rate alpha

Input:
    X - features
    y - target
    theta - parameter/coefficients
    alpha - learning rate
    num_iters - number of iterations
    
Output:
    theta - Updated parameter
    J - Cost function
#######################################################################
"""

import numpy as np

def descent(X, y, theta, alpha, num_iters):
    m = float(len(y))
    for i in range(num_iters):
        """
        Compute the cost function:
        hypothesis = theta' * x (matrix form)
        cost, J = 1/2m * sum((hypothesis - y)**2)
        """
        hypothesis = X.dot(theta)
        error = hypothesis - y
        cost = np.sum(error**2) / (2.0 * m)
        #print("Iteration: {} and cost: {}".format(i,cost))
        """
        Update theta for minimum cost:
        theta = theta - alpha/m * sum((hypothesis - y)*x')
        """
        gradient = (np.transpose(X).dot(error)) / m
        theta = theta - alpha * gradient
    return theta,cost


