"""
Linear regression with one variable.

This script helps in visualizing cost function for different values of theta
"""
import numpy as np
import matplotlib.pyplot as plt
import random

def computeCost(X, y, theta):
    """
    This redundant function is just for plotting values of J vs iterations
    """
    m = len(y)
    hyp = X.dot(theta)
    error = hyp - y
    J = 1.0/ (2.0 * m) * sum(error**2)
    return J

def visualize(X, y, expected_theta):
    theta0 = np.linspace(-10, 10, 100)
    theta1 = np.linspace(-1, 4, 100)

    J_val = np.zeros((len(theta0),len(theta1)))
    
    for i in range(len(theta0)):
        for j in range(len(theta1)):
            theta = np.array([theta0[i], theta1[j]])
            J_val[i,j] = computeCost(X, y, theta)
    J_val = np.transpose(J_val)

    ## lets plot
    fig,ax=plt.subplots(1,1)
    cset = ax.contour(theta0, theta1, J_val, np.logspace(-2, 3, 20))
    fig.colorbar(cset)
    ax.set_xlabel('theta_0')
    ax.set_ylabel('theta_1')
    ax.set_title('Linear regression with one feature')
    ax.plot(expected_theta[0], expected_theta[1], 'rx', markersize=10, linewidth=2)
    plt.show()

def run():
    # load dataset
    data = np.loadtxt('/data/ex1data1.txt', delimiter=',')
    X, y = data[:,0], data[:,1]
    m = len(y) # size of training set
    expected_theta = [-3.6303,1.1664] #value obtained from test.py

    X_padded = np.column_stack((np.ones((m,1)),X))
    J = visualize(X_padded, y, expected_theta)
   
if __name__ == "__main__":
    run()

