"""
Linear regression with one feature

This script performes the "gradient descent" function to predict the profit

Author: Supriya Sudarshan
Version: 14.05.2020
"""
import numpy as np
import matplotlib.pyplot as plt
import gradientDescent as gd

############### Loading the data ###############################

data = np.loadtxt('/data/ex1data1.txt', delimiter = ',')
y = data[:,1] # target
X = data[:,0] # feature

############# Cost and gradient descent #######################

m = len(y) # size of training sets

X_norm = np.column_stack(((np.ones((m,1))), X)) # add intercept to features

# Gradient descent
n = X_norm.shape[1] # number of features
theta = np.zeros(n)
alpha, iters = 0.01, 1500
theta, J = gd.descent(X_norm, y, theta, alpha, iters)

print("Theta found by gradient descent: {}".format(theta))


############# Plot the linear fit ############################

# let's plot the dataset
plt.plot(X, y, 'rx', markersize = 5,label='Training sets')
plt.xlabel('Feature - Population in 10000s')
plt.ylabel('Target - Profit in 10000$ ')

plt.plot(X_norm[:,1], X_norm.dot(theta), ls = '-',label='Linear regression')
plt.title('Linear fit to dataset')
plt.legend(loc = 0)
plt.draw()

############### predictions ####################################

#Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print("For population = 35,000, the profit is: {}".format(predict1*10000))
predict2 = np.array([1, 7]).dot(theta)
print("For population = 70,000, the profit is: {}".format(predict2*10000))

plt.show()

