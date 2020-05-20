"""
Python implementation of Binary Classifier using regularization parameter

Author: Supriya Sudarshan
Version: 20.05.2020

Problem description: Suppose you are the product manager of the factory and you have the test results for some microchips on two different tests. From these two tests, you would like to determine whether the microchips should be accepted or rejected. To help you make the decision, you have a data set of test results on past microchips, from which you can build a logistic regression model.

X : Features - test results
y : Target {0,1} - accept(1) or reject(0)

The script calls "basicFunctions.py" to calculate:
    1. hypothesis, h(x) = sigmoid(theta * x)
    2. Cost function, J(theta) =  costFunctionReg(theta, X, y, lambda)
    3. gradient, del(J) = gradientReg(theta, X, y, lambda)
    3. scikit learn tools to run the optimized algorithm for prediction
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import basicFunctions as bf

####################### Load the dataset ##################################
data = pd.read_csv('Data/ex2data2.txt', sep = ',', header = None, names = ['Test1','Test2','Accept'])

####################### Plot the dataset ################################
pos = data[data['Accept'].isin ([1])]
neg = data[data['Accept'].isin ([0])]

fig, ax = plt.subplots(figsize = (10,5))
ax.scatter(pos['Test1'],pos['Test2'], color = 'b', marker = 'o', label = 'Accepted')
ax.scatter(neg['Test1'],neg['Test2'], color = 'r', marker = '+', label = 'Rejected')
ax.legend(loc = 0)
ax.set_xlabel('Test 1 Score')
ax.set_ylabel('Test 2 Score')  
ax.set_title('Logistic regularization dataset')

######################## Polynomial term additions ########################
"""
From the plot, its difficult to get a linear decision boundary!. 
we need to add more features (polynomial terms) to dataset to obtain the
descent.

To avoid overfitting, lets use regularization technique.
"""
degree = 5
x1 = data['Test1']
x2 = data['Test2']

data.insert(3,'Ones',1)

for i in range(1,degree):
    for j in range(i):
        data['col' + str(i) + str(j)] = np.power(x1,i-j) * np.power(x2,j)

# Now we have 14 columns of new features.
data.drop(['Test1','Test2'], axis = 1, inplace = True)
#print(data[:5])

######################## set features and target ########################

X = data.iloc[:, 1:data.shape[1]]
y = data.iloc[:, 0:1]

X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(X.shape[1])

######################## Cost function Evaluation #########################

regParam = 1 # lambda is a builtin keyword, using regParam (short for regularization parameter)
cost = bf.costFunctionReg(theta, X, y, regParam)
print('Cost at initial theta (zeros) : {}'.format(cost))
print('Expected Cost (approx) is 0.693 \n')

grad = bf.gradientReg(theta, X, y, regParam)
print('Gradient at initial theta (zeros) - first five values : {}'.format(grad[:5]))

################# Regularization and Accurarcy #####################

result = opt.fmin_tnc(func = bf.costFunctionReg, x0 = theta, fprime = bf.gradientReg, args = (X, y, regParam))
print('Minimum theta values: {}'.format(result[0]))

# Lets predict for this optimized theta
theta_min = result[0]
predictions = bf.prediction(theta_min, X)

from sklearn import metrics
print('Predicting accuracy is {} %'.format(metrics.accuracy_score(predictions,y)*100))

plt.show()
