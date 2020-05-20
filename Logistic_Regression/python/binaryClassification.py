"""
Python implementation of Binary Classifier

Author: Supriya Sudarshan
Version: 20.05.2020

Problem description: Given the exam score of students, find if he/she is admitted or not to next course.

X : Features - student scores
y : Target {0,1} - admitted(1) or not admitted(0)

The script calls "basicFunctions.py" to calculate:
    1. hypothesis, h(x) = sigmoid(theta * x)
    2. Cost function, J(theta) =  costFunction(theta, X, y)
    3. Gradient, del(J) = gradient(theta, X, y)
    4. scikit learn tools to run the optimized algorithm for prediction
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import basicFunctions as bf

       
######################### Load Dataset ##########################
data = pd.read_csv('Data/ex2data1.txt',sep = ',',header=None, names = ['Exam1','Exam2','Admitted'])

######################### Plotting data #########################
pos = data[data['Admitted'].isin ([1])]
neg = data[data['Admitted'].isin ([0])]
fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(pos['Exam1'],pos['Exam2'], color = 'b', marker = 'o', label = 'Admitted')
ax.scatter(neg['Exam1'],neg['Exam2'], color = 'r', marker = '+', label = 'Not Admitted')
ax.legend(loc = 0)
ax.set_xlabel('Exam 1 Score')
ax.set_ylabel('Exam 2 Score')
ax.set_title('Dataset for Binary Classification')
#plt.show()

####################### Set the features, target ##########################
data.insert(0,'Ones',1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1] # (100,3)
y = data.iloc[:,cols-1:cols] # (100,1)

###################### Compute Cost and Gradient ########################
X = np.array(X.values)
y = np.array(y.values)
theta = np.zeros(3)

# Let's test the cost function with theta = 0,0,0
cost = bf.costFunction(theta, X, y)
print('Cost at initial theta (zeros): ', cost)
print('Expected cost (approx): 0.693\n')

################## Scipy Optimization for minimum cost ######################

opt_theta = opt.fmin_tnc(func = bf.costFunction, x0 = theta, fprime = bf.gradient, args = (X,y))
cost = bf.costFunction(opt_theta[0], X, y)

print('Cost at theta found by fmin_tnc: ', cost)
print('Expected cost :0.203\n')

################## Predictions and Evaluation ##############################
predict = bf.prediction(opt_theta[0],X)
print('Training Accuracy : {}'. format((y[np.where(predict==y)].size / float(y.size))*100))

plt.show()
