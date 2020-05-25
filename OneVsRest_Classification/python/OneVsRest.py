"""
Multiclass Classification using Logistic Regularization

The application of recognizing handwritten digits(0-9) is an example of Multiclass classification problem. We could solve it by using logistic regression alone with some small changes.
  
Since we need to classify 10 digits with 5000 training sets, lets approach the problem with logistic regularization technique to avoid over-fitting. Our strategy involves training a single classifier per class, with the samples of that class as positive samples and all other samples as negatives. This algorithm is called One-Vs-Rest or One-against-All. 
  
dataset: ex3data1.mat is obtained from "Machine Learning by Stanford University, Coursera".

Version: 25.05.2020
Author: Supriya Sudarshan
"""
import numpy as np
import scipy.io as sio
import scipy.optimize as opt

####################### Load the dataset ############################
data_dict = sio.loadmat('Data/ex3data1.mat')

print('Printing the dataset...\n')
print(data_dict)
print('\nFeature dimensions {} and target dimension {}\n'.format(data_dict['X'].shape, data_dict['y'].shape))
####################### define basic functions #######################
def sigmoid(z):
    """
    This function returns hypothesis
    """
    return 1 / (1 + np.exp(-z))

def cost(theta, X, y, regParam):
    """
    This function returns the cost of using theta
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    term1 = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    term2 = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    reg = (regParam / 2 * len(X)) * np.sum(np.power(theta[:,1:theta.shape[1]], 2))
    return np.sum(term1 - term2) / (len(X)) + reg

def gradient(theta, X, y, regParam):
    """
    This function returns the gradient
    """
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    
    error = sigmoid(X * theta.T) - y
    
    grad = ((X.T * error) / len(X)).T + ((regParam / len(X)) * theta)
    
    grad[0, 0] = np.sum(np.multiply(error, X[:,0])) / len(X) # theta 0 isn't regularized
    
    return np.array(grad).ravel()

def oneVsAll(X, y, num_labels, regParam):
    """
    This function is the implementation of single class classifier.
    Looping over the 10 class will give us the theta estimates
    """
    m, n = X.shape
        
    all_theta = np.zeros((num_labels, n + 1)) # for all 10 classes
    
    X = np.insert(X, 0, values=np.ones(m), axis=1) # add bias to X (5000, 401)
    
    for i in range(1, num_labels + 1): # target is labelled from 1..10
        theta = np.zeros(n + 1) # theta for each class (401,)
        yi = np.array([1 if label == i else 0 for label in y]) 
        yi = yi.reshape(m,1)
        
        # minimize the objective function
        result = opt.fmin_tnc(func = cost, x0 = theta, fprime = gradient, args= (X, yi, regParam))
        
        #fmin = minimize(fun=cost, x0=theta, args=(X, yi, regParam), method='TNC', jac=gradient)
        all_theta[i-1,:] = result[0]
    
    return all_theta #(10, 401)

def predict(X, theta):
    """
    Let's check what is the result of each test input
    """
    X = np.insert(X, 0 ,values = np.ones(X.shape[0]),axis = 1)
    X = np.matrix(X)
    theta = np.matrix(theta)
    
    hyp = sigmoid(X * theta.T)
    # we need to return the location at which hyp is maximum!!

    return (np.argmax(hyp, axis = 1) + 1)


######################### Train the Logistic Classifier #########################################
print('Training the classifier...\n')

num_labels, regParam = 10, 1
"""
X: features will be of dimension (5000, 401) ~ extra bias column
y: target will be of dimension (5000, 1)

all_theta = theta values for entire 10 class (10, 401)
theta = theta for single class (401,)

"""
all_theta = oneVsAll(data_dict['X'], data_dict['y'], num_labels, regParam)
print('theta for which the cost is minimum: \n')
print(all_theta)
print('\n')
############################ Prediction ########################################################

y_pred = predict(data_dict['X'], all_theta)
print('Prediction results ....')
print(y_pred)
print('\n')
correct = []
for (a,b) in zip(y_pred, data_dict['y']):
    if a == b:
        correct.append(1)
    else:
        correct.append(0)

accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {}%'.format(accuracy * 100))






