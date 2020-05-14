"""
Version: 14.05.2020
Author: Supriya Sudarshan

This script computes the solution to linear regression using normal equation

"""
import numpy as np

def normaleq(X, y):
    theta = np.zeros(X.shape[1])
    theta = np.linalg.pinv(np.transpose(X).dot(X)).dot(np.transpose(X).dot(y))
    return theta

if __name__ == "__main__":
    # just the test data
    X = np.arange(9).reshape(3,3) # 3 X 3 matrix
    y = np.transpose(np.array([4,17,28])) # column vector
    
    print(normaleq(X,y))
