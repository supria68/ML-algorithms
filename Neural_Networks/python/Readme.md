# Recognizing handwritten digits using Neural Network

### Packages:  
Numpy, Scipy.io
  
Use either pip install <> or conda install <> for getting the packages.
  
### Case 1: Predicting handwritten digits from already trained neural network(weights given) using feedforward algorithm.
### Datasets:  
All .mat files in the ./Data/ folder are taken from course "Machine learning by Stanford University, Coursera".  
ex3data1.mat -> 5000 sets of handwritten digits  
ex3weights.mat -> Weights for neural network model
  
### Script description:  
1. basicFunctions.py :  
This script contains list of functions to compute sigmoid, hiddenlayer activation units.
2. handwrittenDigitClassify.ipynb :  
Notebook uses functions from "basicFunctions.py" to train and evaluate predictions.  
  
### Result:
Training accuracy of Neural Network with given weights: 97.52%

### Case 2: Predicting handwritten digits by implementing a feed-forward, back propagation neural network model (from scratch)
### Dataset:
All .mat files are taken from course "Machine learning by Stanford University, Coursera".

### Description:
/backpropagation/NeuralNetworks_fromscratch.ipynb:  
This notebook contains the details of building a neural network from scratch

### Result:
Training accuracy of Neural Network : 99.28%
