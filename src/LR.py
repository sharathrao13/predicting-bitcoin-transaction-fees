import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import sys


def lin_reg(training_size):
    print('Extracting Features from txt')
    ds_features = np.loadtxt("features-google.txt", usecols = (0,1,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
    ds_outputs = np.loadtxt("features-google.txt", usecols = (4,))
    ds_features_train, ds_features_test = ds_features[:training_size], ds_features[training_size:]
    ds_outputs_train, ds_outputs_test = ds_outputs[:training_size], ds_outputs[training_size:]
    regr = linear_model.LinearRegression()
    regr.fit(ds_features_train, ds_outputs_train)
    print('Training on set with size of: '+str(training_size))
    print('Coefficients:\n' + str(regr.coef_))
    print("Residual sum of squares: %.2f" % np.mean((regr.predict(ds_features_test) - ds_outputs_test) ** 2))
    
    

if __name__ == "__main__":
    lin_reg(200000)
