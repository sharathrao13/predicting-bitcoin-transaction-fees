import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


def lin_reg(training_size):
    print('Extracting Features from txt')
    ds_features = np.loadtxt("features-google-new.txt", usecols = (0,1,3,5,6,7,10,11,12,14,15,16))
    for k in range(0, 20):
        print (str(ds_features[k][1]))
    scaled_priority = preprocessing.normalize(ds_features[:,1])
    for k in range(0, 20):
        print (" ---   "+str(scaled_priority[0][k]))
    ds_features[:,1] = scaled_priority

if __name__ == "__main__":
    lin_reg(300000)