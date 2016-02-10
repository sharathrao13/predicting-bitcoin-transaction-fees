import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import sys


def lin_reg(training_size):
    print('Extracting Features from txt')
    ds_features = np.loadtxt("features-google-new.txt", usecols = (0,1,3,5,6,7,8,10,11,12,13,14,15,16))
    ds_outputs = np.loadtxt("features-google-new.txt", usecols = (4,))
    
    ds_0 = np.loadtxt("features-google-new.txt", usecols = (0,))
    ds_1 = np.loadtxt("features-google-new.txt", usecols = (1,))
    ds_3 = np.loadtxt("features-google-new.txt", usecols = (3,))
    ds_5 = np.loadtxt("features-google-new.txt", usecols = (5,))
    ds_6 = np.loadtxt("features-google-new.txt", usecols = (6,))
    ds_7 = np.loadtxt("features-google-new.txt", usecols = (7,))
    ds_8 = np.loadtxt("features-google-new.txt", usecols = (8,))
    ds_10 = np.loadtxt("features-google-new.txt", usecols = (10,))
    ds_11 = np.loadtxt("features-google-new.txt", usecols = (11,))
    ds_12 = np.loadtxt("features-google-new.txt", usecols = (12,))
    ds_13 = np.loadtxt("features-google-new.txt", usecols = (13,))
    ds_14 = np.loadtxt("features-google-new.txt", usecols = (14,))
    ds_15 = np.loadtxt("features-google-new.txt", usecols = (15,))
    ds_16 = np.loadtxt("features-google-new.txt", usecols = (16,))
    

    print(str(np.amax(ds_0)))
    print(str(np.amax(ds_1)))
    print(str(np.amax(ds_3)))
    print(str(np.amax(ds_5)))
    print(str(np.amax(ds_6)))
    print(str(np.amax(ds_7)))
    print(str(np.amax(ds_8)))
    print(str(np.amax(ds_10)))
    print(str(np.amax(ds_11)))
    print(str(np.amax(ds_12)))
    print(str(np.amax(ds_13)))
    print(str(np.amax(ds_14)))
    print(str(np.amax(ds_15)))    
    print(str(np.amax(ds_16)))
    

    

if __name__ == "__main__":
    lin_reg(300000)
