import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import mean_squared_error


def lin_reg(training_size):
    
    print('Extracting Features from txt')
    ds_features = np.loadtxt("features-google-new.txt", usecols = (0,1,3,5,6,7,10,11,12,14,15,16))
    ds_outputs = np.loadtxt("features-google-new.txt", usecols = (4,))
    
    print(str(np.amax(ds_features[:,1])))
    print(str(np.amax(ds_features[:,3])))
    print(str(np.amax(ds_features[:,10])))
    print(str(np.amax(ds_features[:,11])))
    
    max_one = np.amax(ds_features[:,1])
    max_two = np.amax(ds_features[:,3])
    max_three = np.amax(ds_features[:,10])
    max_four = np.amax(ds_features[:,11])
    
    for i in range(0, len(ds_features[:,1])):
        if ds_features[i,1] == max_one:
            print (str(i)+" "+str(ds_features[i,1]))
            
    plt.plot(ds_features[:,1])
    plt.ylabel('Priority')
    plt.show()         
    
if __name__ == "__main__":
    lin_reg(300000)
