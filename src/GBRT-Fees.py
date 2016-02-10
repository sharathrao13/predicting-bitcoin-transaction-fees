import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn.metrics import mean_squared_error


def lin_reg(training_size):
    
    print('Extracting Features from txt')
    ds_features = np.loadtxt("features-google-new.txt", usecols = (0,1,3,5,6,7,10,11,12,14,15,16))
    ds_outputs = np.loadtxt("features-google-new.txt", usecols = (4,))
    
    ds_features_train, ds_features_test = ds_features[:training_size], ds_features[training_size:]
    ds_outputs_train, ds_outputs_test = ds_outputs[:training_size], ds_outputs[training_size:]

    params = {'n_estimators': 1000, 'max_depth': 7, 'min_samples_split': 1,
                      'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(ds_features_train, ds_outputs_train)

    mse = mean_squared_error(ds_outputs_test, clf.predict(ds_features_test))
    print("MSE: %.4f" % mse)
    
    
    
    predictions = clf.predict(ds_features_test)
    errors = []
    for i in range(0, len(predictions)):
        if ds_outputs_test[i] > 0:
            errors.append(((predictions[i] - ds_outputs_test[i]) / ds_outputs_test[i]) * 100)
    print(str(np.mean(errors)))
    
    for i in range(0, 100):
        print(str(ds_outputs_test[i])+"---"+str(predictions[i]))
    
    feature_names = ['Size', 'Priority', 'TotalAnFee', 'Children', 'Parents', 'Mempool', 
                     '#Block','NumTxInLastBlock', 'SecondsSinceLastBlock',
                     'IncomingTxRate', 'InputValue', 'OutputValue']

    #Plot feature importance
    feature_importance = clf.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, list(map(lambda x: feature_names[x], sorted_idx)))
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == "__main__":
    lin_reg(300000)
