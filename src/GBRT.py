import numpy as np
import matplotlib.pyplot as plt

from sklearn import ensemble
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
import sys


def lin_reg(training_size):
    print('Extracting Features from txt')
    ds_features = np.loadtxt("features-google-new.txt", usecols = (0,1,3,4,5,6,7,8,11,12,13,14,15,16))
    ds_outputs = np.loadtxt("features-google-new.txt", usecols = (10,))

    ds_features_train, ds_features_test = ds_features[:training_size], ds_features[training_size:]
    ds_outputs_train, ds_outputs_test = ds_outputs[:training_size], ds_outputs[training_size:]

    params = {'n_estimators': 50, 'max_depth': 6, 'min_samples_split': 1,
                      'learning_rate': 0.01, 'loss': 'ls'}
    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(ds_features_train, ds_outputs_train)

    mse = mean_squared_error(ds_outputs_test, clf.predict(ds_features_test))
    print("MSE: %.4f" % mse)

    feature_names = ['Size', 'Priority', 'TotalAnFee', 'FeePerKB', 'Children', 'Parents', 'Mempool', 
                     'MempoolBytes', 'NumTxInLastBlock', 'SecondsSinceLastBlock', 'BlockDiff', 
                     'IncomingTxRate', 'InputValue', 'OutputValue']
    ###############################################################################
    # Plot training deviance

    # compute test set deviance
    test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

    for i, y_pred in enumerate(clf.staged_predict(ds_features_test)):
            test_score[i] = clf.loss_(ds_outputs_test, y_pred)

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title('Deviance')
            plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
                             label='Training Set Deviance')
            plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
                             label='Test Set Deviance')
            plt.legend(loc='upper right')
            plt.xlabel('Boosting Iterations')
            plt.ylabel('Deviance')

    ###############################################################################
    # Plot feature importance
    feature_importance = clf.feature_importances_
    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, map(lambda x: feature_names[x], sorted_idx))
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()


if __name__ == "__main__":
    lin_reg(300000)
