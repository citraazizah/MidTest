#%%allcode
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler




def count_error(X_train, X_test, y_train, y_test):
    knn.fit(X_train, y_train)
    prediksi = knn.predict(X_test)
    if prediksi != y_test:
        return True

    return False

def MinmaxNormalization(data):
    minmax_scaler = MinMaxScaler()

    X = np.array(data)[:, :5]
    y = np.array(data)[:, 5]

    print(X)
    print(y)
    error = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        X_train = list(minmax_scaler.fit_transform(X_train))
        X_test = minmax_scaler.transform(X_test)
        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error Min-Max : ', (error / len(data)) * 100, '%')

def loo_errorratio(data):
    loo = LeaveOneOut()

    X = data[:, :4]
    y = data[:, 4]
    error = 0
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
       
        if (count_error(X_train, X_test, y_train, y_test)):
            error += 1

    print('Error with LOO : ', (error / len(data)) * 100, '%')



iris_data = np.array(pd.read_csv('dataset/iris.csv'))
minmax = MinmaxNormalization(iris_data)
knn = KNeighborsClassifier(n_neighbors=3)
loo_errorratio(iris_data)