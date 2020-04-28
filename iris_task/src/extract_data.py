import pandas as pd
import numpy as np
from sklearn import preprocessing


def extract_data():
    data = pd.read_csv("../iris_data/iris.data", sep=",")
    le = preprocessing.LabelEncoder()
    x1 = list(data["sepal_length"])
    x2 = list(data["sepal_width"])
    x3 = list(data["petal_length"])
    x4 = list(data["petal_width"])
    y = list(le.fit_transform(list(data["class"])))
    x5 = [1] * 150 #Start offset, simular to b in y = ax + b

    return list(zip(x1, x2, x3, x4, x5)), y



def get_sets():
    x, y = extract_data()

    #Get the 30 first for training and 20 last for test
    x_train = x[0:30] + x[50:80] + x[100:130]
    x_test = x[30:50] + x[80:100] + x[130:150]
    y_train = y[0:30] + y[50:80] + y[100:130]
    y_test = y[30:50] + y[80:100] + y[130:150]

    #Get 30 last for training and 20 first for test
    """
    x_train = x[20:50] + x[70:100] + x[120:150]
    x_test = x[0:20] + x[50:70] + x[100:120]
    y_train = y[20:50] + y[70:100] + y[120:150]
    y_test = y[0:20] + y[50:70] + y[100:120]
    """

    t_train = np.zeros((3, len(y_train)))
    t_test = np.zeros((3, len(y_test)))
    for i in range(len(y_train)):
        t_train[y_train[i]][i] = 1

    for i in range(len(y_test)):
        t_test[y_test[i]][i] = 1

    x_train, x_test, y_train, y_test = reshape_data(x_train, x_test, y_train, y_test)

    return  x_train, x_test, y_train, y_test, t_train, t_test


def reshape_data(x_train, x_test, y_train, y_test):
    x_train_new = np.zeros((len(x_train[0]), len(x_train)))
    x_test_new = np.zeros((len(x_test[0]), len(x_test)))
    y_train_new = np.zeros((len(y_train), 1))
    y_test_new = np.zeros((len(y_test), 1))

    for i in range(len(x_train)):
        y_train_new[i] = y_train[i]
        for j in range(len(x_train[0])):
            x_train_new[j][i] = x_train[i][j]

    for i in range(len(x_test)):
        y_test_new[i] = y_test[i]
        for j in range(len(x_test[0])):
            x_test_new[j][i] = x_test[i][j]

    return x_train_new, x_test_new, y_train_new, y_test_new




