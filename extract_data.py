import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.linear_model import SGDClassifier


def extract_data():
    data = pd.read_csv("iris.data", sep=",")
    le = preprocessing.LabelEncoder()
    x1 = list(data["sepal_length"])
    x2 = list(data["sepal_width"])
    x3 = list(data["petal_length"])
    x4 = list(data["petal_width"])
    y = list(le.fit_transform(list(data["class"])))

    return list(zip(x1,x2,x3,x4)), y


def get_sets():
    x, y = extract_data()
    clc1_x_train, clc1_x_test, clc1_y_train, clc1_y_test = sklearn.model_selection.train_test_split(x[0:50], y[0:50],
                                                                                                    test_size=0.4)
    clc2_x_train, clc2_x_test, clc2_y_train, clc2_y_test = sklearn.model_selection.train_test_split(x[50:100], y[50:100],
                                                                                                    test_size=0.4)
    clc3_x_train, clc3_x_test, clc3_y_train, clc3_y_test = sklearn.model_selection.train_test_split(x[100:150], y[100:150],
                                                                                                    test_size=0.4)
    x_train = clc1_x_train + clc2_x_train + clc3_x_train
    x_test = clc1_x_test + clc2_x_test + clc3_x_test
    y_train = clc1_y_train + clc2_y_train + clc3_y_train
    y_test = clc1_y_test + clc2_y_test + clc3_y_test

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


def confusion_matrix(prediction, actual, numClasses):
    confusionMatrix = np.zeros((numClasses,numClasses))
    for i in range(len(prediction)):
        confusionMatrix[int(actual[i])][int(prediction[i])] += 1
    return confusionMatrix


def find_smallest_and_biggest_value(x):
    min = np.inf
    max = -np.inf
    for i in x:
        if i < min:
            min = i
        if i > max:
            max = i
    return min, max

def main():
    x, y = extract_data()
    x1 = np.zeros((len(x), 1))
    x2 = np.zeros((len(x), 1))
    x3 = np.zeros((len(x), 1))
    x4 = np.zeros((len(x), 1))
    print(x[0][0])
    for i in range(len(x)):
        x1[i] = x[i][0]
        x2[i] = x[i][1]
        x3[i] = x[i][2]
        x4[i] = x[i][3]

    min_x1, max_x1 = find_smallest_and_biggest_value(x1)
    min_x2, max_x2 = find_smallest_and_biggest_value(x2)
    min_x3, max_x3 = find_smallest_and_biggest_value(x3)
    min_x4, max_x4 = find_smallest_and_biggest_value(x4)
    print("Min: ", min_x1, " Max: ", max_x1)
    print("Min: ", min_x2, " Max: ", max_x2)
    print("Min: ", min_x3, " Max: ", max_x3)
    print("Min: ", min_x4, " Max: ", max_x4)

main()