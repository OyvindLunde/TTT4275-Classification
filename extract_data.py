import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, linear_model
from sklearn.linear_model import SGDClassifier


def extract_data():
    data = pd.read_csv("iris.data", sep=",")
    print(data.head())
    le = preprocessing.LabelEncoder()
    x1 = list(data["sepal_length"])
    x2 = list(data["sepal_width"])
    x3 = list(data["petal_length"])
    x4 = list(data["petal_width"])
    y = list(le.fit_transform(list(data["class"])))
    return list(zip(x1, x2, x3, x4)), y


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

    t = np.zeros((3, len(y_train)))

    for i in range(len(y_train)):
        t[y_train[i]][i] = 1

    return  x_train, x_test, t, y_test


def confusion_matrix(prediction, actual, numClasses):
    confusionMatrix = np.zeros((numClasses,numClasses))
    for i in range(len(prediction)):
        confusionMatrix[actual[i]][prediction[i]] += 1
    return confusionMatrix

def main():
    x_train, x_test, y_train, y_test = get_sets()

main()