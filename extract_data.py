import pandas as pd
import numpy as np


def extract_data():
    data = pd.read_csv("iris.data", sep=",")
    x1 = list(data["sepal_length"])
    x2 = list(data["sepal_width"])
    x3 = list(data["petal_length"])
    x4 = list(data["petal_width"])
    y = list(data["class"])
    return x1, x2, x3, x4, y


def get_training_set(x1, x2, x3, x4, y):
    x1_train = x1[0:30] + x1[50:80] + x1[100:130]
    x2_train = x2[0:30] + x2[50:80] + x2[100:130]
    x3_train = x3[0:30] + x3[50:80] + x3[100:130]
    x4_train = x4[0:30] + x4[50:80] + x4[100:130]
    y_train = y[0:30] + y[50:80] + y[100:130]
    return x1_train, x2_train, x3_train, x4_train, y_train


def get_test_set(x1, x2, x3, x4, y):
    x1_test = x1[30:50] + x1[80:100] + x1[130:150]
    x2_test = x2[30:50] + x2[80:100] + x2[130:150]
    x3_test = x3[30:50] + x3[80:100] + x3[130:150]
    x4_test = x4[30:50] + x4[80:100] + x4[130:150]
    y_test = y[30:50] + y[80:100] + y[130:150]
    return x1_test, x2_test, x3_test, x4_test, y_test
