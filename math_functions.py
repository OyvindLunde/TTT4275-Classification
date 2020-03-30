import numpy as np
import math


def calculate_MSE(W, t, x):
    g = W.dot(x)
    sum = 0
    for i in range(len(g)):
        diff = np.subtract(g[i], t[i])
        sum += np.dot(diff,diff)

    return sum/2

def calculate_g(W,x):
    z = W.dot(x)
    g = 1/(1+np.exp(-z))
    return g

def matrixMul(x,y):
    z = np.zeros((len(x),len(y)))
    for i in range(len(y)):
        for j in range(len(x)):
            z[j][i] = y[i]*x[j]
    return z

def calculate_MSE_gradient(W, t, x):
    g = calculate_g(W,x)
    mse_grad = g.T-t.T
    g_grad = g.T * (1-g.T) # Element wise
    W_grad = x

    return np.dot(W_grad, mse_grad*g_grad)


def calculate_W(iterations, alpha, t, x):
    W = np.zeros((len(t),len(x)))
    for i in range(iterations):
        W = W - alpha*calculate_MSE_gradient(W,t,x).T
    return W

def predict(x, W, t):
    predictions = np.zeros((len(x[0]), 1))
    g = calculate_g(W, x)
    for i in range(len(g[0])):
        predictions[i] = np.argmax(g[:,i], axis=0)
    return predictions