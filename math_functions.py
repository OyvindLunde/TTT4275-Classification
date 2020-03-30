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
    z = np.dot(W,x)
    g = np.zeros((len(z),len(z[0])))
    for i in range(len(z)):
        for k in range(len(z[0])):
            g[i][k] = 1/(1+math.exp(-z[i][k]))
    return g

def matrixMul(x,y):
    z = np.zeros((len(x),len(y)))

    for i in range(len(y)):
        for j in range(len(x)):
            z[j][i] = y[i]*x[j]

    return z

def calculate_MSE_gradient(W, t, x):
    g = calculate_g(W,x)
    sum = np.zeros((3,4))
    for i in range(len(g)):
        diff = g[:,i]-t[:,i]
        scalar_prod = np.dot(g[:,i],(1-g[:,i]))
        sum += matrixMul(np.dot(diff,scalar_prod), x[:,i])
    return sum

    return np.dot(MSE_gradient, temp)

def calculate_W(iterations, alpha, t, x):
    W = np.zeros((len(t),len(x)))
    for i in range(iterations):
        W = W - alpha*calculate_MSE_gradient(W,t,x)

    return W

