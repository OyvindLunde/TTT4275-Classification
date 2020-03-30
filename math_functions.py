import numpy as np
import math

test = np.array([[1,2,3,4,5], [4,5,6,1,1], [7,8,9,4,3]]) # testing values, completely trash
test2 = np.array([[4,5,6], [7,8,9], [1,4,5], [9,8,10]])
test2 = np.transpose(test2)
x = np.array([[1,1,1,1,1],[5,5,5,5,5],[2,2,2,2,2],[9,9,9,9,9]])
x = np.transpose(x)

def calculate_MSE(W, t, x):
    g = W.dot(x)
    sum = 0
    for i in range(len(g)):
        diff = np.subtract(g[i], t[i])
        sum += np.dot(diff,diff)

    return sum/2

#print(calculate_MSE(test, test2, x))

def calculate_g(W,x):
    print(len(W))
    print(len(W[0]))
    z = np.dot(W,x)
    #print(len(z[0]))
    #print(z)
    g = np.zeros((len(z),len(z[0])))
    #print(g)
    for i in range(len(z)):
        for k in range(len(z[0])):
            g[i][k] = 1/(1+math.exp(-z[i][k]))
    #print(g)
    return g

#print(calculate_g(test,x))

def calculate_MSE_gradient(W, t, x):
    g = calculate_g(W,x)
    MSE_gradient = g-t
    g_gradient = g*(1-g)
    z_gradient = x.T

    return np.dot(MSE_gradient,g_gradient*z_gradient)

#print(calculate_MSE_gradient(test, test2, x))

def calculate_W(iterations, alpha, t, x):
    print(len(x))
    print(len(x[0]))
    W = np.zeros((len(t),len(x)))
    #W = np.ones((len(t),len(x)))
    #print(x)
    for i in range(iterations):
        W = W - alpha*calculate_MSE_gradient(W,t,x)

    return W

