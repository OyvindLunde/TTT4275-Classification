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
    z = np.dot(W,x)
    g = np.zeros((len(z),len(z[0])))
    for i in range(len(z)):
        for k in range(len(z[0])):
            g[i][k] = 1/(1+math.exp(-z[i][k]))

    return g

#print(calculate_g(test,x))

def calculate_MSE_gradient(W, t, x):
    g = calculate_g(W,x)
    ones = [1] * len(x[0])
    sum = 0
    for i in range(len(g)):
        diff = np.subtract(g[i],t[i])
        scalar_prod = g[i].dot(np.subtract(ones,g[i]))
        sum += diff.dot(scalar_prod).dot(np.transpose(x[i]))
    return sum

#print(calculate_MSE_gradient(test, test2, x))

def calculate_W(iterations, alpha, t, x):
    #W = np.zeros((3,5))
    W = np.zeros((len(t),len(x)))
    for i in range(iterations):
        W = W - alpha*calculate_MSE_gradient(W,t,x)

    return W

