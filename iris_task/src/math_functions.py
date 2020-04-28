import numpy as np
import extract_data as ed, math_functions as mf, plotting as plt



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

def predict(x, W):
    predictions = np.zeros((len(x[0]), 1))
    g = calculate_g(W, x)
    for i in range(len(g[0])):
        predictions[i] = np.argmax(g[:,i], axis=0)
    return predictions

def find_total_errors(conf_matrix):
    sum = 0
    for i in range(len(conf_matrix)):
        for j in range(len(conf_matrix[0])):
            if (i != j):
                sum += conf_matrix[i][j]
    return sum


def find_optimal_alpha():
    error_rate = 1
    alpha = 0.1
    best_alpha = 0.1
    iterations = 10000
    x_train, x_test, y_train, y_test, t_train, t_test = ed.get_sets()

    while error_rate > 0.05:    #Return alpha when error rate is small enough, or alpha is too small
        alpha = alpha*0.8
        W = calculate_W(iterations, alpha, t_train, x_train)
        pred_train = predict(x_train, W)
        conf_matrix_train = plt.confusion_matrix(pred_train, y_train, 3)
        total_errors_train = mf.find_total_errors(conf_matrix_train)
        if total_errors_train/90 < error_rate:
            error_rate = total_errors_train / 90
            best_alpha = alpha
        if alpha < 0.0001:
            return best_alpha
    return best_alpha


def find_smallest_and_biggest_value(dataset):
    min = np.inf
    max = -np.inf
    for i in dataset:
        if i < min:
            min = i
        if i > max:
            max = i
    return min, max