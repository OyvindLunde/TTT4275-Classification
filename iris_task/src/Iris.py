from iris_task.src import extract_data as ed, math_functions as mf


def find_total_errors(conf):
    sum = 0
    for i in range(len(conf)):
        for j in range(len(conf[0])):
            if (i != j):
                sum += conf[i][j]
    return sum

def main():
    """total_errors_test = 0
    total_errors_train = 0
    for i in range(10):
        x_train, x_test, y_train, y_test, t_train, t_test = ed.get_sets()
        W = mf.calculate_W(10000, 0.001, t_train, x_train)
        pred_train = mf.predict(x_train, W, t_train)
        pred_test = mf.predict(x_test, W, t_test)
        conf_matrix_train = (ed.confusion_matrix(pred_train, y_train, 3))
        conf_matrix_test = ed.confusion_matrix(pred_test, y_test, 3)
        total_errors_train += find_total_errors(conf_matrix_train)
        total_errors_test += find_total_errors(conf_matrix_test)
    print("Error rate training: ", total_errors_train/900)
    print("Error rate test: ", total_errors_test / 600)"""

    x_train, x_test, y_train, y_test, t_train, t_test = ed.get_sets()
    W = mf.calculate_W(10000, 0.001, t_train, x_train)

    """for i in range(3):
        x = plot.get_feature_vector(1, i, pred_test, x_train)
        plot.histogram(x, 1)"""

    error_rate = 1
    total_errors_test = 0
    alpha = 0.1
    iter = 10000
    while error_rate > 0.02:
        alpha = alpha*0.8
        W = mf.calculate_W(iter, alpha , t_train, x_train)
        pred_test = mf.predict(x_test, W, t_test)
        conf_matrix_test = ed.confusion_matrix(pred_test, y_test, 3)
        total_errors_test = find_total_errors(conf_matrix_test)
        if total_errors_test/60 > error_rate:
            iter *= 2
        error_rate = total_errors_test/60
        print(error_rate)
    print(alpha)
main()