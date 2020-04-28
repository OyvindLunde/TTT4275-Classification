from iris_task.src import extract_data as ed, math_functions as mf, plotting as plt



def main():

    x_train, x_test, y_train, y_test, t_train, t_test = ed.get_sets()
    alpha = mf.find_optimal_alpha()

    W = mf.calculate_W(10000, alpha, t_train, x_train)
    pred_train = mf.predict(x_train, W, t_train)
    pred_test = mf.predict(x_test, W, t_test)
    conf_matrix_test = plt.confusion_matrix(pred_test, y_test, 3)
    conf_matrix_train = plt.confusion_matrix(pred_train, y_train, 3)
    total_errors_train = mf.find_total_errors(conf_matrix_train)
    total_errors_test = mf.find_total_errors(conf_matrix_test)

    print("The optimal alpha is: ", alpha)
    print("Error rate training: ", total_errors_train/90)
    print("Error rate test: ", total_errors_test/60)

    plt.plot_confusion_matrix(conf_matrix_train, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    plt.plot_confusion_matrix(conf_matrix_test, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

    plt.plot_histogram(4, 3, pred_test, x_test)

main()

