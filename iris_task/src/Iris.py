from iris_task.src import extract_data as ed, math_functions as mf, plotting as plt



def main():
    #In ed.get_sets(), you can easily change the training set from the 30 first to the 30 last
    x_train, x_test, y_train, y_test, t_train, t_test = ed.get_sets()
    alpha = mf.find_optimal_alpha()
    iterations = 10000
    num_features = 4 #If no features has been removed
    num_classes = 3

    W = mf.calculate_W(iterations, alpha, t_train, x_train)
    pred_train = mf.predict(x_train, W)
    pred_test = mf.predict(x_test, W)
    conf_matrix_test = plt.confusion_matrix(pred_test, y_test, 3)
    conf_matrix_train = plt.confusion_matrix(pred_train, y_train, 3)
    total_errors_train = mf.find_total_errors(conf_matrix_train)
    total_errors_test = mf.find_total_errors(conf_matrix_test)

    print("The optimal alpha is: ", alpha)
    print("Error rate training: ", total_errors_train/90)
    print("Error rate test: ", total_errors_test/60)

    plt.plot_confusion_matrix(conf_matrix_train, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
    plt.plot_confusion_matrix(conf_matrix_test, ["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

    plt.plot_histogram(num_features, num_classes, pred_test, x_test)

main()

