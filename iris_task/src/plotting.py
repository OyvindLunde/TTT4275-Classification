import matplotlib.pyplot as plt
import extract_data as ed, math_functions as mf
import numpy as np
import itertools


def get_feature_vector(feature_number, class_number, predictions, dataset):
    feature_vector = []
    for i in range(len(dataset[0])):
        if predictions[i] == class_number:
            feature_vector.append(dataset[feature_number][i])

    return feature_vector


def histogram(x, feature_number, class_number):
    #If you have removed features, then remove them from the feature list
    #and remove them from the histogram_bounds() function below
    feuatures = ["sepal length" ,"sepal width", "petal length" ,"petal width"]
    classes = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    min_list, max_list = histogram_bounds()
    num_bins = 10
    plt.hist(x, num_bins, range=[min_list[feature_number][0], max_list[feature_number][0]], facecolor='blue', alpha=0.5)
    plt.title("Class " + classes[class_number] + " with feature " + feuatures[feature_number])
    plt.xlabel("Length [cm]")
    plt.ylabel("Quantity")
    plt.show()


def histogram_bounds():
    #Comment out/delete the feature you want to remove
    x, y =ed. extract_data()
    x1 = np.zeros((len(x), 1))
    x2 = np.zeros((len(x), 1))
    x3 = np.zeros((len(x), 1))
    x4 = np.zeros((len(x), 1))
    for i in range(len(x)):
        x1[i] = x[i][0]
        x2[i] = x[i][1]
        x3[i] = x[i][2]
        x4[i] = x[i][3]

    min_x1, max_x1 = mf.find_smallest_and_biggest_value(x1)
    min_x2, max_x2 = mf.find_smallest_and_biggest_value(x2)
    min_x3, max_x3 = mf.find_smallest_and_biggest_value(x3)
    min_x4, max_x4 = mf.find_smallest_and_biggest_value(x4)

    min_list =  [min_x1, min_x2, min_x3, min_x4]
    max_list = [max_x1, max_x2, max_x3, max_x4]

    return min_list, max_list


def plot_histogram(num_features, num_classes, prediction, dataset):
    for i in range(num_features):
        for j in range(num_classes):
            x = get_feature_vector(i, j, prediction, dataset)
            histogram(x, i, j)


def confusion_matrix(prediction, actual, numClasses):
    confusionMatrix = np.zeros((numClasses,numClasses))
    for i in range(len(prediction)):
        confusionMatrix[int(actual[i])][int(prediction[i])] += 1
    return confusionMatrix


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
