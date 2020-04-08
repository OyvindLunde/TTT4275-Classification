import sys
sys.path.append("..")
from handwritten_numbers_task.src import extract_data as ed
import numpy as np
import itertools
import matplotlib.pyplot as plt
import math

def find_distance(m1, m2):
    dist = 0
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            dist += (m1[i][j]-m2[i][j])**2
    
    return dist


def find_k_nearest_neighbours(test_matrix, train_set, k):
    list = np.zeros((k,2), dtype=int)
    for i in range(k):
        list[i] = [i, find_distance(test_matrix, train_set[i])]
    list = list[np.argsort(list[:, 1])]

    for i in range(k, len(train_set)):
        dist = find_distance(test_matrix, train_set[i])
        if dist < list[k-1][1]:
            list[k-1] = [i,dist]
            list = list[np.argsort(list[:, 1])]

    return list[:,0]

def predict_label(nearest_neighbours, train_labels):
    list = np.zeros((10,1))
    for i in range(len(nearest_neighbours)):
        list[train_labels[nearest_neighbours[i]]] += 1
    return np.argmax(list)

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

