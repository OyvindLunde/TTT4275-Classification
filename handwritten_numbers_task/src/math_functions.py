from handwritten_numbers_task.src import extract_data as ed
import numpy as np
import math
import itertools
import matplotlib.pyplot as plt

def find_distance(m1, m2):
    dist = 0
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            dist += (m1[i][j]-m2[i][j])**2
    
    return dist

train_set, train_labels, test_set, test_labels = ed.get_sets()

reduced_train_set = train_set[0:1000]
reduced_train_labels = train_labels[0:1000]

predicts = np.zeros((10,1))

min_dist = math.inf
index = 0

for i in range(len(predicts)):
    for j in range(len(reduced_train_set)):
        dist = find_distance(reduced_train_set[j], test_set[i])
        if dist < min_dist:
            index = j
            min_dist = dist
    predicts[i] = reduced_train_labels[index]
    min_dist = math.inf


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
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


def confusion_matrix(prediction, actual, numClasses):
    confusionMatrix = np.zeros((numClasses,numClasses))
    for i in range(len(prediction)):
        confusionMatrix[int(actual[i])][int(prediction[i])] += 1
    return confusionMatrix


cm = confusion_matrix(predicts, test_labels[0:10], 10)
plot_confusion_matrix(cm, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])