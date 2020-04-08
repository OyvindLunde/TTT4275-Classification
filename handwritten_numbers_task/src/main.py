from handwritten_numbers_task.src import extract_data as ed
from handwritten_numbers_task.src import math_functions as mf
import numpy as np
import math

def main():
    train_set, train_labels, test_set, test_labels = ed.get_sets()

    reduced_train_set = train_set[0:1000]
    reduced_train_labels = train_labels[0:1000]

    predicts = np.zeros((100, 1))

    min_dist = math.inf
    index = 0

    for i in range(len(predicts)):
        for j in range(len(reduced_train_set)):
            dist = mf.find_distance(reduced_train_set[j], test_set[i])
            if dist < min_dist:
                index = j
                min_dist = dist
        predicts[i] = reduced_train_labels[index]
        min_dist = math.inf

        cm = mf.confusion_matrix(predicts, test_labels[0:len(predicts)], 10)
        mf.plot_confusion_matrix(cm, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])