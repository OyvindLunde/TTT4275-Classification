from handwritten_numbers_task.src import extract_data as ed
from handwritten_numbers_task.src import math_functions as mf
import numpy as np
import math

def main():
    train_set, train_labels, test_set, test_labels = ed.get_sets()

    reduced_train_set = train_set[0:10000]
    reduced_train_labels = train_labels[0:10000]
    predicts = np.zeros((100, 1))

    for i in range(len(predicts)):
        neighbours = mf.find_k_nearest_neighbours(test_set[i],reduced_train_set, 7)
        predicts[i] = mf.predict_label(neighbours, reduced_train_labels)

    cm = mf.confusion_matrix(predicts, test_labels[0:len(predicts)], 10)
    mf.plot_confusion_matrix(cm, ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

main()