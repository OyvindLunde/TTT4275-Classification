#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from iris_task.src import extract_data as ed


def histogram(x, feature_number):
    min_list, max_list = ed.histogram_bounds()
    num_bins = 10
    n, bins, patches = plt.hist(x, num_bins, range=[min_list[feature_number][0], max_list[feature_number][0]], facecolor='blue', alpha=0.5)
    plt.show()

def get_feature_vector(feature_number, class_number, predictions, data_set):
    feature_vector = []
    for i in range(len(data_set[0])):
        if predictions[i] == class_number:
            feature_vector.append(data_set[feature_number][i])

    return feature_vector