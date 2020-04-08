import sys
sys.path.append("..")
from handwritten_numbers_task.src import extract_data as ed
import numpy as np
import math

def find_distance(m1, m2):
    dist = 0
    for i in range(len(m1)):
        for j in range(len(m1[0])):
            dist += (m1[i][j]-m2[i][j])**2
    
    return dist


def find_k_nearest_neighbours(test_matrix, test_set, k):
    list = np.zeros((k,2))
    for i in range(k):
        list[i] = [i, find_distance(test_matrix, test_set[i])]
    list = list[np.argsort(list[:, 1])]

    #print(list)
    for i in range(k, len(test_set)):
        dist = find_distance(test_matrix, test_set[i])
        if dist < list[k-1][1]:
            list[k-1] = [i,dist]
            list = list[np.argsort(list[:, 1])]

    return list[:,0]



train_set, train_labels, test_set, test_labels = ed.get_sets()


reduced_train_set = train_set[0:1000]
reduced_train_labels = train_labels[0:1000]

test = find_k_nearest_neighbours(reduced_train_set[4], reduced_train_set, 7)
print(test)

predicts = np.zeros((100,1))

min_dist = math.inf
index = 0

"""
for i in range(100):
    for j in range(len(reduced_train_set)):
        dist = find_distance(reduced_train_set[j], test_set[i])
        if dist < min_dist:
            index = j
            min_dist = dist
    predicts[i] = reduced_train_labels[index]
    min_dist = math.inf


for i in range(10):
    print("predicted: ", predicts[i])
    print("actual: ", test_labels[i])
    print("________________________")
"""