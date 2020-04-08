from handwritten_numbers_task.src import extract_data as ed
import numpy as np
import math

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

for i in range(10):
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