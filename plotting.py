import numpy as np
#import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def histogram(x):
    num_bins = 10
    n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
    plt.show()


