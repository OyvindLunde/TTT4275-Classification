import struct
import numpy as np

def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def get_sets():
    train_set = read_idx("../number_data/MNist_ttt4275/train_images.bin")
    train_labels = read_idx("../number_data/MNist_ttt4275/train_labels.bin")
    test_set = read_idx("../number_data/MNist_ttt4275/test_images.bin")
    test_labels = read_idx("../number_data/MNist_ttt4275/test_labels.bin")
    return train_set, train_labels, test_set, test_labels

