from keras.datasets import mnist
import numpy as np
import cv2



def load_mnist_data():
    # load mnist data
    data = mnist.load_data()
    data_dict = {
        'train': {'X': data[0][0], 'y': data[0][1]},
        'test': {'X': data[1][0], 'y': data[1][1]}
    }
    for split, sub_dict in data_dict.items():
        X = sub_dict['X']
        # resize X to (num_samples, 14, 14)
        X = np.array([cv2.resize(x, (14, 14)) for x in X])
        y = sub_dict['y']
        # shuffle
        perm = np.random.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]
        # reshape
        X = X.reshape(X.shape[0], -1)
        # normalize
        X = X / 255.0

        # only keep {0, 1, 2}
        mask = (y == 0) | (y == 1) | (y == 2)
        X = X[mask]
        y = y[mask]

        data_dict[split]['X'] = X
        data_dict[split]['y'] = y
    return data_dict['train']['X'], data_dict['train']['y'], data_dict['test']['X'], data_dict['test']['y'] 