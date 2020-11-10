import numpy as np
from .gcForest import *
from time import time


def load_data():
    train_data = np.load()
    train_label = np.load()
    train_weight = np.load()
    test_data = np.load()
    test_label = np.load()
    test_file = np.load()
    return [train_data, train_label, train_weight, test_data, test_label, test_file]


if __name__ == '__main__':
    train_data, train_label, train_weight, test_data, test_label, test_file = load_data()
    clf = gcForest(num_estimator=100, num_forests=4, max_layer=2, max_depth=100, n_fold=5)
    start = time()
    clf.train(train_data, train_label, train_weight)
    end = time()
    print("fitting time: " + str(end - start) + " sec")
    start = time()
    prediction = clf.predict(test_data)
    end = time()
    print("prediction time: " + str(end - start) + " sec")
    result = {}
    for index, item in enumerate(test_file):
        if item not in result:
            result[item] = prediction[index]
        else:
            result[item] = (result[item] + prediction[index]) / 2
    print(result)
