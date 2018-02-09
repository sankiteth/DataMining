import numpy as np
import math

def transform(X):
    return X

def derivative_pegasos(l, w, eta, y, x):
    if y * np.dot(w, x) < 1:
        return (l * w) - (y * x)

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    num_features = 400
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float, x.split(" ")[1:]), value))
    
    w = np.zeros((num_features,), dtype=np.float)
    s = np.ones((num_features,), dtype=np.float)
    eta = 0.01
    l = 1e-12
    
    shuffle_index = y.shape[0]
    for j in range(100):
        i = 0
        while i < shuffle_index:
            data_point = X[i, :]
            data_label = y[i]
            if data_label * np.dot(w, data_point) < 1:
                del_f = derivative_pegasos(l, w, eta, data_label, data_point)
    
                for dim in range(len(data_point)):
                    s[dim] += del_f[dim] ** 2
                    w[dim] -= (eta / (s[dim] ** (0.5)))
            i += 1
            

    yield "key", w  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    # a = np.mean(np.array(values), axis=0)
    # print("Shape yielded by reducer={}".format(a.shape))
    # print(np.random.randn(400).shape)
    yield np.mean(np.array(values), axis=0)  # np.random.randn(400) #
