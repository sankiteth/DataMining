import numpy as np
import math

def transform(X):
    print(X.shape)
    if X.ndim == 1:
        t = np.zeros((1000,))
        for i in range(450):
            dim = (1.0 / np.math.factorial(i)) * np.exp(-0.5 * (np.linalg.norm(X, 2) ** 2) * (np.linalg.norm(X, 2) ** i))
            t[i] = dim
    else:
        
        t = np.zeros((X.shape(0), 1000))
        for row in range(len(X)):
            for i in range(1000):
                t[row] = (1.0 / np.math.facorial(i)) * np.exp(-0.5 * (np.linalg.norm(X[row], 2) ** 2) * (np.linalg.norm(X[row], 2) ** i))
        
    return t

def hinge_loss(label, weights, point):
    return max(0, 1 - label * (np.dot(weights, point)))

def derivative_hinge_loss(label, weights, point):
    if(hinge_loss(label, weights, point) == 0):
        return 0
    else:
        return -label * point

def derivative_pegasos(l, weights, eta, num_corrections, correction):
    return (l * weights) - ((eta * correction) / num_corrections)

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    num_features = 400
    trans_features = 1000
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float, x.split(" ")[1:]), value))
    
    weights = np.zeros((trans_features,), dtype=np.float)
    eta = 100.0
    l = 0.0001
    
    shuffle_index = y.shape[0]  # np.random.permutation(y.shape[0])
    # print('shuffle index = {}'.format(shuffle_index))
    time = 1
    i = 0
    batch = 500
    while i < shuffle_index:
        # do mini batch
        j = 0
        correction = 0.0
        num_corrections = 0
        while j < batch:
            data_point = X[i + j, :]
            data_label = y[i + j]
            trans_data = transform(data_point)
            if data_label * np.dot(weights, trans_data) < 1:
                num_corrections += 1
                correction += (data_label * trans_data)
            j += 1

        eta = 1 / (l * ((time) ** (1 / 2.0)))
        weights_prime = weights - eta * derivative_pegasos(l, weights, eta, num_corrections, correction)
        weights = min(1.0, (1.0 / (l ** (1 / 2.0) * np.linalg.norm(weights_prime, 2)))) * weights_prime
        
        time += 1    
        i += batch

    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    a = np.mean(np.array(values), axis=0)
    # print(a.shape)
    # print(np.random.randn(400).shape)
    yield np.mean(np.array(values), axis=0)  # np.random.randn(400) #

