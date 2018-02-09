import numpy as np
import math

def transform(X):  
    return X

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
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float, x.split(" ")[1:]), value))
    
    weights = np.zeros((num_features,), dtype=np.float)
    eta = 0.1
    beta1 = 0.9
    beta2 = 0.999
    m = 0.0
    v = 0.0
    shuffle_index = y.shape[0]  # np.random.permutation(y.shape[0])
    # print('shuffle index = {}'.format(shuffle_index))
    time = 1
    l = 1
    i = 0
    batch = 1
    while i < shuffle_index:
        # do mini batch
        j = 0
        correction = 0.0
        num_corrections = 0
        while j < batch:
            data_point = X[i + j, :]
            data_label = y[i + j]
            if data_label * np.dot(weights, data_point) < 1:
                num_corrections += 1
                correction += (data_label * data_point)
            j += 1

        if num_corrections > 0:
            grad = derivative_pegasos(l, weights, eta, num_corrections, correction)
        else:
            grad = 0
        m = (beta1 * m) + ((1 - beta1) * grad)
        v = (beta1 * v) + ((1 - beta1) * (grad ** 2))
        m = m / (1 - (beta1 ** (time)))
        v = v / (1 - (beta1 ** (time))) 
        weights -= ( eta / ( (v ** (1 / 2.0)) + 10**-8 ) ) * m
        
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
