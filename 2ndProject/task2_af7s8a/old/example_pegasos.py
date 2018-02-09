import numpy as np
import math

def transform(X):
	return X

def project_L2(w, l):
    """Project to L2-ball, as presented in the lecture."""
    return w * min(1, 1 / (l ** (1 / 2.0) * np.linalg.norm(w, 2)))

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
    eta = 100.0
    l = 1e-12
    
    shuffle_index = y.shape[0]  # np.random.permutation(y.shape[0])
    #print('shuffle index = {}'.format(shuffle_index))
    time = 1
    i = 0
    batch = 10
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

        eta = 1 / ( l * ( (time) ** (1 / 2.0) ) )
        weights = weights - eta * derivative_pegasos(l, weights, eta, num_corrections, correction)
        weights = project_L2(weights, l)
        
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
