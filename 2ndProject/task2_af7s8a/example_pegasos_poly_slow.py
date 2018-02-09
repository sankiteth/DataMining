import numpy as np
import math

def transform(X):
    c = 0.5
    # print(X.shape)
    if X.ndim == 1:
        transformed = np.array((2 * c) ** (0.5) * X)
        for col in range(10):
            transformed = np.append(transformed, [ X[col] * X[i] for i in range(col, 399) ])
        return transformed
    else:
        samples = np.zeros((len(X),4345), dtype=np.float)
        for row in range(len(X)):
            transformed = np.array((2 * c) ** (0.5) * X[row, :])
            for col in range(10):
                transformed = np.append(transformed, [ X[row][col] * X[row][i] for i in range(col, 399) ])
            samples[row, :] = transformed

        #print("Dimension of test data after transform: {0}".format(np.array(samples).shape))
        return samples

def project_L2(w, l):
    """Project to L2-ball, as presented in the lecture."""
    return w * min(1, 1 / (l ** (1 / 2.0) * np.linalg.norm(w, 2)))

def derivative_pegasos(l, weights, eta, num_corrections, correction):
    return (l * weights) - ((eta * correction) / num_corrections)

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    num_features = 400
    trans_features = 4345#799
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float, x.split(" ")[1:]), value))
    
    weights = np.zeros((trans_features,), dtype=np.float)
    eta = 100.0
    l = 0.01
    
    shuffle_index = y.shape[0]  # np.random.permutation(y.shape[0])
    # print('shuffle index = {}'.format(shuffle_index))
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
            trans_data = transform(data_point)
            if data_label * np.dot(weights, trans_data) < 1:
                num_corrections += 1
                correction += (data_label * trans_data)
            j += 1

        eta = 1 / (l * ((time) ** (1 / 2.0)))
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
    #print("Shape yielded by reducer={}".format(a.shape))
    # print(np.random.randn(400).shape)
    yield np.mean(np.array(values), axis=0)  # np.random.randn(400) #

