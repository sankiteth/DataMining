import numpy as np
import math

def transform(X):  
    return X

def derivative_pegasos(l, weights, eta, data_point, data_label):
    return (l * weights) - (data_label * data_point)

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
    num_samples = y.shape[0]
    time = 1
    l = 1
    iteration = 600
    for t in range(iteration):
        i = 0
        grad = 0.0
        for i in range(num_samples):
            # do mini batch
            j = 0
            data_point = X[i + j, :]
            data_label = y[i + j]
            if data_label * np.dot(weights, data_point) < 1:
                grad += derivative_pegasos(l, weights, eta, data_point, data_label)
            
        grad = grad/num_samples
        m = (beta1 * m) + ((1 - beta1) * grad)
        v = (beta2 * v) + ((1 - beta2) * (grad ** 2))
        m = m / (1 - (beta1 ** (time)))
        v = v / (1 - (beta1 ** (time))) 
        weights -= (eta / ((v ** (1 / 2.0)) + 10 ** -8)) * m
        
        time += 1

    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    yield np.mean(np.array(values), axis=0)  # np.random.randn(400) #
