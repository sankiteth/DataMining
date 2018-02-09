# Pegasos
import numpy as np
import math

def transform1(X):
    c = 0.5
    # print(X.shape)
    if X.ndim == 1:
        prod = np.dot(X[np.newaxis].T, np.flipud(X[np.newaxis]))
        # print(prod)
        tri_upper_diag = np.triu(prod, k=0)
        transformed = tri_upper_diag[np.triu_indices(100)]
        return np.append(transformed, X) 
    else:
        samples = np.zeros((len(X), 5450), dtype=np.float)
        for row in range(len(X)):
            prod = np.dot(np.reshape(X[row, :], (len(X[0]), 1)), np.flipud(np.reshape(X[row, :], (1, len(X[0])))))
            tri_upper_diag = np.triu(prod, k=0)
            samples[row, :] = np.append(tri_upper_diag[np.triu_indices(100)], X[row, :])

        print("Dimension of test data after transform: {0}".format(np.array(samples).shape))
        return samples

def transform_No(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X

def hinge_loss(label, weights, point):
    return max(0, 1 - label * (np.dot(weights, point)))

def derivative_hinge_loss(label, weights, point):
    if(hinge_loss(label, weights, point) == 0):
        return 0
    else:
        return -label * point

def derivative_pegasos(l, eta, label, weights, point):
    if(hinge_loss(label, weights, point) == 0):
        return 0
    else:
        return l * weights - eta * label * point

def mapper(key, value):
    # key: None
    # value: one line of input file
    iterations = 500
    l = 1e-12
    # eta = 0.1
    gamma = 0.9
    a_size = 1000
    beta1 = 0.9
    beta2 = 0.999
    m = 0.0
    v = 0.0
    
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float, x.split(" ")[1:]), value))
    
    X = transform(X)
    # print(X)
    num_features = X.shape[1]
    
    weights = np.zeros((num_features,), dtype=np.float)
    # weights.fill(np.sqrt(1/(400*l))-0.01)
    # print("weight norm", np.linalg.norm(weights), weights)
    prev_grad = 0.0
    for t in range(1, iterations):
        # choose At
        shuffle_index = np.random.permutation(X.shape[0])[:a_size]
        
        sum_pegasos = np.zeros((num_features,), dtype=np.float)
        # set eta = 1/(lambda*t)
        eta = 1 / (t * l)
        
        for i in shuffle_index:
            data_point = X[i, :]
            data_label = y[i]
            
            if data_label * np.dot(weights, data_point) < 1:
                sum_pegasos += data_label * data_point
        
        # ADAM
        # grad = (l*weights) - ((eta/a_size)*sum_pegasos)
        # m = (beta1 * m) + ((1 - beta1) * grad)
        # v = (beta2 * v) + ((1 - beta2) * (grad ** 2))
        # m = m / (1 - (beta1 ** (t)))
        # v = v / (1 - (beta1 ** (t))) 
        # weights -= (eta / ((v ** (1 / 2.0)) + 10 ** -8)) * m


        # MOMENTUM METHOD
        grad = (l * weights) - ((eta / a_size) * sum_pegasos)
        grad = gamma * prev_grad + eta * grad
        prev_grad = grad
        weights = weights - grad
    # print(weights)
    weights = min(1, ((1 / l) ** (1 / 2)) / (np.linalg.norm(weights))) * weights

    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    a = np.mean(np.array(values), axis=0)
    # print(a.shape)
    # print(np.random.randn(400).shape)
    yield np.mean(np.array(values), axis=0)  # np.random.randn(400) 
