import numpy as np
import math

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    
    np.random.seed(42)
    x_lines = X.shape[0]
    x_cols = X.shape[1]
    # m = 1000
    # gamma = 0.2
    m = 2000
    gamma = 75
    mu, sigma = np.zeros(x_cols), np.identity(x_cols)
    
    w = np.random.multivariate_normal(mu, sigma, size=m) * np.sqrt(2 * gamma)
    #print(w)
    # w = w / (np.linalg.norm(w))
    b = np.random.uniform(0, 2 * np.pi, size=m)
    # print(b)
    b = np.tile(b.T, (x_lines,1))
    result = np.cos(np.dot(X, w.T) + b) * np.sqrt(2.0/m)
    return result


def transform1(X):
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

    X = transform(X)
    num_trans_features = X.shape[1]
    weights = np.zeros((num_trans_features,), dtype=np.float)
    eta = 100.0
    l = 1e-12
    
    shuffle_index = y.shape[0]  # np.random.permutation(y.shape[0])
    #print('shuffle index = {}'.format(shuffle_index))
    time = 1
    i = 0
    batch = 50
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
        if num_corrections > 0:
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

