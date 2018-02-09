import numpy as np
import math

def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    return X

def hinge_loss(label, weights, point):
    return max(0,1- label * (np.dot(weights, point)))

def derivative_hinge_loss(label, weights, point):
    if(hinge_loss(label,weights,point)==0):
        return 0
    else:
        return -label*point

def derivative_pegasos(l, eta, label, weights, point):
    if(hinge_loss(label,weights,point)==0):
        return 0
    else:
        return l*weights -eta*label*point

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    num_features = 400
    y = np.array(map(lambda x: 1 if x[0]=='+' else -1, value))
    X = np.array(map(lambda x: map(float,x.split(" ")[1:]), value))
    
    weights = np.zeros((num_features,), dtype=np.float)
    eta = 100.0
    l = 0.01
    const = 20
    
    shuffle_index = range(y.shape[0])#np.random.permutation(y.shape[0])
    t = 1
    # A = 0
    
    for i in shuffle_index:
        data_point = X[i,:]
        data_label = y[i]
        # A += 1 if
        eta = eta #/(t**(1/2))
        if data_label*np.dot(weights, data_point)<1:
            w_line = weights - eta*derivative_pegasos(l, eta, data_label, weights, data_point)
            weights = min(1, (1/(l**(1/2))) / np.linalg.norm(w_line))*w_line

        # weights = weights - eta*derivative_hinge_loss(data_label, weights, data_point)
        t += 1

    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    a = np.mean(np.array(values),axis=0)
    print(a.shape)
    print(np.random.randn(400).shape)
    yield np.mean(np.array(values),axis=0)  # np.random.randn(400) #
