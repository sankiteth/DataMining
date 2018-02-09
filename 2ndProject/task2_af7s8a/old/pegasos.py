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
    
    
    y = np.array(map(lambda x: 1 if x[0]=='+' else -1, value))
    X = np.array(map(lambda x: map(float,x.split(" ")[1:]), value))
    
    
    num_features = X.shape[1]

    l = 0.001
    iterations = 100 # 30 #
    a_size = 1
    weights = np.zeros((num_features,), dtype=np.float)
    #weights.fill(np.sqrt(1/(400*l))-0.01)
    #print("weight norm", np.linalg.norm(weights), weights)
    
    for t in range(1,iterations):
        # choose At
        shuffle_index = np.random.permutation(num_features)[:a_size]
        
        sum_pegasos = np.zeros((num_features,), dtype=np.float)
        # set eta = 1/(lambda*t)
        eta = 1/(t*l)
        
        for i in shuffle_index:
            data_point = X[i,:]
            data_label = y[i]
            
            if data_label*np.dot(weights, data_point) < 1:
                sum_pegasos += data_label*data_point
        weights = (1- eta*l)*weights + (eta/a_size)*sum_pegasos

    weights = min(1, ((1/l)**(1/2))/(np.linalg.norm(weights)))*weights
    
    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    a = np.mean(np.array(values),axis=0)
    print(a.shape)
    print(np.random.randn(400).shape)
    yield np.mean(np.array(values),axis=0)  # np.random.randn(400) #
