import numpy as np
import math
from time import time


#iterations=300 
#l=1e-12
#a_size=1000 
#m=5000
#print("iterations, l, a_size, m")
#print(iterations, l, a_size, m)
def transform(X):
    # Make sure this function works for both 1D and 2D NumPy arrays.
    if X.ndim == 1:    
        x_lines = 1
        x_cols = 400
    else:
        x_lines = X.shape[0]
        x_cols = X.shape[1]
    #X = np.float32(X)
    np.random.seed(42)
    # m = 1000
    # gamma = 0.2
    m = 3000
    #gamma = 80
    #mu, sigma = np.zeros(x_cols), np.identity(x_cols) * (1 / np.sqrt(400*gamma*2))
    #w = np.random.gamma(shape=1,scale=22,size=(m,400,)).astype('float32')
    w = np.random.standard_cauchy(size=(m,400,))#.astype('float32')
    #w = np.random.multivariate_normal(mu, sigma, size=m) #* np.sqrt(2 * gamma)
    #print(w)
    # w = w / (np.linalg.norm(w))
    b = np.random.uniform(0, 2 * np.pi, size=m)#.astype('float32')
    # print(b)
    b = np.tile(b.T, (x_lines,1))#.astype('float32')
    result = (np.cos(np.dot(X, w.T) + b) * np.sqrt(2.0/m))#.astype('float16')
    
    if X.ndim == 1:
        result = result.reshape((m))
    return result

def hinge_loss(label, weights, point):
    return max(0,1 - label * (np.dot(weights, point)))

def derivative_hinge_loss(label, weights, point):
    if(hinge_loss(label, weights, point) == 0):
        return 0
    else:
        return - label * point

def derivative_pegasos(l, eta, label, weights, point):
    if(hinge_loss(label,weights,point)==0):
        return 0
    else:
        return l * weights - eta * label * point


def mapper(key, value):
    # key: None
    # value: one line of input file
    
    
    iterations=300 
    l=1e-12
    a_size=500

    
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float,x.split(" ")[1:]), value))
    
    X = transform(X)
    #print("Transformation Done!")
    num_features = X.shape[1]
    

    weights = np.zeros((num_features,), dtype=np.float)
    
    for t in range(1,iterations):
        # choose At
        # shuffle_index = np.random.permutation(num_features)[:a_size]
        shuffle_index = np.random.permutation(num_features)[:a_size]
        sum_pegasos = np.zeros((num_features,), dtype=np.float)
        # set eta = 1/(lambda*t)
        eta = 1 / (t*l)
        
        for i in shuffle_index:
            data_point = X[i,:]
            #data_point = transform(data_point)
            #print("data_point dimension={}".format(data_point.shape))
            data_label = y[i]
            
            if data_label*np.dot(weights, data_point) < 1:
                sum_pegasos += data_label*data_point
    
        weights_old = weights
        weights = (1- eta*l)*weights + (eta/a_size)*sum_pegasos
        #diff = np.max(np.abs(weights-weights_old))
        #if t % 100 == 0:
        #    print("{0}: {1}".format(identity,diff))
        #if (diff < epsilon):
            #break
    weights = (min(1, ((1/l)**(1/2))/(np.linalg.norm(weights)))*weights)#.astype('float16')
    
    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    # Note that we do *not* output a (key, value) pair here.
    #a = np.mean(np.array(values),axis=0)
    yield np.mean(np.array(values),axis=0)  # np.random.randn(400) #
