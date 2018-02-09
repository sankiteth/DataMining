import numpy as np
import math
from time import time

def transform(X):
    '''
    Using Strandard Cauchy trasformation to trasnform input data,
    to increase dimesionality.
    '''
    # Make sure this function works for both 1D and 2D NumPy arrays.
    if X.ndim == 1:    
        x_lines = 1
        x_cols = 400
    else:
        x_lines = X.shape[0]
        x_cols = X.shape[1]

	# Number of dimensions to transform to
    m = 3000

	# Random draws from Standard Cauchy distribution
    w = np.random.standard_cauchy((m, 400)) * np.sqrt(2)

	# Random draws from Uniform distribution
    b = np.random.uniform(0, 2 * np.pi, size=m)
    b = np.tile(b.T, (x_lines,1))

    result = np.cos(np.dot(X, w.T) + b) * np.sqrt(2.0/m)
    
	# Convert result to 1D for 1 dimensional input
    if X.ndim == 1:
        result = result.reshape((m))
    return result

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    # Number of iterations of gradient descent
    iterations = 900

    # Regularization parameter
    l =  1e-13

    # Mini-Batch size
    batch_size = 500

    # Extract labels and pattern for each video from input
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float,x.split(" ")[1:]), value))
    
    # Transform using Cauchy transformation
    X = transform(X)
    num_trans_features = X.shape[1]
    
    # Prameters to be trained
    weights = np.zeros((num_trans_features,), dtype=np.float)
    
    # Doing mini-batch gradient descent
    for t in range(1,iterations):
        # Randomly selecting training data for Stochastic Gradient Descent
        shuffle_index = np.random.permutation(X.shape[0])[:batch_size]
        
        sum_pegasos = np.zeros((num_trans_features,), dtype=np.float)

        # Learning rate
        eta = 1 / (t*l)
        
        # Doing the mini-batch
        for i in shuffle_index:
            data_point = X[i,:]
            data_label = y[i]

            # if classification error, make correction            
            if data_label*np.dot(weights, data_point) < 1:
                sum_pegasos += data_label*data_point
        
        # Update the model parameters
        weights = (1- eta*l)*weights + (eta/batch_size)*sum_pegasos

    # Projecting on the convex constraint set
    weights = min(1, ((1/l)**(1/2))/(np.linalg.norm(weights)))*weights
    
    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    # key: key from mapper used to aggregate
    # values: list of all value for that key
    yield np.mean(np.array(values),axis=0)
