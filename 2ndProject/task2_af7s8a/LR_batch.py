import numpy as np
import math

def transform(X):
    return X

def project_L2(w, l):
    """Project to L2-ball, as presented in the lecture."""
    return w * min(1, 1 / (l ** (1 / 2.0) * np.linalg.norm(w, 2)))

def project_L1(w, a):
    """Project to L1-ball, as described by Duchi et al. [ICML '08]."""
    z = 1.0 / (a * a)
    if np.linalg.norm(w, 1) <= z:
        return w
    mu = -np.sort(-w)
    cs = np.cumsum(mu)
    rho = -1
    for j in range(len(w)):
        if mu[j] - (1.0 / (j + 1)) * (cs[j] - z) > 0:
            rho = j
    theta = (1.0 / (rho + 1)) * (cs[rho] - z)
    return np.sign(w) * np.fmax(w - theta, 0)

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    num_features = 400
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float, x.split(" ")[1:]), value))
    
    weights = np.zeros((num_features,), dtype=np.float)
    eta = 0.1
    l = 0.00000000001
    shuffle_index = y.shape[0]
    iterations = 10
    for t in range(iterations):
        grad = 0
        for i in range(shuffle_index):
            data_point = X[i, :]
            data_label = y[i]
            grad += (1.0 / (1 + np.exp(data_label * np.dot(weights, data_point)))) * data_label * data_point
        grad = grad / shuffle_index
        eta = 1 / np.sqrt(t + 1)
        weights += (eta * grad)
        weights = project_L2(weights, l)

    yield "key", weights  # This is how you yield a key, value pair


def reducer(key, values):
    yield np.mean(np.array(values), axis=0)  # np.random.randn(400) #
