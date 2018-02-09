import numpy as np
import math

def transform_poly(X):
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

def transform(X):
    return X

def project_L2(w, l):
    """Project to L2-ball, as presented in the lecture."""
    return w * min(1, 1 / (l ** (1 / 2.0) * np.linalg.norm(w, 2)))

def derivative_pegasos(l, weights, eta, data_point, data_label):
    return -(data_label * data_point)

def mapper(key, value):
    # key: None
    # value: one line of input file
    
    trans_features = 400  # 799
    y = np.array(map(lambda x: 1 if x[0] == '+' else -1, value))
    X = np.array(map(lambda x: map(float, x.split(" ")[1:]), value))
    
    weights = np.zeros((trans_features,), dtype=np.float)
    eta = 100.0
    l = 1e-12
    gamma = 0.9
    
    shuffle_index = np.random.permutation(X.shape[0]) #y.shape[0] 
    # print('shuffle index = {}'.format(shuffle_index))
    batch = 1
    time = 1
    t = 0
    prev_grad = 0.0
    while t < X.shape[0]:
        # do mini batch
        correction = 0.0
        num_corrections = 0
        for i in range(batch):
            data_point = X[shuffle_index[t+i], :]
            data_label = y[shuffle_index[t+i]]
            trans_data = transform(data_point)
            if data_label * np.dot(weights, trans_data) < 1:
                correction += (data_label * data_point)
                num_corrections += 1
            
            #grad = (l * weights) - ((eta / 500) * correction)
            eta = 1 / (time**0.5)
            grad = (gamma*prev_grad) + eta*(correction/batch)
            prev_grad = grad

        weights = weights + grad
        weights = project_L2(weights, l)

        t += batch
        time += 1

    yield "key", weights


def reducer(key, values):
    yield np.mean(np.array(values), axis=0)
