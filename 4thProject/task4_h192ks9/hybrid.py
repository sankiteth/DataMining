import numpy as np
from numpy import linalg as la

arts = {}
# User information only. Dimension - d = 6
x_t_a = None
d = 6

# User - article interaction information. Dimension - k = 36
z_t_at = None
k = 36

A_0 = np.identity(k, np.float)
A_0_inv = np.identity(k, np.float)
b_0 = np.zeros((k,1))
theta_a = {}
beta = {}

A_a = {}
A_a_inv = {}
B_a = {}
b_a = {}

alpha = 0.1
print("alpha={0}".format(alpha))
last_rec = -1
count = -1

users = np.empty((1000, 6))
user_clusters = None

def set_articles1(articles):
    global arts
    dataset = np.empty((len(articles), 6))
    mapping = {}
    i = 0
    for art in articles:
        dataset[i] = np.array( articles[art] ).reshape((6))
        mapping[i] = art
        i += 1
        
    # Group articles into 5 clusters
    clusters = kmeans(dataset, 2)[0]
    
    dists = cdist(dataset, clusters, 'sqeuclidean')
    rbf_dists = np.exp(dists.dot(-1))
          
    for i in range(len(rbf_dists)):
        sum = reduce((lambda x, y: x + y), rbf_dists[i])
        rbf_dists[i] = map(lambda x: x/sum, rbf_dists[i])
        arts[mapping[i]] = np.insert(rbf_dists[i], 0, 1).reshape(3,1)
        #arts[mapping[i]] = rbf_dists[i].reshape(1,1)

    #print(arts)
    
def set_articles(articles):
    global arts, d
    for art in articles:
        arts[art] = np.array(articles[art]).reshape((d,1))

def update(reward):
    global arts, x_t_a, d, z_t_at, k, A_0, A_0_inv, b_0, theta_a, beta, A_a, A_a_inv, B_a, b_a, alpha, last_rec, count
    if reward != -1:
        #print(reward)
        #count += 1
        #print(count)
        A_0 = np.add( A_0,
                      np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(B_a[last_rec])
                      )
        
        A_0_inv = la.pinv(A_0)
        
        b_0 = np.add( b_0,
                      np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(b_a[last_rec])
                      )
         
        A_a[last_rec] = np.add( A_a[last_rec],
                                x_t_a.dot(np.transpose(x_t_a))
                                )
        
        A_a_inv[last_rec] = la.pinv(A_a[last_rec])
        
        B_a[last_rec] = np.add( B_a[last_rec],
                                x_t_a.dot(np.transpose(z_t_at))
                                )
        
        b_a[last_rec] = np.add( b_a[last_rec],
                                x_t_a*reward
                                )
            
        A_0 = np.add( A_0,
                      z_t_at.dot(np.transpose(z_t_at))
                      )
        A_0 = np.subtract( A_0,
                           np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(B_a[last_rec])
                           )
        
        b_0 = np.add( b_0,
                      reward*z_t_at
                      )
        b_0 = np.subtract( b_0,
                           np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(b_a[last_rec])
                           )
                    
      

def recommend(time, user_features, choices):
    global arts, x_t_a, d, z_t_at, k, A_0, A_0_inv, b_0, theta_a, beta, A_a, A_a_inv, B_a, b_a, alpha, last_rec, count, users, user_clusters   
    count += 1
    
    if count < 10000:
        users[count] = np.array(user_features).reshape((6))
        return -1
    
    if count == 10000:
        user_clusters = kmeans(users, 2)[0]
        return -1

            
    A_t = choices
    max_ucb = -9999999.0
    arg_max_ucb = -1
    
    x_t_a = np.array(user_features).reshape((1,6))
    dists = cdist(x_t_a, user_clusters, 'sqeuclidean')
    rbf_dists = np.exp(dists.dot(-1))
    sum = reduce((lambda x, y: x + y), rbf_dists[0])
    rbf_dists[0] = map(lambda x: x/sum, rbf_dists[0])
    x_t_a = np.insert(rbf_dists[0], 0, 1).reshape(d,1)
    
    beta = A_0_inv.dot(b_0)
    for a in A_t:
        article = arts[a]
        z_t_a = article.dot(np.transpose(x_t_a)).reshape((k,1))
        
        if a not in A_a:
            A_a[a] = np.identity(d, np.float)
            A_a_inv[a] = np.identity(d, np.float)
            B_a[a] = np.zeros((d,k), np.float)
            b_a[a] = np.zeros((d,1), np.float)
        
        theta_a[a] = A_a_inv[a].dot( b_a[a] - ( B_a[a].dot(beta) ) )
        
        s_t_a = np.transpose(z_t_a).dot(A_0_inv).dot(z_t_a)
        
        s_t_a = np.subtract( s_t_a,
                 2*( np.transpose(z_t_a).dot(A_0_inv).dot(np.transpose(B_a[a])).dot(A_a_inv[a]).dot(x_t_a) )
              )
        
        s_t_a = np.add( s_t_a,
                        np.transpose(x_t_a).dot(A_a_inv[a]).dot(x_t_a)
                         )
        
        s_t_a = np.add( s_t_a,
                        np.transpose(x_t_a).dot(A_a_inv[a]).dot(B_a[a]).dot(A_0_inv).dot(np.transpose(B_a[a]))
                        .dot(A_a_inv[a]).dot(x_t_a)
                         )
        
        p_t_a = np.transpose(z_t_a).dot(beta)
        
        p_t_a = np.add( p_t_a,
                        np.transpose(x_t_a).dot(theta_a[a])
            )
        
        p_t_a = np.add( p_t_a,
                        alpha * np.sqrt(s_t_a)
            )
        
        if p_t_a > max_ucb:
            max_ucb = p_t_a
            arg_max_ucb = a
            z_t_at = z_t_a
            
    last_rec = arg_max_ucb  
    return arg_max_ucb
        
        
         