import numpy as np
from numpy import linalg as la

arts = {}
# User information only. Dimension - d = 6
x_t_a = None
d = 6

# User - article interaction information. Dimension - k = 36
z_t_at = None
k = 12

A_0 = np.identity(k, np.float)
A_0_inv = np.identity(k, np.float)
b_0 = np.zeros((k, 1))
theta_a = {}
beta = {}

A_a = {}
A_a_inv = {}
B_a = {}
b_a = {}

alpha = 0.4
print("alpha={0}".format(alpha))
last_rec = -1
    
def set_articles(articles):
    global arts, d
    for art in articles:
        arts[art] = np.array(articles[art][:2]).reshape((2, 1))

def update(reward):
    global arts, x_t_a, d, z_t_at, k, A_0, A_0_inv, b_0, theta_a, beta, A_a, A_a_inv, B_a, b_a, alpha, last_rec
    if reward != -1:
        # print(reward)
        A_0 += np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(B_a[last_rec])         
        
        b_0 += np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(b_a[last_rec])
         
        A_a[last_rec] += x_t_a.dot(np.transpose(x_t_a))

        
        A_a_inv[last_rec] = la.pinv(A_a[last_rec])
        
        B_a[last_rec] += x_t_a.dot(np.transpose(z_t_at))
        
        b_a[last_rec] += (x_t_a * reward)
            
        A_0 += ( z_t_at.dot(np.transpose(z_t_at)) 
               - np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(B_a[last_rec])
               )
        
        A_0_inv = la.pinv(A_0)
        
        b_0 += ( (reward * z_t_at) 
               - np.transpose(B_a[last_rec]).dot(A_a_inv[last_rec]).dot(b_a[last_rec])
               )

def recommend(time, user_features, choices):
    global arts, x_t_a, d, z_t_at, k, A_0, A_0_inv, b_0, theta_a, beta, A_a, A_a_inv, B_a, b_a, alpha, last_rec   
    
    A_t = choices
    max_ucb = -9999999.0
    arg_max_ucb = -1
    
    x_t_a = np.array(user_features).reshape((d, 1))
    
    beta = A_0_inv.dot(b_0)
    for a in A_t:
        article = arts[a]
        z_t_a = article.dot(np.transpose(x_t_a)).reshape((k, 1))
        
        if a not in A_a:
            A_a[a] = np.identity(d, np.float)
            A_a_inv[a] = np.identity(d, np.float)
            B_a[a] = np.zeros((d, k), np.float)
            b_a[a] = np.zeros((d, 1), np.float)
        
        theta_a[a] = A_a_inv[a].dot(b_a[a] - (B_a[a].dot(beta)))
        
        s_t_a = ( np.transpose(z_t_a).dot(A_0_inv).dot(z_t_a) 
                - 2 * (np.transpose(z_t_a).dot(A_0_inv).dot(np.transpose(B_a[a])).dot(A_a_inv[a]).dot(x_t_a))
                + np.transpose(x_t_a).dot(A_a_inv[a]).dot(x_t_a)
                + np.transpose(x_t_a).dot(A_a_inv[a]).dot(B_a[a]).dot(A_0_inv).dot(np.transpose(B_a[a]))
                        .dot(A_a_inv[a]).dot(x_t_a)
                )
        
        p_t_a = (np.transpose(z_t_a).dot(beta)
                + np.transpose(x_t_a).dot(theta_a[a])
                + alpha * np.sqrt(s_t_a)
                )
                        
        if p_t_a > max_ucb:
            max_ucb = p_t_a
            arg_max_ucb = a
            z_t_at = z_t_a
            
    last_rec = arg_max_ucb  
    return arg_max_ucb
        
        
         
