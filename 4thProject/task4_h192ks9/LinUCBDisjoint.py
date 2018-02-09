import numpy as np
from numpy import linalg as la

M_x = {}
M_x_1 = {} 
b_x = {}
w_hat_x = {}
UCB_x = {}

d = 6

z_t = np.empty([d,1], np.float)
last_rec = 0
count = 0

arts = {}

#best alpha = 0.001
alpha = 0.01
reward0 = -3.0
reward1 = 11.9
#print("alpha={0}".format(alpha))

def set_articles(articles):
    global arts, d
    for art in articles:
        arts[art] = np.array(articles[art]).reshape((d,1))
        
    #print(arts)


def update(reward):
    global M_x, M_x_1, b_x, w_hat_x, UCB_x, z_t, last_rec, count, arts, d
    #print('reward in update={0}'.format(reward))
    if reward != -1:
        count += 1
        #print('last_rec in update={0}'.format(count))
        M_x[last_rec] = np.add( M_x[last_rec], z_t.dot( np.transpose(z_t) ) )
        M_x_1[last_rec] = la.pinv(M_x[last_rec])
        if reward == 0:
            b_x[last_rec] = np.add( b_x[last_rec], z_t*reward0 )
        else:
            b_x[last_rec] = np.add( b_x[last_rec], z_t*reward1 )


def recommend(time, user_features, choices):
    global M_x, M_x_1, b_x, w_hat_x, UCB_x, z_t, last_rec, arts, alpha, d

    A_t = choices
    max_ucb = -1.0
    arg_max_ucb = -1
    z_t = np.array(user_features).reshape((d,1))
    for x in A_t:
        # If new article
        if x not in M_x:
            M_x[x] = np.identity(d, np.float)
            M_x_1[x] = np.identity(d, np.float) 
            b_x[x] = np.zeros((d,1), np.float)
            w_hat_x[x] = np.zeros((d,1), np.float)
            UCB_x[x] = 0.0
        
        # set weight
        w_hat_x[x] = np.dot(M_x_1[x], b_x[x])
        
        #delta = 3.142*3.142/(time^2 * 6)
        #alpha = 1 + ( np.sqrt( 0.5 * np.log(2/delta) ) )
        # set upper confidence bound
        UCB_x[x] = np.add( np.transpose(w_hat_x[x]).dot( z_t ),
                            alpha*( np.sqrt( np.transpose( z_t ).dot(M_x_1[x]).dot(z_t)) ) 
                        )
                        
        if UCB_x[x] > max_ucb:
            max_ucb = UCB_x[x]
            arg_max_ucb = x
        
    #print(len(A_t))
    #print(b_x)
    last_rec = arg_max_ucb
    #print('last_rec in recommend={0}'.format(last_rec))
    return arg_max_ucb
