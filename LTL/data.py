import numpy as np
from scipy.linalg import qr
import tensorflow as tf

RANDOM = 2018
sigma = 0.0

class task(object):
    def __init__(self,x,y):
        self.data_x = x
        self.data_y = y
    
    def get_batch(self,batch_size=None):
        if batch_size is None:
            batch_size = len(self.data_x)
        # shuffle the data    
        idx = np.arange(0 , len(self.data_x))
        np.random.shuffle(idx)
        idx = idx[:batch_size]
        # fetch batch_size data
        data_shuffle = [self.data_x[i] for i in idx]
        labels_shuffle = [self.data_y[i] for i in idx]
        return np.asarray(data_shuffle), np.asarray(labels_shuffle)    

# generate data from unit sphere
def sample_spherical(npoints, ndim):
    vec = np.random.randn(ndim,npoints)
    vec /= np.linalg.norm(vec, axis=0)   # check norm ！！！
    return vec.T

def relu(arr):
    return arr*(arr>0)


def generate_data(num_tasks,samples_per_task,dim):
    
    print("Generating data...")
    base_dim = int(dim/2)
    # generate basi representation P
    np.random.seed(seed=RANDOM)
    H = np.random.randn(dim,dim)
    Q, R = qr(H)
    # select dim/2 columns as base representation: already orthonormal
    P = Q[:dim,:base_dim]
    print("Base P is:{}".format(P[:1,:3]))

    # save all the tasks
    my_task = []
    for t in range(num_tasks):
        
        # generate x: flexible
        x = sample_spherical(samples_per_task,dim)
        h = np.matmul(x,P)
        # generate w_hat
        np.random.seed(seed=RANDOM+t)
        w_hat = np.random.randn(base_dim,1)
        w_hat = w_hat/np.sqrt(np.dot(w_hat.T,w_hat))
        # generate y
        y = np.matmul(h,w_hat) + sigma*np.random.randn(samples_per_task,1)
#        # generate w
#        np.random.seed(seed=RANDOM+t)
#        w_hat = np.random.randn(base_dim,1)
#        w_hat = w_hat/np.sqrt(np.dot(w_hat.T,w_hat))
#        w = np.matmul(P,w_hat)
#
#        # generate x: flexible
#        x = sample_spherical(samples_per_task,dim)
#
#        # generate y: flexible
#        y = np.matmul(x,w) + sigma*np.random.randn(samples_per_task,1)
#
        # create task
        my_task.append(task(x,y))
    
    return my_task,P

def generate_test_data(num_tasks,samples_per_task,dim):
    
    print("Generating test data...")
    base_dim = int(dim/2)
    # generate basi representation P
    np.random.seed(seed=RANDOM)
    H = np.random.randn(dim,dim)
    Q, R = qr(H)
    # select dim/2 columns as base representation: already orthonormal
    P = Q[:dim,:base_dim]
    print("Base P is:{}".format(P[:1,:3]))

    # save all the tasks
    my_task = []
    # create train and test for meta learning
    for i in range(2):
        x_all,y_all = [],[]
        for t in range(num_tasks):
            
            # generate x: flexible
            x = sample_spherical(samples_per_task,dim)
            x_all.append(x)
            h = np.matmul(x,P)
            # generate w_hat
            np.random.seed(seed=RANDOM+t)
            w_hat = np.random.randn(base_dim,1)
            w_hat = w_hat/np.sqrt(np.dot(w_hat.T,w_hat))
            # generate y
            y = np.matmul(h,w_hat) + sigma*np.random.randn(samples_per_task,1)
            y_all.append(y)
    #        # generate w
    #        np.random.seed(seed=RANDOM+t)
    #        w_hat = np.random.randn(base_dim,1)
    #        w_hat = w_hat/np.sqrt(np.dot(w_hat.T,w_hat))
    #        w = np.matmul(P,w_hat)
    #
    #        # generate x: flexible
    #        x = sample_spherical(samples_per_task,dim)
    #
    #        # generate y: flexible
    #        y = np.matmul(x,w) + sigma*np.random.randn(samples_per_task,1)
    #
            # create task
        xx = np.concatenate(x_all)
        yy = np.concatenate(y_all)
        my_task.append(task(xx,yy))
    
    return my_task,P
