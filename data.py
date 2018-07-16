import numpy as np
import tensorflow as tf

def down_sampling(dataset,image_h,image_w,sampler=2):
	# sampler: sample for every sampler pixel
	shape = np.shape(dataset.images)
	images = np.reshape(dataset.images,(shape[0],image_h,image_w))
	return (np.reshape(images[:,::sampler,::sampler],(shape[0],-1)),dataset.labels)

# Import MNIST dataset with one-hot encoding of the class labels.
def get_data():
	from tensorflow.examples.tutorials.mnist import input_data
	return input_data.read_data_sets("MNIST_data/", one_hot=True)


class task(object):
    def __init__(self,name,data,index):
        self.data_x = data[0]
        self.data_y = data[1]
        self.name = name
        self.index = index
    
    def get_batch(self,no,batch_size=None):
        if batch_size is None:
            batch_size = self.data_x.shape[0]

        # return permuted MNIST dataset
        if self.name  == "permuted":
            select = np.random.randint(0,self.data_x.shape[0], size=(batch_size))
            p = self.index[no]
            x_select = self.data_x[select,:]
            return x_select[:,p],self.data_y[select,:]
        # return split MNIST dataset
        else:
            index = self.index[no]
            select = np.random.randint(0,index.shape[0], size=(batch_size))
            return self.data_x[index[select],:],self.data_y[index[select],:]

def generate_permuted_MNIST(num_tasks,sampler):

    mnist_raw = get_data()
    mnist_train = down_sampling(mnist_raw.train,28,28,sampler)
    mnist_test = down_sampling(mnist_raw.test,28,28,sampler)

    task_permutation = []
    input_size = mnist_train[0].shape[1]
    for tt in range(num_tasks):
        task_permutation.append(np.random.permutation(input_size))
    return (task("permuted",mnist_train,task_permutation),task("permuted",mnist_test,task_permutation))
	
def generate_split_MNIST(num_tasks,sampler):

	mnist_raw = get_data()
	mnist_train = down_sampling(mnist_raw.train,28,28,sampler)
	mnist_test = down_sampling(mnist_raw.test,28,28,sampler)
	
	train_index =[]
	for i in range(num_tasks):
		location = np.where(mnist_train[1][:,int(10/num_tasks)*i:int(10/num_tasks)*i+int(10/num_tasks)] == 1)
		train_index.append(location[0])

	test_index = [] 
	for i in range(num_tasks):
		location = np.where(mnist_test[1][:,int(10/num_tasks)*i:int(10/num_tasks)*i+int(10/num_tasks)] == 1)
		test_index.append(location[0])
		
	return (task("split",mnist_train,train_index),task("split",mnist_test,test_index))

