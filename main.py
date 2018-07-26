import tensorflow as tf
from MNIST_model import NN_MNIST
import data
from experiment import experiment
import os

tf.reset_default_graph()
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
# number of tasks 
num_tasks = 3
# Parameters
batch_size = 256
iterations = 12*55000/batch_size
log_period_updates = 20
# num of samples to calculate fisher information matrix
fisher_size = 1
loss_option = 0
fisher_true = False
fisher_block = False
fisher_diagonal = False

# experiment 1: CNN on permuted MNIST
learning_rate = 0.1
fisher_multiplier = 20 # 1000 2000 4000
sampler = 1
input_size = 784
output_size = 10
hidden_units = 16
fully_connected_units = 64
data = data.generate_permuted_MNIST(num_tasks,sampler)
model_name = "CNN"
model = NN_MNIST(model_name,input_size,output_size,hidden_units,fully_connected_units)

# experiment 2: CNN on split MNIST

# experiment 3: RNN on permuted MNIST
#learning_rate = 0.2
#fisher_multiplier = 20
#sampler = 4
#input_size = 49
#output_size = 10
#hidden_units = 16
#fully_connected_units = 16
#time_step = 7
#features_per_step = 7
#data = data.generate_permuted_MNIST(num_tasks,sampler)
#model_name = "RNN"
#model = NN_MNIST(model_name,input_size,output_size,hidden_units,fully_connected_units,time_step,features_per_step)

# experiment 4: RNN on generated tasks

def main():
    
    experiment(fisher_multiplier,learning_rate,model,data[0],data[1],num_tasks,iterations,batch_size,log_period_updates,fisher_size,loss_option=loss_option,fisher_true=fisher_true,fisher_block=fisher_block,fisher_diagonal=fisher_diagonal,verbose=True)

    print(tf.trainable_variables())
    return


if __name__ == "__main__":
    main()

