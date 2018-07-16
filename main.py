import tensorflow as tf
from MNIST_model import NN_MNIST
import data
from experiment import experiment


tf.reset_default_graph()

# number of tasks 
num_tasks = 3
# Parameters
batch_size = 256
iterations = 12*55000/batch_size
log_period_updates = 20
# num of samples to calculate fisher information matrix
fisher_size = 200

# experiment 1: CNN on permuted MNIST
#learning_rate = 0.1
#fisher_multiplier = 20
#sampler = 1
#input_size = 784
#output_size = 10
#hidden_units = 100
#fully_connected_units = 64
#data = data.generate_permuted_MNIST(num_tasks,sampler)
#model_name = "CNN"
#model = NN_MNIST(model_name,learning_rate,fisher_multiplier,input_size,output_size,hidden_units,fully_connected_units)

# experiment 2: CNN on split MNIST

# experiment 3: RNN on permuted MNIST
learning_rate = 0.1
fisher_multiplier = 20
sampler = 4
input_size = 49
output_size = 10
hidden_units = 32
fully_connected_units = 32
time_step = 7
features_per_step = 7
data = data.generate_permuted_MNIST(num_tasks,sampler)
model_name = "RNN"
model = NN_MNIST(model_name,learning_rate,fisher_multiplier,input_size,output_size,hidden_units,fully_connected_units,time_step,features_per_step)
# experiment 4: RNN on generated tasks
def main():
    
    experiment(model,data[0],data[1],num_tasks,iterations,batch_size,log_period_updates,fisher_size)

    print(tf.trainable_variables())
    return


if __name__ == "__main__":
    main()

