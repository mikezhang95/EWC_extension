import tensorflow as tf
from model import NN
import data
from experiment import experiment
import os

tf.reset_default_graph()
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'

# number of tasks 
num_tasks_train = 30
num_tasks_test = 20 # value the meta learner
# Parameters
samples_train = 200
samples_test = 200
batch_size = 50
iterations = 200*samples_train/batch_size
log_period_updates = 20

# num of samples to calculate fisher information matrix
fisher_size = 1
loss_option = 0
fisher_true = False
fisher_block = False
fisher_diagonal = False

# experiment 1: CNN on permuted MNIST
learning_rate = 100
fisher_multiplier = 10
input_size = 50
output_size = 1
hidden_units = 25
# generate data
train_data,P = data.generate_data(num_tasks_test,samples_train,input_size)
test_data,_ = data.generate_data(num_tasks_test,samples_test,input_size)
# generate model
model = NN(input_size,output_size,hidden_units=hidden_units)


def main():
    
    experiment(P,fisher_multiplier,learning_rate,model,train_data,test_data,num_tasks_train,num_tasks_test,iterations,batch_size,log_period_updates,fisher_size,loss_option=loss_option,fisher_true=fisher_true,fisher_block=fisher_block,fisher_diagonal=fisher_diagonal,verbose=True)

    print(tf.trainable_variables())
    return


if __name__ == "__main__":
    main()

