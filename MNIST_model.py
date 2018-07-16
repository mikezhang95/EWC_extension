# class for CNN/RNN model on MNIST dataset
import tensorflow as tf
import numpy as np
import functools
# dropout???
# early stopping???

# input
def get_placeholders(input_size,output_size):
    x= tf.placeholder(tf.float32, [None, input_size])
    y_ = tf.placeholder(tf.float32, [None, output_size])
    return x, y_
	
# ease property function
def lazy_property(function):
    attribute = '_cache_' + function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator

# definition
class NN_MNIST(object):
    def __init__(self,name,learning_rate,fisher_multiplier,input_size,output_size,hidden_units=32,fully_connected_units=64,time_step=28,features_per_step=28):
        # initializer
        self.initializer = tf.contrib.layers.xavier_initializer()
        # data
        self.x,self.y_ = get_placeholders(input_size,output_size)
        # model type: RNN or CNN
        self.name = name
        
        # model structure
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.fully_connected_units = fully_connected_units
        self.time_step = time_step
        self.features_per_step = features_per_step
        self.learning_rate = learning_rate
        self.fisher_multiplier = fisher_multiplier

        # model outputs
        self.logits
        self.predictions
        self.cross_entropy
        self.accuracy

        # save variables from current task in vars
        # save variables from past task in star_vars 
        self.vars = []
        for variable in tf.trainable_variables():
            self.vars.append(variable)
        self.star_vars = []
        self.star_vars_vec = []
        return

    @lazy_property
    def logits(self):
  
        if self.name == "CNN":
            # linear layer + softmax & loss
            self.w1 = tf.Variable(self.initializer([self.input_size,self.fully_connected_units]))
            self.b1 = tf.Variable(self.initializer([self.fully_connected_units]))
            self.w2 = tf.Variable(self.initializer([self.fully_connected_units,self.output_size]))
            self.b2 = tf.Variable(self.initializer([self.output_size]))
            logits = self.feed_forward(self.x)

        else:
			# create LSTM/GRU with n_hidden units.
            self.rnn_cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_units)

            self.w1 = tf.Variable(self.initializer([self.hidden_units,self.fully_connected_units]))
            self.b1 = tf.Variable(self.initializer([self.fully_connected_units]))
            self.w2 = tf.Variable(self.initializer([self.fully_connected_units,self.output_size]))
            self.b2 = tf.Variable(self.initializer([self.output_size]))
            
            logits = self.feed_forward(self.x)

        return logits


    @lazy_property
    def predictions(self):
        return tf.nn.softmax(self.logits)

    @lazy_property
    def cross_entropy(self):
		# loss function on task T
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_,logits=self.logits))

    @lazy_property
    def accuracy(self):
		# evalutaion
        correct_prediction = tf.equal(tf.argmax(self.y_,1),tf.argmax(self.predictions,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def calculate_entropy(self,probs):
        entropy = tf.multiply(-probs,tf.log(probs))
        return tf.reduce_mean(tf.reduce_sum(entropy,1))

    # ease notation for forward calculation
    def feed_forward(self,xx):
        if self.name == "CNN":
            h1 = tf.nn.relu(tf.matmul(xx,self.w1)+self.b1)
            logits = tf.matmul(h1,self.w2)+self.b2
        else:
            inputs = tf.reshape(xx, [-1,self.time_step,self.features_per_step])
            # rnn outputs
            outputs, state = tf.nn.dynamic_rnn(self.rnn_cell,inputs,dtype=tf.float32)
            output = tf.reshape(outputs[:,-1,:],(-1,self.hidden_units))
            h0 = tf.nn.relu(output)
            h1 = tf.nn.relu(tf.matmul(h0,self.w1)+self.b1)
            logits = tf.matmul(h1,self.w2)+self.b2
        return logits

    # save the optimal weights after most recent task training
    def star(self):
        self.star_vars = []
        for v in range(len(self.vars)):
            self.star_vars.append(self.vars[v].eval())
        tmp = []
        for v in range(len(self.vars)):
            tmp.append(np.reshape(self.vars[v].eval(),(-1,1)))
        self.star_vars_vec = np.concatenate(tmp,0)
        return
        

    def set_up_fisher(self,fisher_size,empirical=False):
        # iterate the fisher samples
        for i in range(fisher_size):
            xx = tf.reshape(self.x[i:i+1,:], [-1, self.input_size])
  
            if empirical:
                # empirical fisher, true labels
                yy_ = self.y_[i:i+1,:]
            else:
                # true fisher, draw only one sample!
                cla = tf.reshape(tf.stop_gradient(tf.to_int32(tf.multinomial(self.feed_forward(xx), 1))),[-1])
                yy_ = tf.one_hot(cla,self.output_size)
            # calculate the loss for a single sample
            loss_per_sample = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yy_,logits=self.feed_forward(xx)))
            # calculate the gradients of this sample
            grads_per_sample = tf.gradients(loss_per_sample,self.vars) 
  
            if i>0:
                f_cache_tmp = []
                for v in range(len(self.vars)):
                    self.f_diag_cache[v] += tf.square(grads_per_sample[v])
                    self.f_block_cache[v] += tf.matmul(tf.reshape(grads_per_sample[v], (-1,1)), tf.reshape(grads_per_sample[v], (1,-1)))
                    f_cache_tmp.append(tf.reshape(grads_per_sample[v],(-1,1)))
                self.f_cache += tf.concat(f_cache_tmp,0)
            else:
                self.f_diag_cache = []
                self.f_block_cache = []
                f_cache_tmp = []
                for v in range(len(self.vars)):
                    self.f_diag_cache.append(tf.square(grads_per_sample[v]))
                    self.f_block_cache.append(tf.matmul(tf.reshape(grads_per_sample[v], (-1,1)), tf.reshape(grads_per_sample[v], (1,-1))))
                    f_cache_tmp.append(tf.reshape(grads_per_sample[v],(-1,1)))
                self.f_cache = tf.concat(f_cache_tmp,0)

        # divide by fisher size
        for v in range(len(self.vars)):
            self.f_diag_cache[v] /= fisher_size
            self.f_block_cache[v] /= fisher_size
                
        self.f_diag = []
        self.f_block = []
        
        self.f_cache /= fisher_size
        self.f = []
        return

    def compute_fisher(self,sess,fisher_x,fisher_y,fisher_diagonal=True,fisher_block=False,fisher_true=False):
        tmp,tmp2 = [],[]
    def compute_fisher(self,sess,fisher_x,fisher_y,fisher_diagonal=True,fisher_block=False,fisher_true=False):
        tmp,tmp2 = [],[]
        if fisher_diagonal:
            for v in range(len(self.vars)):
                tmp.append(tf.Variable(self.f_diag_cache[v],trainable=False))
        if fisher_block:
            for v in range(len(self.vars)):
                tmp2.append(tf.Variable(self.f_block_cache[v],trainable=False))	
        
        if fisher_block:
            tmp3 = tf.Variable(self.f_cache,trainable=False)	
            sess.run(tf.variables_initializer(tmp3),feed_dict={self.x:fisher_x,slef.y_:fisher_y})

            if len(self.f)>0:
                self.f += tmp3
            else:
                slef.f = tmp3

        sess.run((tf.variables_initializer(tmp),tf.variables_initializer(tmp2)),feed_dict={self.x:fisher_x,self.y_:fisher_y})  

        if len(self.f_diag)>0:
            for v in range(len(tmp)):
                self.f_diag[v]+=tmp[v]
        else:
            for v in range(len(tmp)):
                self.f_diag.append(tmp[v])

        if len(self.f_block)>0:
            for v in range(len(tmp2)):
                self.f_block[v] += tmp2[v]  
            for v in range(len(tmp2)):
                self.f_block.append(tmp2[v])
        return

    # calculate EWC loss: entropy and EWC penalty
    def update_ewc_loss(self):
        self.penalty = tf.reduce_sum(tf.zeros([1],tf.float32))
        # self.penalty2 = tf.reduce_sum(tf.zeros([1],tf.float32))
        if self.star_vars:
            tmp_vars =  [] 
            for v in range(len(self.vars)):
                self.penalty += tf.reduce_sum(tf.multiply(self.f_diag[v],tf.square(tf.subtract(self.vars[v],self.star_vars[v]))))
               # difference = tf.reshape(tf.subtract(self.vars[v],self.star_vars[v]),(-1,1))
                #self.penalty2 += tf.reduce_sum(tf.matmul(tf.transpose(difference),tf.matmul(self.f_block[v],difference)))
               # tmp_vars.append(tf.reshape(self.vars[v],(-1,1)))
           # vars_vec = tf.concat(tmp_vars,0)
           # differnece2 = self.star_vars_vec - vars_vec
           # self.penalty3 = tf.reduce_sum(tf.matmul(difference2.transpose(),tf.matmul(self.f,difference2)))
        # total loss
        self.ewc_loss = self.cross_entropy +  (self.fisher_multiplier / 2) * self.penalty
        # train the model
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        gvs = optimizer.compute_gradients(self.ewc_loss)
        # clip the gradients in [-1,1]
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_step = optimizer.apply_gradients(capped_gvs)
        return

    def train(self,sess,batch_x,batch_y):
        sess.run(self.train_step,feed_dict={self.x:batch_x,self.y_:batch_y})        
        print(sess.run((self.ewc_loss,self.cross_entropy),feed_dict={self.x:batch_x,self.y_:batch_y}))
        return 

    def test(self,sess,test_x,test_y):
        return sess.run(self.accuracy,feed_dict={self.x:test_x,self.y_:test_y})

