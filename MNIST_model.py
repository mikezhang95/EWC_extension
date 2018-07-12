# class for CNN/RNN model on MNIST dataset
import tensorflow as tf
import numpy as np

# dropout???
# early stopping???

# parametrs for the model
input_size = 784
output_size = 10
hidden_units = 32
fully_connected_units = 64
time_step = 28
features_per_step = 28

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

# input
def get_placeholders():
	x = tf.placeholder(tf.float32, [None, input_size])
	y_ = tf.placeholder(tf.float32, [None, output_size])
	return x, y_

# definition
class NN_MNIST(object):
    def __init__(self,x,y_,name):
        # initializer
        self.initializer = tf.contrib.layers.xavier_initializer()
        # data
        self.x = x
        self.y_ = y_
        # model type: RNN or CNN
        self.name = name
        
        # model structure
        self.hidden_units = hidden_units
        self.fully_connected_units = fully_connected_units
        
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
			inputs = tf.reshape(xx, [-1,time_step,features_per_step])
			# rnn outputs
			outputs, state = tf.nn.dynamic_rnn(self.rnn_cell,inputs,dtype=tf.float32)
			output = tf.reshape(outputs[:,-1,:],[-1,self.hidden_units))
			h0 = tf.nn.relu(output)
			h1 = tf.nn.relu(tf.matmul(h0,self.w1)+self.b1)
			logits = tf.matmul(h1,self.w2)+self.b2
	return logits

    # save the optimal weights after most recent task training
    def star(self):
		self.star_vars = []
		for v in range(len(self.vars)):
			self.star_vars.append(self.vars[v].eval())
    return

    def set_up_fisher(self,diagonal_fisher=True, block_fisher=True,empirical=False):
  
        self.f_diag_cache = None
        self.f_block_cache = None
  
        if diagonal_fisher:
			# initialize the dianonal of Fisher Information Matrix
        	self.f_diag_cache = []
        	for v in range(len(self.vars)):
				self.f_diag_cache.append(tf.zeros_like(self.vars[v]))
  
        	self.f_diag = []
        	for v in range(len(self.vars)):
				self.f_diag.append(tf.Variable(self.f_diag_cache[v],trainable=False))
  
        if block_fisher:
        	# initialize the dianonal of Fisher Information Matrix
        	self.f_block_cache = []
        	for v in range(len(self.vars)):
				a = tf.reshape(self.vars[v],[-1,1])
				self.f_block_cache.append(tf.zeros((a.shape[0],a.shape[0])))
  
        	self.f_block = []
        	for v in range(len(self.vars)):
				self.f_block.append(tf.Variable(self.f_block_cache[v],trainable=False))
  
        # iterate the fisher samples
        for i in range(fisher_size):
			xx = tf.reshape(self.x[i:i+1,:], [-1, input_size])
  
			if empirical:
				# empirical fisher, true labels
				yy_ = self.y_[i,:]
			else:
				# true fisher, draw only one sample!
				cla = tf.stop_gradient(tf.to_int32(tf.multinomial(self.feed_forward(xx), 1)))
				yy_ = tf.one_hot(cla,output_size)
  
			# calculate the loss for a single sample
			loss_per_sample = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=yy_,logits=self.feed_forward(xx)))
			# calculate the gradients of this sample
			grads_per_sample = tf.gradients(loss_per_sample,self.vars) 
  
			for v in range(len(self.vars)):
				if self.f_diag_cache is not None:
					self.f_diag_cache[v] += tf.square(grads_per_sample[v])
				if self.f_block_cache is not None:
					self.f_block_cache[v] += tf.matmul(tf.reshape(grads_per_sample[v], (-1,1)), tf.reshape(grads_per_sample[v], (1,-1)))
        # divide by fisher size
        for v in range(len(self.vars)):
			if self.f_diag_cache is not None:
				self.f_diag_cache[v] /= fisher_size
        	if self.f_block_cache is not None:
        		self.f_block_cache[v] /= fisher_size
        return

    def compute_fisher(self,sess,fisher_x,fisher_y):
		tmp,tmp2 = []. []
		if self.f_diag_cache is not None:
			for v in range(len(self.vars)):
				tmp.append(tf.Variable(self.f_diag_cache[v],trainable=False))
		if self.f_block_cache is not None:
			for v in range(len(self.vars)):
				tmp2.append(tf.Variable(self.f_block_cache[v],trainable=False))	

      	sess.run((tf.initializers.variables(tmp),tf.initializers.variables(tmp2)),feed_dict={x:fisher_x,y:fisher_y})  

      	if len(tmp)>0:
      		for v in range(len(self.vars)):
				self.f_diag[v]+=tmp[v]
      	if len(tmp2)>0:
      		for v in range(len(self.vars)):
				self.f_block[v] += tmp2[v]  
		return

    # calculate EWC loss
    # update at the beginning of the task
    def update_ewc_loss(self):
		self.penalty = tf.reduce_sum(tf.zeros([1],tf.float32))
		self.penalty2 = tf.reduce_sum(tf.zeros([1],tf.float32))
		if self.star_vars:
			print("Consider EWC")
			for v in range(len(self.vars)):
				self.penalty += tf.reduce_sum(tf.multiply(self.f_diag[v],tf.square(tf.subtract(self.vars[v],self.star_vars[v]))))
				difference = tf.reshape(tf.subtract(self.vars[v],self.star_vars[v]),(-1,1))
				self.penalty2 += tf.reduce_sum(tf.matmul(tf.transpose(difference),tf.matmul(self.f_block[v],difference)))
		# total loss
		self.ewc_loss = self.cross_entropy +  (fisher_multiplier / 2) * self.penalty2
		# train the model
		self.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.ewc_loss)

		return

