# class for CNN/RNN model on MNIST dataset 
import tensorflow as tf 
import numpy as np 
import functools 
# dropout???  # early stopping???  
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
class NN(object):
    def __init__(self,input_size,output_size,hidden_units=32,fully_connected_units=64):
        # initializer
        self.initializer = tf.contrib.layers.xavier_initializer()
        # data
        self.x,self.y_ = get_placeholders(input_size,output_size)
        
        # model structure
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units

        # model outputs
        self.logits
        self.loss

        # save variables from past task in star_vars 
        self.vars = [self.w0,self.b0]
        #for variable in tf.trainable_variables():
           # self.vars.append(variable)
        self.star_vars = []
        self.star_vars_vec = []
        return

    @lazy_property
    def logits(self):
  
        # linear layer + l2 loss
        self.w0 = tf.Variable(self.initializer([self.input_size,self.hidden_units]))
        self.b0 = tf.Variable(self.initializer([self.hidden_units]))
        self.w1 = tf.Variable(self.initializer([self.hidden_units,self.output_size]))
        self.b1 = tf.Variable(self.initializer([self.output_size]))
        logits = self.feed_forward(self.x)
        
        self.train_variables_list = [self.w1,self.b1]

        self.num_paras = [self.input_size*self.hidden_units+self.hidden_units,self.hidden_units*self.output_size+self.output_size]


        return logits


    @lazy_property
    def loss(self):
		# loss function on task T
        return tf.reduce_mean(tf.losses.mean_squared_error(labels=self.y_,predictions=self.logits))


    # ease notation for forward calculation
    def feed_forward(self,xx):
        # h0 = tf.nn.relu(tf.matmul(xx,self.w0)+self.b0)
        h0 = tf.matmul(xx,self.w0)+self.b0  # 没加relu
        logits = tf.matmul(h0,self.w1)+self.b1
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
        

    def set_up_fisher(self,fisher_size,empirical=True):
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
            loss_per_sample =  tf.reduce_mean(tf.losses.mean_squared_error(labels=yy_,predictions=self.feed_forward(xx)))
            # calculate the gradients of this sample
            grads_per_sample = tf.gradients(loss_per_sample,self.vars) 
  
            if i>0:
                f_cache_tmp = []
                for v in range(len(self.vars)):
                    self.f_diag_cache[v] += tf.square(grads_per_sample[v])
                    self.f_block_cache[v] += tf.matmul(tf.reshape(grads_per_sample[v], (-1,1)), tf.reshape(grads_per_sample[v], (1,-1)))
                    f_cache_tmp.append(tf.reshape(grads_per_sample[v],(-1,1)))
                vec = tf.concat(f_cache_tmp,0)
                self.f_cache += tf.matmul(vec,vec,transpose_b=True)
            else:
                self.f_diag_cache = []
                self.f_block_cache = []
                f_cache_tmp = []
                for v in range(len(self.vars)):
                    self.f_diag_cache.append(tf.square(grads_per_sample[v]))
                    self.f_block_cache.append(tf.matmul(tf.reshape(grads_per_sample[v], (-1,1)), tf.reshape(grads_per_sample[v], (1,-1))))
                    f_cache_tmp.append(tf.reshape(grads_per_sample[v],(-1,1)))
                vec = tf.concat(f_cache_tmp,0)
                self.f_cache = tf.matmul(vec,vec,transpose_b=True)

        # divide by fisher size
        for v in range(len(self.vars)):
            self.f_diag_cache[v] /= fisher_size
            self.f_block_cache[v] /= fisher_size
        self.f_cache /= fisher_size
        
        self.f_diag = []
        self.f_block = []
        self.f = None
        return

    def compute_fisher(self,sess,fisher_x,fisher_y,fisher_diagonal=True,fisher_block=False,fisher_true=False):
        tmp,tmp2,tmp3 = [],[],[]
                
        if fisher_diagonal:
            for v in range(len(self.vars)):
                tmp.append(tf.Variable(self.f_diag_cache[v],trainable=False))
        if fisher_block:
            for v in range(len(self.vars)):
                tmp2.append(tf.Variable(self.f_block_cache[v],trainable=False))	
        if fisher_true:
            tmp3.append(tf.Variable(self.f_cache,trainable=False))	
            #sess.run(tf.variables_initializer([tmp3]),feed_dict={self.x:fisher_x,self.y_:fisher_y})

        tmp_list = tmp+tmp2+tmp3
        
        # initialize these values fisher data
        sess.run(tf.variables_initializer(tmp_list),feed_dict={self.x:fisher_x,self.y_:fisher_y})  

        # add fisher of current task to previous fisher
        if self.f_diag:
            for v in range(len(tmp)):
                self.f_diag[v]+=tmp[v]
        else:
            for v in range(len(tmp)):
                self.f_diag.append(tmp[v])

        if self.f_block:
            for v in range(len(tmp2)):
                self.f_block[v] += tmp2[v]  
        else:
            for v in range(len(tmp2)):
                self.f_block.append(tmp2[v])

        if self.f is not None:
            if len(tmp3)>0:
                self.f += tmp3[0]
        else:
            if len(tmp3)>0:
                self.f = tmp3[0]

        return

    # calculate EWC loss: entropy and EWC penalty
    def update_ewc_loss(self,fisher_multiplier,learning_rate,loss_option=0,freeze=False):
        # loss_option:
        # 0: without EWC penalty 1: with diagonal penalty 2: with block penalty 3: with full penalty
        self.penalty = tf.reduce_sum(tf.zeros([1],tf.float32))

        if self.star_vars:
            tmp_vars =  [] 
            for v in range(len(self.vars)):
                # diagonal fisher
                if loss_option==1:
                    self.penalty += tf.reduce_sum(tf.multiply(self.f_diag[v],tf.square(tf.subtract(self.vars[v],self.star_vars[v]))))
                # block fisher
                if loss_option==2:
                    difference = tf.reshape(tf.subtract(self.vars[v],self.star_vars[v]),(-1,1))
                    self.penalty += tf.reduce_sum(tf.matmul(tf.transpose(difference),tf.matmul(self.f_block[v],difference)))

                # full fisher
                if loss_option==3:
                    tmp_vars.append(tf.reshape(tf.subtract(self.star_vars[v],self.vars[v]),(-1,1)))
            if loss_option==3:
                difference  = tf.reshape(tf.concat(tmp_vars,0),(-1,1))
                self.penalty = tf.reduce_sum(tf.matmul(tf.transpose(difference),tf.matmul(self.f,difference)))

        # total loss
        self.ewc_loss = self.loss +  (fisher_multiplier / 2) * self.penalty

        # train the model
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)        
        
        if freeze==False:
            self.gvs = optimizer.compute_gradients(self.ewc_loss)
        else:
            self.gvs = optimizer.compute_gradients(self.ewc_loss,var_list = self.train_variables_list)

        # clip the gradients in [-1,1]
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
        self.train_step = optimizer.apply_gradients(capped_gvs)
        return 
        
    def train(self,sess,batch_x,batch_y): 
        sess.run(self.train_step,feed_dict={self.x:batch_x,self.y_:batch_y})
        return 


    def get_fisher(self):
        return self.f.eval()

    def test(self,sess,batch_x,batch_y): 
        loss = sess.run(self.loss,feed_dict={self.x:batch_x,self.y_:batch_y})
        return loss


    def get_base(self):
        return self.w0.eval()

