import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt


# evaluation functions
def plot_learning_curves(result):
	# input： list of test performance on different tasks
	num_tasks = len(result)
	iters = len(result[0])
	for t in range(num_tasks):
		n = len(result[t])
		x = [iters-n+i for i in range(n)]
		y = result[t]
		plt.plot(x,y)
		plt.ylim(0,1)

	plt.title("Test_Accuracy on %d tasks"%num_tasks)
	plt.show()
	return

# evaluation functions
def plot_grads(result):
	# input：	
	plt.plot(result)
	#   plt.ylim(0,1)
	plt.title("mean abs grads of last layer")
	plt.show()
	return
	

def experiment(model,train_data,test_data,num_tasks,iterations,batch_size,log_period_updates,fisher_size):

    foo = open("./cache/test_accuracy.txt", "w")

    model.set_up_fisher(fisher_size)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    # save the test accuracy for tasks
    test_result = []
    for t in range(num_tasks):
        test_task =[]
        test_result.append(test_task)
    
	
    for t_tr in range(num_tasks):
	
        print("--->>> Training on task %d"%t_tr)
	
        foo.write("--->>> Training on task %d\n"%t_tr)
        # basic set up at the beginning of the task
        iter = 0
        epoch = 0
        model.update_ewc_loss()
	
        while iter < iterations:

            iter += 1
            # train batch
            batch_x,batch_y = train_data.get_batch(t_tr,batch_size)
            model.train(sess,batch_x,batch_y) 
            # evaluate every 20 iters
            if iter%log_period_updates == 0:
                time.sleep(0.1)
                print("=== {} iterations finished ====".format(iter),end=" ")
                foo.write("=== %d iterations finished ====\n"%(iter))	

                # print(sess.run(model.f_diag_cache[0][200,20],feed_dict={x:batch_x,y_:batch_y}))
                # print(sess.run(model.f_diag[0][200,20]))
      		
                for t_te in range(t_tr+1):
                    # Compute test accuracy	
                    test_x,test_y = test_data.get_batch(t_te)
                    test_accuracy = model.test(sess,test_x,test_y)
                    # Display the results
                    if t_te == t_tr:
                        print("task %d accuracy %f"%(t_te,test_accuracy),end="\r")
                    else:
                        print("task %d accuracy %f"%(t_te,test_accuracy),end=" | ")
                    test_result[t_te].append(test_accuracy)
                    foo.write("task %d accuracy %f\n"%(t_te,test_accuracy)) 
        # compute fisher matrix
        print(" ")
        print("Computing fisher of task{}".format(t_tr))
        fisher_x,fisher_y = train_data.get_batch(t_tr,fisher_size)
        model.compute_fisher(sess,fisher_x,fisher_y)
        # save the parameters of last task
        model.star()
    
    print("Saving the file")
    plot_learning_curves(test_result)
    foo.close()
    return 
