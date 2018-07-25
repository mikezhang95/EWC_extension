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

    for t in range(1,num_tasks):
        n = len(result[t])
        plt.vlines(iters-n,0,1,colors = "c", linestyles = "dashed")

    plt.title("Test_Accuracy on %d tasks"%num_tasks)
    plt.savefig("./cache/test_accuracy")
    #plt.show()
    plt.close()
    return

# evaluation functions
def plot_what(result,name,iters,save=False):
    # input：
    plt.title(name)
    for r in range(result.shape[1]):
        plt.plot(result[:,r])
      
    max_value = np.amax(result)  
    plt.ylim(-max_value,max_value)

    print(iters)
    for t in range(len(iters)-1):
        plt.vlines(iters[t],-max_value,max_value,colors = "c", linestyles = "dashed")

    if save:
        plt.savefig("./cache/"+name)
    # plt.show()
    plt.close()
    return
	
def draw_fisher(f,task_id,iters,save=True):
    absf = abs(f)
    plt.imshow(absf,vmin=np.amin(absf),vmax = np.amax(absf)/2.,cmap="gray_r")

    print(iters)
    num_paras = f.shape[0]
    for t in range(len(iters)-1):
        plt.vlines(iters[t],0,num_paras,colors = "c", linestyles = "dashed")

    for t in range(len(iters)-1):
        plt.hlines(iters[t],0,num_paras,colors = "c", linestyles = "dashed")

    name = "FIM_task_"+str(task_id) 
    plt.title(name)
    if save:
        plt.savefig("./cache/" + name)
 #   plt.show()
    plt.close()
    return

def experiment(fisher_multiplier,learning_rate,model,train_data,test_data,num_tasks,iterations,batch_size,log_period_updates,fisher_size,loss_option=0,fisher_true=False,fisher_block=False,fisher_diagonal=False,verbose=True):

    if verbose:
        bias1= []
        bias2= []
        grads = []
        
    foo = open("./cache/test_accuracy.txt", "w")

    model.set_up_fisher(fisher_size,empirical=False)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())


    # save the test accuracy for tasks
    test_result = []
    iters = []
    for t in range(num_tasks):
        test_task =[]
        test_result.append(test_task)
    
	
    for t_tr in range(num_tasks):
	
        model.update_ewc_loss(fisher_multiplier,learning_rate,loss_option=loss_option)

        print("--->>> Training on task %d"%t_tr)
	
        foo.write("--->>> Training on task %d\n"%t_tr)
        # basic set up at the beginning of the task
        iter = 0
        epoch = 0
	
        while iter < iterations:

            iter += 1
            # train batch
            batch_x,batch_y = train_data.get_batch(t_tr,batch_size)
            model.train(sess,batch_x,batch_y,t_tr) 

            # evaluate every 20 iters
            if iter%log_period_updates == 0:
                time.sleep(0.1)
                print("=== {} iterations finished ====".format(iter),end=" ")
                foo.write("=== %d iterations finished ====\n"%(iter))	

                # print(sess.run(model.f_diag_cache[0][200,20],feed_dict={x:batch_x,y_:batch_y}))
                # print(sess.run(model.f_diag[0][200,20]))
      		
                for t_te in range(t_tr+1):
                    # Compute test accuracy	
                    test_x,test_y = test_data.get_batch(t_te,10000)
                    test_accuracy = model.test(sess,test_x,test_y)
                    # Display the results
                    if t_te == t_tr:
                        print("task %d accuracy %f"%(t_te,test_accuracy),end="\r")
                    else:
                        print("task %d accuracy %f"%(t_te,test_accuracy),end=" | ")
                    test_result[t_te].append(test_accuracy)
                    foo.write("task %d accuracy %f\n"%(t_te,test_accuracy)) 

                if verbose:
                    # print bias among different tasks
                    bias = model.get_bias()
                    bias1.append(bias[0])
                    bias2.append(bias[1])
                    grads.append(model.get_grads(sess,batch_x,batch_y))

        iters.append(iter/log_period_updates)
        # compute fisher matrix
        if t_tr<num_tasks-1:
            print("\n")
            print("Computing fisher of task {}\n".format(t_tr))
            fisher_x,fisher_y = train_data.get_batch(t_tr,fisher_size)
            model.compute_fisher(sess,fisher_x,fisher_y,fisher_diagonal=fisher_diagonal,fisher_block=fisher_block,fisher_true=fisher_true)
            if fisher_true:
                fim = model.get_fisher()
                draw_fisher(fim,t_tr,model.num_paras)


        # save the parameters of last task
        model.star()
    
    if verbose:
        plot_what(np.stack(bias1),"bias1",iters,save=True)
        plot_what(np.stack(bias2),"bias2",iters,save=True)
        # plot_what(np.stack(grads),"graditude of bias2",save=True)
    plot_learning_curves(test_result)
    foo.close()
    return test_result
