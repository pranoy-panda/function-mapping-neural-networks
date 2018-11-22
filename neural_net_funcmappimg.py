#building a neural network 
# architecture input(1x1)->hidden layer(size =10) with reLU activation-> (linear) output layer(1)
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def add_layer(inputs,in_size,out_size,activation_func = None):
    weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
    h1=tf.matmul(inputs,weights)+biases
    if activation_func==None:
        scores=h1
    else:
        scores=activation_func(h1)
    return scores

#real data
x_data=np.linspace(-1,1,500)[:, np.newaxis] #(-1,1,400) range of the values is from -1 to 1 and 400 linearly spaced
                                            #points is returned, np.newaxis makes the dim of x_data->(400,1) insted of (400,)
noise=np.random.normal(0,0.1,x_data.shape) # 0 is the mean and 0.05 is the standard deviation   
# uncomment below code to try out different non-linear functions
#y_data=np.square(x_data)+2.0
y_data = (x_data*x_data)*x_data +2.0*x_data**2 +1.2 +noise
# y_data = (x_data*x_data)*x_data +2.0*x_data**2 +4.5 *np.sin(x_data) + 1.2 +noise
#y_data = np.sqrt(x_data+1)
#y_data = np.sin(5*x_data)+1

# defining placeholders for inputs
xs=tf.placeholder(tf.float32,[None,1]) # None represents the no.of training expamples and 1 is the no. of features
ys=tf.placeholder(tf.float32,[None,1]) # None represents the no.of training expamples and 1 is the no. of output class or no.of outputs


# adding layers
layer1=add_layer(xs,1,40,activation_func=tf.nn.relu)
#layer2=add_layer(layer1,30,5,activation_func=tf.nn.sigmoid)
output=add_layer(layer1,40,1,activation_func=None)

# learning rate and num_iters
learning_rate=0.02
num_iters=1000

# loss calculation and optimzation of loss function and backprop to update the parameters
loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-output),reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

#initialize the variables
init=tf.global_variables_initializer()

# start the session
sess=tf.Session()

sess.run(init) # run the initialization operation

# first plotting real data
fig=plt.figure(1)
ax=fig.add_subplot(1,1,1) # 1subplot of 1 row and 1 col
ax.scatter(x_data,y_data)
plt.ion() 
plt.show()
plt.title('Function mapping using neural networks')
plt.legend('gt')
Loss= []

for i in range(num_iters):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    Loss.append(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
    if i%20==0:
        # visualize the prediction of the model
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        pred_val=sess.run(output,feed_dict={xs:x_data})
        print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        lines=ax.plot(x_data,pred_val,'r-',lw=5,label = 'estimate')
        ax.legend(['estimate','groundTruth'])        
        plt.pause(0.5)
fig= plt.subplot()
fig.plot(np.array(Loss),np.arange(num_iters))
fig.set_ylim(0,7)
plt.show()
