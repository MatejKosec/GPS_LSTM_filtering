import tensorflow as tf
import scipy as sp
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt

#%% Constants
N_TIME = 200
N_HIDDEN = 2
N_INPUT = 2
LR_BASE = 1e-1
BATCH_SIZE = 3
ITRS = 800

#%% Generate a sample
t = sp.linspace(0,10,N_TIME)
true_vx = sp.sin(t)
true_x  = cumtrapz(true_vx,t)
true_x  = sp.hstack([[0],true_x])

noisy_vx = sp.sin(t+sp.random.rand(*t.shape)*0.2)+sp.random.rand(*t.shape)*0.3
noisy_x  = true_x+sp.random.rand(*t.shape)*0.3
noisy_x  = noisy_x

#%%
g1 = tf.Graph()
with g1.as_default():
    #input series placeholder
    x=tf.placeholder(dtype=tf.float32,shape=[None,N_TIME,N_INPUT])
    #input label placeholder
    y=tf.placeholder(dtype=tf.float32,shape=[None,N_TIME,N_INPUT])
    lr=tf.placeholder(dtype=tf.float32,shape=())
    
    
    
    #defining the network as two stacked layers of LSTMs
    lstm_layers=[tf.nn.rnn_cell.BasicLSTMCell(size,forget_bias=1) for size in [N_INPUT]]
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
    
    #Unroll the rnns
    outputs, state = tf.nn.dynamic_rnn(lstm_cell,x,dtype=tf.float32)
    print('Outputs:', outputs.shape)
    
    #loss_function
    loss=tf.log(tf.reduce_mean((y-outputs)**2))
    #optimization
    opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    
    #initialize variables
    init=tf.global_variables_initializer()

#%%


with tf.Session(graph=g1) as sess:
    sess.run(init)
    itr=1
    learning_rate = LR_BASE
    while itr<ITRS:
        batch_x= sp.stack([noisy_x,noisy_vx])
        batch_y= sp.stack([true_x,true_vx])

        batch_x=batch_x.reshape((BATCH_SIZE,N_TIME,N_INPUT))
        batch_y=batch_y.reshape((BATCH_SIZE,N_TIME,N_INPUT))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y, lr:learning_rate})
        
        if itr %20==0:
            learning_rate *= 0.90
            los,out=sess.run([loss,outputs],feed_dict={x:batch_x,y:batch_y,lr:learning_rate})
            print("For iter %i, learning rate %3.6f"%(itr, learning_rate))
            print("Loss ",los)
            print("__________________")

        itr=itr+1
        

#%% Plot the fit        
out_x = sp.squeeze(out[:,:,0])
out_vx = sp.squeeze(out[:,:,0])

plt.figure(figsize=(12,4))
plt.subplot(121)
plt.plot(t,noisy_vx,label='measured')
plt.plot(t,true_vx,label='true')
plt.plot(t,out_vx,label='LSTM')
plt.legend()
plt.subplot(122)
plt.plot(t,noisy_x,label='measured')
plt.plot(t,true_x,label='true')
plt.plot(t,out_x,label='LSTM')
plt.legend()

