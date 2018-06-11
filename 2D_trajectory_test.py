import tensorflow as tf
import functools
from itertools import permutations
from tensorflow.contrib.seq2seq import TrainingHelper, BasicDecoder, dynamic_decode,LuongAttention, AttentionWrapper
import scipy as sp
import numpy as np
import random
from scipy.integrate import cumtrapz
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

#%% Constants
N_TIME = 100
N_HIDDEN = 10
N_INPUT = 4
N_ATTN  = 8 #< N_TIME how many previous steps to attend to
N_PLOTS = 4
N_OUTPUT = 4
LR_BASE = 5e-2
BATCH_SIZE = 38
ITRS = 800
REG = 2e-4

#%% Generate a sample
t = sp.linspace(0,10,N_TIME)

def gen_sample(f, vnoise, xnoise):
    true_vy = [f[1](tprime) for tprime in t[:N_TIME//4]]+[0.0]*(N_TIME//4)+[-f[1](tprime-t[N_TIME//2]) for tprime in t[N_TIME//2:3*N_TIME//4]] +[0.0]*(N_TIME//4)
    true_vx = [0.0]*(N_TIME//4)+[-f[0](tprime-t[N_TIME//4]) for tprime in t[N_TIME//4:N_TIME//2]] + [0.0]*(N_TIME//4) + [f[0](tprime-t[3*N_TIME//4]) for tprime in t[3*N_TIME//4:]]
    true_v  = sp.vstack([true_vx,true_vy])
    true_x  = cumtrapz(true_v,t)
    true_x  = sp.hstack([[[0],[0]],true_x])

    #noisy_v  = true_v+sp.random.randn(*true_v.shape)*vnoise
    noisy_v  = true_v+(sp.random.rand(*true_v.shape)-0.5)*vnoise
    #noisy_x  = true_x+sp.random.randn(*true_x.shape)*xnoise
    noisy_x  = true_x+(sp.random.rand(*true_x.shape)-0.5)*xnoise
<<<<<<< HEAD
=======
    
>>>>>>> ce7f7d394a400190409def1c7004335ae3bc04a8
    
    
    return sp.vstack([true_x,true_v]).T, sp.vstack([noisy_x,noisy_v]).T

#%%
#f1D = [sp.sin,lambda x: sp.cos(x)+1,lambda y: 0.5*y/max(t),lambda z: 0.25*(sp.sin(z)+sp.cos(z)**2)]
f1D = [lambda x: x/max(t),lambda f: -1.7*f/max(t), sp.sin, sp.cos, sp.tanh, lambda z: 0.25*(sp.sin(z)+sp.cos(z)**2), lambda x: -x/max(t)*1.05]
fcouples = list(permutations(f1D,2))
random.shuffle(fcouples)
<<<<<<< HEAD
y_batch, x_batch = list(zip(*[gen_sample(f, 0.05, 0.05) for f in fcouples]))
=======
y_batch, x_batch = list(zip(*[gen_sample(f, 1, 1) for f in fcouples]))
>>>>>>> ce7f7d394a400190409def1c7004335ae3bc04a8
batch_y= sp.stack(y_batch)
batch_x= sp.stack(x_batch)
print(batch_y.shape,batch_x.shape)

#%%
g1 = tf.Graph()
with g1.as_default():
    #input series placeholder
    x=tf.placeholder(dtype=tf.float32,shape=[None,N_TIME,N_INPUT])
    #input label placeholder
    y=tf.placeholder(dtype=tf.float32,shape=[None,N_TIME,N_INPUT])
    #Runtime vars
    batch_size=tf.placeholder(dtype=tf.int32,shape=())
    lr=tf.placeholder(dtype=tf.float32,shape=())
    
    
    #defining the network as stacked layers of LSTMs
    #lstm_layers =[tf.nn.rnn_cell.LSTMCell(size,forget_bias=0.9) for size in [N_HIDDEN]]
    #lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
    lstm_cell =tf.nn.rnn_cell.LSTMCell(N_HIDDEN,forget_bias=0.9)
    
    #Residual weapper
    #lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)    
        
    #UNROLL
    lstm_inputs = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda z: REG*tf.nn.l2_loss(z))(x)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell,lstm_inputs,dtype=tf.float32)

    #Output projection layer
    projection_layer = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda z: REG*tf.nn.l2_loss(z))(outputs)
    predictions = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda z: REG*tf.nn.l2_loss(z))(projection_layer)
    #Final output layer
    predictions = tf.layers.Dense(N_OUTPUT, activation=None,activity_regularizer=lambda z:REG*tf.nn.l2_loss(z))(predictions)
    print('Predictions:', predictions.shape)
    
    #loss_function
    #vel_path = sp.integrate.cumtrapz(predictions[:,:,3], predictions[:,:,1], axis=2, initial=0)

    #print(predictions[:,1:,2].shape)
    #print(tf.concat(( tf.zeros([batch_size,1]), predictions[:,1:,3]), axis=1).shape)
    #print(predictions[:,:,3].shape)
	#np.stack(( np.zeros([100,1], dtype=int), predictions[:,1:,3])
    pos_diff1 = predictions[:,:,1] - tf.concat(( tf.zeros([batch_size,1]), predictions[:,1:,1]), axis=1)
    pos_diff2 = predictions[:,:,2] - tf.concat(( tf.zeros([batch_size,1]), predictions[:,1:,2]), axis=1)
    ##print(pos_diff.shape)
    #vel_path = np.cumsuma((predictions[:,:,3]-[0;predictions[:,:-2,3]])*2.5,axis=2, dtype=, out=None)
    #print('velocity_integrated_path',vel_path.shape)
    loss= tf.reduce_mean((y-predictions)**2) + 0.001*tf.reduce_mean((predictions[:,:,3] - pos_diff1/0.1)**2) #+ tf.reduce_mean((predictions[:,:,4] - pos_diff2/0.1)**2)
    ##optimization
    opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    print('Compiled loss and trainer')
    
    #initialize variables
    init=tf.global_variables_initializer()
    print('Added initializer')

    #Count the trainable parameters 
    shapes = [functools.reduce(lambda x,y: x*y,variable.get_shape()) for variable in tf.trainable_variables()]
    print('Nparams: ', functools.reduce(lambda x,y: x+y, shapes))
#%% TRAINING
train_batch_x = batch_x[:BATCH_SIZE,:,:]
train_batch_y = batch_y[:BATCH_SIZE,:,:]
test_batch_x = batch_x[BATCH_SIZE:,:,:]
test_batch_y = batch_y[BATCH_SIZE:,:,:]        
with tf.Session(graph=g1) as sess:
    sess.run(init)
    itr=1
    learning_rate = LR_BASE
    while itr<ITRS:

        sess.run(opt, feed_dict={x: train_batch_x, y: train_batch_y, lr:learning_rate, batch_size: train_batch_x.shape[0]})
        
        if itr %20==0:
            learning_rate *= 0.95
            los,out=sess.run([loss,predictions],feed_dict={x:train_batch_x,y: train_batch_y,lr:learning_rate, batch_size: train_batch_x.shape[0]})
            print("For iter %i, learning rate %3.6f"%(itr, learning_rate))
            print("Loss ".ljust(12),los)
            los2,out2=sess.run([loss,predictions],feed_dict={x:test_batch_x,y: test_batch_y, batch_size:test_batch_x.shape[0]})
            print("DEV Loss ".ljust(12),los2)
            print("_"*80)

        itr=itr+1
        
    
    out  = sp.concatenate([out,out2],axis=0)
    
           

#%% Compute the EKF results
from KalmanFilterClass import LinearKalmanFilter2D, Data
batch_kalman = []
for i in range(batch_y.shape[0]):
    deltaT = sp.mean(t[1:] - t[0:-1])
    state0 = sp.squeeze(batch_x[i,0,:])
    P0     = sp.identity(4)*0.1
    F0     = sp.array([[1, 0, deltaT, 0],\
                       [0, 1, 0, deltaT],\
                       [0, 0, 1, 0],\
                       [0, 0, 0, 1]])
    H0     = sp.identity(4)
    Q0     = sp.diagflat([0.0005,0.0005,0.1,0.1])
<<<<<<< HEAD
    R0     = sp.diagflat([0.07,0.07,0.07,0.07])
=======
    R0     = sp.diagflat([7,7,7,7])
>>>>>>> ce7f7d394a400190409def1c7004335ae3bc04a8


    data = Data(sp.squeeze(batch_x[i,:,0]),sp.squeeze(batch_x[i,:,1]),sp.squeeze(batch_x[i,:,2]),sp.squeeze(batch_x[i,:,3]),[],[])
    filter1b = LinearKalmanFilter2D(F0, H0, P0, Q0, R0, state0)
    kalman_data = filter1b.process_data(data)
    batch_kalman.append(sp.vstack([kalman_data.x[1:],kalman_data.y[1:], kalman_data.vx[1:], kalman_data.vy[1:]]).T)
    
xk_batch = sp.stack(batch_kalman)
xk_batch - batch_y
print('Kalman loss;'.ljust(12), sp.mean(pow(xk_batch[BATCH_SIZE:,:,:] - batch_y[BATCH_SIZE:,:,:],2)))
print(xk_batch.shape)
#%% Plot the fit    
plt.figure(figsize=(14,16))

for batch_idx in range(BATCH_SIZE,BATCH_SIZE+N_PLOTS):
    out_xc = sp.squeeze(out[batch_idx,:,0])
    out_yc = sp.squeeze(out[batch_idx,:,1])
    out_vxc = sp.squeeze(out[batch_idx,:,2])
    out_vyc = sp.squeeze(out[batch_idx,:,3])
    
    noisy_xc  = batch_x[batch_idx,:,0]
    noisy_yc  = batch_x[batch_idx,:,1]
    noisy_vxc = batch_x[batch_idx,:,2]
    noisy_vyc = batch_x[batch_idx,:,3]
    
    true_xc = batch_y[batch_idx,:,0]
    true_yc = batch_y[batch_idx,:,1]
    true_vxc = batch_y[batch_idx,:,2]
    true_vyc = batch_y[batch_idx,:,3]
    
    ekf_xc  = sp.squeeze(xk_batch[batch_idx,:,0])
    ekf_yc  = sp.squeeze(xk_batch[batch_idx,:,1])
    ekf_vxc = sp.squeeze(xk_batch[batch_idx,:,2])
    ekf_vyc = sp.squeeze(xk_batch[batch_idx,:,3])
    
    
    l2 = lambda x,y: pow(x**2 + y**2,0.5)
    
    plot_idx = batch_idx-BATCH_SIZE
    plt.subplot(20+(N_PLOTS)*100 + plot_idx*2+1)
    if batch_idx == 0: plt.title('Location x')
    plt.plot(true_xc,true_yc,lw=2,label='true')
    plt.plot(noisy_xc,noisy_yc,lw=1,label='measured')
    plt.plot(ekf_xc,ekf_yc,lw=1,label='Linear KF')
    plt.plot(out_xc,out_yc,lw=1,label='LSTM')
    plt.grid(which='both')
<<<<<<< HEAD
    plt.gca().equal()
=======
    #plt.gca().equal()
>>>>>>> ce7f7d394a400190409def1c7004335ae3bc04a8
    plt.ylabel('x[m]')
    plt.xlabel('time[s]')
    plt.legend()
    
    plt.subplot(20+(N_PLOTS)*100 + plot_idx*2+2)
    if batch_idx == 0: plt.title('Velocity Norm')
    #plt.plot(t,true_vxc,lw=2,label='true')
    #plt.plot(t,noisy_vxc,lw=1,label='measured')
    #plt.plot(t,ekf_vxc,lw=1,label='Linear KF')
    #plt.plot(t,out_vxc,lw=1,label='LSTM')
    plt.plot(t,l2(true_vxc,true_vyc),lw=2,label='true')
    plt.plot(t,l2(noisy_vxc,noisy_vyc),lw=1,label='measured')
    plt.plot(t,l2(ekf_vxc,ekf_vyc),lw=1,label='Linear KF')
    plt.plot(t,l2(out_vxc,out_vyc),lw=1,label='LSTM')
    plt.ylabel('vx[m/s]')
    plt.xlabel('time[s]')
    plt.grid(which='both')
    plt.legend()
    
plt.savefig('2Dexample.png',dpi=200)
    
