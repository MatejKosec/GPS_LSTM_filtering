from __future__ import print_function, division
import scipy as sp 
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

import functools

#%% Constants
N_TIME = 125
#file 1 has 663 timesteps
#file 2 has 628 timesteps
N_HIDDEN = 30
N_INPUT = 6
N_PLOTS = 25
N_OUTPUT = 6
LR_BASE = (2e-3)#*0.99**100
BATCH_SIZE = 60
ITRS = 80 #1000
REG = 1.5e-2
DROPOUT1= 0.1
DROPOUT2= 0.1
DECAY = 0.99
RESTORE_CHECKPOINT  = False
PERFORM_TRAINING = True
SAVE_DIR = './checkpoints'

def read_file(file_name):
	with open(file_name,"r") as data_file:
        	#lines = data_file.readlines()
        	line = data_file.readline()
        	row = 0
        	x = 0
        	all_data = []
        	for line in data_file:
        		numbers = line.split(',')
        		#ts, t_diffs, xs,ys,zs,vxs,vys,vzs, x_ts,y_ts,z_ts,vx_ts,vy_ts,vz_ts = [each_line for each_line in numbers]
        		line_items = [each_line for each_line in numbers]
        		data = []
        		for ino,item in enumerate(line_items):
        			if ino == len(line_items):
        				data.append(float(item[:-2].strip()))
        				break
        			data.append(float(item.strip()))
        		all_data.append(data)
        		row = row+1
        	#print(row)
	
	data_reduced = np.vstack(all_data)[:,:]
	data_reduced[:,2:5] = data_reduced[:,2:5] - data_reduced[1,2:5]
	data_reduced[:,8:11] = data_reduced[:,8:11] - data_reduced[1,8:11]
	
	#Slicing data to get more training data for the dense layers
	slice_size = 125
	sliced_data = []
	for i in range(row//slice_size):
		sliced_data.append(np.array(data_reduced[i*slice_size:(i+1)*slice_size,:]))
	
	return sliced_data


#%%
g1 = tf.Graph()
with g1.as_default():
    #input series placeholder
    x=tf.placeholder(dtype=tf.float32,shape=[None,N_TIME,N_INPUT])
    #input label placeholder
    y=tf.placeholder(dtype=tf.float32,shape=[None,N_TIME,N_INPUT])
    #Dropout needs to know if training
    is_training = tf.placeholder_with_default(True, shape=())
    #Runtime vars
    batch_size=tf.placeholder(dtype=tf.int32,shape=())
    lr=tf.placeholder(dtype=tf.float32,shape=())
    tf.set_random_seed(0)
    
    
    #defining the network as stacked layers of LSTMs
    lstm_cell =tf.nn.rnn_cell.LSTMCell(N_HIDDEN,forget_bias=0.99)
    
    #Residual weapper
    #lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)    
    
    #Dropout wrapper
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=tf.maximum(1-DROPOUT2,1-tf.cast(is_training,tf.float32)),\
                                              output_keep_prob=tf.maximum(1-DROPOUT2,1-tf.cast(is_training,tf.float32)))
        
    #UNROLL
    lstm_inputs = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda z: REG*tf.nn.l2_loss(z))(x)
    outputs, state = tf.nn.dynamic_rnn(lstm_cell,lstm_inputs,dtype=tf.float32)

    #Output projection layer
    projection_layer = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda z: REG*tf.nn.l2_loss(z))(outputs)
    projection_layer = tf.layers.dropout(projection_layer,rate=DROPOUT1, training=is_training)
    predictions = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda z: REG*tf.nn.l2_loss(z))(projection_layer)
    predictions = tf.layers.dropout(predictions,rate=DROPOUT1, training=is_training)
    #Final output layer
    predictions = tf.layers.Dense(N_OUTPUT, activation=None,activity_regularizer=lambda z:REG*tf.nn.l2_loss(z))(predictions)
    print('Predictions:', predictions.shape)
    
    #loss_function
    loss= tf.reduce_mean((y-predictions)**2)
    test_loss_summary = tf.reduce_mean((y-predictions)**2,axis=[1])
    #optimization
    opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    print('Compiled loss and trainer')
    
    #initialize variables
    init=tf.global_variables_initializer()
    print('Added initializer')

    #Count the trainable parameters 
    shapes = [functools.reduce(lambda x,y: x*y,variable.get_shape()) for variable in tf.trainable_variables()]
    print('Nparams: ', functools.reduce(lambda x,y: x+y, shapes))
    saver = tf.train.Saver()
#%% TRAINING
data = []
#Add saver
for i in range(10):
	file_name = './NewData/Oval_circ1_N'+str(i+1)+'.txt'
	data.extend(read_file(file_name))
	file_name = './NewData/Oval_circ2_N'+str(i+1)+'.txt'
	data.extend(read_file(file_name))
#data contains [t, t_diff, x,y,z,vx,vy,vz, x_t,y_t,z_t,vx_t,vy_t,vz_t]

all_data = sp.stack(data)
batch_x = all_data[:,:,2:8]
batch_y = all_data[:,:,8:]


train_batch_x = batch_x[:BATCH_SIZE,:,:]
train_batch_y = batch_y[:BATCH_SIZE,:,:]
test_batch_x = batch_x[BATCH_SIZE:,:,:]
test_batch_y = batch_y[BATCH_SIZE:,:,:]        
#Save losses for plotting of progress
dev_loss_plot = []
tra_loss_plot = []
lr_plot = []

with tf.Session(graph=g1) as sess:
    if RESTORE_CHECKPOINT:
          saver.restore(sess, SAVE_DIR+"/model.ckpt")
    else:
        sess.run(init)
    itr=0
    learning_rate = LR_BASE
    while itr<ITRS and PERFORM_TRAINING:
        
        #Do somme minibatching
        mini_size = 32
        for i in range(0,train_batch_x.shape[0],mini_size):
            start = i
            end   = min(i+mini_size,train_batch_x.shape[0])
            sess.run(opt, feed_dict={x: train_batch_x[start:end], y: train_batch_y[start:end], lr:learning_rate, batch_size: start-end})
        lr_plot.append(learning_rate)
        
        if itr %20==0:
            learning_rate *= DECAY
            los,out=sess.run([loss,predictions],feed_dict={x:train_batch_x,y: train_batch_y,lr:learning_rate, batch_size: train_batch_x.shape[0],is_training: False})
            tra_loss_plot.append(los)
            print("For iter %i, learning rate %3.6f"%(itr, learning_rate))
            print("Loss ".ljust(12),los)
            los2,out2=sess.run([loss,predictions],feed_dict={x:test_batch_x,y: test_batch_y, batch_size:test_batch_x.shape[0],\
                               is_training: False})
            dev_loss_plot.append(los2)
            print("DEV Loss ".ljust(12),los2)
            if itr %100==0 and itr > 0:
                save_path = saver.save(sess, SAVE_DIR+"/model.ckpt")
                print("Model saved in path: %s" % save_path)
            print("_"*80)
        

        itr=itr+1
    dev_losses = sess.run([test_loss_summary],feed_dict={x:test_batch_x,y: test_batch_y, batch_size:test_batch_x.shape[0],\
                               is_training: False})[0]
    
    out  = sp.concatenate([out,out2],axis=0)


#%% Compute the EKF results
from KalmanFilterClass import LinearKalmanFilter3D, Data3D
batch_kalman = []
for i in range(batch_y.shape[0]):
    deltaT = 1.0
    state0 = sp.squeeze(batch_x[i,0,:])
    P0     = sp.identity(6)*0.01
    F0     = sp.array([[1, 0, 0, deltaT, 0, 0],\
                       [0, 1, 0, 0, deltaT, 0],\
                       [0, 0, 1, 0, 0, deltaT],\
                       [0, 0, 0, 1, 0, 0],\
                       [0, 0, 0, 0, 1, 0],
                       [0, 0, 0, 0, 0, 1]])
    H0     = sp.identity(6)
    Q0     = sp.diagflat([50,50,50,0.01,0.01,0.01])
    R0     = sp.diagflat([10,10,10,5,5,5])
    filter3d = LinearKalmanFilter3D(F0, H0, P0, Q0, R0, state0)

    #batch_x = sp.cast(batch_x, sp.float64)
    data = Data3D(sp.squeeze(batch_x[i,:,0]),sp.squeeze(batch_x[i,:,1]),sp.squeeze(batch_x[i,:,2]),sp.squeeze(batch_x[i,:,3]),sp.squeeze(batch_x[i,:,4]),sp.squeeze(batch_x[i,:,5]),[],[])
    filter3d = LinearKalmanFilter3D(F0, H0, P0, Q0, R0, state0)
    kalman_data = filter3d.process_data(data)
    batch_kalman.append(sp.vstack([kalman_data.x[1:],kalman_data.y[1:],kalman_data.z[1:], kalman_data.vx[1:], kalman_data.vy[1:], kalman_data.vz[1:]]).T)
    
xk_batch = sp.stack(batch_kalman)
#xk_batch - batch_y
print('Kalman loss;'.ljust(12), sp.mean(pow(xk_batch[BATCH_SIZE:,:,:] - batch_y[BATCH_SIZE:,:,:],2)))
print(xk_batch.shape)
 
    #%%
for i in range(test_batch_x.shape[0]):
	print(i)
	plt.figure(figsize=(14,4))
	plt.subplot(231)
	plt.title('Training progress ylog plot')
	plt.gca().set_yscale('log')
	plt.plot(range(0,ITRS,20),dev_loss_plot,label='dev loss')
	plt.plot(range(0,ITRS,20),tra_loss_plot,label='train loss')
	plt.xlabel('Adam iteration')
	plt.ylabel('L2 fitting loss')
	plt.grid(which='both')
	plt.legend()
	
	plt.subplot(232)
	plt.title('Position Horizontal')
	plt.plot(test_batch_y[i,:,0],test_batch_y[i,:,1],label='True')
	plt.plot(test_batch_x[i,:,0],test_batch_x[i,:,1],label='Measured')
	plt.plot(xk_batch[i,:,0],xk_batch[i,:,1],label='Linear KF')
	plt.plot(out2[i,:,0],out2[i,:,1],label='LSTM')
	plt.axis('equal')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.grid(which='both')
	plt.legend()
	
	plt.subplot(233)
	plt.title('Position Vertical')
	plt.plot(test_batch_y[i,:,2],label='True')
	plt.plot(test_batch_x[i,:,2],label='Measured')
	plt.plot(xk_batch[i,:,2],label='Linear KF')
	plt.plot(out2[i,:,2],label='LSTM')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.grid(which='both')
	plt.legend()
	
	plt.subplot(234)
	plt.title('Velocity x')
	plt.plot(test_batch_y[i,:,3],label='True')
	plt.plot(test_batch_x[i,:,3],label='Measured')
	plt.plot(xk_batch[i,:,3],label='Linear KF')
	plt.plot(out2[i,:,3],label='LSTM')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.grid(which='both')
	plt.legend()
	
	plt.subplot(235)
	plt.title('Velocity y')
	plt.plot(test_batch_y[i,:,4],label='True')
	plt.plot(test_batch_x[i,:,4],label='Measured')
	plt.plot(xk_batch[i,:,4],label='Linear KF')
	plt.plot(out2[i,:,4],label='LSTM')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.grid(which='both')
	plt.legend()
	
	plt.subplot(236)
	plt.title('Velocity z')
	plt.plot(test_batch_y[i,:,5],label='True')
	plt.plot(test_batch_x[i,:,5],label='Measured')
	plt.plot(xk_batch[i,:,5],label='Linear KF')
	plt.plot(out2[i,:,5],label='LSTM')
	plt.xlabel('x axis')
	plt.ylabel('y axis')
	plt.grid(which='both')
	plt.legend()
	
	#plt.subplot(23)
	#plt.title('Learning Rate')
	#plt.gca().set_yscale('log')
	#plt.plot(range(len(lr_plot)),lr_plot,label='Exponentially decayed to %i percent every 20 iterations'%(DECAY*100))
	#plt.xlabel('Adam iteration')
	#plt.ylabel('L2 fitting loss')
	#plt.grid(which='both')
	#plt.legend()
	plt.savefig('training_progress2D'+str(i)+'.png',bbox_inches='tight', dpi=200)