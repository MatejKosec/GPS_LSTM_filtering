from __future__ import print_function, division
import scipy as sp 
import tensorflow as tf
from scipy import stats
from scipy.integrate import cumtrapz
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from decimal import Decimal
import numpy as np

pdf  = sp.stats.multivariate_normal.pdf
cdf  = sp.stats.multivariate_normal.cdf
import functools

#%% Constants
N_TIME = 625
#file 1 has 663 timesteps
#file 2 has 628 timesteps
N_HIDDEN = 30
N_INPUT = 6
N_PLOTS = 25
N_OUTPUT = 6
LR_BASE = 2e-3
BATCH_SIZE = 15
ITRS = 100
REG = 1.5e-1
DROPOUT1= 0.10
DROPOUT2= 0.10
DECAY = 0.95
RESTORE_CHECKPOINT  = True
PERFORM_TRAINING = True
SAVE_DIR = './checkpoints'

def read_file(file_name):
	data_file = open(file_name,"r")
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
				data.append(Decimal(item[:-2].strip()))
				break
			data.append(Decimal(item.strip()))
		all_data.append(data)
		row = row+1
	print(row)
	data_file.close()
	data_reduced = np.vstack(all_data)[:N_TIME,:]
	data_reduced[:,2:5] = data_reduced[:,2:5] - data_reduced[1,2:5]
	data_reduced[:,8:11] = data_reduced[:,8:11] - data_reduced[1,8:11]

	return data_reduced[:N_TIME,:]


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
for i in range(8):
	file_name = 'Oval_circ1_N'+str(i+1)+'.txt'
	data.append(read_file(file_name))
	file_name = 'Oval_circ2_N'+str(i+1)+'.txt'
	data.append(read_file(file_name))
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
            if itr %100==0:
                save_path = saver.save(sess, SAVE_DIR+"/model.ckpt")
                print("Model saved in path: %s" % save_path)
            print("_"*80)
        

        itr=itr+1
    dev_losses = sess.run([test_loss_summary],feed_dict={x:test_batch_x,y: test_batch_y, batch_size:test_batch_x.shape[0],\
                               is_training: False})[0]
    
    out  = sp.concatenate([out,out2],axis=0)
    
    #%%
plt.figure(figsize=(14,4))
plt.subplot(221)
plt.title('Training progress ylog plot')
plt.gca().set_yscale('log')
plt.plot(range(0,ITRS,20),dev_loss_plot,label='dev loss')
plt.plot(range(0,ITRS,20),tra_loss_plot,label='train loss')
plt.xlabel('Adam iteration')
plt.ylabel('L2 fitting loss')
plt.grid(which='both')
plt.legend()
plt.subplot(222)
plt.title('Learning rate')

plt.plot(out2[0,:,0],out2[0,:,1],label='Test path output')
plt.plot(test_batch_x[0,:,0],test_batch_x[0,:,1],label='Input path')
plt.plot(test_batch_y[0,:,0],test_batch_y[0,:,1],label='True path')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.grid(which='both')
plt.legend()
plt.subplot(223)
plt.title('Position')

plt.plot(out2[0,:,3],out2[0,:,4],label='Test velocity output')
plt.plot(test_batch_x[0,:,3],test_batch_x[0,:,4],label='Test velocity input')
plt.plot(test_batch_y[0,:,3],test_batch_y[0,:,4],label='True velocity')
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.grid(which='both')
plt.legend()
plt.subplot(224)
plt.title('Velocity')
#plt.gca().set_yscale('log')
plt.plot(range(len(lr_plot)),lr_plot,label='Exponentially decayed to %i percent every 20 iterations'%(DECAY*100))
plt.xlabel('Adam iteration')
plt.ylabel('L2 fitting loss')
plt.grid(which='both')
plt.legend()
plt.savefig('training_progress2D.png',bbox_inches='tight', dpi=200)
