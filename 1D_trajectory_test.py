import tensorflow as tf
import functools
from tensorflow.contrib.seq2seq import TrainingHelper, BasicDecoder, dynamic_decode,LuongAttention, AttentionWrapper
import scipy as sp
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt


#%% Constants
N_TIME = 80
N_HIDDEN = 10
N_INPUT = 2
N_ATTN  = 12 #< N_TIME how many previous steps to attend to
N_OUTPUT = 2
LR_BASE = 1e-1
BATCH_SIZE = 4
ITRS = 800
REG = 1e-5

#%% Generate a sample
t = sp.linspace(0,10,N_TIME)
def gen_sample(f, vnoise, xnoise):
    true_vx = f(t)
    true_x  = cumtrapz(true_vx,t)
    true_x  = sp.hstack([[0],true_x])
    
    noisy_vx = f(t+sp.random.randn(*t.shape)*xnoise)+sp.random.randn(*t.shape)*vnoise
    noisy_x  = true_x+sp.random.randn(*t.shape)*xnoise
    noisy_x  = noisy_x
    
    return sp.stack([true_x,true_vx]).T, sp.stack([noisy_x,noisy_vx]).T

#%%
y_batch, x_batch = list(zip(*[gen_sample(f, 0.3, 0.2) for f in [sp.sin,lambda x: sp.cos(x)+1,lambda x: 0.5*x/max(t),lambda x: 0.25*(sp.sin(x)+sp.cos(x)**2)]]))
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
    lr=tf.placeholder(dtype=tf.float32,shape=())
    
    
    #defining the network as stacked layers of LSTMs
    #lstm_layers =[tf.nn.rnn_cell.LSTMCell(size,forget_bias=0.9) for size in [N_HIDDEN]]
    #lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
    lstm_cell =tf.nn.rnn_cell.LSTMCell(N_HIDDEN,forget_bias=0.9)
    
    #Self attention mechanism
    Wattn = tf.get_variable('attentionWeights', dtype =tf.float32, shape=[N_HIDDEN,N_HIDDEN],\
                            initializer=tf.contrib.layers.xavier_initializer())
    Wcont = tf.get_variable('ContextWeights', dtype =tf.float32, shape=[2*N_HIDDEN,N_HIDDEN],\
                            initializer=tf.contrib.layers.xavier_initializer())
    
    #Initialize the LSTM
    state = lstm_cell.zero_state(BATCH_SIZE, tf.float32)
    for i in range(N_TIME):
        #Feed inputs to the lstm
        output, state = lstm_cell(x[:,i,:], state)
        #No attention options for first timestep (replicate measurement)
        if i == 0:
            outputs = tf.expand_dims(output,axis=1)
        else:
            #Transpose the output for processing
            output = tf.expand_dims(output,axis=1)
            raw_output = output #save for later
            
            #The context window
            windowed_outputs = outputs[:,max(i-N_ATTN,0):i,:]
            
            #Two step context weighing
            attention_logits = tf.tensordot(output,Wattn,axes=[[2],[0]])
            attention_logits = tf.transpose(tf.matmul(attention_logits,tf.transpose(windowed_outputs,[0,2,1])),[0,2,1])
            attention_weights = tf.nn.softmax(attention_logits,axis=1)
            #Compute the context state
            context = tf.reduce_sum(attention_weights*windowed_outputs,axis=1,keepdims=True)
            
            #Mix context and raw output
            output  = tf.nn.tanh(tf.tensordot(tf.concat([raw_output,context],axis=2),Wcont,axes=[[2],[0]]))
            outputs = tf.concat([outputs,output],axis=1)
            
            #Set the state  for the LSTM to the attended output
            state = tf.nn.rnn_cell.LSTMStateTuple(tf.squeeze(output),tf.squeeze(raw_output))
            
    print('Unrolled')
    
    #Skip connection to original input
    outputs = tf.concat([outputs,x],axis=2)
    
    #Output projection layer
    projection_layer = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda x: REG*tf.nn.l2_loss(x))(outputs)
    predictions = tf.layers.Dense(N_HIDDEN, activation=tf.nn.relu,activity_regularizer=lambda x: REG*tf.nn.l2_loss(x))(projection_layer)
    #Final output layer
    predictions = tf.layers.Dense(N_OUTPUT, activation=None,activity_regularizer=lambda x:REG*tf.nn.l2_loss(x))(predictions)
    print('Predictions:', predictions.shape)
    
    #loss_function
    loss= tf.reduce_mean((y-predictions)**2)
    #optimization
    opt=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    print('Compiled loss and trainer')
    
    #initialize variables
    init=tf.global_variables_initializer()
    print('Added initializer')

    #Count the trainable parameters 
    shapes = [functools.reduce(lambda x,y: x*y,variable.get_shape()) for variable in tf.trainable_variables()]
    print('Nparams: ', functools.reduce(lambda x,y: x+y, shapes))
#%%
with tf.Session(graph=g1) as sess:
    sess.run(init)
    itr=1
    learning_rate = LR_BASE
    while itr<ITRS:

        sess.run(opt, feed_dict={x: batch_x, y: batch_y, lr:learning_rate})
        
        if itr %20==0:
            learning_rate *= 0.93
            los,out=sess.run([loss,predictions],feed_dict={x:batch_x,y:batch_y,lr:learning_rate})
            print("For iter %i, learning rate %3.6f"%(itr, learning_rate))
            print("Loss ",los)
            print("__________________")

        itr=itr+1
#%% Compute the EKF results
from KalmanFilterClass import LinearKalmanFilter1D, Data1D
batch_kalman = []
deltaT = sp.mean(t[1:] - t[0:-1])
state0 = sp.array([0, 0]).T
P0     = sp.identity(2)*0.1
F0     = sp.array([[1, deltaT],\
                   [0, 1]])
H0     = sp.identity(2)
Q0     = sp.diagflat([0.005,0.0001])
R0     = sp.diagflat([0.25,0.001])

for i in range(BATCH_SIZE):
    data = Data1D(sp.squeeze(batch_x[i,:,0]),sp.squeeze(batch_x[i,:,1]),[])
    filter1b = LinearKalmanFilter1D(F0, H0, P0, Q0, R0, state0)
    kalman_data = filter1b.process_data(data)
    batch_kalman.append(sp.vstack([kalman_data.x[1:], kalman_data.vx[1:]]).T)
    
xk_batch = sp.stack(batch_kalman)
print(xk_batch.shape)
#%% Plot the fit    
plt.figure(figsize=(14,16))
for batch_idx in range(BATCH_SIZE):
    out_x = sp.squeeze(out[batch_idx,:,0])
    out_vx = sp.squeeze(out[batch_idx,:,1])
    noisy_x = batch_x[batch_idx,:,0]
    noisy_vx = batch_x[batch_idx,:,1]
    true_x = batch_y[batch_idx,:,0]
    true_vx = batch_y[batch_idx,:,1]
    
    plt.subplot(20+(BATCH_SIZE)*100 + batch_idx*2+1)
    if batch_idx == 0: plt.title('Location x')
    plt.plot(t,true_x,lw=2,label='true')
    plt.plot(t,noisy_x,lw=1,label='measured')
    plt.plot(t,sp.squeeze(xk_batch[batch_idx,:,0]),lw=1,label='Linear KF')
    plt.plot(t,out_x,lw=1,label='LSTM')
    plt.grid(which='both')
    plt.ylabel('x[m]')
    plt.xlabel('time[s]')
    plt.legend()
    
    plt.subplot(20+(BATCH_SIZE)*100 + batch_idx*2+2)
    if batch_idx == 0: plt.title('Velocity x')
    plt.plot(t,true_vx,lw=2,label='true')
    plt.plot(t,noisy_vx,lw=1,label='measured')
    plt.plot(t,sp.squeeze(xk_batch[batch_idx,:,1]),lw=1,label='Linear KF')
    plt.plot(t,out_vx,lw=1,label='LSTM')
    plt.ylabel('vx[m/s]')
    plt.xlabel('time[s]')
    plt.grid(which='both')
    plt.legend()
    
plt.savefig('1Dexample.png',dpi=200)
    
