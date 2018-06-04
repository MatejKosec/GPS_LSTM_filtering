import tensorflow as tf
import functools
from itertools import permutations
from tensorflow.contrib.seq2seq import TrainingHelper, BasicDecoder, dynamic_decode,LuongAttention, AttentionWrapper
import scipy as sp
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt


#%% Constants
N_TIME = 100
N_HIDDEN = 12
N_INPUT = 4
N_ATTN  = 8 #< N_TIME how many previous steps to attend to
N_PLOTS = 4
N_OUTPUT = 4
LR_BASE = 1e-1
BATCH_SIZE = 12
ITRS = 800
REG = 2e-5

#%% Generate a sample
t = sp.linspace(0,10,N_TIME)
def gen_sample(f, vnoise, xnoise):
    true_vy = [f[1](tprime) for tprime in t[:N_TIME//4]]+[0.0]*(N_TIME//4)+[-f[1](tprime) for tprime in t[N_TIME//2:3*N_TIME//4]] +[0.0]*(N_TIME//4)
    true_vx = [0.0]*(N_TIME//4)+[f[0](tprime) for tprime in t[N_TIME//4:N_TIME//2]] + [0.0]*(N_TIME//4) + [-f[0](tprime) for tprime in t[3*N_TIME//4:]]
    true_v  = sp.vstack([true_vx,true_vy])
    true_x  = cumtrapz(true_v,t)
    true_x  = sp.hstack([[[0],[0]],true_x])

    noisy_v  = true_v+sp.random.randn(*true_v.shape)*vnoise
    noisy_x  = true_x+sp.random.randn(*true_x.shape)*xnoise
    
    
    return sp.vstack([true_x,true_v]).T, sp.vstack([noisy_x,noisy_v]).T

#%%
#f1D = [sp.sin,lambda x: sp.cos(x)+1,lambda y: 0.5*y/max(t),lambda z: 0.25*(sp.sin(z)+sp.cos(z)**2)]
f1D = [lambda x: x/max(t),lambda f: -0.7*f/max(t), sp.sin, sp.cos]
fcouples = permutations(f1D,2)
y_batch, x_batch = list(zip(*[gen_sample(f, 0.1, 0.2) for f in fcouples]))
batch_y= sp.stack(y_batch)
batch_x= sp.stack(x_batch)
print(batch_y.shape,batch_x.shape)

#%%
g1 = tf.Graph()
with g1.as_default():
    #input series placeholder
    x=tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,N_TIME,N_INPUT])
    #input label placeholder
    y=tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,N_TIME,N_INPUT])
    lr=tf.placeholder(dtype=tf.float32,shape=())
    
    
    #defining the network as stacked layers of LSTMs
    #lstm_layers =[tf.nn.rnn_cell.LSTMCell(size,forget_bias=0.9) for size in [N_HIDDEN]]
    #lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers)
    lstm_cell =tf.nn.rnn_cell.LSTMCell(N_HIDDEN,forget_bias=0.95)
    
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
from KalmanFilterClass import LinearKalmanFilter2D, Data
batch_kalman = []
for i in range(BATCH_SIZE):
    deltaT = sp.mean(t[1:] - t[0:-1])
    state0 = sp.squeeze(batch_x[i,0,:])
    P0     = sp.identity(4)*0.1
    F0     = sp.array([[1, 0, deltaT, 0],\
                       [0, 1, 0, deltaT],\
                       [0, 0, 1, 0],\
                       [0, 0, 0, 1]])
    H0     = sp.identity(4)
    Q0     = sp.diagflat([0.005,0.005,0.0001,0.0001])
    R0     = sp.diagflat([0.25,0.25,0.001,0.001])


    data = Data(sp.squeeze(batch_x[i,:,0]),sp.squeeze(batch_x[i,:,1]),sp.squeeze(batch_x[i,:,2]),sp.squeeze(batch_x[i,:,3]),[],[])
    filter1b = LinearKalmanFilter2D(F0, H0, P0, Q0, R0, state0)
    kalman_data = filter1b.process_data(data)
    batch_kalman.append(sp.vstack([kalman_data.x[1:],kalman_data.y[1:], kalman_data.vx[1:], kalman_data.vy[1:]]).T)
    
xk_batch = sp.stack(batch_kalman)
xk_batch - batch_y
print('Kalman loss; ', sp.mean(pow(xk_batch - batch_y,2)))
print(xk_batch.shape)
#%% Plot the fit    
plt.figure(figsize=(14,16))

for batch_idx in range(N_PLOTS):
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
    
    plt.subplot(20+(N_PLOTS)*100 + batch_idx*2+1)
    if batch_idx == 0: plt.title('Location x')
    plt.plot(true_xc,true_yc,lw=2,label='true')
    plt.plot(noisy_xc,noisy_yc,lw=1,label='measured')
    plt.plot(ekf_xc,ekf_yc,lw=1,label='Linear KF')
    plt.plot(out_xc,out_yc,lw=1,label='LSTM')
    plt.grid(which='both')
    plt.ylabel('x[m]')
    plt.xlabel('time[s]')
    plt.legend()
    
    plt.subplot(20+(N_PLOTS)*100 + batch_idx*2+2)
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
    
