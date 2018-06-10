from __future__ import print_function, division
import scipy as sp 
import tensorflow as tf
from scipy import stats
from scipy.integrate import cumtrapz
from scipy import interpolate
from matplotlib import pyplot as plt
pdf  = sp.stats.multivariate_normal.pdf
cdf  = sp.stats.multivariate_normal.cdf
import functools

#%% Constants
N_TIME = 100
N_HIDDEN = 30
N_INPUT = 4
N_PLOTS = 4
N_OUTPUT = 4
LR_BASE = 5e-3
BATCH_SIZE = 80
ITRS = 800
REG = 1.1e-3
DROPOUT1= 0.02

#Noise parameters
VNOISE_MU    = [1.0,5.0]
VNOISE_SCALE = [0.9,1.4]
XNOISE_SCALE1= [0.9,2.0]
XNOISE_SCALE2= [0.9,2.0]
XNOISE_MU1   = [0.0,0.0]
XNOISE_MU2   = [4.0,6.0]
NOISE_ALPHA  = [0, 2*sp.pi]

sp.random.seed(0)
#%%
# Create a bimodal gaussian distribution an implemnt a function to sample from it
class bimodal_gaussian_2D(object):
    
    def __init__(self,loc1,loc2,scale1,scale2,xmin,xmax,npts=100,plot=False):
    
        #Sample spacec for plotting and interpolating
        x_eval = sp.linspace(xmin,xmax,npts)
        y_eval = sp.linspace(xmin,xmax,npts)
        if plot: print('Done with linspace')
        x_eval,y_eval = sp.meshgrid(x_eval,y_eval)
        xy_eval = sp.dstack((x_eval,y_eval))
        if plot: print('Done with dstack')
        #Create a bimodal pdf
        bimodal_pdf = pdf(xy_eval, mean=loc1, cov=scale1)*0.5 + \
                      pdf(xy_eval, mean=loc2, cov=scale2)*0.5
        if plot: print('Done with pdf')
        bimodal_cdf = cdf(xy_eval, mean=loc1, cov=scale1)*0.5 + \
                      cdf(xy_eval, mean=loc2, cov=scale2)*0.5
        if plot: print('Done with cdf')
        
        
        #Make sure the cdf is bounded before interpolating the inverse
        bimodal_cdf[:,0]=0
        bimodal_cdf[:,-1]=1
        self.ppf = interpolate.interp2d(x_eval,y_eval,bimodal_cdf,kind='linear')
        if plot: print('Done building interpolator')
        
        #Store the data
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.bimodal_pdf = bimodal_pdf
        self.bimodal_cdf = bimodal_cdf
        return 
        
    #Sample the distribution for any given shape of input array (same as rand function)
    #ppf is an interpolation (approximate)
    def sample(self, *shape):
        samples = sp.random.rand(*shape)
        samples = self.ppf(samples)
        return samples
    

    
#Example of distribution
if __name__ == "__main__":        
    loc1 = [0.0,0.0]
    scale1 = [0.5,0.3]
    loc2 = [2,3]
    scale2 = [0.7,0.4]
    noise_dist = bimodal_gaussian_2D(loc1,loc2,scale1,scale2,-5,5,80,plot=True)
#%%
if __name__ == "__main__":        
    x_eval = noise_dist.x_eval
    y_eval = noise_dist.y_eval
    bimodal_pdf = noise_dist.bimodal_pdf
    bimodal_cdf = noise_dist.bimodal_cdf
    plt.figure(figsize=(9,6))
    plt.title('2D Bimodal distribution example')
    plt.contour(x_eval,y_eval, bimodal_pdf,10)
    plt.annotate('True location peak', loc1, [i-1.6 for i in loc1],\
                 arrowprops=dict(facecolor='black', shrink=0.005))
    plt.annotate('Multipath location peak', loc2, [i-1.6 for i in loc2],\
                 arrowprops=dict(facecolor='black', shrink=0.005))
    plt.grid(which='both')
    plt.ylabel('$\Delta$ y [m] ')
    plt.xlabel('$\Delta$ x [m] ')
    plt.colorbar()
    #plt.savefig('bimodal_distribution_example_2D.png',bbox_inches='tight',dpi=100)

#%% Generate true labels and noisy data for a given time-frame
t = sp.linspace(0,10,N_TIME)
def gen_sample(vx,vy,noise_alpha, vnoise_sigma, xnoise_mu1,xnoise_mu2, xnoise_sigma1,xnoise_sigma2):
    true_vy = [vx for _ in t[:N_TIME//4]]+[0.0]*(N_TIME//4)+[-vx for _ in t[N_TIME//2:3*N_TIME//4]] +[0.0]*(N_TIME//4)
    true_vx = [0.0]*(N_TIME//4)+[vy for _ in t[N_TIME//4:N_TIME//2]] + [0.0]*(N_TIME//4) + [-vy for _ in t[3*N_TIME//4:]]
    true_v  = sp.vstack([true_vx,true_vy])
    true_x  = cumtrapz(true_v,t,initial=0)
    
    #Velocity only has Gaussian noise
    noisy_v = true_v+sp.random.randn(*true_v.shape)*vnoise_sigma
    
    #Position has bimodal noise that keeps constant orientation (as if building is throwing a shadow)
    noise_dist = bimodal_gaussian(xnoise_mu1,xnoise_mu2,xnoise_sigma1,xnoise_sigma2,-10,10,150)
    noisy_delta  = noise_dist.sample(*t.shape) #1D samples
    
    #Now project the 1D noise onto the 2D space
    noisy_x = true_x + sp.array([[sp.cos(noise_alpha)],[sp.sin(noise_alpha)]])*noisy_delta
    
    return sp.stack([true_x,true_vx]).T, sp.stack([noisy_x,noisy_v]).T

#%% Each sample will contain a trajectory of constant veloity and varying noise distribution
#Sample random noise distributions in a given range
#TODO incorporate noise_alpha
N_SAMPLES  = BATCH_SIZE+N_PLOTS
noise_alpha  = (NOISE_ALPHA[1]-NOISE_ALPHA[0])*sp.random.rand(N_SAMPLES) + NOISE_ALPHA[0]
vxnoise_mu    = (VXNOISE_MU[1]-VXNOISE_MU[0])*sp.random.rand(N_SAMPLES) + VXNOISE_MU[0]
vxnoise_sigma = (VXNOISE_SCALE[1]-VXNOISE_SCALE[0])*sp.random.rand(N_SAMPLES)+VXNOISE_SCALE[0]
vynoise_mu    = (VYNOISE_MU[1]-VYNOISE_MU[0])*sp.random.rand(N_SAMPLES) + VYNOISE_MU[0]
vynoise_sigma = (VYNOISE_SCALE[1]-VYNOISE_SCALE[0])*sp.random.rand(N_SAMPLES)+VYNOISE_SCALE[0]
xnoise_mu1   = (XNOISE_MU1[1]-XNOISE_MU1[0])*sp.random.rand(N_SAMPLES) + XNOISE_MU1[0]
left_right = 2*((sp.random.rand(N_SAMPLES)>0.5)-0.5)
left_right[-1] = 1
left_right[-2] = -1
left_right[-3] = 1
left_right[-4] = -1
xnoise_mu2   = left_right*((XNOISE_MU2[1]-XNOISE_MU2[0])*sp.random.rand(N_SAMPLES) + XNOISE_MU2[0])
xnoise_scale1 = (XNOISE_SCALE1[1]-XNOISE_SCALE1[0])*sp.random.rand(N_SAMPLES) + XNOISE_SCALE1[0]
xnoise_scale2 = (XNOISE_SCALE2[1]-XNOISE_SCALE2[0])*sp.random.rand(N_SAMPLES) + XNOISE_SCALE2[0]

batch_generation_inputs = zip(vnoise_mu,noise_alpha,vnoise_sigma,xnoise_mu1,xnoise_mu2,xnoise_scale1,xnoise_scale2)

y_batch, x_batch = list(zip(*[gen_sample(*generator) for generator in batch_generation_inputs]))
batch_y= sp.stack(y_batch)
batch_x= sp.stack(x_batch)
print(batch_y.shape,batch_x.shape)

if True:
    plt.figure(figsize=(14,16))
    for batch_idx in range(N_PLOTS):
        noisy_x = batch_x[batch_idx,:,0]
        noisy_vx = batch_x[batch_idx,:,1]
        true_x = batch_y[batch_idx,:,0]
        true_vx = batch_y[batch_idx,:,1]
        
        plt.subplot(20+(N_PLOTS)*100 + batch_idx*2+1)
        if batch_idx == 0: plt.title('Location x')
        plt.plot(t,true_x,lw=2,label='true')
        plt.plot(t,noisy_x,lw=1,label=r'measured ($\mu =$ [%3.2f, %3.2f], $\sigma =$ [%3.2f, %3.2f])'\
                                                 %(xnoise_mu1[batch_idx],xnoise_mu2[batch_idx],\
                                                   xnoise_scale1[batch_idx],xnoise_scale2[batch_idx]))
        plt.grid(which='both')
        plt.ylabel('x[m]')
        plt.xlabel('time[s]')
        plt.legend()
        
        plt.subplot(20+(N_PLOTS)*100 + batch_idx*2+2)
        if batch_idx == 0: plt.title('Velocity x')
        plt.plot(t,true_vx,lw=2,label='true')
        plt.plot(t,noisy_vx,lw=1,label='measured')
        plt.ylabel('vx[m/s]')
        plt.xlabel('time[s]')
        plt.ylim([0,10])
        plt.grid(which='both')
        plt.legend()
        
    plt.savefig('bimodal_example_data_1D.png',dpi=200)

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
    lstm_cell =tf.nn.rnn_cell.LSTMCell(N_HIDDEN,forget_bias=0.9)
    
    #Residual weapper
    #lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)    
        
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
#%% TRAINING
train_batch_x = batch_x[:BATCH_SIZE,:,:]
train_batch_y = batch_y[:BATCH_SIZE,:,:]
test_batch_x = batch_x[BATCH_SIZE:,:,:]
test_batch_y = batch_y[BATCH_SIZE:,:,:]        
#Save losses for plotting of progress
dev_loss_plot = []
tra_loss_plot = []
lr_plot = []

with tf.Session(graph=g1) as sess:
    sess.run(init)
    itr=0
    learning_rate = LR_BASE
    while itr<ITRS:

        sess.run(opt, feed_dict={x: train_batch_x, y: train_batch_y, lr:learning_rate, batch_size: train_batch_x.shape[0]})
        lr_plot.append(learning_rate)
        
        if itr %20==0:
            learning_rate *= 0.93
            los,out=sess.run([loss,predictions],feed_dict={x:train_batch_x,y: train_batch_y,lr:learning_rate, batch_size: train_batch_x.shape[0],is_training: False})
            tra_loss_plot.append(los)
            print("For iter %i, learning rate %3.6f"%(itr, learning_rate))
            print("Loss ".ljust(12),los)
            los2,out2=sess.run([loss,predictions],feed_dict={x:test_batch_x,y: test_batch_y, batch_size:test_batch_x.shape[0],\
                               is_training: False})
            dev_loss_plot.append(los2)
            print("DEV Loss ".ljust(12),los2)
            print("_"*80)

        itr=itr+1
    dev_losses = sess.run([test_loss_summary],feed_dict={x:test_batch_x,y: test_batch_y, batch_size:test_batch_x.shape[0],\
                               is_training: False})[0]
    
    out  = sp.concatenate([out,out2],axis=0)
    
#%%
plt.figure(figsize=(14,4))
plt.subplot(121)
plt.title('Training progress ylog plot')
plt.gca().set_yscale('log')
plt.plot(range(0,ITRS,20),dev_loss_plot,label='dev loss')
plt.plot(range(0,ITRS,20),tra_loss_plot,label='train loss')
plt.xlabel('Adam iteration')
plt.ylabel('L2 fitting loss')
plt.grid(which='both')
plt.legend()
plt.subplot(122)
plt.title('Learning rate')
#plt.gca().set_yscale('log')
plt.plot(range(len(lr_plot)),lr_plot,label='Exponentially decayed to 93% every 20 iterations')
plt.xlabel('Adam iteration')
plt.ylabel('L2 fitting loss')
plt.grid(which='both')
plt.legend()
plt.savefig('training_progress1D.png',bbox_inches='tight', dpi=200)



#%% Compute the EKF results
from KalmanFilterClass import LinearKalmanFilter1D, Data1D
batch_kalman = []
deltaT = sp.mean(t[1:] - t[0:-1])

P0     = sp.identity(2)*0.01
F0     = sp.array([[1, deltaT],\
                   [0, 1]])
H0     = sp.identity(2)
Q0     = sp.diagflat([0.0001,0.00001])
R0     = sp.diagflat([1.5,0.01])

for i in range(batch_y.shape[0]):
    data = Data1D(sp.squeeze(batch_x[i,:,0]),sp.squeeze(batch_x[i,:,1]),[])
    state0 = sp.array([0, batch_x[i,0,1]]).T
    filter1b = LinearKalmanFilter1D(F0, H0, P0, Q0, R0, state0)
    kalman_data = filter1b.process_data(data)
    batch_kalman.append(sp.vstack([kalman_data.x[1:], kalman_data.vx[1:]]).T)
    
xk_batch = sp.stack(batch_kalman)
print(xk_batch.shape)
print('Kalman loss;'.ljust(12), sp.mean(pow(xk_batch[BATCH_SIZE:,:,:] - batch_y[BATCH_SIZE:,:,:],2)))
print(xk_batch.shape)
#%% Plot the fit    
plt.figure(figsize=(14,16))
N_PLOTS = 2
for batch_idx in range(BATCH_SIZE,BATCH_SIZE+N_PLOTS):
    out_xc = sp.squeeze(out[batch_idx,:,0])
    out_vxc = sp.squeeze(out[batch_idx,:,1])
    
    noisy_xc  = batch_x[batch_idx,:,0]
    noisy_vxc = batch_x[batch_idx,:,1]
    
    true_xc = batch_y[batch_idx,:,0]
    true_vxc = batch_y[batch_idx,:,1]
    
    ekf_xc  = sp.squeeze(xk_batch[batch_idx,:,0])
    ekf_vxc = sp.squeeze(xk_batch[batch_idx,:,1])
    
    x_eval = sp.linspace(-10,10,100)
    #Create a bimodal pdf
    loc1 = xnoise_mu1[batch_idx]
    loc2 = xnoise_mu2[batch_idx]
    scale1 = xnoise_scale1[batch_idx]
    scale2 = xnoise_scale2[batch_idx]
    bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1)*0.5 + \
                  pdf(x_eval, loc=loc2, scale=scale2)*0.5

    #Grab the LSTM and kalman losses for annotating 
    lstm_loss = dev_losses[batch_idx-BATCH_SIZE]
    kalman_loss = sp.mean(pow(xk_batch[batch_idx-BATCH_SIZE,:,:] - batch_y[batch_idx-BATCH_SIZE,:,:],2),axis=0)
    
    plot_idx = batch_idx-BATCH_SIZE
    plt.subplot(30+(N_PLOTS)*100 + plot_idx*3+1)
    if batch_idx == BATCH_SIZE: plt.title('Position filtering')
    plt.plot(t,true_xc,lw=2,label='True')
    plt.text(5,0,'LSTM loss:    %3.2f \nKalman loss: %3.2f'%(lstm_loss[0],kalman_loss[0]),
                                                            fontsize=12,color='white',\
                                                            bbox=dict(facecolor='green', alpha=0.8))
    plt.plot(t,noisy_xc,lw=1,label='Measured')
    plt.plot(t,ekf_xc,lw=1,label='Linear KF')
    plt.plot(t,out_xc,lw=1,label='LSTM')
    plt.grid(which='both')
    #plt.gca().equal()
    plt.ylabel('x[m]')
    plt.xlabel('time[s]')
    plt.legend()
    
    plt.subplot(30+(N_PLOTS)*100 + plot_idx*3+2)
    if batch_idx == BATCH_SIZE: plt.title('Velocity filtering (Gaussian noise)')
    plt.plot(t,true_vxc,lw=2,label='True')
    plt.plot(t,noisy_vxc,lw=1,label='Measured')
    plt.plot(t,ekf_vxc,lw=1,label='Linear KF')
    plt.plot(t,out_vxc,lw=1,label='LSTM')
    plt.text(5,0,'LSTM loss:    %3.2f\nKalman loss: %3.2f'%(lstm_loss[1],kalman_loss[1]),
                                                            fontsize=12,color='white',\
                                                            bbox=dict(facecolor='green', alpha=0.8))
    plt.ylabel('vx[m/s]')
    plt.xlabel('time[s]')
    plt.grid(which='both')
    plt.legend()
    
    plt.subplot(30+(N_PLOTS)*100 + plot_idx*3+3)
    if batch_idx == BATCH_SIZE: plt.title('Position noise distribution')
    plt.grid(which='both')
    plt.plot(x_eval,bimodal_pdf,'g', label='pdf')
    plt.fill_between(x_eval,bimodal_pdf,0,color='g',alpha=0.4)
    plt.ylim([0,0.2])
    peak1 = 0.5/(pow(2*sp.pi,0.5)*scale1)
    peak2 = 0.5/(pow(2*sp.pi,0.5)*scale2)
    side = left_right[batch_idx]
    plt.annotate('True peak', (loc1,peak1), (-5+side*5,peak1*1.2), \
                 arrowprops=dict(facecolor='black', shrink=0.005))
    plt.annotate('Multipath peak', (loc2,peak2), (1+side*1,peak2*0.7),\
                 arrowprops=dict(facecolor='black', shrink=0.005))
    plt.xlabel(r'$\Delta$ position x [m]')
    plt.legend()
    
plt.savefig('bimodal_results_example1D.png',dpi=200)