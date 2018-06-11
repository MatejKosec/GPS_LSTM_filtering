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
LR_BASE = 1e-2
BATCH_SIZE = 100
ITRS = 800
REG = 1e-2
DROPOUT1= 0.1

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
        x_eval_space = sp.linspace(xmin,xmax,npts)
        y_eval_space = sp.linspace(xmin,xmax,npts)
        if plot: print('Done with linspace')
        x_eval,y_eval = sp.meshgrid(x_eval_space,y_eval_space)
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
        bimodal_cdf[-1,-1]=1
        bimodal_cdf[-1,0]=0
        self.ppfx = interpolate.interp1d(bimodal_cdf[-1,:],x_eval_space)
        self.ppfix = interpolate.interp1d(x_eval_space,sp.arange(npts))
        if plot: print('Done building interpolator')
        
        #Store the data
        self.x_eval = x_eval
        self.y_eval = y_eval
        self.x_eval_space = x_eval_space
        self.y_eval_space = y_eval_space
        self.bimodal_pdf = bimodal_pdf
        self.bimodal_cdf = bimodal_cdf
        return 
        
    #Sample the distribution for any given shape of input array (same as rand function)
    #ppf is an interpolation (approximate)
    def sample(self, *shape):
        bimodal_partial_cdf= cumtrapz(self.bimodal_pdf,initial=0,axis=0)
        #First sample in the x coordinate
        samplesx  = self.ppfx(sp.random.rand(*shape))
        #Next sample in the ybin
        bin_index = self.ppfix(samplesx)
        
        #compute samples inside the ybin   
        def compute_sample(ysample,xsample,binindex):
            upper_index = sp.int32(sp.ceil(binindex))
            lower_index = sp.int32(sp.floor(binindex))
            
            ppy_upper = interpolate.interp1d(bimodal_partial_cdf[:,upper_index],self.y_eval_space)
            ppy_lower = interpolate.interp1d(bimodal_partial_cdf[:,lower_index],self.y_eval_space)
            
            a = bimodal_partial_cdf[:,upper_index]
            b = bimodal_partial_cdf[:,lower_index]
            
            samples_upper = ppy_upper(ysample*(max(a)-min(a))*0.9999 + min(a)*1.001)
            samples_lower = ppy_lower(ysample*(max(b)-min(b))*0.9999 + min(b)*1.001)
            
            #Lerp over the lower and upper
            a = self.x_eval_space[upper_index]
            b = self.x_eval_space[lower_index]
            
            return samples_lower + (samples_upper-samples_lower)/(a-b)*(xsample-b)
        
        #Vectorize and sample in ybin
        samplesy = sp.random.rand(*shape)
        compute_samples = sp.vectorize(compute_sample)
        samplesy = compute_samples(samplesy,samplesx,bin_index)
        
        #Stack the values
        samples = sp.stack([samplesx,samplesy])
        return samples.T
    

    
#Example of distribution
if __name__ == "__main__":        
    loc1 = [0.0,0.0]
    scale1 = [0.8,0.5]
    loc2 = [4,3]
    scale2 = [0.8,0.5]
    noise_dist = bimodal_gaussian_2D(loc1,loc2,scale1,scale2,-8,8,80,plot=True)

    
#%%
if __name__ == "__main__":        
    x_eval = noise_dist.x_eval
    y_eval = noise_dist.y_eval
    bimodal_pdf = noise_dist.bimodal_pdf
    bimodal_cdf = noise_dist.bimodal_cdf
    plt.figure(figsize=(9,6))
    plt.title('2D Bimodal distribution example')
    plt.contour(x_eval,y_eval, bimodal_pdf,sp.logspace(-6,0,20))
    plt.colorbar()
    a = noise_dist.sample(500)
    plt.scatter(a[:,0],a[:,1],label='Example samples')
    plt.annotate('True location peak', loc1, [i-2.6 for i in loc1],\
                 arrowprops=dict(facecolor='black', shrink=0.005),
                 bbox=dict(facecolor='white', alpha=0.8))
    plt.annotate('Multipath location peak', loc2, [i-2.6 for i in loc2],\
                 arrowprops=dict(facecolor='black', shrink=0.005),
                 bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(which='both')
    plt.legend()
    plt.ylabel('$\Delta$ y [m] ')
    plt.xlabel('$\Delta$ x [m] ')
    
    plt.savefig('bimodal_distribution_example_2D.png',bbox_inches='tight',dpi=100)

#%% Generate true labels and noisy data for a given time-frame
t = sp.linspace(0,10,N_TIME)
def gen_sample(v,vnoise_sigma, xnoise_mu1,xnoise_mu2, xnoise_sigma1,xnoise_sigma2):
    vx = v[0]
    vy = v[1]
    true_vy = [vx for _ in t[:N_TIME//4]]+[0.0]*(N_TIME//4)+[-vx for _ in t[N_TIME//2:3*N_TIME//4]] +[0.0]*(N_TIME//4)
    true_vx = [0.0]*(N_TIME//4)+[vy for _ in t[N_TIME//4:N_TIME//2]] + [0.0]*(N_TIME//4) + [-vy for _ in t[3*N_TIME//4:]]
    true_v  = sp.vstack([true_vx,true_vy])
    true_x  = cumtrapz(true_v,t,initial=0)
    
    #Velocity only has Gaussian noise
    noisy_v = true_v+sp.random.randn(*true_v.shape)*sp.reshape(vnoise_sigma,[2,1])
    
    #Position has bimodal noise that keeps constant orientation (as if building is throwing a shadow)
    noise_dist = bimodal_gaussian_2D(xnoise_mu1,xnoise_mu2,xnoise_sigma1,xnoise_sigma2,-10,10,150)
    noisy_x  = true_x + noise_dist.sample(*t.shape).T #1D samples
    print(noisy_x.shape,noisy_v.shape)
    return sp.vstack([true_x,true_v]).T, sp.vstack([noisy_x,noisy_v]).T


#%% Each sample will contain a trajectory of constant veloity and varying noise distribution
#Sample random noise distributions in a given range
#TODO incorporate noise_alpha
N_SAMPLES  = BATCH_SIZE+N_PLOTS
noise_alpha  = (NOISE_ALPHA[1]-NOISE_ALPHA[0])*sp.random.rand(N_SAMPLES,2) + NOISE_ALPHA[0]
vnoise_mu    = (VNOISE_MU[1]-VNOISE_MU[0])*sp.random.rand(N_SAMPLES,2) + VNOISE_MU[0]
vnoise_sigma = (VNOISE_SCALE[1]-VNOISE_SCALE[0])*sp.random.rand(N_SAMPLES,2)+VNOISE_SCALE[0]
xnoise_mu1   = (XNOISE_MU1[1]-XNOISE_MU1[0])*sp.random.rand(N_SAMPLES,2) + XNOISE_MU1[0]
left_right = 2*((sp.random.rand(N_SAMPLES,2)>0.5)-0.5)
xnoise_mu2   = left_right*((XNOISE_MU2[1]-XNOISE_MU2[0])*sp.random.rand(N_SAMPLES,2) + XNOISE_MU2[0])
xnoise_scale1 = (XNOISE_SCALE1[1]-XNOISE_SCALE1[0])*sp.random.rand(N_SAMPLES,2) + XNOISE_SCALE1[0]
xnoise_scale2 = (XNOISE_SCALE2[1]-XNOISE_SCALE2[0])*sp.random.rand(N_SAMPLES,2) + XNOISE_SCALE2[0]

batch_generation_inputs = zip(vnoise_mu,vnoise_sigma,xnoise_mu1,xnoise_mu2,xnoise_scale1,xnoise_scale2)

y_batch, x_batch = list(zip(*[gen_sample(*generator) for generator in batch_generation_inputs]))
batch_y= sp.stack(y_batch)
batch_x= sp.stack(x_batch)
print(batch_y.shape,batch_x.shape)

if False:
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
        
    plt.savefig('bimodal_example_data_2D.png',dpi=200)

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
plt.savefig('training_progress2D.png',bbox_inches='tight', dpi=200)



#%% Compute the EKF results
from KalmanFilterClass import LinearKalmanFilter2D, Data
batch_kalman = []
for i in range(batch_y.shape[0]):
    deltaT = sp.mean(t[1:] - t[0:-1])
    state0 = sp.squeeze(batch_x[i,0,:])
    state0[0] = 0
    state0[1] = 0
    P0     = sp.identity(4)*0.0001
    F0     = sp.array([[1, 0, deltaT, 0],\
                       [0, 1, 0, deltaT],\
                       [0, 0, 1, 0],\
                       [0, 0, 0, 1]])
    H0     = sp.identity(4)
    Q0     = sp.diagflat([0.0001,0.0001,0.1,0.1])
    R0     = sp.diagflat([6.0,6.0,1.,1.])


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
N_PLOTS = 3
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
    plt.axis('equal')
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
        
plt.savefig('bimodal_results_example2D.png',dpi=200)