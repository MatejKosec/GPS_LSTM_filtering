from __future__ import print_function, division
import scipy as sp 
from scipy import stats
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt

#%%
N_TIME= 100

#%%

#Import normal distribution
pdf = sp.stats.norm.pdf
cdf = sp.stats.norm.cdf

#Sample spacec for plotting
x_eval = sp.linspace(-5,5,100)
loc1 = 0.0
scale1 = 0.3
loc2 = 2
scale2 = 0.5

#Create 
from scipy.stats import rv_continuous
class bimodal_gaussian(rv_continuous):
    def __init__(self,loc1,loc2,scale1,scale2,name):
        self.loc1 = loc1
        self.loc2 = loc2
        self.scale1 = scale1
        self.scale2 = scale2
        #Call constructor of parent class
        super(bimodal_gaussian,self).__init__(name=name)
        
    def _pdf(self, x):
        value = pdf(x,loc=self.loc1,scale=self.scale1)*0.5+\
                pdf(x,loc=self.loc2,scale=self.scale2)*0.5
        return value
    

bimodal = bimodal_gaussian(loc1,loc2,scale1,scale2,name='bimodal_example')


#Create a bimodal pdf
bimodal_pdf = pdf(x_eval, loc=loc1, scale=scale1)*0.5 + \
              pdf(x_eval, loc=loc2, scale=scale2)*0.5
              
bimodal_cdf = cdf(x_eval, loc=loc1, scale=scale1)*0.5 + \
              cdf(x_eval, loc=loc2, scale=scale2)*0.5


plt.plot(x_eval,bimodal_pdf)
plt.plot(x_eval,bimodal_cdf)

#Generate some ranodm samples
samples = sp.random.rand(100)
samples = bimodal.ppf(samples)


plt.ylim([0,1])
plt.grid(which='both')


#%% Generate a sample
t = sp.linspace(0,10,N_TIME)
def gen_sample(f, vnoise, xnoise):
    true_vx = f(t)
    true_x  = cumtrapz(true_vx,t)
    true_x  = sp.hstack([[0],true_x])
    
    noisy_vx = true_vx+sp.random.randn(*t.shape)*vnoise
    noisy_x  = true_x +sp.random.randn(*t.shape)*xnoise
    noisy_x  = noisy_x
    
    return sp.stack([true_x,true_vx]).T, sp.stack([noisy_x,noisy_vx]).T

#%%
y_batch, x_batch = list(zip(*[gen_sample(f, 0.3, 0.2) for f in [sp.sin,lambda x: sp.cos(x)+1,lambda x: 0.5*x/max(t),lambda x: 0.25*(sp.sin(x)+sp.cos(x)**2)]]))
batch_y= sp.stack(y_batch)
batch_x= sp.stack(x_batch)
print(batch_y.shape,batch_x.shape)
