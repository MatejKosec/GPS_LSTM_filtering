from matplotlib import pyplot as plt
import scipy as sp


#%%
plt.figure(figsize=(10,8))
ax = plt.gca()
pos1 = 0.5
pos2 = 0.01
plt.text(pos1,pos2,r'$[x, \dot{x}]$'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=0.8,pad=1),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')

pos1 = 0.5
pos2 = 0.1
plt.text(pos1,pos2,'Input proj. layer \n(dim$==30$)'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=0.8,pad=1),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')


pos1 = 0.5
pos2 = 0.25
plt.text(pos1,pos2,'LSTM layer \n(dim$=30$)'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=0.8,pad=1),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')

pos1 = 0.5
pos2 = 0.35
plt.text(pos1,pos2,'Output proj. layer \n(dim$=30$)'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=0.8, pad=1),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')



#%%