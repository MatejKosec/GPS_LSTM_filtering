from matplotlib import pyplot as plt
import scipy as sp


#%%
plt.figure(figsize=(10,8))
ax = plt.gca()
pos1 = 0.5
pos2 = 0.00
plt.text(pos1,pos2,r'$[x, \dot{x}]$'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=1,pad=5),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')

pos1 = 0.5
pos2 = 0.12
plt.text(pos1,pos2,'Dense layer \n(dim$==30$)'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=1,pad=5),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')


pos1 = 0.5
pos2 = 0.24
plt.text(pos1,pos2,'LSTM layer \n(dim$=30$)'.center(15),fontsize=12,color='white',\
         bbox=dict(facecolor='grey', alpha=1,pad=5),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')

pos1 = 0.5
pos2 = 0.36
plt.text(pos1,pos2,'Dense layer \n(dim$=30$)'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=1, pad=5),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')

pos1 = 0.5
pos2 = 0.48
plt.text(pos1,pos2,'Dense layer \n(dim$=30$)'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=1, pad=5),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')

pos1 = 0.5
pos2 = 0.60
plt.text(pos1,pos2,'Dense layer \n(dim$=30$)'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=1, pad=5),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')

pos1 = 0.5
pos2 = 0.80
plt.text(pos1,pos2,'L2 loss'.center(15),fontsize=12,color='black',\
         bbox=dict(facecolor='white', alpha=1, pad=5),
         transform=ax.transAxes,
         verticalalignment='bottom', horizontalalignment='center')
plt.vlines([0.5],[0],[0.6] )
ax.axis('off')
plt.savefig('architecture_plot.png',dpi=300,bbox_inches='tight')

#%%