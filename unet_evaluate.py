#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 13:16:23 2019

@author: nikos
"""

import numpy as np
import pandas as pd
#import matplotlib as mpl
import matplotlib.pyplot as plt
#import h5py
from keras.models import load_model
#from keras.backend import clear_session
#from scipy.stats import iqr
#import scipy.stats
#from scipy.stats import ttest_ind

def dice2D(a,b):
    #https://stackoverflow.com/questions/31273652/how-to-calculate-dice-coefficient-for-measuring-accuracy-of-image-segmentation-i
    #https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    intersection = np.sum(a[b==1])
    dice = (2*intersection)/(np.sum(a)+np.sum(b))
    if (np.sum(a)+np.sum(b))==0: #black/empty masks
        dice=1.0
    return(dice)

X_ts = np.load('./data/X_ts.npy')
Y_ts = np.load('./data/Y_ts.npy')#.astype('float32')#to match keras predicted mask

Ntest=len(X_ts)

df = pd.read_csv('./data/training_validation_test_splits.csv')
well_ts = df[df['split']=='test']['well'].tolist()
#Y_ts is a binary mask
#np.unique(Y_ts)
#array([ 0.,  1.], dtype=float32)

#%% get predicted masks for test set
model = load_model('./trained_models/unet_div8_495K.hdf5')

Y_ts_hat = model.predict(X_ts,batch_size=1)

#%% convert predicted mask to binary
threshold=0.5
Y_ts_hat[Y_ts_hat<threshold]=0
Y_ts_hat[Y_ts_hat>=threshold]=1

#%% calculate dice
dice = []
for i in range(Ntest):
    print(i)
    dice.append(dice2D(Y_ts[i,:,:,0],Y_ts_hat[i,:,:,0]))
dice = np.array(dice)

#%%
fig, axes = plt.subplots(Ntest,4,figsize=(3*3,Ntest*3))
for i in range(Ntest):
    axes[i,0].imshow(X_ts[i,:,:,0], cmap='gray')
    axes[i,0].set_xticks([])
    axes[i,0].set_yticks([])
    axes[i,0].set_title('input 1: w1')
    axes[i,0].set_ylabel(well_ts[i],{'fontsize':16})
    
    axes[i,1].imshow(X_ts[i,:,:,1], cmap='gray')
    axes[i,1].set_xticks([])
    axes[i,1].set_yticks([])
    axes[i,1].set_title('input 2: w2')
    
    axes[i,2].imshow(Y_ts[i,:,:,0], cmap='gray')
    axes[i,2].set_xticks([])
    axes[i,2].set_yticks([])
    axes[i,2].set_title('True Mask')
    
    axes[i,3].imshow(Y_ts_hat[i,:,:,0], cmap='gray')
    axes[i,3].set_xticks([])
    axes[i,3].set_yticks([])
    axes[i,3].set_title('Predicted Mask')
plt.savefig('test_set_all.png',bbox_inces='tight',dpi=100)

#%%
fig, axes = plt.subplots(Ntest,2,figsize=(4*3,Ntest*4))
for i in range(Ntest):

    axes[i,0].imshow(Y_ts[i,:,:,0], cmap='gray')
    axes[i,0].set_xticks([])
    axes[i,0].set_yticks([])
    axes[i,0].set_title('True',{'fontsize':16})
    axes[i,0].set_ylabel(well_ts[i],{'fontsize':16})
    
    axes[i,1].imshow(Y_ts_hat[i,:,:,0], cmap='gray')
    axes[i,1].set_xticks([])
    axes[i,1].set_yticks([])
    axes[i,1].set_title('Predicted, dice='+str(np.round(dice[i],2)),{'fontsize':16})
plt.savefig('test_set_predictions.png',bbox_inces='tight',dpi=100)

print('mean dice:', dice.mean())
#0.9023061252373422
print('median dice:', np.median(dice))
#0.9082371504531912

#%% now do a boxplot

plt.figure(figsize=(6,6))
plt.boxplot(dice)
plt.title('Test set performance',{'fontsize':16})
plt.ylabel('Dice score',{'fontsize':16})
plt.xticks([])
plt.savefig('test_set_dice.png',bbox_inces='tight',dpi=100)



