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
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
from keras.layers import BatchNormalization
#from keras.layers import Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.layers.merge import concatenate #Concatenate (capital C) not working 

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

filepath = 'unet_div8_495K'

#%% set-up the UNET model

#model parameters
bnorm_axis = -1
#filter sizes of the original model
nfilters = np.array([64, 128, 256, 512, 1024])

#downsize the UNET for this example.
#the smaller network is faster to train
#and produces excellent results on the dataset at hand
nfilters = (nfilters/8).astype('int')

#input
input_tensor = Input(shape=X_ts.shape[1:], name='input_tensor')

####################################
# encoder (contracting path)
####################################
#encoder block 0
e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(input_tensor)
e0 = BatchNormalization(axis=bnorm_axis)(e0)
e0 = Activation('relu')(e0)
e0 = Conv2D(filters=nfilters[0], kernel_size=(3,3), padding='same')(e0)
e0 = BatchNormalization(axis=bnorm_axis)(e0)
e0 = Activation('relu')(e0)

#encoder block 1
e1 = MaxPooling2D((2, 2))(e0)
e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
e1 = BatchNormalization(axis=bnorm_axis)(e1)
e1 = Activation('relu')(e1)
e1 = Conv2D(filters=nfilters[1], kernel_size=(3,3), padding='same')(e1)
e1 = BatchNormalization(axis=bnorm_axis)(e1)
e1 = Activation('relu')(e1)

#encoder block 2
e2 = MaxPooling2D((2, 2))(e1)
e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
e2 = BatchNormalization(axis=bnorm_axis)(e2)
e2 = Activation('relu')(e2)
e2 = Conv2D(filters=nfilters[2], kernel_size=(3,3), padding='same')(e2)
e2 = BatchNormalization(axis=bnorm_axis)(e2)
e2 = Activation('relu')(e2)

#encoder block 3
e3 = MaxPooling2D((2, 2))(e2)
e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
e3 = BatchNormalization(axis=bnorm_axis)(e3)
e3 = Activation('relu')(e3)
e3 = Conv2D(filters=nfilters[3], kernel_size=(3,3), padding='same')(e3)
e3 = BatchNormalization(axis=bnorm_axis)(e3)
e3 = Activation('relu')(e3)

#encoder block 4
e4 = MaxPooling2D((2, 2))(e3)
e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
e4 = BatchNormalization(axis=bnorm_axis)(e4)
e4 = Activation('relu')(e4)
e4 = Conv2D(filters=nfilters[4], kernel_size=(3,3), padding='same')(e4)
e4 = BatchNormalization(axis=bnorm_axis)(e4)
e4 = Activation('relu')(e4)
#e4 = MaxPooling2D((2, 2))(e4)

####################################
# decoder (expansive path)
####################################

#decoder block 3
d3=UpSampling2D((2, 2),)(e4)
d3=concatenate([e3,d3], axis=-1)#skip connection
d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
d3=BatchNormalization(axis=bnorm_axis)(d3)
d3=Activation('relu')(d3)
d3=Conv2DTranspose(nfilters[3], (3, 3), padding='same')(d3)
d3=BatchNormalization(axis=bnorm_axis)(d3)
d3=Activation('relu')(d3)

#decoder block 2
d2=UpSampling2D((2, 2),)(d3)
d2=concatenate([e2,d2], axis=-1)#skip connection
d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
d2=BatchNormalization(axis=bnorm_axis)(d2)
d2=Activation('relu')(d2)
d2=Conv2DTranspose(nfilters[2], (3, 3), padding='same')(d2)
d2=BatchNormalization(axis=bnorm_axis)(d2)
d2=Activation('relu')(d2)

#decoder block 1
d1=UpSampling2D((2, 2),)(d2)
d1=concatenate([e1,d1], axis=-1)#skip connection
d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
d1=BatchNormalization(axis=bnorm_axis)(d1)
d1=Activation('relu')(d1)
d1=Conv2DTranspose(nfilters[1], (3, 3), padding='same')(d1)
d1=BatchNormalization(axis=bnorm_axis)(d1)
d1=Activation('relu')(d1)

#decoder block 0
d0=UpSampling2D((2, 2),)(d1)
d0=concatenate([e0,d0], axis=-1)#skip connection
d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
d0=BatchNormalization(axis=bnorm_axis)(d0)
d0=Activation('relu')(d0)
d0=Conv2DTranspose(nfilters[0], (3, 3), padding='same')(d0)
d0=BatchNormalization(axis=bnorm_axis)(d0)
d0=Activation('relu')(d0)

#output
#out_class = Dense(1)(d0)
out_class = Conv2D(1, (1, 1), padding='same')(d0)
out_class = Activation('sigmoid',name='output')(out_class)

#create and compile the model
model=Model(inputs=input_tensor,outputs=out_class)
model.compile(loss={'output':'binary_crossentropy'},
              metrics={'output':'accuracy'},
              optimizer='adam')

#%% get predicted masks for test set
#model = load_model('./trained_models/'+filepath+'.hdf5')

model.load_weights('./trained_models/'+filepath+'.hdf5')
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
fig, axes = plt.subplots(Ntest,4,figsize=(4*4,Ntest*4))
for i in range(Ntest):
    axes[i,0].imshow(X_ts[i,:,:,0], cmap='gray')
    axes[i,0].set_xticks([])
    axes[i,0].set_yticks([])
    axes[i,0].set_title('input 1: w1',{'fontsize':16})
    axes[i,0].set_ylabel(well_ts[i],{'fontsize':16})
    
    axes[i,1].imshow(X_ts[i,:,:,1], cmap='gray')
    axes[i,1].set_xticks([])
    axes[i,1].set_yticks([])
    axes[i,1].set_title('input 2: w2',{'fontsize':16})
    
    axes[i,2].imshow(Y_ts[i,:,:,0], cmap='gray')
    axes[i,2].set_xticks([])
    axes[i,2].set_yticks([])
    axes[i,2].set_title('True',{'fontsize':16})
    
    axes[i,3].imshow(Y_ts_hat[i,:,:,0], cmap='gray')
    axes[i,3].set_xticks([])
    axes[i,3].set_yticks([])
    axes[i,3].set_title('Predicted, dice='+str(np.round(dice[i],2)),{'fontsize':16})
plt.savefig('./figures/test_set_all_'+filepath+'.png',bbox_inces='tight',dpi=100)

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
plt.savefig('./figures/test_set_predictions_'+filepath+'.png',bbox_inces='tight',dpi=100)

print('mean dice:', dice.mean())
#mean dice: 0.916088477076695
print('median dice:', np.median(dice))
#median dice: 0.9239170480386971
#np.save('./data/dice_'+filepath+'.npy',dice)

#%% now do a boxplot

plt.figure(figsize=(7,7))
plt.boxplot(dice)
plt.title('Test set performance',{'fontsize':16})
plt.ylabel('Dice score',{'fontsize':16})
plt.xticks([])
plt.ylim(0.8,1)
plt.savefig('./figures/test_set_dice_'+filepath+'.png',bbox_inces='tight',dpi=100)



