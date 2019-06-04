#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 15 11:33:01 2019

@author: nikos
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import h5py
from keras.preprocessing import image# for RGB images
import os
#import imageio
from sklearn.model_selection import train_test_split
import cv2# cv2.imread() for grayscale images

import matplotlib.pyplot as plt
from mpl_toolkits import axes_grid1
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

#%% load the images
img_folder = './data/BBBC010_v2_images'
msk_folder = './data/BBBC010_v1_foreground'
target_height = 400
target_width = 400
Nimages = 100#100 images, each image has 2 channels

# load the filenames of all images
# Note: delete the __MACOSX folder in the img_folder first
img_filenames = np.array(sorted(os.listdir(img_folder)))#sort to alphabetical order
assert len(img_filenames)==Nimages*2#2 channels

wells = [f.split('_')[6] for f in img_filenames]
wells = np.sort(np.unique(wells))#e.g. A01, A02, ..., E04
channels = [1,2]

#%%load the images
#images, 2 channels
X = np.zeros(shape=(Nimages,target_height,target_width,2),dtype='float32')
Y = np.zeros(shape=(Nimages,target_height,target_width,1),dtype='float32')

i=0
for w in  wells:
    print('loading image ',i+1)
    for c in channels:
        key = w+'_w'+str(c)
        img_file = None
        for f in img_filenames:
            if key in f:
                img_file=f
                break;
        print(img_file)
        #cv2 is better for grayscale images, use 
        #load the image
        img = cv2.imread(img_folder+'/'+img_file,-1)
        #resize
        img=cv2.resize(img,(target_width,target_height))
        #normalize to 0-1
        img=img/img.max()
        X[i,:,:,c-1]=img
    print('loading mask')
    img = cv2.imread(msk_folder+'/'+w+'_binary.png',cv2.IMREAD_GRAYSCALE)
    #resize
    img=cv2.resize(img,(target_width,target_height))
    #normalize to 0-1
    img=img/img.max()
    #create binary image from [0,1] to {0,1}, using 0.5 as threshold
    img[img<0.5]=0
    img[img>=0.5]=1
    Y[i,:,:,0]=img
    i=i+1
    print()#add a blank line for readability

#double-check that the masks are binary
assert np.array_equal(np.unique(Y), [0,1])
    
#%% plot as a sanity check

#plot channel 0
img=0    
fig, axes = plt.subplots(10,10)
for i in range(10):
    for j in range(10):
        axes[i,j].imshow(X[img,:,:,0],cmap='gray')
        axes[i,j].set_title(wells[img])
        img=img+1

#plot channel 1
img=0    
fig, axes = plt.subplots(10,10)
for i in range(10):
    for j in range(10):
        axes[i,j].imshow(X[img,:,:,1],cmap='gray')
        axes[i,j].set_title(wells[img])
        img=img+1   

#%%
i=4
plt.figure(figsize=(3*6,6))
plt.subplot(1,3,1)
im=plt.imshow(X[i,:,:,0],cmap='gray')
add_colorbar(im)
plt.title(wells[i]+' w1')

plt.subplot(1,3,2)
im=plt.imshow(X[i,:,:,1],cmap='gray')
add_colorbar(im)
plt.title(wells[i]+' w2')

plt.subplot(1,3,3)
im=plt.imshow(Y[i,:,:,0],cmap='gray')
add_colorbar(im)
plt.title(wells[i]+' mask')
plt.savefig('example_image.png',dpi=100,bbox_inches='tight')

#%% split into train, validation and test sets

ix = np.arange(len(wells))

ix_tr, ix_val_ts = train_test_split(ix,train_size=60, random_state=0)
ix_val, ix_ts = train_test_split(ix_val_ts,train_size=20, random_state=0)

#sanity check, no overlap between train, validation and test sets
assert len(np.intersect1d(ix_tr,ix_val))==0
assert len(np.intersect1d(ix_tr,ix_ts))==0
assert len(np.intersect1d(ix_val,ix_ts))==0

X_tr = X[ix_tr,:]
Y_tr = Y[ix_tr,:]

X_val = X[ix_val,:]
Y_val = Y[ix_val,:]

X_ts = X[ix_ts,:]
Y_ts = Y[ix_ts,:]

fnames_tr = wells[ix_tr].tolist()
fnames_val = wells[ix_val].tolist()
fnames_ts = wells[ix_ts].tolist()

fname_split = ['train']*len(fnames_tr)+['validation']*len(fnames_val)+['test']*len(fnames_ts)
df=pd.DataFrame({'well':fnames_tr+fnames_val+fnames_ts,
              'split':fname_split})

#save to disk
df.to_csv('./data/training_validation_test_splits.csv',index=False)

np.save('./Data/X_tr.npy',X_tr)
np.save('./Data/X_val.npy',X_val)
np.save('./Data/X_ts.npy',X_ts)

np.save('./Data/Y_tr.npy',Y_tr)
np.save('./Data/Y_val.npy',Y_val)
np.save('./Data/Y_ts.npy',Y_ts)



