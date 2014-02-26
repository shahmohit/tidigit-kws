# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:19:00 2014

@author: mohit
"""
import h5py
import os
import numpy as np

fdir = '../data'
mfccpath = os.path.join(fdir,'train.hdf5')
f = h5py.File(mfccpath,'r+')

tmp = f['mfcc']
feats = np.zeros(shape(tmp))
tmp.read_direct(feats)

tmp = f['spkrid']
spkrs = np.zeros(shape(tmp)[0])
tmp.read_direct(spkrs)

tmp = f['spkr_mean']
means = np.zeros(shape(tmp))
tmp.read_direct(means)

tmp = f['spkr_std']
stds = np.zeros(shape(tmp))
tmp.read_direct(stds)

spkrs = spkrs.astype('int')
num_spkrs = np.max(spkrs)+1

for i in range(num_spkrs):
    idx = np.where(spkrs==i)[0]
    tmp = np.take(feats,idx,axis=0)
    feats[idx,:] = (tmp - means[i,:])/stds[i,:]

globalmean = np.mean(feats,axis=0).reshape(1,-1)
globalstd = np.std(feats,axis=0).reshape(1,-1)

data = f.create_dataset('global_mean',np.shape(globalmean))
data[:,:] = globalmean 

data = f.create_dataset('global_std',np.shape(globalstd))
data[:,:] = globalstd 

f.close()

# Write global stats to test file
mfccpath = os.path.join(fdir,'test.hdf5')
f = h5py.File(mfccpath,'r+')

data = f.create_dataset('global_mean',np.shape(globalmean))
data[:,:] = globalmean 

data = f.create_dataset('global_std',np.shape(globalstd))
data[:,:] = globalstd

f.close()