# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:19:00 2014

@author: mohit
"""
import h5py
import os
import numpy as np

exp = 'test'
fdir = '../data'
mfccpath = os.path.join(fdir,exp+'.hdf5')
f = h5py.File(mfccpath,'r+')

tmp = f['mfcc']
feats = np.zeros(shape(tmp))
tmp.read_direct(feats)
tmp = f['spkrid']
spkrs = np.zeros(shape(tmp)[0])
tmp.read_direct(spkrs)

spkrs = spkrs.astype('int')
num_spkrs = np.max(spkrs)+1

dims = np.shape(feats)[1]

means = np.zeros((num_spkrs,dims))
stds = np.zeros((num_spkrs,dims))

for i in range(num_spkrs):
    idx = np.where(spkrs==i)[0]
    tmp = np.take(feats,idx,axis=0)
    means[i,:] = np.mean(tmp,axis=0)
    stds[i,:] = np.std(tmp,axis=0)

data = f.create_dataset('spkr_mean', np.shape(means))
data[:,:] = means
data = f.create_dataset('spkr_std', np.shape(stds))
data[:,:] = stds
f.close()
