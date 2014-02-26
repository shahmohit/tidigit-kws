# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 21:19:00 2014

@author: mohit
"""
import h5py
import os
import numpy as np

left = 15
right = 15

dims = 13
input_size = (left*dims) + dims + (right*dims)

exp = 'test'
fdir = '../data'
mfccpath = os.path.join(fdir,exp+'.hdf5')
f = h5py.File(mfccpath,'r')

tmp = f['mfcc']
feats = np.zeros(np.shape(tmp))
tmp.read_direct(feats)

tmp = f['segments']
boundary = np.zeros(np.shape(tmp)[0])
tmp.read_direct(boundary)
boundary = boundary.astype('int')

tmp = f['spkrid']
spkrs = np.zeros(np.shape(tmp)[0])
tmp.read_direct(spkrs)
spkrs = spkrs.astype('int')
num_spkrs = np.max(spkrs) + 1

tmp = f['uttid']
utts = np.zeros(np.shape(tmp)[0])
tmp.read_direct(utts)
utts = utts.astype('int')
num_utts = np.max(utts) + 1

tmp = f['spkr_mean']
means = np.zeros(np.shape(tmp))
tmp.read_direct(means)

tmp = f['spkr_std']
stds = np.zeros(np.shape(tmp))
tmp.read_direct(stds)

tmp = f['global_mean']
gmean = np.zeros(np.shape(tmp))
tmp.read_direct(gmean)

tmp = f['global_std']
gstd = np.zeros(np.shape(tmp))
tmp.read_direct(gstd)

f.close()


path = os.path.join(fdir,exp+'.context.hdf5')
f = h5py.File(path,'w')
data = f.create_dataset('mfcc',(np.shape(feats)[0],input_size))

for i in range(num_utts):
    b = int(boundary[i])
    idx = np.where(utts==i)[0]
    spkrid = spkrs[idx[0]]
    tmp = np.take(feats,idx,axis=0)
    tmp = (tmp - means[spkrid,:])/stds[spkrid,:]
    tmp = (tmp - gmean)/gstd    
    frames = np.shape(tmp)[0]
    ft = np.zeros((left+right+1)).astype('int')
    ft[0:left] = 0
    ft[left] = 0
    ft[(left+1):(left+right+1)] = np.arange(1,right+1)
    ft = list(ft)
    data[b,:] = (np.take(tmp,ft,axis=0)).ravel()    
    for j in range(1,frames):
        ft.pop(0)
        r = j+right
        if (r < frames):
            ft.append(right+j)
        else:
            ft.append(frames-1)
        data[b+j,:] = (np.take(tmp,ft,axis=0)).ravel()

f.close()