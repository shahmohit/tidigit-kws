# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 01:05:46 2012

@author: mohit
"""

import struct
import numpy as np
import pickle
import os
import h5py

exp = 'test'
dims = 13
hdir = '../fe'

adir = '../alignments'

path = os.path.join(adir,exp)
afi = open(path,'r')
alignments = pickle.load(afi)
afi.close()

keys = alignments.keys()
keys.sort()

fdir = '../data'
mfccpath = os.path.join(fdir,exp+'.hdf5')
f = h5py.File(mfccpath,'r+')
labels = f['labels']
boundary = f['segments']
duration = f['durations']
if 'mfcc' not in f.keys():
    dataset = f.create_dataset('mfcc', (len(labels),dims))
else:
    dataset = f['mfcc']

for i in range(len(keys)):
    key = keys[i]
    fname = os.path.join(hdir,key+'.htk')
    inf = open(fname,'rb')
    tmp = inf.read(4)
    frames = list(struct.unpack('>I',tmp))[0]
    d = int(duration[i])
    b = int(boundary[i])
    inf.read(8)
    mfcc = np.zeros((frames,dims))
    for j in range(frames):
        for k in range(dims):
            tmp = inf.read(4)            
            mfcc[j,k] = float(list(struct.unpack('>f',tmp))[0])    
    inf.close()
    dataset[b:(b+d),:] = mfcc

f.close()