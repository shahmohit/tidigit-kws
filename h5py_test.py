# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 05:09:57 2014

@author: mohit
"""

import os
import h5py
import numpy as np
import time

chunk = 10000
bsize = 100
exp = 'train'
fdir = '../data'
path = os.path.join(fdir,exp+'.hdf5')
fi = h5py.File(path,'r')
feats = fi['mfcc']
r,c = np.shape(feats)
times = []
for i in range(0,r,chunk):
    t1 = time.time()
    s = np.shape(feats[i:(i+chunk),:])
    tmp = np.zeros((s[0],c))
    feats.read_direct(tmp,np.s_[i:(i+chunk),:])
    idx = np.arange(s[0])
    np.random.shuffle(idx)
    tmp = np.take(tmp,idx,axis=0)
    print np.mean(tmp)
    t2 = time.time()
    times.append(t2-t1)
fi.close()
