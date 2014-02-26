# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:54:37 2014

@author: mohit
"""

import os
import h5py
import numpy as np
import pickle
import matplotlib.pyplot as plt

'''
def chunks(csize,prev):
    tmp = np.zeros((csize,dims))
    data.read_direct(tmp,np.s_[prev:(prev+csize),:])
    return tmp

w = np.load('../params/weights.npy')
b = np.load('../params/out.npy')
w_2 = np.load('../params/weights_layer2.npy')
b_2 = np.load('../params/out_layer2.npy')

dims,hiddens = np.shape(w)
chunksize = 10000
keyword = 1

# Test labels
path = os.path.join('../data','test.hdf5')
fi = h5py.File(path,'r')
data = fi['labels']
labels = np.zeros(np.shape(data)).astype('int')
data.read_direct(labels)

data = fi['uttid']
utt_id = np.zeros(np.shape(data)).astype('int')
data.read_direct(utt_id)
num_utts = np.max(utt_id) + 1

testlabels = np.zeros(np.shape(data)).astype('int')
idx = np.where(labels==keyword)[0]
testlabels[idx] = 1
idx = np.where(labels!=keyword)[0]
testlabels[idx] = 0

fi.close()

path = '../classifiers/'+str(keyword)+'.pkl'
fi = open(path,'rb')
clf = pickle.load(fi)
fi.close()


# Test Set Evaluation

path = os.path.join('../data','test.context.hdf5')
fi = h5py.File(path,'r')
data = fi['mfcc']                
testsize = np.shape(data)[0]

prev = 0
chunk_idx = 0
pred = []
for chunk in np.arange(0,testsize,chunksize):
    csize,_ = np.shape(data[chunk:(chunk+chunksize),:])
    chunkdata = chunks(csize,prev)
    prev += np.shape(chunkdata)[0]
    hid = 1/(1+(np.exp(-np.dot(chunkdata,w) - b.T)))        
    hid2 = 1/(1+(np.exp(-np.dot(hid,w_2) - b_2.T)))
    predtmp = clf.predict_proba(hid2)
    pred += list(predtmp)
    chunk_idx += 1

fi.close()
'''
utt = 5377
idx = np.where(utt_id==utt)[0]
p = np.take(pred,idx,axis=0)
words = np.take(labels,idx)
l = np.take(testlabels,idx)

plt.plot(l)
plt.plot(p[:,1])
'''
tstcm = np.zeros((2,2))
for p,t in zip(pred,testlabels):
    tstcm[t,p] += 1

wr = np.trace(tstcm)/np.sum(tstcm)
print wr
print np.diag(tstcm)/np.sum(tstcm,axis=1)
'''